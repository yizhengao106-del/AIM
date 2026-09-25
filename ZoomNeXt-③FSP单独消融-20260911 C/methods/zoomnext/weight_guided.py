"""Experimental shallow-conditioned scale selection with independent FDE/edge ablations."""
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from .layers import FrequencyDetailEnhancer, EdgeReconstructionHead
from .ops import resize_to
from .zoomnext import EffB1_ZoomNeXt


class ShallowGuidedMHSIU(nn.Module):
    """Preserve MHSIU's values; add a shallow-conditioned residual to scale logits.

    Zero-initialized output projection starts with exactly the original weights.
    The guide is a feature, not a ground-truth mask, at both train and test time.
    """

    def __init__(self, core, dim, guide_dim):
        super().__init__()
        self.core = core
        hidden = max(16, dim // 4)
        self.guide_proj = nn.Conv2d(guide_dim, hidden, 1)
        self.target_proj = nn.Conv2d(dim, hidden, 1)
        self.delta = nn.Sequential(
            nn.Conv2d(2 * hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, 3 * core.num_groups, 1),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, l, m, s, guide, return_attention=False):
        c = self.core
        l = c.conv_l_pre(l)
        l = F.adaptive_max_pool2d(l, m.shape[-2:]) + F.adaptive_avg_pool2d(l, m.shape[-2:])
        s = resize_to(c.conv_s_pre(s), tgt_hw=m.shape[-2:])
        l, m, s = c.conv_l(l), c.conv_m(m), c.conv_s(s)
        lms = torch.cat([l, m, s], dim=1)
        attn = rearrange(c.conv_lms(lms), 'b (nb ng d) h w -> (b ng) (nb d) h w', nb=3, ng=c.num_groups)
        # Keep original logits and softmax over the three scales within each head.
        for layer in list(c.trans.children())[:-1]:
            attn = layer(attn)
        guide = self.guide_proj(guide)
        if guide.shape[-2] >= m.shape[-2] and guide.shape[-1] >= m.shape[-1]:
            guide = F.adaptive_avg_pool2d(guide, m.shape[-2:])
        else:
            guide = resize_to(guide, tgt_hw=m.shape[-2:])
        delta = self.delta(torch.cat([self.target_proj(m), guide], dim=1))
        delta = rearrange(delta, 'b (ng nb) h w -> (b ng) nb h w', ng=c.num_groups, nb=3)
        attn = c.trans[-1](attn + delta)
        values = rearrange(c.initial_merge(lms), 'b (nb ng d) h w -> (b ng) nb d h w', nb=3, ng=c.num_groups)
        out = rearrange((attn.unsqueeze(2) * values).sum(1), '(b ng) d h w -> b (ng d) h w', ng=c.num_groups)
        if return_attention:
            return out, rearrange(attn, '(b ng) nb h w -> b ng nb h w', ng=c.num_groups)
        return out


class EffB1_ZoomNeXt_WeightGuided(EffB1_ZoomNeXt):
    def __init__(self, pretrained, num_frames=1, input_norm=True, mid_dim=64,
                 siu_groups=4, hmu_groups=6, use_weight_guidance=True,
                 use_frequency=False, use_edge=False, guided_stages=(3, 2),
                 edge_loss_weight=0.3, **kwargs):
        super().__init__(pretrained=pretrained, num_frames=num_frames,
                         input_norm=input_norm, mid_dim=mid_dim,
                         siu_groups=siu_groups, hmu_groups=hmu_groups, **kwargs)
        self.use_weight_guidance = bool(use_weight_guidance)
        self.use_frequency = bool(use_frequency)
        self.use_edge = bool(use_edge)
        self.guided_stages = tuple(guided_stages)
        if len(set(self.guided_stages)) != len(self.guided_stages) or any(s not in (1, 2, 3, 4, 5) for s in self.guided_stages):
            raise ValueError('guided_stages must contain unique stages in 1..5')
        # Create in historical FDER order, before additional guidance parameters.
        if self.use_frequency:
            self.fde_2 = FrequencyDetailEnhancer(mid_dim)
            self.fde_1 = FrequencyDetailEnhancer(mid_dim)
        self.edge_head = EdgeReconstructionHead(mid_dim) if self.use_edge else None
        self.edge_loss_weight = edge_loss_weight
        if self.use_weight_guidance:
            for stage in self.guided_stages:
                name = f'siu_{stage}'
                setattr(self, name, ShallowGuidedMHSIU(getattr(self, name), mid_dim, self.embed_dims[0]))

    def set_loss_config(self, loss_cfg=None):
        super().set_loss_config(loss_cfg)
        self.edge_loss_weight = self.loss_cfg.get('edge_loss_weight', self.edge_loss_weight)

    def body_with_edge(self, data):
        lf = self.normalize_encoder(data['image_l'])
        mf = self.normalize_encoder(data['image_m'])
        sf = self.normalize_encoder(data['image_s'])
        x = None
        for stage in (5, 4, 3, 2, 1):
            tra = getattr(self, f'tra_{stage}')
            l, m, s = tra(lf[stage-1]), tra(mf[stage-1]), tra(sf[stage-1])
            siu = getattr(self, f'siu_{stage}')
            if self.use_weight_guidance and stage in self.guided_stages:
                lms = siu(l=l, m=m, s=s, guide=mf[0])
            else:
                lms = siu(l=l, m=m, s=s)
            if self.use_frequency and stage in (2, 1):
                lms = getattr(self, f'fde_{stage}')(lms)
            if x is not None:
                lms = lms + resize_to(x, tgt_hw=lms.shape[-2:])
            x = getattr(self, f'hmu_{stage}')(lms)
        edge = self.edge_head(x) if self.use_edge else None
        return self.predictor(x), edge

    def body(self, data):
        return self.body_with_edge(data)[0]

    def forward(self, data, iter_percentage=1, **kwargs):
        logits, edge = self.body_with_edge(data)
        if not self.training:
            return logits
        prob = logits.sigmoid()
        bce = F.binary_cross_entropy_with_logits(logits, data['mask'])
        coef = self.get_coef(iter_percentage=iter_percentage, method='cos', milestones=(0, 1))
        ual = coef * (1 - (2 * prob - 1).abs().pow(2)).mean()
        loss = bce + ual
        text = f'bce: {bce.item():.5f} powual_{coef:.5f}: {ual.item():.5f}'
        vis = dict(sal=prob)
        if edge is not None:
            edge_loss = F.binary_cross_entropy_with_logits(edge, self.mask_to_edge(data['mask']))
            loss = loss + self.edge_loss_weight * edge_loss
            text += f' edge_{self.edge_loss_weight:.2f}: {edge_loss.item():.5f}'
            vis['edge'] = edge.sigmoid()
        return dict(vis=vis, loss=loss, loss_str=text)
