"""Add shallow guidance to the recovered FEDER decoder, without altering it."""
import torch
from torch import nn

from .ops import ConvBNReLU, resize_to
from .zoomnext import EffB1_ZoomNeXt_FDER


class ShallowGuidance(nn.Module):
    """Historical SG: projected shallow encoder detail gates a decoder residual."""

    def __init__(self, guide_dim, out_dim, squeeze_ratio=4):
        super().__init__()
        hidden_dim = max(out_dim // squeeze_ratio, 16)
        self.guide_proj = ConvBNReLU(guide_dim, out_dim, 3, 1, 1)
        self.target_proj = ConvBNReLU(out_dim, out_dim, 3, 1, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(2 * out_dim, hidden_dim, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, out_dim, 1, bias=False),
            nn.Sigmoid(),
        )
        self.fuse = ConvBNReLU(out_dim, out_dim, 3, 1, 1)

    def forward(self, x, guide):
        guide = resize_to(self.guide_proj(guide), tgt_hw=x.shape[-2:])
        target = self.target_proj(x)
        gate = self.gate(torch.cat([target, guide], dim=1))
        return self.fuse(target + gate * guide) + x


class EffB1_ZoomNeXt_FDER_SG(EffB1_ZoomNeXt_FDER):
    def __init__(self, pretrained, num_frames=1, input_norm=True, mid_dim=64,
                 siu_groups=4, hmu_groups=6, use_sg=True, **kwargs):
        super().__init__(pretrained=pretrained, num_frames=num_frames,
                         input_norm=input_norm, mid_dim=mid_dim,
                         siu_groups=siu_groups, hmu_groups=hmu_groups, **kwargs)
        self.use_sg = use_sg
        if use_sg:
            self.sg_2 = ShallowGuidance(self.embed_dims[1], mid_dim)
            self.sg_1 = ShallowGuidance(self.embed_dims[0], mid_dim)

    def body_with_edge(self, data):
        l_trans_feats = self.normalize_encoder(data['image_l'])
        m_trans_feats = self.normalize_encoder(data['image_m'])
        s_trans_feats = self.normalize_encoder(data['image_s'])

        l, m, s = self.tra_5(l_trans_feats[4]), self.tra_5(m_trans_feats[4]), self.tra_5(s_trans_feats[4])
        lms = self.siu_5(l=l, m=m, s=s)
        x = self.hmu_5(lms)

        l, m, s = self.tra_4(l_trans_feats[3]), self.tra_4(m_trans_feats[3]), self.tra_4(s_trans_feats[3])
        lms = self.siu_4(l=l, m=m, s=s)
        x = self.hmu_4(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m, s = self.tra_3(l_trans_feats[2]), self.tra_3(m_trans_feats[2]), self.tra_3(s_trans_feats[2])
        lms = self.siu_3(l=l, m=m, s=s)
        x = self.hmu_3(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m, s = self.tra_2(l_trans_feats[1]), self.tra_2(m_trans_feats[1]), self.tra_2(s_trans_feats[1])
        lms = self.siu_2(l=l, m=m, s=s)
        lms = self.fde_2(lms)
        fused = lms + resize_to(x, tgt_hw=lms.shape[-2:])
        if self.use_sg:
            fused = self.sg_2(fused, m_trans_feats[1])
        x = self.hmu_2(fused)

        l, m, s = self.tra_1(l_trans_feats[0]), self.tra_1(m_trans_feats[0]), self.tra_1(s_trans_feats[0])
        lms = self.siu_1(l=l, m=m, s=s)
        lms = self.fde_1(lms)
        fused = lms + resize_to(x, tgt_hw=lms.shape[-2:])
        if self.use_sg:
            fused = self.sg_1(fused, m_trans_feats[0])
        x = self.hmu_1(fused)

        edge_logits = self.edge_head(x)
        logits = self.predictor(x)
        return logits, edge_logits
