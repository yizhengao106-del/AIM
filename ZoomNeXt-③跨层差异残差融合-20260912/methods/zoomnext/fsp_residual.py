"""Experimental adjacent-level residual correction, inspired by feature interaction.

This is not the historical FSP block or a reproduction of the FSPNet paper.
"""
import torch
from torch import nn

from .ops import resize_to
from .zoomnext import EffB1_ZoomNeXt


class DifferenceGuidedResidualFusion(nn.Module):
    """Keep curr + resized prev; learn a gated correction conditioned on both.

    Difference is a feature cue, not a calibrated uncertainty or error estimate.
    Zero-initializing only the output projection preserves initial base behavior
    while allowing the output projection to learn on the first optimizer step.
    """
    def __init__(self, channels, use_difference=True):
        super().__init__()
        hidden = max(16, channels // 2)
        self.use_difference = bool(use_difference)
        self.curr_proj = nn.Sequential(nn.Conv2d(channels, hidden, 1),
                                       nn.GroupNorm(1, hidden), nn.GELU())
        self.prev_proj = nn.Sequential(nn.Conv2d(channels, hidden, 1),
                                       nn.GroupNorm(1, hidden), nn.GELU())
        self.gate = nn.Sequential(nn.Conv2d(3 * hidden, hidden, 1), nn.GELU(),
                                  nn.Conv2d(hidden, channels, 1), nn.Sigmoid())
        self.refine = nn.Sequential(
            nn.Conv2d(2 * hidden, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.GELU())
        self.out = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, curr, prev):
        prev = resize_to(prev, tgt_hw=curr.shape[-2:])
        c, p = self.curr_proj(curr), self.prev_proj(prev)
        difference = (c - p).abs() if self.use_difference else torch.zeros_like(c)
        gate = self.gate(torch.cat([c, p, difference], dim=1))
        correction = self.out(self.refine(torch.cat([c, p], dim=1)))
        return curr + prev + gate * correction


class EffB1_ZoomNeXt_FSPResidual(EffB1_ZoomNeXt):
    def __init__(self, pretrained, num_frames=1, input_norm=True, mid_dim=64,
                 siu_groups=4, hmu_groups=6, use_fsp_residual=True,
                 fusion_stages=(3, 2), use_difference=True, **kwargs):
        super().__init__(pretrained=pretrained, num_frames=num_frames,
                         input_norm=input_norm, mid_dim=mid_dim,
                         siu_groups=siu_groups, hmu_groups=hmu_groups, **kwargs)
        self.use_fsp_residual = bool(use_fsp_residual)
        self.fusion_stages = tuple(fusion_stages)
        self.use_difference = bool(use_difference)
        if (not self.fusion_stages or len(set(self.fusion_stages)) != len(self.fusion_stages)
                or any(s not in (1, 2, 3, 4) for s in self.fusion_stages)):
            raise ValueError('fusion_stages must be nonempty unique stages in 1..4')
        if self.use_fsp_residual:
            for stage in self.fusion_stages:
                setattr(self, f'fusion_{stage}',
                        DifferenceGuidedResidualFusion(mid_dim, self.use_difference))

    def body(self, data):
        lf = self.normalize_encoder(data['image_l'])
        mf = self.normalize_encoder(data['image_m'])
        sf = self.normalize_encoder(data['image_s'])
        x = None
        for stage in (5, 4, 3, 2, 1):
            tra = getattr(self, f'tra_{stage}')
            l, m, s = tra(lf[stage - 1]), tra(mf[stage - 1]), tra(sf[stage - 1])
            curr = getattr(self, f'siu_{stage}')(l=l, m=m, s=s)
            if x is not None:
                if self.use_fsp_residual and stage in self.fusion_stages:
                    curr = getattr(self, f'fusion_{stage}')(curr, x)
                else:
                    curr = curr + resize_to(x, tgt_hw=curr.shape[-2:])
            x = getattr(self, f'hmu_{stage}')(curr)
        return self.predictor(x)
