"""Block copied verbatim from the retained historical local FSP implementation."""
import torch
from torch import nn
from .ops import ConvBNReLU, resize_to


class FeatureShrinkageDecoderBlock(nn.Module):
    def __init__(self, in_dim, squeeze_ratio=4):
        super().__init__()
        hidden_dim = max(in_dim // squeeze_ratio, 16)
        self.curr_proj = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.prev_proj = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(2 * in_dim, hidden_dim, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, 2 * in_dim, 1, bias=False),
            nn.Sigmoid(),
        )
        self.adjacent_fuse = ConvBNReLU(2 * in_dim, in_dim, 3, 1, 1)
        self.pyramid = nn.ModuleList(
            [
                ConvBNReLU(in_dim, in_dim, 3, 1, 1),
                ConvBNReLU(in_dim, in_dim, 3, 1, 2, dilation=2),
                ConvBNReLU(in_dim, in_dim, 3, 1, 3, dilation=3),
            ]
        )
        self.pyramid_fuse = ConvBNReLU(3 * in_dim, in_dim, 1)

    def forward(self, curr, prev=None):
        curr = self.curr_proj(curr)
        if prev is not None:
            prev = resize_to(prev, tgt_hw=curr.shape[-2:])
            prev = self.prev_proj(prev)
            curr_gate, prev_gate = self.gate(torch.cat([curr, prev], dim=1)).chunk(2, dim=1)
            curr = self.adjacent_fuse(torch.cat([curr * curr_gate, prev * prev_gate], dim=1))

        pyramid = torch.cat([branch(curr) for branch in self.pyramid], dim=1)
        return self.pyramid_fuse(pyramid) + curr
