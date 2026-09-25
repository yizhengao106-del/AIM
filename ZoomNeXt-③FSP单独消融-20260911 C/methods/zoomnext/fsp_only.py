"""Historical FSP-inspired decoder on original ZoomNeXt; no SG/FDE/edge/MSCA."""
from .zoomnext import EffB1_ZoomNeXt
from .fsp_legacy_block import FeatureShrinkageDecoderBlock
from .ops import resize_to


class EffB1_ZoomNeXt_FSP_Only(EffB1_ZoomNeXt):
    def __init__(self, pretrained, num_frames=1, input_norm=True, mid_dim=64,
                 siu_groups=4, hmu_groups=6, use_fsp=True, **kwargs):
        super().__init__(pretrained=pretrained, num_frames=num_frames,
                         input_norm=input_norm, mid_dim=mid_dim,
                         siu_groups=siu_groups, hmu_groups=hmu_groups, **kwargs)
        self.use_fsp = bool(use_fsp)
        if self.use_fsp:
            for stage in (5, 4, 3, 2, 1):
                setattr(self, f'fsp_{stage}', FeatureShrinkageDecoderBlock(mid_dim))

    def body(self, data):
        lf = self.normalize_encoder(data['image_l'])
        mf = self.normalize_encoder(data['image_m'])
        sf = self.normalize_encoder(data['image_s'])
        x = None
        for stage in (5, 4, 3, 2, 1):
            tra = getattr(self, f'tra_{stage}')
            l, m, s = tra(lf[stage-1]), tra(mf[stage-1]), tra(sf[stage-1])
            curr = getattr(self, f'siu_{stage}')(l=l, m=m, s=s)
            if self.use_fsp:
                curr = getattr(self, f'fsp_{stage}')(curr, x)
            elif x is not None:
                curr = curr + resize_to(x, tgt_hw=curr.shape[-2:])
            x = getattr(self, f'hmu_{stage}')(curr)
        return self.predictor(x)
