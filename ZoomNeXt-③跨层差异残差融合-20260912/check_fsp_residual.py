"""Meaningful CPU checks: identity, learnability, difference conditioning, isolation."""
import torch
from mmengine import Config
from methods import EffB1_ZoomNeXt, EffB1_ZoomNeXt_FSPResidual
from methods.zoomnext.fsp_residual import DifferenceGuidedResidualFusion
from methods.zoomnext.ops import resize_to


def main():
    torch.set_num_threads(2)
    torch.manual_seed(112358)
    block = DifferenceGuidedResidualFusion(64)
    curr, prev = torch.randn(2,64,8,12), torch.randn(2,64,4,6)
    expected = curr + resize_to(prev, tgt_hw=curr.shape[-2:])
    torch.testing.assert_close(block(curr, prev), expected, rtol=0, atol=0)
    opt = torch.optim.SGD(block.parameters(), lr=.01)
    (block(curr,prev)*torch.randn_like(curr)).sum().backward()
    assert torch.count_nonzero(block.out.weight.grad), 'Zero init prevents learning'
    opt.step(); opt.zero_grad()
    y = block(curr, prev)
    assert not torch.equal(y, expected)
    y.square().mean().backward()
    for name in ('curr_proj', 'prev_proj', 'gate', 'refine'):
        grads = [p.grad for p in getattr(block,name).parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads), name
        assert any(torch.count_nonzero(g) for g in grads), name
    block.use_difference = False
    assert (block(curr,prev)-y).abs().max() > 1e-8, 'Difference has no effect after update'
    del block, y, opt
    print('PASS: zero-init identity, first-step output gradient, later upstream gradients, difference affects correction', flush=True)

    data = {f'image_{k}': torch.rand(2,3,h,w) for k,h,w in
            [('l',96,144),('m',64,96),('s',32,48)]}
    data['mask'] = (torch.rand(2,1,64,96)>.5).float()
    ref_cfg = Config.fromfile('configs/icod_train_fder_restored.py')
    for suffix in ('', '_nodiff', '_baseline'):
        cfg = Config.fromfile(f'configs/icod_train_fsp_residual{suffix}.py')
        for key in ('train','test','base_seed','pretrained','loss_cfg'):
            assert cfg[key] == ref_cfg[key], key
        base = EffB1_ZoomNeXt(pretrained=False)
        model = EffB1_ZoomNeXt_FSPResidual(pretrained=False, **cfg.model_cfg)
        model.set_loss_config(cfg.loss_cfg)
        incompatible = model.load_state_dict(base.state_dict(), strict=False)
        assert not incompatible.unexpected_keys
        assert all(k.startswith(('fusion_3.', 'fusion_2.')) for k in incompatible.missing_keys)
        assert bool(incompatible.missing_keys) == model.use_fsp_residual
        assert not any(hasattr(model,n) for n in ('sg_1','sg_2','fde_1','fde_2','edge_head','fsp_1'))
        assert model.__class__.forward is EffB1_ZoomNeXt.forward
        base.eval(); model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(data), base(data), rtol=0, atol=0)
        base.train(); model.train()
        torch.manual_seed(73)
        a = base(data, iter_percentage=.4)
        calls = {3:0, 2:0}
        handles=[]
        if model.use_fsp_residual:
            for stage in calls:
                def hook(module, inputs, output, stage=stage):
                    calls[stage] += 1
                handles.append(getattr(model,f'fusion_{stage}').register_forward_hook(hook))
        torch.manual_seed(73)
        b = model(data, iter_percentage=.4)
        for handle in handles:
            handle.remove()
        torch.testing.assert_close(a['loss'], b['loss'], rtol=0, atol=0)
        assert calls == ({3:1,2:1} if model.use_fsp_residual else {3:0,2:0})
        assert set(b['vis']) == {'sal'} and 'edge' not in b['loss_str']
        assert b['vis']['sal'].shape == data['mask'].shape and torch.isfinite(b['loss'])
        b['loss'].backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if model.use_fsp_residual:
            for stage in calls:
                assert torch.count_nonzero(getattr(model,f'fusion_{stage}').out.weight.grad)
        print('PASS:', suffix or 'difference', 'exact initial original logits/loss; finite backward; stages3,2 only; BCE+UAL', flush=True)
        del a, b, base, model
    print('PASS: all residual-fusion checks')


if __name__ == '__main__':
    main()
