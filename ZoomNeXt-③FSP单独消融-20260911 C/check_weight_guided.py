"""Regression and ablation checks; CPU, no data or downloaded weights."""
import copy
import torch
from mmengine import Config
from methods import EffB1_ZoomNeXt, EffB1_ZoomNeXt_FDER, EffB1_ZoomNeXt_WeightGuided
from methods.zoomnext.layers import MHSIU
from methods.zoomnext.weight_guided import ShallowGuidedMHSIU


def assert_same(reference, candidate, data):
    reference.eval()
    candidate.eval()
    with torch.no_grad():
        assert torch.equal(reference(data), candidate(data)), 'Baseline output changed'
    reference.train()
    candidate.train()
    torch.manual_seed(22)
    ref = reference(data, iter_percentage=0.4)
    torch.manual_seed(22)
    new = candidate(data, iter_percentage=0.4)
    assert torch.equal(ref['loss'], new['loss']), 'Baseline loss changed'
    for key in ref['vis']:
        assert torch.equal(ref['vis'][key], new['vis'][key])


def main():
    torch.set_num_threads(2)
    torch.manual_seed(13)
    core = MHSIU(16, 4).eval()
    guided = ShallowGuidedMHSIU(copy.deepcopy(core), 16, 8).eval()
    l, m, s = torch.randn(2, 16, 12, 18), torch.randn(2, 16, 8, 12), torch.randn(2, 16, 4, 6)
    guide = torch.randn(2, 8, 16, 24)
    y, attn = guided(l, m, s, guide, return_attention=True)
    assert torch.equal(y, core(l, m, s)), 'Zero guidance not equal to MHSIU'
    assert attn.shape == (2, 4, 3, 8, 12)
    assert (attn >= 0).all() and torch.allclose(attn.sum(2), torch.ones_like(attn[:, :, 0]))
    optimizer = torch.optim.SGD(guided.parameters(), lr=0.1)
    (y * torch.randn_like(y)).sum().backward()
    assert torch.count_nonzero(guided.delta[-1].weight.grad)
    optimizer.step()
    optimizer.zero_grad()
    guide = guide.detach().requires_grad_()
    changed = guided(l, m, s, guide)
    changed.square().mean().backward()
    assert torch.count_nonzero(guided.guide_proj.weight.grad)
    assert guide.grad is not None and torch.count_nonzero(guide.grad)
    assert not torch.allclose(changed, guided(l, m, s, torch.zeros_like(guide))), 'Guide has no effect after update'
    del guided, optimizer, y, changed

    data = {f'image_{key}': torch.rand(2, 3, h, w)
            for key, h, w in (('l', 96, 144), ('m', 64, 96), ('s', 32, 48))}
    data['mask'] = (torch.rand(2, 1, 64, 96) > 0.5).float()
    for ref_cls, fde, edge in ((EffB1_ZoomNeXt, False, False), (EffB1_ZoomNeXt_FDER, True, True)):
        ref = ref_cls(pretrained=False)
        off = EffB1_ZoomNeXt_WeightGuided(pretrained=False, use_weight_guidance=False, use_frequency=fde, use_edge=edge)
        off.load_state_dict(ref.state_dict(), strict=True)
        assert_same(ref, off, data)
        on = EffB1_ZoomNeXt_WeightGuided(pretrained=False, use_frequency=fde, use_edge=edge)
        weights = on.state_dict()
        old = ref.state_dict()
        for key in weights:
            source = key.replace('.core.', '.')
            if source in old:
                weights[key] = old[source]
        on.load_state_dict(weights, strict=True)
        assert_same(ref, on, data)
        del ref, off, on

    variants = {'baseline': (0, 0, 0), 'guidance': (1, 0, 0), 'frequency': (0, 1, 0),
                'edge': (0, 0, 1), 'fder': (0, 1, 1), 'guidance_frequency': (1, 1, 0),
                'guidance_edge': (1, 0, 1), 'full': (1, 1, 1)}
    ref_cfg = Config.fromfile('configs/icod_train_fder_restored.py')
    for name, flags in variants.items():
        cfg = Config.fromfile(f'configs/wg_{name}.py')
        for key in ('train', 'test', 'base_seed', 'pretrained', 'loss_cfg'):
            assert cfg[key] == ref_cfg[key], (name, key)
        assert tuple(cfg.model_cfg[k] for k in ('use_weight_guidance', 'use_frequency', 'use_edge')) == flags
        model = EffB1_ZoomNeXt_WeightGuided(pretrained=False, **cfg.model_cfg)
        model.set_loss_config(cfg.loss_cfg)
        assert hasattr(model, 'fde_1') == bool(flags[1])
        assert (model.edge_head is not None) == bool(flags[2])
        result = model(data, iter_percentage=0.4)
        assert result['vis']['sal'].shape == data['mask'].shape
        assert ('edge' in result['vis']) == bool(flags[2])
        assert torch.isfinite(result['loss'])
        result['loss'].backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        if flags[0]:
            assert torch.count_nonzero(model.siu_3.delta[-1].weight.grad)
        if flags[1]:
            assert any(torch.count_nonzero(p.grad) for p in model.fde_1.parameters() if p.grad is not None)
        if flags[2]:
            assert any(torch.count_nonzero(p.grad) for p in model.edge_head.parameters() if p.grad is not None)
        model.eval()
        with torch.no_grad():
            assert torch.isfinite(model(data)).all()
        print('PASS variant:', name, flags, flush=True)
        del result, model, grads
    print('PASS: exact baseline/FDER equivalence, zero-init identity, normalized head weights, guide dependence/gradients, 8 ablations')


if __name__ == '__main__':
    main()
