"""CPU checks for the standalone historical FSP ablation."""
import torch
from mmengine import Config
from methods import EffB1_ZoomNeXt, EffB1_ZoomNeXt_FSP_Only


def main():
    torch.set_num_threads(2)
    torch.manual_seed(112358)
    cfg = Config.fromfile('configs/icod_train_fsp_only.py')
    base_cfg = Config.fromfile('configs/icod_train_fder_restored.py')
    for key in ('train', 'test', 'base_seed', 'pretrained', 'loss_cfg'):
        assert cfg[key] == base_cfg[key], key
    assert cfg.model_name == 'EffB1_ZoomNeXt_FSP_Only' and cfg.model_cfg == dict(use_fsp=True)
    data = {f'image_{k}': torch.rand(2, 3, h, w) for k,h,w in
            [('l',96,144), ('m',64,96), ('s',32,48)]}
    data['mask'] = (torch.rand(2,1,64,96) > .5).float()
    base = EffB1_ZoomNeXt(pretrained=False)
    off = EffB1_ZoomNeXt_FSP_Only(pretrained=False, use_fsp=False)
    off.load_state_dict(base.state_dict(), strict=True)
    base.eval(); off.eval()
    with torch.no_grad():
        torch.testing.assert_close(base(data), off(data), rtol=0, atol=0)
    base.train(); off.train()
    torch.manual_seed(99)
    a = base(data, iter_percentage=.4)
    torch.manual_seed(99)
    b = off(data, iter_percentage=.4)
    torch.testing.assert_close(a['loss'], b['loss'], rtol=0, atol=0)
    del base, off, a, b
    model = EffB1_ZoomNeXt_FSP_Only(pretrained=False, **cfg.model_cfg)
    model.set_loss_config(cfg.loss_cfg)
    assert model.__class__.forward is EffB1_ZoomNeXt.forward
    assert not any(hasattr(model,n) for n in ('sg_1','sg_2','fde_1','fde_2','edge_head'))
    calls = {s:0 for s in (5,4,3,2,1)}
    handles=[]
    for stage in calls:
        def hook(module, inputs, output, stage=stage):
            calls[stage] += 1
        handles.append(getattr(model,f'fsp_{stage}').register_forward_hook(hook))
    result = model(data, iter_percentage=.4)
    assert all(n == 1 for n in calls.values()), calls
    for handle in handles:
        handle.remove()
    assert result['vis']['sal'].shape == data['mask'].shape
    assert set(result['vis']) == {'sal'} and 'edge' not in result['loss_str']
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    for stage in calls:
        grads=[p.grad for p in getattr(model,f'fsp_{stage}').parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g) for g in grads)
    model.eval()
    with torch.no_grad():
        output=model(data)
        assert output.shape == data['mask'].shape and torch.isfinite(output).all()
    print('PASS: unchanged training settings; FSP-off exact original logits/loss; FSP-only 5 stages once each; finite gradients and inference; original BCE+UAL; no SG/FDE/edge')


if __name__ == '__main__':
    main()
