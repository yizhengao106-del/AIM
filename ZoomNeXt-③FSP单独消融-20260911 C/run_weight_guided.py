"""Run ONE named ablation, freezing source/config/data paths before training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['baseline', 'guidance', 'frequency', 'edge',
                        'fder', 'guidance_frequency', 'guidance_edge', 'full'], required=True)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--data-cfg', default='dataset.yaml')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    # Set visibility before any import that could initialize CUDA.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['MPLBACKEND'] = 'Agg'
    import torch
    import yaml
    from mmengine import Config
    from methods import EffB1_ZoomNeXt_WeightGuided
    assert torch.cuda.is_available(), 'Requested CUDA device unavailable'
    cfg = Config.fromfile(f'configs/wg_{args.variant}.py')
    assert cfg.model_name == 'EffB1_ZoomNeXt_WeightGuided'
    assert not cfg.get('load_from'), 'Ablations should start from the same encoder initialization'
    expected = {
        'baseline': (0, 0, 0), 'guidance': (1, 0, 0), 'frequency': (0, 1, 0),
        'edge': (0, 0, 1), 'fder': (0, 1, 1), 'guidance_frequency': (1, 1, 0),
        'guidance_edge': (1, 0, 1), 'full': (1, 1, 1),
    }[args.variant]
    model = EffB1_ZoomNeXt_WeightGuided(pretrained=False, **cfg.model_cfg)
    actual = (model.use_weight_guidance, model.use_frequency, model.use_edge)
    assert actual == expected, (actual, expected)
    print('MODEL:', type(model).__name__, 'VARIANT:', args.variant, 'FLAGS (guidance/frequency/edge):', actual, flush=True)
    print('VISIBLE GPU:', args.gpu, torch.cuda.get_device_name(0), flush=True)
    del model
    with open(args.data_cfg, encoding='utf-8') as f:
        datasets = yaml.safe_load(f)
    counts = {}
    for name in cfg.train.data.names + cfg.test.data.names:
        item = datasets[name]
        item['root'] = str(Path(item['root']).expanduser().resolve())
        names = {}
        for kind in ('image', 'mask'):
            folder = Path(item['root']) / item[kind]['path']
            assert folder.is_dir(), f'Missing {folder}'
            names[kind] = {p.stem for p in folder.glob('*' + item[kind]['suffix']) if p.is_file()}
            assert names[kind], f'No {kind} files: {folder}'
        counts[name] = len(names['image'] & names['mask'])
        assert counts[name] > 0, f'No pairs: {name}'
        print('DATA:', name, 'images', len(names['image']), 'masks', len(names['mask']), 'pairs', counts[name], flush=True)
        if names['image'] != names['mask']:
            print('DATA WARNING: unmatched files; existing trainer uses the intersection. This is not proof of a valid standard split.', flush=True)
    total = sum(counts[n] for n in cfg.train.data.names)
    if total != 4040:
        print(f'DATA WARNING: {total} training pairs; inherited schedule was configured for 4040. Preserved for comparison, not corrected silently.', flush=True)
    subprocess.run([sys.executable, 'check_weight_guided.py'], check=True)
    runs = root / 'weight_guided_runs'
    runs.mkdir(exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%d_%H%M%S_') + args.variant + '_', dir=runs))
    source = run / 'source'
    source.mkdir()
    for folder in ('methods', 'utils', 'configs'):
        shutil.copytree(root / folder, source / folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('main_for_image.py', 'requirements.txt', 'run_weight_guided.py', 'check_weight_guided.py'):
        shutil.copy2(root / name, source / name)
    cfg.dump(str(source / 'selected_config.py'))
    (source / 'dataset.yaml').write_text(yaml.safe_dump(datasets, allow_unicode=True), encoding='utf-8')
    hashes = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source.rglob('*') if p.is_file()}
    (run / 'source.sha256.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    (run / 'run.json').write_text(json.dumps(dict(variant=args.variant, model=cfg.model_name,
        flags=cfg.model_cfg.to_dict(), gpu=args.gpu, dataset_pairs=counts,
        torch=torch.__version__), indent=2), encoding='utf-8')
    with (run / 'environment.txt').open('w') as f:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
    (root / 'last_weight_guided_run.txt').write_text(str(run), encoding='utf-8')
    print('SNAPSHOT/RESULTS:', run, flush=True)
    cmd = [sys.executable, '-u', 'main_for_image.py', '--config', 'selected_config.py',
           '--model-name', cfg.model_name, '--data-cfg', 'dataset.yaml',
           '--output-dir', str(run / 'results'), '--info', f'wg_{args.variant}']
    if cfg.pretrained:
        cmd.append('--pretrained')
    subprocess.run(cmd, cwd=source, check=True)


if __name__ == '__main__':
    main()
