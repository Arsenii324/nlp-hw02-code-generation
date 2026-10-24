"""Freeze allowlisted TPU probe source and private metadata; never upload/launch."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CACHE_SHA = '41efd8551f4e9d679a4f06b4e6f86163c15af52c5f6711deacd765b7c06d9a9b'
MEMBERS = ('src/nlp_hw02/__init__.py', 'src/nlp_hw02/tpu_pilot.py',
           'src/nlp_hw02/model.py', 'src/nlp_hw02/tokenizer.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage(tools, kernel, dataset_id, kernel_id, accelerator, weights, steps=20000, chunk=5):
    tools, kernel = Path(tools), Path(kernel)
    for path in (tools, kernel):
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise RuntimeError('Preserve existing staging: '+str(path))
    if tools.resolve() == kernel.resolve():
        raise ValueError('Tools and kernel directories must differ')
    if not accelerator.startswith('Tpu') or any(len(i.split('/')) != 2 for i in (dataset_id, kernel_id)):
        raise ValueError('Explicit TPU accelerator and owner/slug IDs required')
    for identity in (dataset_id, kernel_id):
        slug = identity.split('/')[1]
        if not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*', slug) or len(slug) > 50:
            raise ValueError('Use lowercase hyphenated slug <=50 chars')
    # Kaggle can normalize a new kernel ID from title. Preserve the exact slug.
    kernel_title = kernel_id.split('/')[1].replace('-', ' ')
    dataset_title = dataset_id.split('/')[1].replace('-', ' ')
    if not 20 <= steps <= 20000 or not 1 <= chunk <= 20 or steps % chunk:
        raise ValueError('Require20..20000 steps divisible by chunk1..20')
    for name in MEMBERS:
        p = ROOT/name
        if not p.is_file() or p.is_symlink():
            raise RuntimeError('Missing or symlinked source: '+name)
        compile(p.read_text(), str(p), 'exec')
    driver = ROOT/'scripts/tpu_pilot_driver.py'
    compile(driver.read_text(), str(driver), 'exec')
    tools.mkdir(parents=True, exist_ok=True); kernel.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tools/'tpu_source.bin', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in MEMBERS:
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 2, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (ROOT/name).read_bytes())
    shutil.copyfile(driver, kernel/'run.py')
    # Audit the real trained checkpoint; never create synthetic training inputs.
    import sys
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT/'src'))
    from nlp_hw02.model import ModelConfig, Seq2Seq
    weights = Path(weights)
    expected = 'd09a8611bd6ef8ab2d0711d6999aa419efe50454d1bb12252069c59bc20c8a85'
    if digest(weights) != expected:
        raise RuntimeError('Require verified immutable B034 selected weights')
    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    model = Seq2Seq(ModelConfig(**config))
    model.load_state_dict(checkpoint['model'], strict=True)
    arrays = {n: v.detach().cpu().float().numpy() for n, v in model.state_dict().items()}
    if sum(a.size for a in arrays.values()) != 17524256 or not all(np.isfinite(a).all() for a in arrays.values()):
        raise RuntimeError('Invalid B034 weights')
    if checkpoint['step'] != 300000 or not config['learnable_xsa']:
        raise RuntimeError('Unexpected B034 recipe')
    np.savez(tools/'initial.npz', **arrays)
    (tools/'initial.json').write_text(json.dumps({'config': config, 'step': checkpoint['step'],
        'source_torch_sha256': expected, 'weights_sha256': digest(tools/'initial.npz'),
        'lineage': 'B034 selected300k', 'optimizer_moments_reset': True}, indent=2))
    manifest = {'schema': 1, 'source_members': {n: digest(ROOT/n) for n in MEMBERS},
                'checksums': {n: digest(tools/n) for n in ('tpu_source.bin', 'initial.npz', 'initial.json')},
                'driver_sha256': digest(driver), 'cache_manifest_sha256': CACHE_SHA,
                'cpu_kernel_version': 'arsen4ikvar/nlp-hw02-prepare-v2/2',
                'requested_accelerator': accelerator,
                'kernel_id': kernel_id, 'kernel_title': kernel_title,
                'tool_dataset_id': dataset_id, 'tool_dataset_title': dataset_title,
                'hardware_assumption': 'single-host TPU; >=2 visible JAX devices; count divides64; discover actual type/count, otherwise fail',
                'probe': {'steps': steps, 'chunk': chunk, 'batch': 64, 'seed': 42,
                          'optimizer': 'muon', 'learnable_xsa': True, 'initial_step': 300000},
                'bounds': {'platform_timeout_seconds': 1800, 'parent_wall_seconds': 1740,
                           'cpu_preflight_seconds': 60,
                           'child_wall_seconds': 1500, 'startup_seconds': 120,
                           'compile_seconds': 900, 'conversion_seconds': 100},
                'hypothesis': 'Compact NS5 loop and a 900-second compile allowance enable finite learned-XSA B034 warm-start updates plus recoverable native state on all visible TPU devices; compile speed and throughput unverified.',
                'limitations': ['B034 weights with reset moments/RNG, BF16, fixed shapes and pinned2M cache; not exact CUDA continuation',
                                'native JAX state saved and resumable; CUDA moments not converted', 'no quality selection or test generation',
                                'new checkpoint requires one-T4 exact notebook verification'],
                'no_install_no_retry_no_submission': True}
    for path in (tools, kernel):
        (path/'tpu_probe_manifest.json').write_text(json.dumps(manifest, indent=2))
    (tools/'dataset-metadata.json').write_text(json.dumps({'id': dataset_id, 'title': dataset_title,
                                                          'licenses': [{'name': 'other'}]}, indent=2))
    (kernel/'kernel-metadata.json').write_text(json.dumps({'id': kernel_id, 'title': kernel_title,
        'code_file': 'run.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
        'enable_gpu': False, 'enable_tpu': True, 'enable_internet': False, 'machine_shape': accelerator,
        'dataset_sources': [dataset_id], 'competition_sources': [],
        'kernel_sources': ['arsen4ikvar/nlp-hw02-prepare-v2']}, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tools', required=True); p.add_argument('--kernel', required=True)
    p.add_argument('--dataset-id', required=True); p.add_argument('--kernel-id', required=True)
    p.add_argument('--accelerator', required=True); p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--chunk', type=int, default=5); p.add_argument('--weights', required=True)
    a = p.parse_args()
    print(json.dumps(stage(a.tools, a.kernel, a.dataset_id, a.kernel_id, a.accelerator, a.weights, a.steps, a.chunk), indent=2))


if __name__ == '__main__':
    main()
