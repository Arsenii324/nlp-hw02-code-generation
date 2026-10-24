"""Strict restoration of retained training state into a freshly initialized run."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import torch
from .model import ModelConfig


def config_dict(config):return asdict(ModelConfig(**config))


def restore_training(model,optimizers,scaler,rng,directory,*,seed,data,batch_size=64,allow_orthogonalizer_change=False):
    """Restore all state; never infer optimizer state from inference weights."""
    directory=Path(directory)
    metadata=json.loads((directory/'run_config.json').read_text())
    if metadata['torch']!=str(torch.__version__) or metadata['cuda']!=torch.version.cuda:
        raise ValueError('Exact resumption requires the same Torch/CUDA versions')
    if metadata['data']!=data or metadata['seed']!=seed or metadata['batch_size']!=batch_size:
        raise ValueError('Resume data, seed or batch size mismatch')
    info=metadata['optimizer']
    old_method=info.get('orthogonalizer')
    if old_method is None and info['kind']=='hybrid grouped FP16 Muon+AdamW':old_method='newton_schulz'
    new_method=getattr(optimizers[0],'orthogonalizer_name',None)
    method_change=None
    if old_method=='stock_PyTorch':old_method=None
    if old_method!=new_method:
        if not allow_orthogonalizer_change or (old_method,new_method)!=('newton_schulz','polar_express_v5_5'):
            raise ValueError('Optimizer method change is not exact resumption; declare the controlled comparison')
        method_change={'from':old_method,'to':new_method,'momentum_and_other_state_restored':True}
    state=torch.load(directory/'last_training.pt',map_location='cpu',weights_only=True)
    if config_dict(state['config'])!=asdict(model.cfg):raise ValueError('Resume model config mismatch')
    if state['seed']!=seed or state['dataset_revision']!=data['dataset_revision']:
        raise ValueError('Resume checkpoint provenance mismatch')
    step=state['step'];updates=state['optimizer_updates'];skips=state['skipped_updates']
    if (not all(type(x) is int and x>=0 for x in (step,updates,skips)) or
        step<=0 or updates+skips!=step or state['sampled_examples']!=step*batch_size):
        raise ValueError('Resume counters inconsistent')
    if not all(math.isfinite(state[key]) for key in ('validation_loss','best_validation_loss')):
        raise ValueError('Nonfinite resume loss')
    if len(optimizers)!=len(state['optimizers']):raise ValueError('Resume optimizer count mismatch')
    for optimizer,saved in zip(optimizers,state['optimizers']):
        for entry in saved['state'].values():
            if any(isinstance(value,torch.Tensor) and not torch.isfinite(value).all() for value in entry.values()):
                raise ValueError('Nonfinite resume optimizer state')
        fresh=optimizer.state_dict()
        if len(saved['param_groups'])!=len(fresh['param_groups']):raise ValueError('Resume parameter groups mismatch')
        for a,b in zip(saved['param_groups'],fresh['param_groups']):
            if len(a['params'])!=len(b['params']):raise ValueError('Resume parameter partition mismatch')
            # LR is intentionally inherited; everything else identifies the recipe.
            for key in ('betas','eps','weight_decay','momentum','nesterov','ns_steps',
                        'ns_dtype','adjust_lr_fn','lr_multiplier','fused','amsgrad','maximize'):
                if a.get(key)!=b.get(key):raise ValueError(f'Resume optimizer setting mismatch: {key}')
    best=torch.load(directory/'model.pt',map_location='cpu',weights_only=True)
    if (config_dict(best['config'])!=asdict(model.cfg) or best['seed']!=seed or
        best['dataset_revision']!=data['dataset_revision'] or
        not 0<best['step']<=step or best['validation_loss']!=state['best_validation_loss']):
        raise ValueError('Resume best checkpoint/provenance mismatch')
    # Validate both roles strictly before overwriting the freshly built model.
    if set(best['model'])!=set(model.state_dict()) or set(state['model'])!=set(model.state_dict()):
        raise ValueError('Resume parameter names mismatch')
    for name,parameter in model.state_dict().items():
        if parameter.shape!=best['model'][name].shape or parameter.shape!=state['model'][name].shape:
            raise ValueError('Resume parameter shape mismatch')
        if not torch.isfinite(best['model'][name]).all() or not torch.isfinite(state['model'][name]).all():
            raise ValueError('Nonfinite resume model')
    cuda_rng=state['rng_torch_cuda']
    expected_cuda_devices=torch.cuda.device_count() if next(model.parameters()).is_cuda else 0
    if len(cuda_rng)!=expected_cuda_devices:raise ValueError('Resume CUDA RNG device count mismatch')
    model.load_state_dict(state['model'],strict=True)
    for optimizer,saved in zip(optimizers,state['optimizers']):optimizer.load_state_dict(saved)
    scaler.load_state_dict(state['scaler'])
    rng.bit_generator.state=state['rng_numpy_generator']
    torch.set_rng_state(state['rng_torch_cpu'])
    if cuda_rng:torch.cuda.set_rng_state_all(cuda_rng)
    return {'step':step,'optimizer_updates':updates,'skipped_updates':skips,
        'sampled_examples':state['sampled_examples'],'best_validation_loss':state['best_validation_loss'],
        'last_validation_loss':state['validation_loss'],'best_step':best['step'],
        'orthogonalizer_change':method_change,
        'parent_directory':str(directory)}
