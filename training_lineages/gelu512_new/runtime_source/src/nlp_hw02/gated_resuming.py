"""Strict zero-reset continuation of gated warmstart full state (origin300000).

Do not apply this loader to inference-only CE-best weights. The original generic
loader assumes zero-origin counters and rejects this deliberately offset schema.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch

ORIGIN=300000
PARENT_B034_SHA='d09a8611bd6ef8ab2d0711d6999aa419efe50454d1bb12252069c59bc20c8a85'


def require(ok,message):
    if not ok:raise ValueError(message)


def absolute_lr(step):return 6e-4*min(step/200,math.sqrt(200/max(step,200)))


def optimizer_order(model,optimizers):
    names={id(p):name for name,p in model.named_parameters()};seen=[];groups=[]
    aliases={}
    for name,p in model.named_parameters(remove_duplicate=False):aliases.setdefault(id(p),[]).append(name)
    for optimizer in optimizers:
        partition=[]
        for group in optimizer.param_groups:
            current=[names[id(p)] for p in group['params']];partition.append(current);seen.extend(id(p) for p in group['params'])
        groups.append(partition)
    require(len(seen)==len(set(seen)) and set(seen)==set(names),'Optimizer coverage/alias partition differs')
    return {'groups':groups,'aliases':{names[k]:v for k,v in aliases.items() if len(v)>1}}


def replay_sample_prefix(state,deadline,progress=None):
    """Audit actual saved trace using a SEPARATE generator, never advance train RNG."""
    import time
    count=state['step']-ORIGIN;replay=np.random.default_rng(state['seed']);digest=hashlib.sha256()
    for index in range(count):
        if index%1000==0:
            require(time.monotonic()<deadline,'Whole-worker deadline during actual saved sample-prefix audit')
            if progress is not None:progress(index,count)
        indices=replay.integers(0,6000000,size=64);digest.update(indices.astype('<i8',copy=False).tobytes())
    require(replay.bit_generator.state==state['rng_numpy_generator'],'Replayed actual saved Generator state differs; do not restore or train')
    require(digest.hexdigest()==state['sampled_indices_sha256'],'Replayed actual sample-prefix digest differs')
    return digest


def restore_rng(saved,rng):
    # Call LAST immediately before the first resumed training iteration.
    rng.bit_generator.state=saved['numpy']
    torch.set_rng_state(saved['cpu'])
    torch.cuda.set_rng_state_all(saved['cuda'])


def restore_gated_training(model,optimizers,scaler,rng,parent,*,seed,data,optimizer_info,lane,deadline,progress=None):
    parent=Path(parent);metadata=json.loads((parent/'run_config.json').read_text())
    require(metadata['torch']==str(torch.__version__) and metadata['cuda']==torch.version.cuda,'Exact trained Torch/CUDA required')
    require(metadata['data']==data and metadata['seed']==seed==42 and metadata['batch_size']==64,'Exact data/seed/batch required')
    require(metadata['config']==asdict(model.cfg) and metadata['optimizer']==optimizer_info,'Config/optimizer source partition differs')
    require(metadata['warmstart']['parent_step']==ORIGIN and metadata['warmstart']['parent_model_sha256']==PARENT_B034_SHA and metadata['warmstart']['lane']==lane,'Actual reset-at300k lineage differs')
    require(bool(model.cfg.gated_repeats)==(lane=='treatment'),'Lane/gated topology differs')
    order=optimizer_order(model,optimizers)
    if 'continuation' in metadata:require(metadata['continuation']['optimizer_order']==order,'Continued optimizer name/order/alias binding differs')
    # Parent SHA/source/terminal identities are checked by worker contract first.
    state=torch.load(parent/'last_training.pt',map_location='cpu',weights_only=True)
    best=torch.load(parent/'model.pt',map_location='cpu',weights_only=True)
    require(state['config']==asdict(model.cfg) and state['seed']==seed and state['dataset_revision']==data['dataset_revision'] and state['warmstart']==metadata['warmstart'],'Fullstate config/provenance differs')
    step=state['step'];updates=state['optimizer_updates'];skips=state['skipped_updates'];seen=state['sampled_examples']
    require(all(type(n) is int and n>=0 for n in (step,updates,skips,seen)) and step>ORIGIN,'Integer absolute/cumulative counters required')
    require(updates+skips==step-ORIGIN and seen==(step-ORIGIN)*64,'Gated counters must use300k reset origin, not absolute step')
    require(all(math.isfinite(state[k]) for k in ('validation_loss','best_validation_loss')),'Nonfinite fullstate CE')
    require(len(optimizers)==len(state['optimizers'])==2,'Exact Muon+AdamW optimizer pair required')
    if 'lr_absolute_step' in state:require(state['lr_absolute_step']==step,'Saved explicit LR absolute step differs')
    for optimizer_index,(optimizer,saved) in enumerate(zip(optimizers,state['optimizers'])):
        fresh=optimizer.state_dict();require(len(saved['param_groups'])==len(fresh['param_groups']),'Optimizer group count differs')
        for old,new,actual in zip(saved['param_groups'],fresh['param_groups'],optimizer.param_groups):
            require(len(old['params'])==len(new['params'])==len(actual['params']),'Optimizer ordered partition differs')
            for key in ('betas','eps','weight_decay','momentum','nesterov','ns_steps','ns_dtype','adjust_lr_fn','lr_multiplier','fused','amsgrad','maximize'):
                require(old.get(key)==new.get(key),'Optimizer setting differs:'+key)
            require(old['lr']==absolute_lr(step)*old['lr_multiplier'],'Saved group LR does not equal absolute checkpoint step')
            for key,parameter in zip(old['params'],actual['params']):
                require(key in saved['state'],'Missing trained optimizer parameter state')
                entry=saved['state'][key]
                require(set(entry)==({'momentum_buffer'} if optimizer_index==0 else {'step','exp_avg','exp_avg_sq'}),'Exact trained optimizer state fields differ')
                if optimizer_index==1:require(isinstance(entry['step'],torch.Tensor) and entry['step'].numel()==1 and float(entry['step'])==updates,'Adam per-parameter update counter differs')
                for field,value in saved['state'][key].items():
                    if isinstance(value,torch.Tensor):
                        require(bool(torch.isfinite(value).all()),'Nonfinite optimizer state')
                        if field in ('momentum_buffer','exp_avg','exp_avg_sq','max_exp_avg_sq'):
                            require(value.shape==parameter.shape and value.dtype==parameter.dtype,'Optimizer moment shape/dtype differs:'+field)
        ids=[key for group in saved['param_groups'] for key in group['params']]
        require(len(ids)==len(set(ids)) and set(ids)==set(saved['state']),'Optimizer state key coverage differs')
    require(best['config']==asdict(model.cfg) and best['seed']==seed and best['dataset_revision']==data['dataset_revision'] and type(best['step']) is int and ORIGIN<=best['step']<=step and best['validation_loss']==state['best_validation_loss'],'Inherited CE-best inference role differs')
    if 'best_step' in state:require(state['best_step']==best['step'],'Saved best role step differs')
    metrics=json.loads((parent/'metrics.json').read_text())
    require(metrics['absolute_completed_step']==step and metrics['completed_steps']==step-ORIGIN and metrics['sampled_examples']==seen and metrics['optimizer_updates']==updates and metrics['skipped_updates']==skips,'Actual endpoint metrics/fullstate counters differ')
    require(metrics['selected_step']==best['step'] and metrics['best_validation_loss']==best['validation_loss'] and metrics['final_validation_loss']==state['validation_loss'] and metrics['sampled_indices_sha256']==state['sampled_indices_sha256'],'Actual endpoint/CE-best metrics roles differ')
    fresh_state=model.state_dict()
    for saved,label in [(state['model'],'training endpoint'),(best['model'],'inherited inference best')]:
        require(set(saved)==set(fresh_state),'State names differ:'+label)
        for name,value in fresh_state.items():require(saved[name].shape==value.shape and saved[name].dtype==value.dtype and bool(torch.isfinite(saved[name]).all()),'State shape/dtype/finite differs:'+label+':'+name)
        # Validate duplicated aliases BEFORE either role could overwrite a model.
        for names in order['aliases'].values():
            require(all(torch.equal(saved[names[0]],saved[n]) for n in names[1:]),'Checkpoint alias tensors disagree:'+label)
    if lane=='treatment':
        for name in ('encoder_repeat_gates','decoder_repeat_gates'):
            require(bool(state['model'][name].ne(0).any()),'Training endpoint gates are still zero')
    cuda_rng=state['rng_torch_cuda'];require(len(cuda_rng)==torch.cuda.device_count()==1,'One visible T4 saved RNG state required')
    for value in [state['rng_torch_cpu'],*cuda_rng]:require(value.dtype==torch.uint8 and value.ndim==1 and value.device.type=='cpu','Invalid saved RNG tensor')
    digest=replay_sample_prefix(state,deadline,progress)
    saved_rng={'numpy':state['rng_numpy_generator'],'cpu':state['rng_torch_cpu'],'cuda':cuda_rng}
    model.load_state_dict(state['model'],strict=True)  # NEVER load best['model'] for training.
    for optimizer,saved in zip(optimizers,state['optimizers']):optimizer.load_state_dict(saved)
    scaler.load_state_dict(state['scaler'])
    require(scaler.state_dict()==state['scaler'],'AMP scaler restore did not round-trip')
    result={'absolute_step':step,'lr_absolute_step':step,'warmstart_origin':ORIGIN,'warmstart_iterations':step-ORIGIN,
        'sampled_examples':seen,'optimizer_updates':updates,'skipped_updates':skips,'best_validation_loss':state['best_validation_loss'],
        'last_validation_loss':state['validation_loss'],'best_step':best['step'],'sampled_indices_sha256':state['sampled_indices_sha256'],
        'optimizer_order':order,'warmstart':state['warmstart'],'past_lineage':'Fresh optimizer/scaler/RNG reset at300000; no pre-reset optimizer/data lineage reconstructed',
        'saved_lr_step_in_original_schema':'Derived and verified against group LR' if 'lr_absolute_step' not in state else 'Explicit and verified'}
    # Actual training RNG remains unrestored until caller finishes setup/CE audit.
    del state,best
    return result,saved_rng,digest
