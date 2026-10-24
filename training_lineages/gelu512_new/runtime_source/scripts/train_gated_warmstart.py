"""B034 weights-only warm start; fresh optimizer/scaler/RNG in both lanes.

The absolute LR schedule starts at 300k. Only existing six-million-row cached
training targets are used. This is deliberately not full-state resumption.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from nlp_hw02.model import ModelConfig, Seq2Seq
from nlp_hw02.optim import make_optimizers
from nlp_hw02.tokenizer import BOS, PAD
from nlp_hw02.train import atomic_save, shifted, validation_loss
from nlp_hw02.training_arrays import load_training_arrays
from nlp_hw02.progress import ProgressWriter

B034_SHA256='d09a8611bd6ef8ab2d0711d6999aa419efe50454d1bb12252069c59bc20c8a85'
B034_CONFIG=ModelConfig(vocab_size=2048,d_model=384,heads=4,encoder_layers=4,
    decoder_layers=4,ff=1536,dropout=.1,max_source=256,max_target=192,
    learnable_xsa=True)
TREATMENT_CONFIG=ModelConfig(**{**asdict(B034_CONFIG),'loop_repeats':2,
    'loop_order':'adjacent','loop_share_scope':'attention','gated_repeats':True,
    'repeat_residual_scale':.5})


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_frozen_parent(path):
    path=Path(path)
    if sha(path/'model.pt')!=B034_SHA256:raise ValueError('Unexpected B034 weights SHA256')
    checkpoint=torch.load(path/'model.pt',map_location='cpu',weights_only=True)
    config=json.loads((path/'run_config.json').read_text())['config']
    metrics=json.loads((path/'metrics.json').read_text())
    data=json.loads((path/'data_stats.json').read_text())
    if (ModelConfig(**config)!=B034_CONFIG or ModelConfig(**checkpoint['config'])!=B034_CONFIG or
        checkpoint['seed']!=42 or checkpoint['step']!=300000 or
        checkpoint['dataset_revision']!=data['dataset_revision'] or
        checkpoint['validation_loss']!=metrics['best_validation_loss'] or
        metrics['selected_step']!=300000 or data['train_rows']!=6000000):
        raise ValueError('Frozen B034 checkpoint/metadata mismatch')
    return checkpoint,data,metrics


def expand_state(parent,model):
    """Repeat-major storage, adjacent execution; attention parameters stay aliased."""
    source=parent['model'];expanded={}
    for name,destination in model.state_dict().items():
        old=name
        for prefix,count in (('encoder.',4),('decoder.',4)):
            if name.startswith(prefix):
                index,suffix=name[len(prefix):].split('.',1)
                old=f'{prefix}{int(index)%count}.{suffix}'
                break
        if name.endswith('_repeat_gates'):expanded[name]=torch.zeros_like(destination)
        elif old in source and destination.shape==source[old].shape:expanded[name]=source[old]
        else:raise ValueError(f'No compatible B034 source for {name} <- {old}')
    model.load_state_dict(expanded,strict=True)
    return model


def atomic_json(value,path):
    path=Path(path);tmp=Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)


@torch.inference_mode()
def initial_prefix_parity(parent,model,src,tgt,device):
    """Actual first two held-out records, teacher-forcing and cached prefixes."""
    reference=Seq2Seq(B034_CONFIG);reference.load_state_dict(parent['model'])
    reference=reference.to(device).eval();model.eval()
    src=src[:2].to(device);tgt=tgt[:2,:16].to(device);inputs=shifted(tgt)
    context=(torch.autocast('cuda',dtype=torch.float16) if device.type=='cuda' else torch.autocast('cpu',enabled=False))
    with context:
        baseline=reference(src,inputs);expanded=model(src,inputs)
        error=float((baseline-expanded).abs().max())
        refmem,refvalid=reference.encode(src);memory,valid=model.encode(src)
        refcache=reference.new_cache(refmem);cache=model.new_cache(memory)
        cached_error=0.;forward_cached_error=0.
        for t in range(inputs.shape[1]):
            base=reference.decode_step(inputs[:,t],t,refcache,refvalid)
            result=model.decode_step(inputs[:,t],t,cache,valid)
            cached_error=max(cached_error,float((base-result).abs().max()))
            forward_cached_error=max(forward_cached_error,float((expanded[:,t]-result).abs().max()))
    record={'rows':2,'prefix_tokens':16,'dtype':'fp16_autocast' if device.type=='cuda' else 'fp32',
        'max_abs_logit_error':error,'max_abs_cached_logit_error':cached_error,
        'expanded_forward_cached_max_abs_error':forward_cached_error,
        'effective_decoder_depth':len(cache),'distinct_self_kv_storages':len({pair[0].data_ptr() for pair in cache}),
        'reference_parameters':sum(p.numel() for p in reference.parameters()),
        'treatment_parameters':sum(p.numel() for p in model.parameters()),
        'xsa_scalar_count':sum(p.numel() for n,p in model.named_parameters() if n.endswith('.xsa_alpha'))}
    if error!=0. or cached_error!=0.:raise ValueError(f'Initial B034 function parity failed: {record}')
    if record['distinct_self_kv_storages']!=len(cache):raise ValueError('Self KV caches alias across depth')
    if record['xsa_scalar_count']!=32:raise ValueError('Learned XSA aliases lost')
    del reference,baseline,expanded,refmem,memory,refcache,cache
    if device.type=='cuda':torch.cuda.empty_cache()
    return record


def run(args):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    cfg=B034_CONFIG if args.lane=='control' else TREATMENT_CONFIG
    limit=45000 if args.lane=='control' else 30000
    new_steps=args.steps or limit
    if not 1000<=new_steps<=limit:raise ValueError('Steps outside declared lane horizon')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    data_dir=Path(args.data);parent=Path(args.parent)
    progress=ProgressWriter(output/'progress.json');progress.update(0,'initializing',force=True)
    checkpoint,parent_data,parent_metrics=read_frozen_parent(parent)
    for name in ('tokenizer.json','validation.json','validation_legacy.json','data_stats.json'):
        if (parent/name).exists():
            if sha(parent/name)!=sha(data_dir/name):raise ValueError(f'Parent/cache mismatch: {name}')
            shutil.copyfile(parent/name,output/name)
    data_stats=json.loads((output/'data_stats.json').read_text())
    arrays,targets,loading=load_training_arrays(data_dir,resident=args.resident,
        phase=lambda phase:progress.update(0,phase,force=True))
    if len(arrays)!=6000000:raise ValueError('Expected pinned 6M cache')
    from tokenizers import Tokenizer
    from nlp_hw02.inference import padded_tokens
    tokenizer=Tokenizer.from_file(str(output/'tokenizer.json'))
    validation=json.loads((output/'validation.json').read_text())
    va=padded_tokens(tokenizer,[r['question'] for r in validation],cfg.max_source)
    vb=padded_tokens(tokenizer,[r['code'] for r in validation],cfg.max_target)
    legacy=None
    if (output/'validation_legacy.json').exists():
        records=json.loads((output/'validation_legacy.json').read_text())
        legacy=(padded_tokens(tokenizer,[r['question'] for r in records],cfg.max_source),
                padded_tokens(tokenizer,[r['code'] for r in records],cfg.max_target))
    progress.update(0,'model_initialization',force=True)
    torch.manual_seed(args.seed);np.random.seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    model=Seq2Seq(cfg)
    if args.lane=='control':model.load_state_dict(checkpoint['model'],strict=True)
    else:expand_state(checkpoint,model)
    model=model.cuda()
    parity=initial_prefix_parity(checkpoint,model,va,vb,torch.device('cuda'))
    del checkpoint
    optimizers,opt_info=make_optimizers(model,'muon_grouped')
    scaler=torch.amp.GradScaler('cuda');rng=np.random.default_rng(args.seed)
    # Reset both train streams after constructor/parity work, identically by lane.
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    progress.update(0,'initial_validation',force=True)
    initial=validation_loss(model,va,vb)
    expected=parent_metrics['best_validation_loss']
    if not math.isclose(initial,expected,rel_tol=0,abs_tol=2e-6):
        raise ValueError(f'Warm-start initial CE mismatch: {initial} vs {expected}')
    metadata={'config':asdict(cfg),'parameters':sum(p.numel() for p in model.parameters()),
        'torch':str(torch.__version__),'cuda':torch.version.cuda,'python':platform.python_version(),
        'gpu':torch.cuda.get_device_name(0),'gpu_count':torch.cuda.device_count(),
        'seed':args.seed,'max_steps':300000+new_steps,'new_iterations':new_steps,'batch_size':64,
        'optimizer':opt_info,'schedule':'absolute warmup200 inverse_sqrt, starting300000',
        'data':data_stats,'array_loading':loading,'initial_prefix_parity':parity,
        'warmstart':{'kind':'weights only; fresh optimizers/scaler/RNG','parent_model_sha256':B034_SHA256,
            'parent_step':300000,'lane':args.lane,'same_numpy_sample_stream_seed':args.seed}}
    atomic_json(metadata,output/'run_config.json')
    def log(row):
        record={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**row}
        with (output/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    def save_best(loss,absolute):
        atomic_save({'config':asdict(cfg),'model':model.state_dict(),'step':absolute,
            'validation_loss':loss,'seed':args.seed,'dataset_revision':data_stats['dataset_revision']},output/'model.pt')
    seen=updates=skips=completed=0;sample_hash=hashlib.sha256();best=initial;best_step=300000
    def save_last(val,absolute):
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise RuntimeError(f'Nonfinite checkpoint parameters at {absolute}')
        atomic_save({'config':asdict(cfg),'model':model.state_dict(),'optimizers':[o.state_dict() for o in optimizers],
            'scaler':scaler.state_dict(),'step':absolute,'validation_loss':val,
            'rng_numpy_generator':rng.bit_generator.state,'rng_torch_cpu':torch.get_rng_state(),
            'rng_torch_cuda':torch.cuda.get_rng_state_all(),'sampled_examples':seen,
            'best_validation_loss':best,'optimizer_updates':updates,'skipped_updates':skips,
            'sampled_indices_sha256':sample_hash.hexdigest(),'seed':args.seed,
            'dataset_revision':data_stats['dataset_revision'],'warmstart':metadata['warmstart']},output/'last_training.pt')
    def gates(step,with_grad=False):
        rows={}
        for name,p in model.named_parameters():
            if name.endswith('_repeat_gates') or name.endswith('.xsa_alpha'):
                rows[name]={'values':p.detach().float().cpu().tolist()}
                if with_grad:rows[name]['unscaled_clipped_gradient']=None if p.grad is None else p.grad.detach().float().cpu().tolist()
        log({'event':'gate_diagnostics','step':step,'absolute_step':300000+step,'gates':rows})
    log({'event':'array_loading',**loading})
    log({'event':'initial_validation','step':0,'absolute_step':300000,'loss':initial,
        'expected_parent_loss':expected,'absolute_error':abs(initial-expected),'prefix_parity':parity,
        'optimizer_state':'fresh','scaler_state':'fresh','rng_state':'fresh'})
    gates(0);save_best(initial,300000)
    stopped=False
    def request_stop(signum,frame):
        nonlocal stopped;stopped=True
    previous=signal.signal(signal.SIGTERM,request_stop)
    model.train();started=time.monotonic();last=started;loss_sum=token_count=0
    for new_step in range(1,new_steps+1):
        if time.monotonic()-started>=args.seconds or stopped:break
        absolute=300000+new_step
        indices=rng.integers(0,len(arrays),size=64);sample_hash.update(indices.astype('<i8',copy=False).tobytes())
        aa=np.asarray(arrays[indices],dtype=np.int64);bb=np.asarray(targets[indices],dtype=np.int64)
        aa=aa[:,:int((aa!=0).sum(1).max())];bb=bb[:,:int((bb!=0).sum(1).max())]
        a=torch.from_numpy(aa).cuda();b=torch.from_numpy(bb).cuda()
        lr=6e-4*min(absolute/200,math.sqrt(200/max(absolute,200)))
        for opt in optimizers:
            for group in opt.param_groups:group['lr']=lr*group['lr_multiplier']
            opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            logits=model(a,shifted(b));loss=F.cross_entropy(logits.flatten(0,1),b.flatten(),ignore_index=PAD)
        if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at {absolute}')
        scaler.scale(loss).backward()
        for opt in optimizers:scaler.unscale_(opt)
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if new_step==1:
            gates(new_step,with_grad=True)
            copied={n:None if p.grad is None else float(p.grad.detach().abs().max()) for n,p in model.named_parameters()
                if n.startswith(('encoder.4.ff.','decoder.4.ff.'))}
            log({'event':'initial_copied_ffn_gradients','step':new_step,'max_abs_gradients':copied})
        if bool(torch.isfinite(grad_norm)):
            for opt in optimizers:scaler.step(opt)
            updates+=1
        else:skips+=1
        scaler.update();count=int(b.ne(PAD).sum());loss_sum+=loss.item()*count;token_count+=count;seen+=64;completed=new_step
        progress.update(completed,'training',optimizer_updates=updates,skipped_updates=skips)
        if new_step%100==0:
            now=time.monotonic();log({'event':'train','step':new_step,'absolute_step':absolute,
                'loss':loss_sum/token_count,'lr':lr,'elapsed_seconds':now-started,
                'steps_per_second':100/(now-last),'target_tokens_per_second':token_count/(now-last),
                'grad_norm':float(grad_norm),'peak_memory_gb':torch.cuda.max_memory_allocated()/1e9})
            loss_sum=token_count=0;last=now
        if new_step%1000==0:
            progress.update(completed,'validation',force=True);val=validation_loss(model,va,vb)
            record={'event':'validation','step':new_step,'absolute_step':absolute,'loss':val,
                'elapsed_seconds':time.monotonic()-started,'optimizer_updates':updates,'skipped_updates':skips,
                'sampled_indices_sha256':sample_hash.hexdigest()}
            if legacy is not None:record['legacy_validation_loss']=validation_loss(model,*legacy)
            log(record);gates(new_step)
            progress.update(completed,'checkpoint',force=True)
            if val<best:best=val;best_step=absolute;save_best(val,absolute)
            save_last(val,absolute)
            if new_step==30000:
                matched=output/'matched_330k';matched.mkdir()
                for name in ('model.pt','tokenizer.json','validation.json','data_stats.json','run_config.json'):
                    shutil.copyfile(output/name,matched/name)
                atomic_save({'config':asdict(cfg),'model':model.state_dict(),'step':absolute,
                    'validation_loss':val,'seed':args.seed,'dataset_revision':data_stats['dataset_revision']},matched/'final_model.pt')
                atomic_json({'selected_step':best_step,'completed_steps':new_step,'absolute_completed_step':absolute,
                    'best_validation_loss':best,'final_validation_loss':val,'sampled_examples':seen,
                    'validation_generation_indices':parent_metrics['validation_generation_indices'],
                    'optimizer_updates':updates,'skipped_updates':skips,'sampled_indices_sha256':sample_hash.hexdigest()},matched/'metrics.json')
                log({'event':'matched_snapshot','step':new_step,'selected_step':best_step,'validation_loss':best})
            progress.update(completed,'training',force=True)
            log({'event':'checkpoint_completed','step':new_step,'absolute_step':absolute,
                'elapsed_seconds':time.monotonic()-started})
    progress.update(completed,'final_validation',force=True);final=validation_loss(model,va,vb)
    if final<best:best=final;best_step=300000+completed;save_best(final,best_step)
    progress.update(completed,'checkpoint',force=True);save_last(final,300000+completed)
    atomic_save({'config':asdict(cfg),'model':model.state_dict(),'step':300000+completed,
        'validation_loss':final,'seed':args.seed,'dataset_revision':data_stats['dataset_revision']},output/'final_model.pt')
    elapsed=time.monotonic()-started
    metrics={'initial_validation_loss':initial,'final_validation_loss':final,'best_validation_loss':best,
        'selected_step':best_step,'completed_steps':completed,'absolute_completed_step':300000+completed,
        'new_iterations':completed,'sampled_examples':seen,'optimizer_updates':updates,'skipped_updates':skips,
        'training_elapsed_seconds':elapsed,'lane':args.lane,'sampled_indices_sha256':sample_hash.hexdigest(),
        'validation_generation_indices':parent_metrics['validation_generation_indices'],'stop_requested':stopped,'matched_horizon_snapshot':'matched_330k' if completed>=30000 else None}
    atomic_json(metrics,output/'metrics.json');gates(completed)
    log({'event':'finished','step':completed,'absolute_step':300000+completed,**metrics})
    progress.update(completed,'finished',force=True,optimizer_updates=updates,skipped_updates=skips)
    signal.signal(signal.SIGTERM,previous)
    return metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--lane',choices=['control','treatment'],required=True)
    p.add_argument('--parent',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seconds',type=int,default=1000000);p.add_argument('--steps',type=int)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--resident',action='store_true')
    print(json.dumps(run(p.parse_args())),flush=True)


if __name__=='__main__':main()
