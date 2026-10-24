"""Single-GPU, time-bounded baseline with atomic checkpointing and loss validation."""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import signal
import time
import numpy as np
import torch
from torch.nn import functional as F
from .model import ModelConfig, Seq2Seq, load_model
from .tokenizer import BOS, PAD
from .inference import padded_tokens, generate_questions, syntax_valid


def atomic_save(value,path):
    temp=Path(str(path)+'.tmp')
    torch.save(value,temp); os.replace(temp,path)


def shifted(target):
    return torch.cat([torch.full_like(target[:,:1],BOS),target[:,:-1]],1)


@torch.inference_mode()
def validation_loss(model,src,tgt,batch=64):
    model.eval(); total=0.; count=0
    device=next(model.parameters()).device
    for i in range(0,len(src),batch):
        a=src[i:i+batch].to(device); b=tgt[i:i+batch].to(device)
        with torch.autocast('cuda',dtype=torch.float16):
            logits=model(a,shifted(b))
            loss=F.cross_entropy(logits.flatten(0,1),b.flatten(),ignore_index=PAD,reduction='sum')
        total+=loss.item(); count+=b.ne(PAD).sum().item()
    model.train()
    return total/count


def train(output, cfg=None, seconds=3600, max_steps=20000, seed=42, optimizer_kind='adamw', data_dir=None, resident_data=False, resume_dir=None, allow_orthogonalizer_change=False, data_transition_manifest=None, expected_initial_validation=None):
    output=Path(output); cfg=cfg or ModelConfig()
    if resume_dir is not None and output.exists() and any(output.iterdir()):
        raise ValueError('Resume requires a fresh output directory')
    if allow_orthogonalizer_change and resume_dir is None:
        raise ValueError('An orthogonalizer-change comparison must declare its parent resume directory')
    if data_transition_manifest is not None and (resume_dir is None or allow_orthogonalizer_change):
        raise ValueError('Data transition requires parent and unchanged optimizer')
    output.mkdir(parents=True,exist_ok=True)
    data_dir=Path(data_dir) if data_dir is not None else output
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for this experiment')
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(2)
    rng=np.random.default_rng(seed)
    from .progress import ProgressWriter
    from .training_arrays import load_training_arrays
    progress=ProgressWriter(output/'progress.json')
    step=0;optimizer_updates=skipped_updates=0;stop_requested=False
    def request_stop(signum,frame):
        nonlocal stop_requested
        stop_requested=True
    previous_handler=signal.signal(signal.SIGTERM,request_stop)
    progress.update(0,'initializing',force=True)
    src,tgt,array_loading=load_training_arrays(data_dir,resident=resident_data,
        phase=lambda name:progress.update(0,name,force=True))
    progress.update(0,'model_initialization',force=True)
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(data_dir/'tokenizer.json'))
    validation=json.loads((data_dir/'validation.json').read_text())
    if data_dir!=output:
        import shutil
        for name in ['tokenizer.json','validation.json','data_stats.json']:
            shutil.copyfile(data_dir/name,output/name)
    va=padded_tokens(tok,[r['question'] for r in validation],cfg.max_source)
    vb=padded_tokens(tok,[r['code'] for r in validation],cfg.max_target)
    legacy_arrays=None
    if (data_dir/'validation_legacy.json').exists():
        legacy=json.loads((data_dir/'validation_legacy.json').read_text())
        legacy_arrays=(padded_tokens(tok,[r['question'] for r in legacy],cfg.max_source),
                       padded_tokens(tok,[r['code'] for r in legacy],cfg.max_target))
        if data_dir!=output:
            import shutil
            shutil.copyfile(data_dir/'validation_legacy.json',output/'validation_legacy.json')
    model=Seq2Seq(cfg).cuda()
    from .optim import make_optimizers
    optimizers,optimizer_info=make_optimizers(model,optimizer_kind)
    scaler=torch.amp.GradScaler('cuda')
    resume=None
    if resume_dir is not None:
        from .resuming import restore_training
        restore_data=json.loads((output/'data_stats.json').read_text());transition=None
        if data_transition_manifest is not None:
            from .data_transition import validate_transition
            restore_data,transition=validate_transition(data_transition_manifest,resume_dir,data_dir)
        resume=restore_training(model,optimizers,scaler,rng,resume_dir,seed=seed,
            data=restore_data,allow_orthogonalizer_change=allow_orthogonalizer_change)
        if transition is not None:
            if resume['step']!=transition['absolute_transition_step']:raise ValueError('Transition step mismatch')
            resume['controlled_data_transition']=transition
        if max_steps<=resume['step']:raise ValueError('Resume must request new iterations')
        step=resume['step'];optimizer_updates=resume['optimizer_updates'];skipped_updates=resume['skipped_updates']
        import shutil
        shutil.copyfile(Path(resume_dir)/'model.pt',output/'model.pt')
    events=output/'train.jsonl'
    def log(record):
        record={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**record}
        with events.open('a') as f: f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    def log_xsa_gates(current_step):
        gates={name:p.detach().float().tanh().cpu().tolist()
               for name,p in model.named_parameters() if name.endswith('.xsa_alpha')}
        if gates: log({'event':'xsa_gates','step':current_step,'tanh_alpha':gates})
    norm_records=[];norm_profiling=False
    if cfg.loop_repeats>1:
        def record_norm(name):
            def hook(module,inputs,result):
                if norm_profiling:
                    norm_records.append({'module':name,
                        'input_rms':float(inputs[0].detach().float().square().mean().sqrt()),
                        'output_rms':float(result.detach().float().square().mean().sqrt())})
            return hook
        for name,module in model.named_modules():
            parts=name.split('.')
            if parts[0] in ('encoder','decoder') and (len(parts)==2 or
                    (len(parts)==3 and parts[-1] in ('self_attn','cross','ff'))):
                module.register_forward_hook(record_norm(name))
    tracked_names={'encoder.0.self_attn.q.weight','encoder.0.ff.0.weight','encoder.0.ff.3.weight',
        'encoder.0.ff.up_gate.weight','encoder.0.ff.down.weight','embedding.weight'}
    if cfg.loop_repeats>1 and (not cfg.share_loop_weights or cfg.loop_share_scope=='attention'):
        for suffix in ('self_attn.q.weight','ff.0.weight','ff.3.weight'):
            tracked_names.add(f'encoder.{cfg.encoder_layers}.{suffix}')
    metadata={'config':asdict(cfg),'parameters':sum(p.numel() for p in model.parameters()),
        'torch':torch.__version__,'cuda':torch.version.cuda,'python':platform.python_version(),
        'gpu':torch.cuda.get_device_name(0),'gpu_count':torch.cuda.device_count(),
        'seed':seed,'train_seconds_budget':seconds,'max_steps':max_steps,'batch_size':64,
        'optimizer':optimizer_info,'schedule':'warmup=200, inverse_sqrt decay',
        'data':json.loads((output/'data_stats.json').read_text()),'array_loading':array_loading,
        'resumption':resume}
    (output/'run_config.json').write_text(json.dumps(metadata,indent=2))
    log({'event':'array_loading',**array_loading})
    progress.update(step,'initial_validation',force=True)
    initial=validation_loss(model,va,vb)
    if expected_initial_validation is not None:
        if not math.isfinite(expected_initial_validation) or not math.isclose(initial,expected_initial_validation,rel_tol=1e-6,abs_tol=1e-7):
            raise ValueError('Restored initial validation differs from pinned parent')
        log({'event':'parent_initial_validation_verified','step':step,'expected':expected_initial_validation,
             'actual':initial,'absolute_error':abs(initial-expected_initial_validation),'rel_tol':1e-6,'abs_tol':1e-7})
    log_xsa_gates(step)
    log({'event':'initial_validation','step':step,'loss':initial,**{'parameters':metadata['parameters']}})
    best=resume['best_validation_loss'] if resume is not None else initial
    resumed_step=step
    started=time.monotonic(); last=started; loss_total=token_count=0; seen=resume['sampled_examples'] if resume is not None else 0
    def save_best(loss,step):
        atomic_save({'config':asdict(cfg),'model':model.state_dict(),'step':step,'validation_loss':loss,
            'seed':seed,'dataset_revision':metadata['data']['dataset_revision']},output/'model.pt')
    progress.update(step,'checkpoint',force=True)
    if resume is None:save_best(initial,0)
    progress.update(step,'training',force=True)
    phase_totals={k:0. for k in ('input_seconds','transfer_seconds','forward_seconds','backward_seconds','optimizer_seconds')}
    for step in range(resumed_step+1,max_steps+1):
        if time.monotonic()-started>=seconds or stop_requested: break
        batch_start=time.monotonic()
        indices=rng.integers(0,len(src),size=64)
        aa=np.array(src[indices],dtype=np.int64); bb=np.array(tgt[indices],dtype=np.int64)
        aa=aa[:,:int((aa!=0).sum(1).max())]; bb=bb[:,:int((bb!=0).sum(1).max())]
        input_end=time.monotonic()
        a=torch.from_numpy(aa).cuda(); b=torch.from_numpy(bb).cuda()
        transfer_end=time.monotonic()
        lr=6e-4*min(step/200,math.sqrt(200/max(step,200)))
        for optimizer in optimizers:
            for group in optimizer.param_groups: group['lr']=lr*group['lr_multiplier']
            optimizer.zero_grad(set_to_none=True)
        norm_profiling=cfg.loop_repeats>1 and step in (1,1000,10000)
        norm_records.clear()
        with torch.autocast('cuda',dtype=torch.float16):
            logits=model(a,shifted(b))
            loss=F.cross_entropy(logits.flatten(0,1),b.flatten(),ignore_index=PAD)
        norm_profiling=False
        if norm_records:
            log({'event':'loop_norms','step':step,'residual_scale':cfg.residual_scale,
                'records':norm_records,'note':'Block RMS includes residual scaling/dropout; branch module RMS is before outer residual dropout/scale, includes padding. Diagnostic synchronization included in this step time.'})
        if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss at step {step}')
        forward_end=time.monotonic()
        scaler.scale(loss).backward()
        for optimizer in optimizers: scaler.unscale_(optimizer)
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        tracked={name:p.detach().clone() for name,p in model.named_parameters()
                 if optimizer_kind.startswith('muon') and step in [resumed_step+1,200,1000,10000] and name in tracked_names}
        # Keep hybrid optimizers coherent when FP16 gradients overflow.
        gradients_finite=bool(torch.isfinite(grad_norm))
        backward_end=time.monotonic()
        if gradients_finite:
            for optimizer in optimizers: scaler.step(optimizer)
            optimizer_updates+=1
        else: skipped_updates+=1
        scaler.update()
        if tracked:
            updates={name:float((p.detach()-tracked[name]).float().square().mean().sqrt())
                     for name,p in model.named_parameters() if name in tracked}
            log({'event':'update_rms','step':step,'rms':updates,'lr':lr,'muon_lr':5*lr})
        n=int(b.ne(PAD).sum()); loss_total+=loss.item()*n; token_count+=n; seen+=64
        optimizer_end=time.monotonic()
        step_times={'input_seconds':input_end-batch_start,'transfer_seconds':transfer_end-input_end,
            'forward_seconds':forward_end-transfer_end,'backward_seconds':backward_end-forward_end,
            'optimizer_seconds':optimizer_end-backward_end}
        for key,value in step_times.items():phase_totals[key]+=value
        progress.update(step,'training',optimizer_updates=optimizer_updates,skipped_updates=skipped_updates,
            last_step_seconds=optimizer_end-batch_start,last_step_phases=step_times)
        if step-resumed_step<=3:log({'event':'step_timing','step':step,**step_times})
        if step%100==0:
            now=time.monotonic()
            log({'event':'train','step':step,'loss':loss_total/token_count,'lr':lr,
                 'elapsed_seconds':now-started,'steps_per_second':100/(now-last),
                 'target_tokens_per_second':token_count/(now-last),'grad_norm':float(grad_norm),
                 'peak_memory_gb':torch.cuda.max_memory_allocated()/1e9,'phase_seconds':dict(phase_totals)})
            loss_total=token_count=0; last=now
            phase_totals={k:0. for k in phase_totals}
        if step%1000==0:
            progress.update(step,'validation',force=True)
            val=validation_loss(model,va,vb)
            record={'event':'validation','step':step,'loss':val,'elapsed_seconds':time.monotonic()-started,
                    'optimizer_updates':optimizer_updates,'skipped_updates':skipped_updates}
            if legacy_arrays is not None: record['legacy_validation_loss']=validation_loss(model,*legacy_arrays)
            log(record)
            log_xsa_gates(step)
            progress.update(step,'checkpoint',force=True)
            if val<best: best=val; save_best(val,step)
            atomic_save({'config':asdict(cfg),'model':model.state_dict(),'optimizers':[o.state_dict() for o in optimizers],
                'scaler':scaler.state_dict(),'step':step,'validation_loss':val,
                'rng_numpy_generator':rng.bit_generator.state,'rng_torch_cpu':torch.get_rng_state(),
                'rng_torch_cuda':torch.cuda.get_rng_state_all(),'sampled_examples':seen,
                'best_validation_loss':best,'optimizer_updates':optimizer_updates,'skipped_updates':skipped_updates,
                'seed':seed,'dataset_revision':metadata['data']['dataset_revision']},output/'last_training.pt')
            progress.update(step,'training',force=True)
    # Record the completed step, including a time-limited exit before its update.
    completed=seen//64
    progress.update(completed,'final_validation',force=True)
    val=validation_loss(model,va,vb)
    if val<best: best=val; save_best(val,completed)
    training_elapsed=time.monotonic()-started
    log_xsa_gates(completed)
    log({'event':'finished','step':completed,'final_validation_loss':val,'best_validation_loss':best,
         'elapsed_seconds':training_elapsed,'sampled_examples':seen,
         'optimizer_updates':optimizer_updates,'skipped_updates':skipped_updates,'stop_requested':stop_requested})
    progress.update(completed,'checkpoint',force=True)
    atomic_save({'config':asdict(cfg),'model':model.state_dict(),'optimizers':[o.state_dict() for o in optimizers],
                'scaler':scaler.state_dict(),'step':completed,'validation_loss':val,
                'rng_numpy_generator':rng.bit_generator.state,'rng_torch_cpu':torch.get_rng_state(),
                'rng_torch_cuda':torch.cuda.get_rng_state_all(),'sampled_examples':seen,
                'best_validation_loss':best,'optimizer_updates':optimizer_updates,'skipped_updates':skipped_updates,
                'seed':seed,'dataset_revision':metadata['data']['dataset_revision']},output/'last_training.pt')
    signal.signal(signal.SIGTERM,previous_handler)
    progress.update(completed,'post_training',force=True)
    post_started=time.monotonic()
    model,checkpoint=load_model(output/'model.pt')
    subset_indices=np.random.default_rng(20261001).permutation(len(validation))[:256] if legacy_arrays is not None else np.arange(min(256,len(validation)))
    records=[validation[int(i)] for i in subset_indices]
    predictions,timing=generate_questions(model,tok,[r['question'] for r in records])
    from .arithmetic import evaluate_code
    reference_values=[evaluate_code(r['code']) for r in records]
    prediction_values=[evaluate_code(p) for p in predictions]
    scorable=sum(r.supported for r in reference_values)
    agreeing=sum(r.supported and p.supported and math.isclose(r.value,p.value,rel_tol=1e-9,abs_tol=1e-9)
                 for r,p in zip(reference_values,prediction_values))
    metrics={'initial_validation_loss':initial,'final_validation_loss':val,'best_validation_loss':best,
        'selected_step':checkpoint['step'],'completed_steps':completed,'sampled_examples':seen,
        'resumed_from_step':resumed_step,'new_iterations':completed-resumed_step,
        'optimizer_updates':optimizer_updates,'skipped_updates':skipped_updates,
        'array_loading':array_loading,'training_elapsed_seconds':training_elapsed,'post_training_seconds':time.monotonic()-post_started,
        'validation_generation_rows':len(records),
        'validation_generation_indices':[int(i) for i in subset_indices],
        'numeric_proxy':{'reference_supported':scorable,'prediction_supported':sum(p.supported for p in prediction_values),
            'agreeing':agreeing,'agreement_over_supported_references':agreeing/scorable if scorable else None,
            'agreement_over_all_rows':agreeing/len(records),
            'note':'Bounded AST single-reference numeric agreement; ignores print arguments. Synthetic references may conflict or be wrong; not official pass@1.'},
        'validation_exact_text_match':sum(p==r['code'] for p,r in zip(predictions,records))/len(records),
        'validation_python_syntax_valid_fraction':sum(map(syntax_valid,predictions))/len(records),
        'validation_generation_timing':timing,'pass_at_1':None,
        'metric_note':'Exact text match and AST syntax are proxies; no execution tests/labels for test split.'}
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2))
    with (output/'validation_predictions.jsonl').open('w') as f:
        for r,p in zip(records,predictions): f.write(json.dumps({**r,'prediction':p},ensure_ascii=False)+'\n')
    print(json.dumps(metrics),flush=True)
    progress.update(completed,'finished',force=True)
    return metrics
