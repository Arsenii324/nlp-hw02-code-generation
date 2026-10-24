"""Source-only exact continuation; requires future actual-bound original terminal.

Training tensors come from last_training.pt; inherited model.pt remains only the
CE-best inference candidate. No architecture/optimizer/data-layout change here.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

ORIGINAL_SOURCE_SHA='eefda7a0c41df4aa992f5cbde4620d7486b58ee86214c9d708698979ea2c5d83'
ORIGINAL_DRIVER_SHA='2dd4ebd12cdea6ea670cd7967fc092f7bf3ad2f82a89606d7c9574d22db8caf3'
ORIGINAL_MANIFEST_SHA='65a2693da6db3d45413e5f7e199d91cc6fe3d1237ecde49edb0af57bc8779256'
ORIGIN=300000
FROZEN_RUNTIME={'scripts/train_gated_warmstart.py': '653cbbf90925f78a2a2ba8570dd07ab58db96e10a4fe9190c320eb5b1a13db33', 'src/nlp_hw02/model.py': '07e58176c39eed8ed03134074282d74589be97120e6d6569a15f27eee1538b26', 'src/nlp_hw02/optim.py': 'eed8391c8385b5d09d52384fc20c975cdb1332c45e04bbb36a2b8682f7a140e9', 'src/nlp_hw02/optim_batched.py': '9161d39b568ba472fe132b13e50bc906537104e61aca30a0f7de89efff8e246f', 'src/nlp_hw02/train.py': '1c98897afe264db8ff1803f586ce183adc799dd9d4382dd63805aa0f980bd406', 'src/nlp_hw02/training_arrays.py': '4e5bd548a59693ab16b7fae6048e40688cd6cbbc59b332e4eccaa1606efa3ec8', 'src/nlp_hw02/progress.py': '825026a39e5744e8f96af0ecbd0f91d137f943f0b45cfe0fd1dbbb0742c39c63'}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_json(value,path):
    path=Path(path);tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def verify_binding(args):
    c=json.loads(args.contract.read_text())
    require(c['schema']=='gated_exact_continuation_v1' and c['verified_actual_values'] is True,'Unbound source-only continuation; real terminal/fullstate bindings required')
    require(c['lane']==args.lane and c['new_steps']==args.new_steps and args.new_steps>=1000,'Explicit future lane/new horizon differs')
    require(c['counter_origin']==ORIGIN and c['data_layout']=='original_uniform_batch64','No bucketing/targets/recipe change in exact continuation')
    require(sha(args.terminal)==c['terminal_sha256'],'Actual original terminal hash differs')
    terminal=json.loads(args.terminal.read_text())
    require(terminal['training_complete'] is True and terminal['source_sha256']==ORIGINAL_SOURCE_SHA and terminal['driver_sha256']==ORIGINAL_DRIVER_SHA,'Exact completed original source/driver required')
    require(terminal['artifact_sha256']['experiment-manifest.json']==ORIGINAL_MANIFEST_SHA,'Original training manifest identity differs')
    observed={row['id']:row for row in terminal['lanes']}[args.lane]
    expected=45000 if args.lane=='control' else 30000
    require(observed['exit']==0 and observed['stopped'] is False and observed['observation']['exact'] is True and observed['observation']['steps']==expected,'Actual original lane incomplete')
    required={'last_training.pt','model.pt','run_config.json','metrics.json','tokenizer.json','validation.json','validation_legacy.json','data_stats.json'}
    require(set(c['parent_artifact_sha256'])==required,'Exact original endpoint/best/small-metadata allowlist required')
    for name,digest in c['parent_artifact_sha256'].items():
        require(terminal['artifact_sha256'][f'run-{args.lane}/{name}']==digest and sha(args.parent/name)==digest,'Actual original parent role bytes differ:'+name)
    root=Path(__file__).resolve().parents[1]
    require(all(c['runtime_source_sha256'].get(name)==digest for name,digest in FROZEN_RUNTIME.items()),'Frozen model/optimizer/training recipe source changed')
    for name,digest in c['runtime_source_sha256'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe bound source path')
        require(sha(root/name)==digest,'Exact source/order implementation differs:'+name)
    require('scripts/train_gated_continuation_b.py' in c['runtime_source_sha256'] and 'src/nlp_hw02/gated_resuming.py' in c['runtime_source_sha256'],'New continuation source identities required')
    require(sha(args.data/'cache_manifest.json')==c['cache_manifest_sha256']=='8ce0939534bdb5f2b791d19d0ab13f6b4c79402b8627fecad4e9902cd08bb13c','Original6M cache identity differs')
    cache=json.loads((args.data/'cache_manifest.json').read_text());require(cache['complete'] is True,'Incomplete cache')
    require(cache['checksums']==c['cache_checksums'],'Bound cache checksums differ')
    for name in ('tokenizer.json','validation.json','validation_legacy.json','data_stats.json'):
        require(sha(args.data/name)==c['cache_checksums'][name]==c['parent_artifact_sha256'][name],'Parent/current cache metadata differs:'+name)
    return c


def run(args):
    startup=time.monotonic();c=verify_binding(args)
    require(math.isfinite(args.deadline) and time.monotonic()<args.deadline,'Finite whole-worker absolute deadline required')
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from train_gated_warmstart import B034_CONFIG,TREATMENT_CONFIG
    from nlp_hw02.model import Seq2Seq
    from nlp_hw02.optim import make_optimizers
    from nlp_hw02.gated_resuming import restore_gated_training,restore_rng,absolute_lr
    from nlp_hw02.train import atomic_save,shifted,validation_loss
    from nlp_hw02.tokenizer import PAD
    from nlp_hw02.inference import padded_tokens
    from nlp_hw02.training_arrays import load_training_arrays
    from nlp_hw02.progress import ProgressWriter
    require(torch.cuda.is_available() and torch.cuda.device_count()==1 and 'T4' in torch.cuda.get_device_name(0),'One visible T4 required')
    torch.set_num_threads(2);cfg=B034_CONFIG if args.lane=='control' else TREATMENT_CONFIG
    output=args.output;output.mkdir(parents=True,exist_ok=False);progress=ProgressWriter(output/'progress.json');progress.update(0,'initializing',force=True)
    for name in ('tokenizer.json','validation.json','validation_legacy.json','data_stats.json'):shutil.copyfile(args.parent/name,output/name)
    data=json.loads((output/'data_stats.json').read_text())
    arrays,targets,loading=load_training_arrays(args.data,resident=args.resident,phase=lambda name:progress.update(0,name,force=True))
    require(len(arrays)==6000000 and arrays.dtype==targets.dtype==np.uint16 and arrays.shape==(6000000,256) and targets.shape==(6000000,192),'Exact original6M arrays required')
    tok=Tokenizer.from_file(str(output/'tokenizer.json'));validation=json.loads((output/'validation.json').read_text())
    va=padded_tokens(tok,[r['question'] for r in validation],256);vb=padded_tokens(tok,[r['code'] for r in validation],192)
    records=json.loads((output/'validation_legacy.json').read_text());legacy=(padded_tokens(tok,[r['question'] for r in records],256),padded_tokens(tok,[r['code'] for r in records],192))
    progress.update(0,'model_initialization',force=True)
    model=Seq2Seq(cfg).cuda();optimizers,opt_info=make_optimizers(model,'muon_grouped');scaler=torch.amp.GradScaler('cuda');rng=np.random.default_rng()
    restored,saved_rng,sample_hash=restore_gated_training(model,optimizers,scaler,rng,args.parent,seed=42,data=data,optimizer_info=opt_info,lane=args.lane,deadline=args.deadline,
        progress=lambda done,total:progress.update(0,'sample_prefix_audit',force=True,replayed_warmstart_iterations=done,total_warmstart_iterations=total))
    start_step=restored['absolute_step'];require(start_step==c['parent_absolute_step'],'Actual fullstate step differs from explicit binding')
    seen=restored['sampled_examples'];updates=restored['optimizer_updates'];skips=restored['skipped_updates'];best=restored['best_validation_loss'];best_step=restored['best_step']
    progress.update(0,'initial_validation',force=True);initial=validation_loss(model,va,vb)
    require(math.isfinite(initial) and math.isclose(initial,restored['last_validation_loss'],rel_tol=1e-6,abs_tol=1e-7),'Restored endpoint CE differs; no optimizer-on-best fallback')
    shutil.copyfile(args.parent/'model.pt',output/'model.pt')  # Inherited inference role only.
    metadata=json.loads((args.parent/'run_config.json').read_text())
    metadata.update(max_steps=start_step+args.new_steps,new_iterations=args.new_steps,array_loading=loading,
        continuation={'schema':'gated_exact_continuation_v1','parent_absolute_step':start_step,'counter_origin':ORIGIN,
            'parent_terminal_sha256':c['terminal_sha256'],'parent_last_training_sha256':c['parent_artifact_sha256']['last_training.pt'],
            'inherited_inference_best_sha256':c['parent_artifact_sha256']['model.pt'],'optimizer_order':restored['optimizer_order'],
            'runtime_source_sha256':c['runtime_source_sha256'],'contract_sha256':sha(args.contract),'zero_reset':True,
            'schedule':'same absolute LR6e-4*min(step/200,sqrt(200/max(step,200))); no duplicate warmup',
            'backend_limit':'Saved state restored, not a bitwise cross-hardware guarantee; old source did not save determinism/backend flags.',
            'current_numpy_version':np.__version__,'historical_numpy_version':'Not recorded by original source; separate prefix replay verifies saved draw state/digest'})
    atomic_json(metadata,output/'run_config.json')
    def log(row):
        record={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**row}
        with (output/'train.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps(record,allow_nan=False),flush=True)
    log({'event':'initial_validation','step':0,'absolute_step':start_step,'warmstart_iterations':start_step-ORIGIN,'loss':initial,
        'expected_parent_endpoint_ce':restored['last_validation_loss'],'restored':restored,'startup_seconds':time.monotonic()-startup,'optimizer_state':'restored, not reset','scaler_state':'restored','rng_state':'saved state restored last before first resumed iteration'})
    stopped=False
    def request_stop(signum,frame):
        nonlocal stopped;stopped=True
    previous=signal.signal(signal.SIGTERM,request_stop)
    completed=0;started=time.monotonic();last=started;loss_sum=token_count=0;checkpoint_cost=0.;admission=None
    target=start_step+args.new_steps
    def save_best(loss,absolute):
        atomic_save({'config':asdict(cfg),'model':model.state_dict(),'step':absolute,'validation_loss':loss,'seed':42,'dataset_revision':data['dataset_revision']},output/'model.pt')
    def save_last(val,absolute):
        require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),'Nonfinite checkpoint model')
        atomic_save({'config':asdict(cfg),'model':model.state_dict(),'optimizers':[o.state_dict() for o in optimizers],'scaler':scaler.state_dict(),
            'step':absolute,'lr_absolute_step':absolute,'validation_loss':val,'rng_numpy_generator':rng.bit_generator.state,
            'rng_torch_cpu':torch.get_rng_state(),'rng_torch_cuda':torch.cuda.get_rng_state_all(),'sampled_examples':seen,
            'best_validation_loss':best,'best_step':best_step,'optimizer_updates':updates,'skipped_updates':skips,'sampled_indices_sha256':sample_hash.hexdigest(),
            'seed':42,'dataset_revision':data['dataset_revision'],'warmstart':metadata['warmstart'],
            'continuation':{'schema':'gated_exact_continuation_v1','parent_absolute_step':start_step,'since_resume_iterations':absolute-start_step,
                'warmstart_origin':ORIGIN,'warmstart_iterations':absolute-ORIGIN}},output/'last_training.pt')
    model.train();restore_rng(saved_rng,rng);del saved_rng  # LAST setup operation before training; no actual-RNG draws in replay.
    for new_step in range(1,args.new_steps+1):
        now=time.monotonic();remaining=args.deadline-now;rate=completed/(now-started) if completed else None;eta=(args.new_steps-completed)/rate if rate else None
        save_reserve=max(180.,1.5*checkpoint_cost);near_finish=eta is not None and eta<=180 and eta+max(30.,1.5*checkpoint_cost)<remaining
        if stopped or (remaining<=save_reserve and not near_finish):break
        absolute=start_step+new_step
        indices=rng.integers(0,len(arrays),size=64);sample_hash.update(indices.astype('<i8',copy=False).tobytes())
        aa=np.asarray(arrays[indices],dtype=np.int64);bb=np.asarray(targets[indices],dtype=np.int64)
        aa=aa[:,:int((aa!=0).sum(1).max())];bb=bb[:,:int((bb!=0).sum(1).max())]
        a=torch.from_numpy(aa).cuda();b=torch.from_numpy(bb).cuda();lr=absolute_lr(absolute)
        for opt in optimizers:
            for group in opt.param_groups:group['lr']=lr*group['lr_multiplier']
            opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            logits=model(a,shifted(b));loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),b.flatten(),ignore_index=PAD)
        require(bool(torch.isfinite(loss)),'Nonfinite resumed loss')
        scaler.scale(loss).backward()
        for opt in optimizers:scaler.unscale_(opt)
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if bool(torch.isfinite(grad_norm)):
            for opt in optimizers:scaler.step(opt)
            updates+=1
        else:skips+=1
        scaler.update();count=int(b.ne(PAD).sum());loss_sum+=loss.item()*count;token_count+=count;seen+=64;completed=new_step
        progress.update(completed,'training',absolute_step=absolute,warmstart_iterations=absolute-ORIGIN,optimizer_updates=updates,skipped_updates=skips)
        if new_step%100==0:
            now=time.monotonic();log({'event':'train','step':new_step,'absolute_step':absolute,'warmstart_iterations':absolute-ORIGIN,
                'loss':loss_sum/token_count,'lr':lr,'elapsed_seconds':now-started,'steps_per_second':100/(now-last),'grad_norm':float(grad_norm) if bool(torch.isfinite(grad_norm)) else None,'grad_norm_finite':bool(torch.isfinite(grad_norm))})
            loss_sum=token_count=0;last=now
        if absolute%1000==0 or new_step==1000:
            cost_start=time.monotonic();progress.update(completed,'validation',force=True,absolute_step=absolute);val=validation_loss(model,va,vb)
            legacy_val=validation_loss(model,*legacy);log({'event':'validation','step':new_step,'absolute_step':absolute,'loss':val,'legacy_validation_loss':legacy_val,
                'optimizer_updates':updates,'skipped_updates':skips,'sampled_indices_sha256':sample_hash.hexdigest(),'elapsed_seconds':time.monotonic()-started})
            if val<best:best=val;best_step=absolute;save_best(val,absolute)
            progress.update(completed,'checkpoint',force=True,absolute_step=absolute);save_last(val,absolute);checkpoint_cost=max(checkpoint_cost,time.monotonic()-cost_start)
            elapsed=time.monotonic()-started;log({'event':'checkpoint_completed','step':new_step,'absolute_step':absolute,'elapsed_seconds':elapsed,'checkpoint_and_CE_seconds':checkpoint_cost})
            if new_step>=1000 and admission is None:
                rate=new_step/elapsed;eta=max(0,args.new_steps-new_step)/rate;remaining=args.deadline-time.monotonic();admission=eta+max(180.,1.5*checkpoint_cost)<=remaining
                log({'event':'budget_admission','step':new_step,'absolute_step':absolute,'actual_NEW_steps':new_step,'measured_seconds_including_CE_fullsave':elapsed,
                    'eta_seconds':eta,'remaining_seconds':remaining,'save_reserve_seconds':max(180.,1.5*checkpoint_cost),'admitted':admission})
                if not admission:stopped=True
            progress.update(completed,'training',force=True,absolute_step=absolute)
    absolute=start_step+completed;progress.update(completed,'final_validation',force=True,absolute_step=absolute);final=validation_loss(model,va,vb)
    if final<best:best=final;best_step=absolute;save_best(final,absolute)
    progress.update(completed,'checkpoint',force=True,absolute_step=absolute);save_last(final,absolute)
    atomic_save({'config':asdict(cfg),'model':model.state_dict(),'step':absolute,'validation_loss':final,'seed':42,'dataset_revision':data['dataset_revision']},output/'final_model.pt')
    metrics={'selected_step':best_step,'best_validation_loss':best,'final_validation_loss':final,'parent_absolute_step':start_step,'absolute_completed_step':absolute,
        'counter_origin':ORIGIN,'completed_steps':absolute-ORIGIN,'warmstart_iterations':absolute-ORIGIN,'since_resume_iterations':completed,
        'since_resume_sampled_examples':completed*64,'sampled_examples':seen,'optimizer_updates':updates,'skipped_updates':skips,
        'since_resume_optimizer_updates':updates-restored['optimizer_updates'],'since_resume_skipped_updates':skips-restored['skipped_updates'],
        'sampled_indices_sha256':sample_hash.hexdigest(),'lr_absolute_step':absolute,'training_elapsed_seconds':time.monotonic()-started,
        'whole_worker_seconds':time.monotonic()-startup,'requested_new_iterations':args.new_steps,'exact_requested_horizon_complete':completed==args.new_steps,
        'budget_admitted':admission,'stop_requested':stopped,'lane':args.lane,'warmstart':metadata['warmstart'],
        'validation_generation_indices':json.loads((args.parent/'metrics.json').read_text())['validation_generation_indices']}
    atomic_json(metrics,output/'metrics.json');log({'event':'finished','step':completed,'absolute_step':absolute,**metrics})
    progress.update(completed,'finished',force=True,absolute_step=absolute,optimizer_updates=updates,skipped_updates=skips);signal.signal(signal.SIGTERM,previous)
    return metrics


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lane',choices=('control','treatment'),required=True);p.add_argument('--contract',type=Path,required=True);p.add_argument('--terminal',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--new-steps',type=int,required=True);p.add_argument('--deadline',type=float,required=True);p.add_argument('--resident',action='store_true');a=p.parse_args()
    print(json.dumps(run(a),allow_nan=False),flush=True)
