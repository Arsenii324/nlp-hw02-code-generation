"""Source-only recipe-aware uniform/bucket continuation after exact gated/control restore.

Explicit fixedfraction LR branch; no targets/filtering/optimizer/scaler/RNG reset. Fullstates commit at consumed windows
using two durable slots; partial windows only write NONRESUMABLE diagnostics.
"""
import argparse
from copy import deepcopy
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

REQUIRED_PARENT={'last_training.pt','model.pt','run_config.json','metrics.json','tokenizer.json','validation.json','validation_legacy.json','data_stats.json'}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def fsync_directory(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.next')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path);fsync_directory(path.parent)


def atomic_torch_save(value,path):
    import torch
    temporary=path.with_name(path.name+'.next')
    with temporary.open('wb') as stream:
        torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path);fsync_directory(path.parent)


def safe_relative(name):
    return isinstance(name,str) and bool(name) and not Path(name).is_absolute() and '..' not in Path(name).parts


def safe_run_id(value):
    return isinstance(value,str) and 1<=len(value)<=64 and value[0] in 'abcdefghijklmnopqrstuvwxyz' and all(ch in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for ch in value)


def verify_binding(args):
    c=json.loads(args.contract.read_text())
    require(c['schema']=='half_low_lr_continuation_a_v1' and c['verified_actual_values'] is True,'Actual bound lowLR contract required')
    require(safe_run_id(args.run_id) and c['run_id']==args.run_id and args.lane==c['lane']=='control','Exact control run identity required')
    require(c['parent_kind']=='cooldown_D_half_endpoint' and c['layout']=='bucket' and c['parent_absolute_step']==417384 and c['counter_origin']==300000,'TypedD endpoint417384 only; never original345/best415656')
    require(c['lr_fraction'] in (.125,.25) and c['new_steps']==args.new_steps and 1024<=args.new_steps<=16384 and args.new_steps%64==0,'Explicit lowLR fraction/cap required')
    require(c['admission_mode']=='paired_measured64_shortening' and c['data_layout']=='gated_window64_v1','Paired common measured window policy required')
    require(sha(args.terminal)==c['parent_terminal_sha256']=='71e8ba12573751cef9bef71a036cf73138bfc3bdef104f7cdda12753acbb74ed','ActualD terminal differs')
    terminal=json.loads(args.terminal.read_text())
    require(terminal['complete'] is terminal['training_complete'] is terminal['artifact_hashes_complete'] is True and
            bool(terminal['cleanup']) and all(r['reaped'] and r['group_exited'] for r in terminal['cleanup']),'Complete immutable D groups required')
    require(terminal['source_sha256']==c['parent_source_sha256']=='a8a41094ab5128d33b0b7b5eb95054fa08bf56f8b8c2dcaf80344985fb60e201' and terminal['driver_sha256']==c['parent_driver_sha256']=='fee6590e61ca5d5fee1eb8996abca87a74686b3dbc07a9df45dff18f1f4ea4d4','Exact savedD source/driver required')
    require(c['parent_run_path']=='run-cooldown-half/durable-slot1' and args.parent.resolve()==(args.terminal.parent/c['parent_run_path']).resolve(),'ActualD endpointslot path differs')
    require(c['parent_manifest_path']=='gated_cooldown_continuation_manifest.json' and c['parent_manifest_sha256']=='6d2898f293e4bd54536ecfd793c8fd3beb32cee0aff17905c8a4d9440f110b7c' and
            terminal['artifact_sha256'][c['parent_manifest_path']]==c['parent_manifest_sha256'] and sha(args.terminal.parent/c['parent_manifest_path'])==c['parent_manifest_sha256'],'ActualD manifest differs')
    require(set(c['parent_artifact_sha256'])==REQUIRED_PARENT and c['parent_artifact_sha256']['last_training.pt']=='93cd78737719bf87a8060c27d6079f7510b77d47ac37fc45085069d53fd6929b' and c['parent_artifact_sha256']['model.pt']=='edb1b328116abb82240b094093108e533a0ce026a283f7f5463602631f253089','Exact endpoint/fullstate vs inheritedbest roles required')
    for name,h in c['parent_artifact_sha256'].items():require(terminal['artifact_sha256'][c['parent_run_path']+'/'+name]==h and sha(args.parent/name)==h,'Actual parent bytes differ:'+name)
    runtime=Path(__file__).resolve().parents[1]
    require(sha(args.source_archive)==c['runtime_archive_sha256'],'Bound current sourcearchive differs')
    require(all(c['runtime_source_sha256'].get(n)==v for n,v in {**FROZEN_RUNTIME,'src/nlp_hw02/gated_resuming.py':'59edf37be73c37e63f924dae987072c4b2ee2c83d8bd9e2c8135c6ec3edc5361'}.items()),'Original model/optimizer/math/restore changed')
    require('src/nlp_hw02/half_low_lr_bridge_a.py' in c['runtime_source_sha256'] and 'scripts/train_half_low_lr_a.py' in c['runtime_source_sha256'],'Typed bridge/currentworker identities missing')
    for name,h in c['runtime_source_sha256'].items():require(safe_relative(name) and sha(runtime/name)==h,'Runtime member differs:'+name)
    require(sha(args.data/'cache_manifest.json')==c['cache_manifest_sha256']=='8ce0939534bdb5f2b791d19d0ab13f6b4c79402b8627fecad4e9902cd08bb13c','Original6M cache differs')
    cache=json.loads((args.data/'cache_manifest.json').read_text());require(cache['complete'] is True and cache['checksums']==c['cache_checksums'],'Bound6M checksums differ')
    for name in ('tokenizer.json','validation.json','validation_legacy.json','data_stats.json'):require(sha(args.data/name)==c['cache_checksums'][name]==c['parent_artifact_sha256'][name],'Exact parent/cache small metadata required')
    require(args.coordination.is_dir(),'Parent must create owned shared coordination directory')
    return c


def run(args):
    startup=time.monotonic();c=verify_binding(args)
    require(math.isfinite(args.deadline) and time.monotonic()<args.deadline,'Finite remaining whole-worker deadline required')
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from train_gated_warmstart import B034_CONFIG,TREATMENT_CONFIG
    from nlp_hw02.model import Seq2Seq
    from nlp_hw02.optim import make_optimizers
    from nlp_hw02.half_low_lr_bridge_a import (recipe_for,restore_window_training,restore_rng,absolute_lr,
        validate_window_parent,draw_window,segment_record,fingerprint)
    from nlp_hw02.train import shifted,validation_loss
    from nlp_hw02.tokenizer import PAD
    from nlp_hw02.inference import padded_tokens
    from nlp_hw02.training_arrays import load_training_arrays
    from nlp_hw02.progress import ProgressWriter
    recipe,recipe_sha=recipe_for(c['layout'])
    require(c['recipe']==recipe and c['recipe_sha256']==recipe_sha,'Explicit unchanged layout recipe required')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1 and 'T4' in torch.cuda.get_device_name(0),'Exactly one visible T4 required')
    torch.set_num_threads(2);cfg=B034_CONFIG if args.lane=='control' else TREATMENT_CONFIG
    out=args.output;out.mkdir(parents=True,exist_ok=False);progress=ProgressWriter(out/'progress.json');progress.update(0,'initializing',force=True)
    data=json.loads((args.parent/'data_stats.json').read_text())
    arrays,targets,loading=load_training_arrays(args.data,resident=args.resident,phase=lambda phase:progress.update(0,phase,force=True))
    require(arrays.shape==(6000000,256) and targets.shape==(6000000,192) and arrays.dtype==targets.dtype==np.uint16,'Exact6M cached arrays required')
    tok=Tokenizer.from_file(str(args.parent/'tokenizer.json'));validation=json.loads((args.parent/'validation.json').read_text());legacyrows=json.loads((args.parent/'validation_legacy.json').read_text())
    va=padded_tokens(tok,[r['question'] for r in validation],256);vb=padded_tokens(tok,[r['code'] for r in validation],192)
    legacy=(padded_tokens(tok,[r['question'] for r in legacyrows],256),padded_tokens(tok,[r['code'] for r in legacyrows],192))
    progress.update(0,'model_initialization',force=True);model=Seq2Seq(cfg).cuda();optimizers,opt_info=make_optimizers(model,'muon_grouped');scaler=torch.amp.GradScaler('cuda');rng=np.random.default_rng()
    restored,saved_rng,whole_raw,saved_recipe=restore_window_training(model,optimizers,scaler,rng,args.parent,seed=42,data=data,optimizer_info=opt_info,lane=args.lane,deadline=args.deadline,
        progress=lambda done,total:progress.update(0,'raw_prefix_audit',force=True,replayed_iterations=done,total_iterations=total))
    start=restored['absolute_step'];require(start==c['parent_absolute_step'],'Explicit parent absolute step differs')
    old_metadata=json.loads((args.parent/'run_config.json').read_text());old_metrics=json.loads((args.parent/'metrics.json').read_text())
    layout_rng,history,window_origin=validate_window_parent(saved_recipe,old_metadata,old_metrics,c,restored,saved_rng)
    seen=restored['sampled_examples'];updates=restored['optimizer_updates'];skips=restored['skipped_updates'];best=restored['best_validation_loss'];best_step=restored['best_step']
    progress.update(0,'initial_validation',force=True);initial=validation_loss(model,va,vb)
    require(math.isfinite(initial) and math.isclose(initial,restored['last_validation_loss'],rel_tol=1e-6,abs_tol=1e-7),'Restored endpoint CE differs; never train inherited best')
    shutil.copyfile(args.parent/'model.pt',out/'model.pt')
    # Saved D groupLR0 was already checked by the unchanged strict saved-LR oracle.
    require(all(group['lr']==0. for optimizer in optimizers for group in optimizer.param_groups),'D endpoint scheduled groupLR must be0')
    for optimizer in optimizers:
        for group in optimizer.param_groups:group['lr']=c['lr_fraction']*absolute_lr(start)*group['lr_multiplier']
    metadata=deepcopy(old_metadata)
    metadata.pop('cooldown_plan',None);metadata.pop('cooldown_plan_sha256',None)
    metadata['low_lr_branch']={'schema':'half_low_lr_branch_a_v1','fraction':c['lr_fraction'],'branch_origin_absolute_step':start,'counter_origin':300000,
        'parent_schedule':'Verified completed halfcooldown endpoint groupLR0; explicit LR-only intervention before NEW0 checkpoint',
        'parent_cooldown_plan':old_metadata['cooldown_plan'],'parent_sampler_checkpoint':deepcopy(saved_recipe),
        'initial_layout_rng_sha256':fingerprint(layout_rng.bit_generator.state),'future_resume':'Requires explicit typed lowLR saved-LR oracle; never feed newstates to original cooldown LR oracle'}
    metadata.update(max_steps=start+args.new_steps,new_iterations=args.new_steps,array_loading=loading,
        continuation={'schema':'gated_window_continuation_v1','run_id':args.run_id,'lane':args.lane,'layout':c['layout'],'parent_absolute_step':start,'counter_origin':ORIGIN,'optimizer_order':restored['optimizer_order'],
            'parent_terminal_sha256':c['parent_terminal_sha256'],'parent_last_training_sha256':c['parent_artifact_sha256']['last_training.pt'],
            'zero_reset':True,'single_arm_protocol':None,'paired_protocol':'half_low_lr_pair_a_v1','schedule':'fixedfraction*absolute_inverse_sqrt; no optimizer/scaler/RNG reset','lr_fraction':c['lr_fraction'],'current_numpy_version':np.__version__,
            'root_bound_NEW_cap':args.new_steps,'admission_mode':c['admission_mode'],
            'backend_limit':'Saved computational states restored, followed by explicit grouping intervention; not bitwise trajectory/quality equivalence'},
        window_intervention={'recipe':recipe,'recipe_sha256':recipe_sha,'window_origin_step':window_origin,'runtime_archive_sha256':c['runtime_archive_sha256'],
            'runtime_source_sha256':c['runtime_source_sha256'],'parent_kind':c['parent_kind'],'contract_sha256':sha(args.contract),
            'objective_warning':'Existing6M bucket/window/raw/layout state continues unchanged; only the declared LRfraction is an intervention; no bitwise crossGPU trajectory claim'})
    atomic_json(out/'run_config.json',metadata)
    completed=0;committed=0;active_slot=None;latest=None;checkpoint_cost=0.;admitted=None;admission=None;target=args.new_steps;partial=False;stopped=False;windows=[]
    raw_segment=hashlib.sha256();applied_segment=hashlib.sha256();raw_start=fingerprint(saved_rng['numpy']);layout_start=fingerprint(layout_rng.bit_generator.state)
    def log(row):
        record={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**row}
        with (out/'train.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps(record,allow_nan=False),flush=True)
    def save_best(loss,absolute):
        atomic_torch_save({'config':asdict(cfg),'model':model.state_dict(),'step':absolute,'validation_loss':loss,'seed':42,'dataset_revision':data['dataset_revision']},out/'model.pt')
    def checkpoint(val,label,rng_override=None):
        nonlocal active_slot,committed,latest
        require(completed%64==0 and not partial,'Never commit partial-window fullstate')
        require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),'Nonfinite checkpoint model')
        require(all(bool(torch.isfinite(value).all()) for optimizer in optimizers for entry in optimizer.state.values() for value in entry.values() if isinstance(value,torch.Tensor)),'Nonfinite checkpoint optimizer state')
        require(all(not isinstance(value,(int,float)) or math.isfinite(value) for value in scaler.state_dict().values()),'Nonfinite checkpoint scaler')
        absolute=start+completed;raw_state=rng_override['numpy'] if rng_override else rng.bit_generator.state
        cpu_state=rng_override['cpu'] if rng_override else torch.get_rng_state();cuda_state=rng_override['cuda'] if rng_override else torch.cuda.get_rng_state_all()
        segment=segment_record(c,start,absolute,raw_segment,applied_segment,whole_raw,raw_start,fingerprint(raw_state),layout_start,fingerprint(layout_rng.bit_generator.state))
        segment.update(worker_contract_sha256=sha(args.contract),root_bound_NEW_cap=args.new_steps,admission_mode=c['admission_mode'],admitted_NEW_horizon=target if admission else None)
        sampler={'schema':'gated_window_checkpoint_v1','recipe':recipe,'recipe_sha256':recipe_sha,'window_origin_step':window_origin,'absolute_step':absolute,
            'at_consumed_window_boundary':True,'pending_rows':0,'layout_rng_state':deepcopy(layout_rng.bit_generator.state),'segments':history+[segment],
            'runtime_archive_sha256':c['runtime_archive_sha256'],'runtime_source_sha256':c['runtime_source_sha256'],'admission':admission,'historical_parent_sampler_checkpoint':deepcopy(saved_recipe)}
        metrics={'selected_step':best_step,'best_validation_loss':best,'final_validation_loss':val,'parent_absolute_step':start,'absolute_completed_step':absolute,
            'counter_origin':ORIGIN,'completed_steps':absolute-ORIGIN,'warmstart_iterations':absolute-ORIGIN,'since_resume_iterations':completed,
            'since_resume_sampled_examples':completed*64,'sampled_examples':seen,'optimizer_updates':updates,'skipped_updates':skips,
            'since_resume_optimizer_updates':updates-restored['optimizer_updates'],'since_resume_skipped_updates':skips-restored['skipped_updates'],
            'sampled_indices_sha256':whole_raw.hexdigest(),'lr_absolute_step':absolute,'validation_generation_indices':old_metrics['validation_generation_indices'],
            'sampler_checkpoint':sampler,'low_lr_branch':metadata['low_lr_branch'],'run_id':args.run_id,'lane':args.lane,'layout':c['layout'],'durable_checkpoint_label':label,'warmstart':metadata['warmstart'],
            'root_bound_NEW_cap':args.new_steps,'admission_mode':c['admission_mode'],'admitted_NEW_horizon':target if admission else None}
        slot=1-active_slot if active_slot is not None else 0;directory=out/('durable-slot'+str(slot));directory.mkdir(exist_ok=True)
        for name in ('tokenizer.json','validation.json','validation_legacy.json','data_stats.json'):
            if not (directory/name).exists():shutil.copyfile(args.parent/name,directory/name)
        atomic_json(directory/'run_config.json',metadata)
        atomic_torch_save({'config':asdict(cfg),'model':model.state_dict(),'optimizers':[o.state_dict() for o in optimizers],'scaler':scaler.state_dict(),
            'step':absolute,'lr_absolute_step':absolute,'validation_loss':val,'rng_numpy_generator':deepcopy(raw_state),'rng_torch_cpu':cpu_state,'rng_torch_cuda':cuda_state,
            'sampled_examples':seen,'best_validation_loss':best,'best_step':best_step,'optimizer_updates':updates,'skipped_updates':skips,
            'sampled_indices_sha256':whole_raw.hexdigest(),'seed':42,'dataset_revision':data['dataset_revision'],'warmstart':metadata['warmstart'],'sampler_checkpoint':sampler,'low_lr_branch':metadata['low_lr_branch']},directory/'last_training.pt')
        source_model=out/'model.pt';destination_model=directory/'model.pt'
        # If the durable role already points to the current best model, do not
        # create a second name for the same inode: POSIX rename(src,dst) is a
        # no-op when both names resolve to the same existing file.
        if not destination_model.exists() or not os.path.samefile(source_model,destination_model):
            owned_link=directory/('model.pt.link-'+str(absolute)+'-'+str(os.getpid()))
            require(not owned_link.exists(),'Invocation-owned model link path already exists')
            os.link(source_model,owned_link)
            if destination_model.exists() and os.path.samefile(owned_link,destination_model):
                # Remove only the temporary link created above, after proving
                # the destination already names the same file.
                os.unlink(owned_link)
            else:
                os.replace(owned_link,destination_model)
        atomic_json(directory/'metrics.json',metrics)
        for name in REQUIRED_PARENT:
            with (directory/name).open('rb') as stream:os.fsync(stream.fileno())
        fsync_directory(directory)
        hashes={name:sha(directory/name) for name in sorted(REQUIRED_PARENT)}
        pointer={'schema':'gated_window_durable_pointer_v1','slot':directory.name,'absolute_step':absolute,'since_resume_iterations':completed,
            'recipe_sha256':recipe_sha,'contract_sha256':sha(args.contract),'files_sha256':hashes,'metrics':metrics}
        atomic_json(out/'latest_checkpoint.json',pointer)  # Commit AFTER every role is durable.
        active_slot=slot;committed=completed;latest=pointer
        return pointer
    log({'event':'initial_validation','step':0,'absolute_step':start,'loss':initial,'restored':restored,'startup_seconds':time.monotonic()-startup,'intervention':metadata['window_intervention']})
    def request_stop(signum,frame):
        nonlocal stopped;stopped=True
    previous=signal.signal(signal.SIGTERM,request_stop)
    progress.update(0,'initial_checkpoint',force=True);initial_checkpoint_started=time.monotonic();checkpoint(initial,'initial',saved_rng);checkpoint_cost=time.monotonic()-initial_checkpoint_started
    started=time.monotonic();model.train();restore_rng(saved_rng,rng);del saved_rng  # LAST setup before first actual train draw.
    while completed<target:
        now=time.monotonic();remaining=args.deadline-now;recent=max([w['seconds'] for w in windows[-3:]] or [30.]);save_reserve=max(180.,1.5*checkpoint_cost)
        rate=completed/(now-started) if completed else None;eta=(target-completed)/rate if rate else None
        near_finish=eta is not None and eta<=180 and eta+max(30.,1.5*checkpoint_cost)<remaining
        if stopped or (remaining<=save_reserve+1.25*recent and not near_finish):break
        progress.update(completed,'training',force=True,absolute_step=start+completed,committed_new_iterations=committed)
        torch.cuda.synchronize();window_started=time.monotonic();raw,applied,guard=draw_window(rng,layout_rng,arrays,targets,c['layout'])
        raw_bytes=raw.astype('<i8',copy=False).tobytes();applied_bytes=applied.astype('<i8',copy=False).tobytes()
        useful_source=useful_target=allocated_source=allocated_target=0;loss_sum=0.;before_updates=updates;before_skips=skips;cursor=0;norm=None
        for batch_index in range(64):
            now=time.monotonic();estimated_tail=recent*(64-batch_index)/64*1.25+max(30.,1.5*checkpoint_cost)
            if now+30>=args.deadline or (stopped and now+estimated_tail>=args.deadline):partial=True;break
            indices=applied[batch_index*64:(batch_index+1)*64]
            aa=np.asarray(arrays[indices],dtype=np.int64);bb=np.asarray(targets[indices],dtype=np.int64)
            aa=aa[:,:int((aa!=0).sum(1).max())];bb=bb[:,:int((bb!=0).sum(1).max())]
            a=torch.from_numpy(aa).cuda();b=torch.from_numpy(bb).cuda();absolute=start+completed+1;lr=c['lr_fraction']*absolute_lr(absolute)
            for optimizer in optimizers:
                for group in optimizer.param_groups:group['lr']=lr*group['lr_multiplier']
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                loss=torch.nn.functional.cross_entropy(model(a,shifted(b)).flatten(0,1),b.flatten(),ignore_index=PAD)
            require(bool(torch.isfinite(loss)),'Nonfinite window loss')
            scaler.scale(loss).backward()
            for optimizer in optimizers:scaler.unscale_(optimizer)
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);finite=bool(torch.isfinite(grad_norm));norm=float(grad_norm) if finite else None
            if finite:
                for optimizer in optimizers:scaler.step(optimizer)
                updates+=1
            else:skips+=1
            scaler.update();count=int(b.ne(PAD).sum());loss_sum+=loss.item()*count;useful_source+=int(a.ne(PAD).sum());useful_target+=count;allocated_source+=a.numel();allocated_target+=b.numel()
            seen+=64;completed+=1;cursor=batch_index+1
            progress.update(completed,'training',absolute_step=start+completed,window_cursor=cursor,committed_new_iterations=committed,sampler_frontier_pending_rows=(64-cursor)*64)
        torch.cuda.synchronize();window_seconds=time.monotonic()-window_started
        if partial:
            atomic_torch_save({'schema':'gated_window_partial_model_NONRESUMABLE_v1','config':asdict(cfg),'model':model.state_dict(),'step':start+completed,
                'seed':42,'dataset_revision':data['dataset_revision'],'nonresumable_reason':'Ahead-of-consumption raw/layout RNG and partial optimizer frontier; use last committed durable pointer, not this model'},out/'partial_model_NONRESUMABLE.pt')
            atomic_json(out/'partial_progress_NONRESUMABLE.json',{'complete':False,'since_resume_iterations':completed,'absolute_step':start+completed,
                'last_durable_pointer':latest,'consumed_in_window_rows':cursor*64,'raw_rows':raw.tolist(),'applied_rows':applied.tolist(),
                'ahead_raw_generator_state':rng.bit_generator.state,'ahead_layout_rng_state':layout_rng.bit_generator.state,
                'can_resume_this_model':False,'raw_prefix_digest_excludes_partial_window':whole_raw.hexdigest()})
            log({'event':'partial_window_NONRESUMABLE','step':completed,'absolute_step':start+completed,'window_cursor':cursor,'committed_new_iterations':committed});break
        require(cursor==64 and completed%64==0,'Consumed-window counter differs')
        whole_raw.update(raw_bytes);raw_segment.update(raw_bytes);applied_segment.update(applied_bytes)
        row={'event':'window_completed','step':completed,'absolute_step':start+completed,'window':len(windows),'seconds':window_seconds,**guard,
            'raw_window_sha256':hashlib.sha256(raw_bytes).hexdigest(),'applied_window_sha256':hashlib.sha256(applied_bytes).hexdigest(),
            'duplicate_draws':4096-len(np.unique(raw)),'whole_raw_sha256':whole_raw.hexdigest(),'useful_source_tokens':useful_source,'useful_target_tokens':useful_target,
            'allocated_source_positions':allocated_source,'allocated_target_positions':allocated_target,
            'source_padding_positions':allocated_source-useful_source,'target_padding_positions':allocated_target-useful_target,
            'token_mean_training_loss':loss_sum/useful_target,'optimizer_updates':updates-before_updates,'AMP_skips':skips-before_skips,'last_grad_norm':norm,
            'timing_scope':'Complete synchronized window includes rawguard/lookup/sort/transfers/train; postwindow SHA/duplicate/log costs in whole measured rate'}
        windows.append(row);log(row)
        if completed%1024==0 or completed==target or stopped:
            cost_started=time.monotonic();progress.update(completed,'validation',force=True);val=validation_loss(model,va,vb);legacy_val=validation_loss(model,*legacy)
            require(math.isfinite(val) and math.isfinite(legacy_val),'Nonfinite checkpoint CE')
            if val<best:best=val;best_step=start+completed;save_best(val,best_step)
            progress.update(completed,'checkpoint',force=True);checkpoint(val,'periodic' if completed<target else 'endpoint');checkpoint_cost=max(checkpoint_cost,time.monotonic()-cost_started)
            elapsed=time.monotonic()-started
            if completed>=1024 and admitted is None:
                request={'schema':'half_low_lr_prefix_request_a_v1','run_id':args.run_id,'lr_fraction':c['lr_fraction'],'contract_sha256':sha(args.contract),
                    'prefix_NEW_iterations':completed,'absolute_step':start+completed,'checkpoint_inclusive_seconds':elapsed,'measured_steps_per_second':completed/elapsed,
                    'checkpoint_cost_seconds':checkpoint_cost,'worker_deadline_monotonic':args.deadline,'ready_monotonic':time.monotonic(),
                    'raw_segment_sha256':raw_segment.hexdigest(),'applied_segment_sha256':applied_segment.hexdigest(),'pointer_sha256':sha(out/'latest_checkpoint.json')}
                request_path=args.coordination/(args.run_id+'-request.json');atomic_json(request_path,request)
                wait_end=min(args.deadline-max(180.,1.5*checkpoint_cost),time.monotonic()+120.)
                decision_path=args.coordination/'common_horizon.json'
                while not decision_path.is_file() and time.monotonic()<wait_end and not stopped:time.sleep(.25)
                require(decision_path.is_file() and time.monotonic()<wait_end and not stopped,'Common horizon unavailable before bounded barrier/save reserve')
                decision=json.loads(decision_path.read_text());target=decision['common_NEW_horizon']
                require(decision['schema']=='half_low_lr_common_horizon_a_v1' and decision['requests_sha256'][args.run_id]==sha(request_path) and
                    type(target) is int and completed<=target<=args.new_steps and target%64==0 and decision['root_bound_NEW_cap']==args.new_steps,'Actual paired horizon/request differs')
                admitted=True
                admission={'mode':'paired_measured64_shortening','root_bound_NEW_cap':args.new_steps,'admitted_NEW_horizon':target,'actual_NEW_steps':completed,
                    'admitted':True,'request':request,'decision':decision,'decision_sha256':sha(decision_path)}
                metadata['continuation']['admission']=deepcopy(admission);metadata['max_steps']=start+target;metadata['new_iterations']=target
                atomic_json(out/'run_config.json',metadata);atomic_json(out/'admission.json',admission)
                checkpoint(val,'admission_metadata');checkpoint_cost=max(checkpoint_cost,time.monotonic()-cost_started)
                log({'event':'budget_admission','step':completed,**admission,'post_metadata_fullsave_elapsed_seconds':time.monotonic()-started})
            elapsed=time.monotonic()-started;log({'event':'checkpoint_completed','step':completed,'absolute_step':start+completed,'elapsed_seconds':elapsed,
                'checkpoint_and_CE_seconds':checkpoint_cost,'validation_loss':val,'legacy_validation_loss':legacy_val,'latest_checkpoint':latest['slot'],
                'admitted_NEW_horizon':target if admission else None,'root_bound_NEW_cap':args.new_steps})
            if stopped:break
    if not partial and completed!=committed:
        progress.update(completed,'final_validation',force=True);val=validation_loss(model,va,vb);require(math.isfinite(val),'Nonfinite final CE')
        if val<best:best=val;best_step=start+completed;save_best(val,best_step)
        progress.update(completed,'checkpoint',force=True);checkpoint(val,'controlled_stop')
    result={'schema':'half_low_lr_worker_result_a_v1','single_arm_protocol':None,'paired_protocol':'half_low_lr_pair_a_v1','run_id':args.run_id,'lane':args.lane,'layout':c['layout'],'parent_kind':c['parent_kind'],'parent_absolute_step':start,
        'requested_new_iterations':args.new_steps,'admission_mode':c['admission_mode'],'admission':admission,'admitted_NEW_horizon':target if admission else None,
        'actual_model_iterations':completed,'durable_new_iterations':committed,
        'durable_absolute_step':start+committed,'partial_nonresumable':partial,'exact_requested_horizon_complete':not partial and completed==committed==args.new_steps,
        'admitted_horizon_complete':not partial and admitted is True and admission is not None and completed==committed==target,
        'durable_fullstate_available':latest is not None,'latest_checkpoint':latest,'budget_admitted':admitted,'stop_requested':stopped,
        'whole_worker_seconds':time.monotonic()-startup,'training_and_checkpoints_seconds':time.monotonic()-started,
        'recipe_sha256':recipe_sha,'windows_completed':len(windows),'selected_step':best_step,'best_validation_loss':best,
        'past_lineage':'Original optimizer/RNG reset at300k remains explicit; no pre-reset history reconstructed',
        'intervention_warning':metadata['window_intervention']['objective_warning']}
    atomic_json(out/'worker_result.json',result);log({'event':'finished','step':completed,**result});progress.update(completed,'finished',force=True,committed_new_iterations=committed)
    signal.signal(signal.SIGTERM,previous);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--lane',choices=('control','treatment'),required=True);p.add_argument('--contract',type=Path,required=True);p.add_argument('--terminal',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--source-archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--new-steps',type=int,required=True);p.add_argument('--deadline',type=float,required=True);p.add_argument('--coordination',type=Path,required=True);p.add_argument('--resident',action='store_true');a=p.parse_args()
    print(json.dumps(run(a),allow_nan=False),flush=True)
