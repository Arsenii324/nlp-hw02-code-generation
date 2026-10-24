"""Source-only recipe-aware uniform/bucket continuation after exact gated/control restore.

Source-only retained1024 prefix then common-horizon cooldown; no targets/filtering/reset. Fullstates commit at consumed windows
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
    require(c['schema']=='gated_cooldown_continuation_a_v1' and c['verified_actual_values'] is True,'Unbound window continuation: actual roles/hashes required')
    require(safe_run_id(args.run_id) and c['run_id']==args.run_id,'Explicit safe bound run identity required')
    require(c['layout']=='bucket' and args.lane=='control' and args.new_steps==81920 and c['parent_kind']=='original_uniform_gated','Initial cooldown source supports original345k control/bucket/rootcap81920 only; descendants require explicit future adapter')
    from nlp_hw02.gated_cooldown_schedule_a import prospective_plan,digest
    require(c['cooldown_plan']==prospective_plan(c['cooldown_arm']) and c['cooldown_plan_sha256']==digest(c['cooldown_plan']),'Immutable prospective schedule plan differs')
    require(args.coordination.is_dir() and args.coordination.resolve().parent==args.output.resolve().parent and args.coordination.name=='cooldown-coordination','Parent-owned shared coordination root required')
    require(c['lane']==args.lane and c['new_steps']==args.new_steps and args.new_steps>=1024 and args.new_steps%64==0,'Root-bound NEW horizon must be>=1024 and64-aligned')
    require(c['counter_origin']==ORIGIN and c['data_layout']=='gated_window64_v1','Wrong counter/data intervention')
    require(c['admission_mode'] in ('fixed','measured64_shortening'),'Explicit root-bound admission mode required')
    require(type(c['allow_incomplete_window_parent']) is bool,'Explicit incomplete-parent policy required')
    require(sha(args.terminal)==c['parent_terminal_sha256'],'Actual parent terminal hash differs')
    terminal=json.loads(args.terminal.read_text())
    require(safe_relative(c['parent_run_path']) and args.parent.resolve()==(args.terminal.parent/c['parent_run_path']).resolve() and args.parent.resolve().is_relative_to(args.terminal.parent.resolve()),'Actual parent slot path differs/escapes')
    require(terminal['source_sha256']==c['parent_source_sha256'] and terminal['driver_sha256']==c['parent_driver_sha256'],'Actual parent source/driver differs')
    require(safe_relative(c['parent_manifest_path']) and terminal['artifact_sha256'][c['parent_manifest_path']]==c['parent_manifest_sha256'] and sha(args.terminal.parent/c['parent_manifest_path'])==c['parent_manifest_sha256'],'Actual parent manifest differs')
    if c['parent_kind']=='original_uniform_gated':
        require(not c['allow_incomplete_window_parent'] and terminal['training_complete'] is True,'Original source requires completed original training')
        require(c['parent_source_sha256']==ORIGINAL_SOURCE_SHA and c['parent_driver_sha256']==ORIGINAL_DRIVER_SHA and c['parent_manifest_path']=='experiment-manifest.json' and c['parent_manifest_sha256']==ORIGINAL_MANIFEST_SHA,'Original frozen source lineage differs')
        require(c['parent_run_path']=='run-'+args.lane,'Matched snapshot is not an original endpoint/fullstate parent')
        observed={row['id']:row for row in terminal['lanes']}[args.lane];expected=45000 if args.lane=='control' else 30000
        require(observed['exit']==0 and observed['stopped'] is False and observed['observation']['exact'] is True and observed['observation']['steps']==expected,'Actual original lane incomplete')
    elif c['parent_kind']=='same_recipe_window':
        require(c['parent_source_sha256']==c['runtime_archive_sha256'] and c['parent_driver_sha256']==c['runtime_source_sha256']['scripts/kaggle_gated_window_continuation_b.py'],'Descendant must use exact same reviewed runtime')
        require(terminal['artifact_hashes_complete'] is True and not terminal.get('cleanup_incomplete') and bool(terminal['cleanup']) and all(r['reaped'] and r['group_exited'] for r in terminal['cleanup']),'Descendant parent must be terminal and immutable with verified groups')
        require(c['allow_incomplete_window_parent'] or terminal['training_complete'] is True,'Incomplete descendant parent requires explicit root permission')
    else:raise ValueError('Unsupported parent recipe kind')
    require(set(c['parent_artifact_sha256'])==REQUIRED_PARENT,'Exact endpoint/best/metadata allowlist required')
    for name,digest in c['parent_artifact_sha256'].items():
        require(terminal['artifact_sha256'][c['parent_run_path']+'/'+name]==digest and sha(args.parent/name)==digest,'Actual parent bytes differ:'+name)
    runtime=Path(__file__).resolve().parents[1]
    require(sha(args.source_archive)==c['runtime_archive_sha256'],'Exact current runtime archive differs')
    frozen={**FROZEN_RUNTIME,'src/nlp_hw02/gated_resuming.py':'59edf37be73c37e63f924dae987072c4b2ee2c83d8bd9e2c8135c6ec3edc5361'}
    require(all(c['runtime_source_sha256'].get(n)==v for n,v in frozen.items()),'Original model/optimizer/strict restore implementation changed')
    require(all(n in c['runtime_source_sha256'] for n in ('src/nlp_hw02/gated_cooldown_recipe_b.py','src/nlp_hw02/gated_cooldown_schedule_a.py','scripts/gated_cooldown_coordination_b.py','scripts/train_gated_cooldown_prefix_c.py','scripts/kaggle_gated_cooldown_prefix_c.py')),'New recipe source identities missing')
    for name,digest in c['runtime_source_sha256'].items():
        require(safe_relative(name) and sha(runtime/name)==digest,'Actual runtime source differs:'+name)
    require(sha(args.data/'cache_manifest.json')==c['cache_manifest_sha256']=='8ce0939534bdb5f2b791d19d0ab13f6b4c79402b8627fecad4e9902cd08bb13c','Original6M cache identity differs')
    cache=json.loads((args.data/'cache_manifest.json').read_text());require(cache['complete'] is True and cache['checksums']==c['cache_checksums'],'Complete bound cache checksums differ')
    for name in ('tokenizer.json','validation.json','validation_legacy.json','data_stats.json'):
        require(sha(args.data/name)==c['cache_checksums'][name]==c['parent_artifact_sha256'][name],'Parent/current metadata differs:'+name)
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
    from nlp_hw02.gated_cooldown_recipe_b import (recipe_for,restore_window_training,restore_rng,absolute_lr,
        validate_window_parent,draw_window,segment_record,fingerprint)
    from nlp_hw02.gated_cooldown_schedule_a import scheduled_lr,checkpoint_schedule,PREFIX
    from gated_cooldown_coordination_b import request_record,await_binding
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
    metadata=deepcopy(old_metadata);metadata.update(max_steps=start+args.new_steps,new_iterations=args.new_steps,array_loading=loading,
        continuation={'schema':'gated_window_continuation_v1','run_id':args.run_id,'lane':args.lane,'layout':c['layout'],'parent_absolute_step':start,'counter_origin':ORIGIN,'optimizer_order':restored['optimizer_order'],
            'parent_terminal_sha256':c['parent_terminal_sha256'],'parent_last_training_sha256':c['parent_artifact_sha256']['last_training.pt'],
            'zero_reset':True,'schedule':'retained1024 absoluteISQRT then immutableH remaining-envelope cooldown; no warmup/reset','current_numpy_version':np.__version__,
            'root_bound_NEW_cap':args.new_steps,'admission_mode':c['admission_mode'],
            'backend_limit':'Saved computational states restored, followed by explicit grouping intervention; not bitwise trajectory/quality equivalence'},
        window_intervention={'recipe':recipe,'recipe_sha256':recipe_sha,'window_origin_step':window_origin,'runtime_archive_sha256':c['runtime_archive_sha256'],
            'runtime_source_sha256':c['runtime_source_sha256'],'parent_kind':c['parent_kind'],'contract_sha256':sha(args.contract),
            'objective_warning':'Original draw order retained; inactive layout RNG unchanged; exact backend trajectory remains unproved' if c['layout']=='uniform' else 'Uniform raw draws preserved; grouping changes batch token means, gradients, dropout shapes and optimizer trajectory'})
    metadata.update(cooldown_plan=deepcopy(c['cooldown_plan']),cooldown_plan_sha256=c['cooldown_plan_sha256'])
    atomic_json(out/'run_config.json',metadata)
    cooldown_binding=None;cooldown_decision=None;postrelease_anchor=None;postrelease_anchor_completed=None;prefix_checkpoint_inclusive_seconds=None
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
            'runtime_archive_sha256':c['runtime_archive_sha256'],'runtime_source_sha256':c['runtime_source_sha256'],'admission':admission}
        sampler['cooldown_checkpoint']=checkpoint_schedule(absolute,c['cooldown_plan'],cooldown_binding)
        metrics={'selected_step':best_step,'best_validation_loss':best,'final_validation_loss':val,'parent_absolute_step':start,'absolute_completed_step':absolute,
            'counter_origin':ORIGIN,'completed_steps':absolute-ORIGIN,'warmstart_iterations':absolute-ORIGIN,'since_resume_iterations':completed,
            'since_resume_sampled_examples':completed*64,'sampled_examples':seen,'optimizer_updates':updates,'skipped_updates':skips,
            'since_resume_optimizer_updates':updates-restored['optimizer_updates'],'since_resume_skipped_updates':skips-restored['skipped_updates'],
            'sampled_indices_sha256':whole_raw.hexdigest(),'lr_absolute_step':absolute,'validation_generation_indices':old_metrics['validation_generation_indices'],
            'sampler_checkpoint':sampler,'cooldown_checkpoint':sampler['cooldown_checkpoint'],'run_id':args.run_id,'lane':args.lane,'layout':c['layout'],'durable_checkpoint_label':label,'warmstart':metadata['warmstart'],
            'root_bound_NEW_cap':args.new_steps,'admission_mode':c['admission_mode'],'admitted_NEW_horizon':target if admission else None}
        slot=1-active_slot if active_slot is not None else 0;directory=out/('durable-slot'+str(slot));directory.mkdir(exist_ok=True)
        for name in ('tokenizer.json','validation.json','validation_legacy.json','data_stats.json'):
            if not (directory/name).exists():shutil.copyfile(args.parent/name,directory/name)
        atomic_json(directory/'run_config.json',metadata)
        atomic_torch_save({'config':asdict(cfg),'model':model.state_dict(),'optimizers':[o.state_dict() for o in optimizers],'scaler':scaler.state_dict(),
            'step':absolute,'lr_absolute_step':absolute,'validation_loss':val,'rng_numpy_generator':deepcopy(raw_state),'rng_torch_cpu':cpu_state,'rng_torch_cuda':cuda_state,
            'sampled_examples':seen,'best_validation_loss':best,'best_step':best_step,'optimizer_updates':updates,'skipped_updates':skips,
            'sampled_indices_sha256':whole_raw.hexdigest(),'seed':42,'dataset_revision':data['dataset_revision'],'warmstart':metadata['warmstart'],'sampler_checkpoint':sampler,'cooldown_checkpoint':sampler['cooldown_checkpoint']},directory/'last_training.pt')
        # Current best inode is immutable after atomic replacement; link it into the inactive slot.
        link=directory/('model.pt.link-'+str(absolute));os.link(out/'model.pt',link);os.replace(link,directory/'model.pt')
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
        torch.cuda.synchronize();window_started=time.monotonic()
        if completed==PREFIX and cooldown_binding is not None:require(not stopped and time.monotonic()<cooldown_decision['barrier_cutoff_monotonic'],'Common barrier elapsed before firstpostprefix draw; retain boundprefix')
        raw,applied,guard=draw_window(rng,layout_rng,arrays,targets,c['layout'])
        raw_bytes=raw.astype('<i8',copy=False).tobytes();applied_bytes=applied.astype('<i8',copy=False).tobytes()
        useful_source=useful_target=allocated_source=allocated_target=0;loss_sum=0.;before_updates=updates;before_skips=skips;cursor=0;norm=None
        for batch_index in range(64):
            now=time.monotonic();estimated_tail=recent*(64-batch_index)/64*1.25+max(30.,1.5*checkpoint_cost)
            if now+30>=args.deadline or (stopped and now+estimated_tail>=args.deadline):partial=True;break
            indices=applied[batch_index*64:(batch_index+1)*64]
            aa=np.asarray(arrays[indices],dtype=np.int64);bb=np.asarray(targets[indices],dtype=np.int64)
            aa=aa[:,:int((aa!=0).sum(1).max())];bb=bb[:,:int((bb!=0).sum(1).max())]
            a=torch.from_numpy(aa).cuda();b=torch.from_numpy(bb).cuda();absolute=start+completed+1;lr=scheduled_lr(absolute,c['cooldown_plan'],cooldown_binding)
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
            if completed==PREFIX and admitted is None:
                prefix_elapsed=time.monotonic()-started
                request=request_record(args.run_id,c['cooldown_arm'],sha(args.contract),c['cooldown_plan_sha256'],sha(out/'latest_checkpoint.json'),prefix_elapsed,PREFIX/prefix_elapsed,checkpoint_cost,raw_segment.hexdigest(),applied_segment.hexdigest())
                def bind_checkpoint(binding,released):
                    nonlocal cooldown_binding,target,admitted,admission,checkpoint_cost
                    cooldown_binding=deepcopy(binding);target=binding['common_NEW_horizon'];admitted=True
                    decision=json.loads((args.coordination/'common_decision.json').read_text())
                    require(sha(args.coordination/'common_decision.json')==binding['decision_sha256'],'Common decision changed before binding save')
                    admission={'mode':c['admission_mode'],'root_bound_NEW_cap':args.new_steps,'admitted_NEW_horizon':target,'actual_NEW_steps':PREFIX,'admitted':True,
                        'coordination':'common1024_min64_a_v1','common_decision':decision,'decision_sha256':binding['decision_sha256'],'release_sha256':binding['release_sha256']}
                    metadata['continuation']['admission']=deepcopy(admission);metadata['max_steps']=start+target;metadata['new_iterations']=target
                    atomic_json(out/'run_config.json',metadata);atomic_json(out/'admission.json',admission)
                    cost=time.monotonic();checkpoint(val,'common_release_bound' if released else 'common_decision_bound');checkpoint_cost=max(checkpoint_cost,time.monotonic()-cost)
                    return sha(out/'latest_checkpoint.json')
                progress.update(completed,'shared_admission_wait',force=True)
                cooldown_binding,cooldown_decision=await_binding(args.coordination,args.run_id,request,atomic_json,args.deadline,lambda:stopped,bind_checkpoint)
                postrelease_anchor=time.monotonic();postrelease_anchor_completed=completed;prefix_checkpoint_inclusive_seconds=prefix_elapsed
                log({'event':'budget_admission','step':completed,**admission,'checkpoint_inclusive_prefix_seconds':prefix_elapsed,'barrier_and_bound_save_charged_seconds':time.monotonic()-started-prefix_elapsed})
            elif completed==PREFIX+1024 and completed<target:
                guard_now=time.monotonic();require(postrelease_anchor is not None and postrelease_anchor_completed==PREFIX and completed-postrelease_anchor_completed==1024,'Actual postrelease1024-attempt block required')
                postrelease_seconds=guard_now-postrelease_anchor;prefix_seconds=prefix_checkpoint_inclusive_seconds
                require(all(math.isfinite(v) and v>0 for v in (postrelease_seconds,prefix_seconds,checkpoint_cost)),'Actual positive checkpoint-inclusive block/save costs required')
                seconds_per_iteration=max(prefix_seconds/1024,postrelease_seconds/1024);measured_rate=1/seconds_per_iteration
                remaining=args.deadline-guard_now;reserve=max(180.,1.5*checkpoint_cost);remaining_iterations=target-completed
                central_eta=remaining_iterations*seconds_per_iteration;central_forecast=central_eta+reserve;conservative_forecast=1.25*central_eta+reserve
                reject=central_forecast>remaining
                recheck={'schema':'gated_cooldown_postprefix_recheck_d_v1','run_id':args.run_id,'NEW_step':completed,'common_NEW_horizon':target,'guard_monotonic':guard_now,'worker_deadline_monotonic':args.deadline,
                    'postrelease_anchor_monotonic':postrelease_anchor,'postrelease_anchor_NEW_step':postrelease_anchor_completed,'prefix_checkpoint_inclusive_seconds':prefix_seconds,'postrelease_checkpoint_inclusive_seconds':postrelease_seconds,
                    'prefix_rate':1024/prefix_seconds,'postrelease_rate':1024/postrelease_seconds,'future_seconds_per_attempt':seconds_per_iteration,'future_rate':measured_rate,'rate_policy':'slower of actualprefix1024 and actualpostrelease1024, each checkpointinclusive',
                    'remaining_iterations':remaining_iterations,'remaining_worker_seconds':remaining,'running_max_checkpoint_and_CE_seconds':checkpoint_cost,'final_save_reserve_seconds':reserve,
                    'central_eta_seconds':central_eta,'central_forecast_plus_save_seconds':central_forecast,'conservative125_forecast_plus_save_seconds':conservative_forecast,
                    'central_headroom_seconds':remaining-central_forecast,'conservative125_headroom_seconds':remaining-conservative_forecast,'decision':'reject' if reject else 'continue',
                    'reason':'Actual centralETA plus save reserve cannot fit immutableH' if reject else 'CentralETA plus actual save reserve fits; initial uncertainty may be consumed; H unchanged',
                    'uncertainty_policy':'Initial common admission1.25 unchanged; recheck1.25 diagnostic only; ordinary window/deadline/save/stall/nearfinish guards unchanged; future throughput not guaranteed'}
                atomic_json(args.coordination/(args.run_id+('-postprefix_reject.json' if reject else '-postprefix_continue.json')),recheck);log({'event':'postprefix_cost_recheck',**recheck})
                if reject:stopped=True
            elapsed=time.monotonic()-started;log({'event':'checkpoint_completed','step':completed,'absolute_step':start+completed,'elapsed_seconds':elapsed,
                'checkpoint_and_CE_seconds':checkpoint_cost,'validation_loss':val,'legacy_validation_loss':legacy_val,'latest_checkpoint':latest['slot'],
                'admitted_NEW_horizon':target if admission else None,'root_bound_NEW_cap':args.new_steps})
            if stopped:break
    if not partial and completed!=committed:
        progress.update(completed,'final_validation',force=True);val=validation_loss(model,va,vb);require(math.isfinite(val),'Nonfinite final CE')
        if val<best:best=val;best_step=start+completed;save_best(val,best_step)
        progress.update(completed,'checkpoint',force=True);checkpoint(val,'controlled_stop')
    result={'schema':'gated_window_worker_result_v1','run_id':args.run_id,'lane':args.lane,'layout':c['layout'],'parent_kind':c['parent_kind'],'parent_absolute_step':start,
        'requested_new_iterations':args.new_steps,'admission_mode':c['admission_mode'],'admission':admission,'admitted_NEW_horizon':target if admission else None,
        'actual_model_iterations':completed,'durable_new_iterations':committed,
        'durable_absolute_step':start+committed,'partial_nonresumable':partial,'exact_requested_horizon_complete':not partial and completed==committed==args.new_steps,
        'admitted_horizon_complete':not partial and admitted is True and admission is not None and completed==committed==target,
        'durable_fullstate_available':latest is not None,'latest_checkpoint':latest,'budget_admitted':admitted,'stop_requested':stopped,
        'whole_worker_seconds':time.monotonic()-startup,'training_and_checkpoints_seconds':time.monotonic()-started,
        'recipe_sha256':recipe_sha,'windows_completed':len(windows),'selected_step':best_step,'best_validation_loss':best,
        'past_lineage':'Original optimizer/RNG reset at300k remains explicit; no pre-reset history reconstructed',
        'intervention_warning':metadata['window_intervention']['objective_warning'],'cooldown_plan':c['cooldown_plan'],'cooldown_binding':cooldown_binding}
    atomic_json(out/'worker_result.json',result);log({'event':'finished','step':completed,**result});progress.update(completed,'finished',force=True,committed_new_iterations=committed)
    signal.signal(signal.SIGTERM,previous);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--lane',choices=('control','treatment'),required=True);p.add_argument('--contract',type=Path,required=True);p.add_argument('--terminal',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--source-archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--new-steps',type=int,required=True);p.add_argument('--deadline',type=float,required=True);p.add_argument('--coordination',type=Path,required=True);p.add_argument('--resident',action='store_true');a=p.parse_args()
    print(json.dumps(run(a),allow_nan=False),flush=True)
