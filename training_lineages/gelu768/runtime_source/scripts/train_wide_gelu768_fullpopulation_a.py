"""Own GELU768F endpoint continuation with explicit full-population PCG64 segment."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib,json,math,os
from pathlib import Path
import shutil,signal,time

CACHE='695ad7d4acf0a9ff3b95249439ada6033fdc6db8693eaca33fbad1959fe9d9f9'
LOADER='6f3c2a6582fa2f9124ac8c48925248f1468c5cdaa593301466cac58ec2fd3b6c'
MATH={'src/nlp_hw02/model.py':'07e58176c39eed8ed03134074282d74589be97120e6d6569a15f27eee1538b26','src/nlp_hw02/optim.py':'eed8391c8385b5d09d52384fc20c975cdb1332c45e04bbb36a2b8682f7a140e9','src/nlp_hw02/optim_batched.py':'9161d39b568ba472fe132b13e50bc906537104e61aca30a0f7de89efff8e246f','src/nlp_hw02/train.py':'1c98897afe264db8ff1803f586ce183adc799dd9d4382dd63805aa0f980bd406','src/nlp_hw02/training_arrays.py':'4e5bd548a59693ab16b7fae6048e40688cd6cbbc59b332e4eccaa1606efa3ec8','src/nlp_hw02/wide_gelu768_fullpopulation_a.py':'0c9f27068bfeebe3f27f49b90625b8e405dec97983a08877bc42fc56c7cf3534','src/nlp_hw02/wide_gelu768_origin0_resume_a.py':LOADER,'src/nlp_hw02/wide_scratch_recipe_a.py':'21df6a54934c346cdaab69378c41292fe0c150b4cb837aed634edc183e403de0','src/nlp_hw02/compile_engine_observer_a.py':'ff88ec5d7699f545c4ee43d15e920a9cbcaaa7d3f0189ca1a7d5ec97632abd84'}
ROLES={'last_training.pt','model.pt','metrics.json','run_config.json','tokenizer.json','validation.json','validation_legacy.json','data_stats.json'}
def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def fsync_directory(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
def atomic_json(path,value):
    p=Path(path);tmp=p.with_name(p.name+'.next')
    with tmp.open('w') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p);fsync_directory(p.parent)
def atomic_torch_save(value,path):
    import torch
    tmp=path.with_name(path.name+'.next')
    with tmp.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path);fsync_directory(path.parent)
def verify(args):
    c=json.loads(args.contract.read_text());require(c['schema']=='wide_gelu768_fullpopulation_training_contract_a_v1' and c['verified_actual_values'] is True,'Actual root-bound continuation contract required')
    require(c['parent_kind']=='native_wide_origin0_F' and c['architecture']=='gelu768' and c['hardware_policy']=='same_parent_training_hardware' and c['seed']==42 and c['counter_origin']==0,'Explicit strict parent/hardware/origin policy required')
    require(type(c['new_steps']) is int and 1024<=c['new_steps']<=2000000 and c['new_steps']%64==0 and c['new_steps']==args.new_steps,'Fixed64-aligned root NEW horizon required')
    require(c['engine']=='eager' and c['schedule']['mode']=='linear_cooldown' and c['admission_mode']=='fixed_horizon_physical_feasibility','No implicit engine or schedule shortening')
    runtime=Path(__file__).resolve().parents[1];require(sha(args.source_archive)==c['runtime_archive_sha256'],'Actual current archive differs')
    require(all(c['runtime_source_sha256'].get(n)==h for n,h in MATH.items()),'Frozen model/optimizer/loader/draw/math differs')
    require({'scripts/train_wide_gelu768_fullpopulation_a.py','src/nlp_hw02/wide_continuation_schedule_a.py'}<=set(c['runtime_source_sha256']),'Continuation sources missing')
    for n,h in c['runtime_source_sha256'].items():
        p=Path(n);require(not p.is_absolute() and '..' not in p.parts and sha(runtime/p)==h,'Actual runtime member differs:'+n)
    require(sha(args.parent_contract)==c['parent_binding_sha256'],'Actual typed parent contract differs')
    require(not args.resident and all(type(c[k]) is int and 0<=c[k]<2**64 for k in ('population_raw_seed','population_layout_seed')),'Read-only mmap and explicit root new-segment seeds required')
    from nlp_hw02.wide_gelu768_fullpopulation_a import verify_cache
    verify_cache(args.data,c,args.deadline)
    require(math.isfinite(args.deadline) and time.monotonic()<args.deadline,'Finite future work deadline required')
    return c

def run(args):
    startup=time.monotonic();c=verify(args)
    from nlp_hw02.wide_gelu768_origin0_resume_a import verify_parent,restore_wide_origin0,restore_rng_last
    parent_contract=json.loads(args.parent_contract.read_text());descriptor,files,pointer,pc,accepted_parent=verify_parent(parent_contract)
    require(parent_contract['architecture']==c['architecture'],'Declared actual parent architecture differs')
    require(pointer['absolute_step']==pc['new_steps']==accepted_parent['actual_endpoint_step']==400000 and accepted_parent['selected_best_vs_endpoint_roles_checked'] is True,'Only actual complete originalF400k endpoint with distinct CEbest is supported')
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from nlp_hw02.model import Seq2Seq,ModelConfig
    from nlp_hw02.optim import make_optimizers
    from nlp_hw02.wide_scratch_recipe_a import fingerprint,absolute_lr
    from nlp_hw02.wide_gelu768_fullpopulation_a import ROWS,ORIGIN,recipe_for,draw_window,compare_parent_data
    from nlp_hw02.wide_continuation_schedule_a import validate_schedule,scheduled_lr
    from nlp_hw02.train import validation_loss,shifted
    from nlp_hw02.inference import padded_tokens
    from nlp_hw02.tokenizer import PAD
    from nlp_hw02.training_arrays import load_training_arrays
    from nlp_hw02.progress import ProgressWriter
    from nlp_hw02.gated_resuming import optimizer_order
    from nlp_hw02.compile_engine_observer_a import register_engine,observe_engine
    require(str(torch.__version__)=='2.10.0+cu128' and torch.version.cuda=='12.8' and torch.cuda.device_count()==1,'Exact software and one actual visibleGPU required')
    torch.set_num_threads(2);out=args.output;out.mkdir(parents=True,exist_ok=False);progress=ProgressWriter(out/'progress.json');progress.update(0,'loading_arrays',force=True)
    arrays=np.load(args.data/'train_source.npy',mmap_mode='r',allow_pickle=False);targets=np.load(args.data/'train_target.npy',mmap_mode='r',allow_pickle=False);row_ids=np.load(args.data/'train_row_ids.npy',mmap_mode='r',allow_pickle=False)
    require(arrays.shape==(ROWS,256) and targets.shape==(ROWS,192) and arrays.dtype==targets.dtype==np.uint16 and row_ids.shape==(ROWS,2) and row_ids.dtype==np.uint32 and not any(x.flags.writeable for x in (arrays,targets,row_ids)),'Exact pinned read-only full-population tokens/rowIDs required')
    loading={'mode':'read_only_mmap','resident_bytes':0,'token_payload_bytes':arrays.nbytes+targets.nbytes,'rowID_payload_bytes':row_ids.nbytes,'single_population_payload_bytes':arrays.nbytes+targets.nbytes+row_ids.nbytes,'old6M_arrays_read':False}
    data=json.loads((args.data/'data_stats.json').read_text());tok=Tokenizer.from_file(str(args.data/'tokenizer.json'));validation=json.loads((args.data/'validation.json').read_text());legacyrows=json.loads((args.data/'validation_legacy.json').read_text())
    va=padded_tokens(tok,[r['question'] for r in validation],256);vb=padded_tokens(tok,[r['code'] for r in validation],192);legacy=(padded_tokens(tok,[r['question'] for r in legacyrows],256),padded_tokens(tok,[r['code'] for r in legacyrows],192))
    cfg=ModelConfig(**pc['config']);model=Seq2Seq(cfg).cuda();optimizers,opt_info=make_optimizers(model,'muon_grouped');scaler=torch.amp.GradScaler('cuda');rng=np.random.default_rng(42);layout_rng=np.random.default_rng(20261022)
    restored,saved_rng,layout_state=restore_wide_origin0(model,optimizers,scaler,rng,parent_contract,optimizer_info=opt_info,deadline=args.deadline,progress=lambda n:progress.update(0,'parent_prefix_replay',parent_attempts_verified=n))
    order=optimizer_order(model,optimizers);start=restored['absolute_step'];H=args.new_steps;schedule=validate_schedule(c['schedule'],start,H)
    parent_data=json.loads(files['data_stats.json'].read_text());compare_parent_data(parent_data,data)
    for n in ('tokenizer.json','validation.json','validation_legacy.json'):require(sha(args.data/n)==sha(files[n]),'Parent/cache small role differs:'+n)
    current_hw=restored['current_hardware'];parent_hw=restored['parent_hardware'];require(current_hw['name']==parent_hw['name'] and list(current_hw['capability'])==list(parent_hw['compute_capability']),'Cross-hardware continuation requires future explicit policy')
    inherited_raw_state_sha=fingerprint(saved_rng['numpy']);inherited_layout_state_sha=fingerprint(layout_state)
    rng=np.random.default_rng(c['population_raw_seed']);layout_rng=np.random.default_rng(c['population_layout_seed']);phase_layout_state=deepcopy(layout_rng.bit_generator.state);phase_saved_rng={**saved_rng,'numpy':deepcopy(rng.bit_generator.state)}
    recipe,recipe_sha=recipe_for(start,c['population_raw_seed'],c['population_layout_seed']);raw_start=fingerprint(phase_saved_rng['numpy']);layout_start=fingerprint(phase_layout_state)
    transition={'schema':'wide_gelu768_fullpopulation_transition_a_v1','parent_cache_manifest_sha256':'8ce0939534bdb5f2b791d19d0ab13f6b4c79402b8627fecad4e9902cd08bb13c','cache_manifest_sha256':CACHE,'parent_rows':6000000,'rows':ROWS,'parent_absolute_step':start,'new_population_attempt_origin':0,'inherited_raw_state_sha256':inherited_raw_state_sha,'inherited_layout_state_sha256':inherited_layout_state_sha,'new_raw_state_sha256':raw_start,'new_layout_state_sha256':layout_start,'inherited_Torch_CPU_RNG_sha256':hashlib.sha256(saved_rng['cpu'].numpy().tobytes()).hexdigest(),'inherited_Torch_CUDA_RNG_sha256':[hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in saved_rng['cuda']],'reset_only_population_raw_and_layout_generators':True,'optimizer_moments_scaler_or_Torch_RNG_reset':False,'exact_old6M_example_stream_continuation':False,'old_arrays_needed_for_parent_prefix_replay':False,'origin':ORIGIN,'population_binding':c['population_binding']}
    observe_engine(torch);calibration_started=time.monotonic();progress.update(0,'initial_endpoint_validation',force=True);initial=validation_loss(model,va,vb)
    require(math.isfinite(initial) and math.isclose(initial,restored['last_validation_loss'],rel_tol=1e-6,abs_tol=1e-7),'Actual restored endpoint CE differs on samehardware')
    engine=register_engine(model,c['engine']);best=restored['best_validation_loss'];best_step=restored['best_step'];shutil.copyfile(files['model.pt'],out/'model.pt');require(sha(out/'model.pt')==parent_contract['projection']['model.pt']['sha256'],'Inherited CEbest bytes changed')
    parent_updates=restored['optimizer_updates'];parent_skips=restored['skipped_updates'];parent_seen=restored['sampled_examples'];updates=parent_updates;skips=parent_skips;seen=parent_seen
    saved_base=absolute_lr(start);first_base=scheduled_lr(schedule,1);lr_transition={'saved_schedule':'fresh200warmup_absoluteISQRT_fraction1','saved_base_lr':saved_base,'new_initial_base_lr':scheduled_lr(schedule,0),'first_update_base_lr':first_base,'first_update_minus_saved_lr':first_base-saved_base,'first_update_ratio_to_saved_lr':first_base/saved_base,'group_multipliers':[[g['lr_multiplier'] for g in o.param_groups] for o in optimizers]}
    require(scheduled_lr(schedule,0)==saved_base and all(g['lr']==saved_base*g['lr_multiplier'] for o in optimizers for g in o.param_groups),'Restored savedLR must pass unchanged before intervention')
    metadata={'schema':'wide_gelu768_fullpopulation_run_a_v1','training_family':'wide_gelu768_fullpopulation_continuation_a_v1','config':asdict(cfg),'architecture':c['architecture'],'seed':42,'batch_size':64,'data':data,'parent_data_stats':parent_data,'population_transition':transition,'optimizer':opt_info,'optimizer_order':order,'torch':str(torch.__version__),'cuda':torch.version.cuda,'numpy':np.__version__,'hardware':{'name':current_hw['name'],'compute_capability':current_hw['capability'],'visible':1},'engine':engine,'counter_origin':0,'parent_absolute_step':start,'initial_endpoint_validation_loss':initial,'parent':restored,'parent_binding_sha256':sha(args.parent_contract),'parent_contract_sha256':pointer['contract_sha256'],'parent_source_archive_sha256':descriptor['archive'],'recipe':recipe,'recipe_sha256':recipe_sha,'schedule':schedule,'LR_transition':lr_transition,'array_loading':loading,'parameters':descriptor['parameters'],'unique_xsa_scalars':96,'runtime_archive_sha256':c['runtime_archive_sha256'],'runtime_source_sha256':c['runtime_source_sha256'],'contract_sha256':sha(args.contract),'root_bound_NEW_horizon':H,'admission_mode':c['admission_mode'],'fullstate_resume_policy':'ThisfamilyneedsitsownsavedLR/sampleradapter; origin0Floadercannotresumeit','targets':'Unchangedoriginal11811850eligiblecode/EOS/caps;nofilter/teacher/generatedtargets','initialization':'Strictparentendpoint/moments/scaler/RNG—notfreshreset','inherited_best_initial_sha256':sha(out/'model.pt')}
    atomic_json(out/'run_config.json',metadata);completed=committed=0;active_slot=None;latest=None;checkpoint_cost=0.;admission=None;admitted=None;partial=False;stopped=False;stop_reason=None;windows=[];raw_segment=hashlib.sha256();applied_segment=hashlib.sha256();global_raw_segment=hashlib.sha256();global_applied_segment=hashlib.sha256()
    def log(row):
        record={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**row}
        with (out/'train.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps(record,allow_nan=False),flush=True)
    def save_best(val,step):atomic_torch_save({'config':asdict(cfg),'model':model.state_dict(),'step':step,'validation_loss':val,'seed':42,'dataset_revision':data['dataset_revision']},out/'model.pt')
    def checkpoint(val,label,rng_override=None):
        nonlocal active_slot,committed,latest
        require(completed%64==0 and not partial,'Nevercommitpartialwindow');absolute=start+completed;require(updates+skips==absolute and seen==absolute*64,'Cumulative origin0 counters differ')
        require(updates-parent_updates+skips-parent_skips==completed,'New attempt accounting differs')
        require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),'Nonfinite model')
        require(all(bool(torch.isfinite(v).all()) for o in optimizers for x in o.state.values() for v in x.values() if isinstance(v,torch.Tensor)),'Nonfinite optimizer moments')
        require(all(not isinstance(v,(int,float)) or math.isfinite(v) for v in scaler.state_dict().values()),'Nonfinite scaler state')
        raw=deepcopy(rng_override['numpy'] if rng_override else rng.bit_generator.state);cpu=rng_override['cpu'] if rng_override else torch.get_rng_state();cuda=rng_override['cuda'] if rng_override else torch.cuda.get_rng_state_all()
        sampler={'schema':'wide_gelu768_fullpopulation_sampler_a_v1','counter_origin':0,'absolute_step':absolute,'parent_absolute_step':start,'new_iterations':completed,'pending_rows':0,'at_consumed_window_boundary':True,'recipe':recipe,'recipe_sha256':recipe_sha,'new_raw_segment_sha256':raw_segment.hexdigest(),'new_applied_segment_sha256':applied_segment.hexdigest(),'new_global_raw_rowID_segment_sha256':global_raw_segment.hexdigest(),'new_global_applied_rowID_segment_sha256':global_applied_segment.hexdigest(),'rowIDs_sha256':c['cache_checksums']['train_row_ids.npy'],'population_transition':transition,'raw_generator_start_sha256':raw_start,'raw_generator_end_sha256':fingerprint(raw),'layout_rng_start_sha256':layout_start,'layout_rng_end_sha256':fingerprint(layout_rng.bit_generator.state),'layout_rng_state':deepcopy(layout_rng.bit_generator.state),'parent_sampler':restored['parent_sampler'],'parent_raw_digest':restored['parent_raw_digest'],'parent_applied_row_digest':restored['parent_applied_row_digest'],'parent_applied_digest_scope':restored['applied_digest_scope'],'no_combined_historical_hasher_state_claim':True,'schedule':schedule,'admission':admission}
        metrics={'schema':'wide_gelu768_fullpopulation_metrics_a_v1','counter_origin':0,'selected_step':best_step,'best_validation_loss':best,'final_validation_loss':val,'absolute_completed_step':absolute,'completed_steps':absolute,'parent_absolute_step':start,'since_resume_iterations':completed,'sampled_examples':seen,'optimizer_updates':updates,'skipped_updates':skips,'since_resume_optimizer_updates':updates-parent_updates,'since_resume_skipped_updates':skips-parent_skips,'lr_absolute_step':absolute,'sampler_checkpoint':sampler,'checkpoint_label':label,'root_bound_NEW_horizon':H,'schedule':schedule}
        slot=1-active_slot if active_slot is not None else 0;directory=out/('durable-slot'+str(slot));directory.mkdir(exist_ok=True)
        for n in ('tokenizer.json','validation.json','validation_legacy.json','data_stats.json'):
            if not (directory/n).exists():shutil.copyfile(args.data/n,directory/n)
        atomic_json(directory/'run_config.json',metadata)
        atomic_torch_save({'schema':'wide_gelu768_fullpopulation_training_state_a_v1','config':asdict(cfg),'model':model.state_dict(),'optimizers':[o.state_dict() for o in optimizers],'optimizer_order':order,'scaler':scaler.state_dict(),'step':absolute,'lr_absolute_step':absolute,'validation_loss':val,'best_validation_loss':best,'best_step':best_step,'rng_numpy_generator':raw,'rng_torch_cpu':cpu,'rng_torch_cuda':cuda,'sampled_examples':seen,'optimizer_updates':updates,'skipped_updates':skips,'seed':42,'dataset_revision':data['dataset_revision'],'counter_origin':0,'parent_absolute_step':start,'since_resume_iterations':completed,'sampler_checkpoint':sampler,'population_transition':transition,'schedule':schedule,'runtime_archive_sha256':c['runtime_archive_sha256'],'runtime_source_sha256':c['runtime_source_sha256'],'parent_binding_sha256':sha(args.parent_contract)},directory/'last_training.pt')
        destination=directory/'model.pt'
        if not destination.exists() or not os.path.samefile(out/'model.pt',destination):
            link=directory/('model.pt.link-'+str(completed)+'-'+str(os.getpid()));require(not link.exists(),'Ownedmodellinkexists');os.link(out/'model.pt',link)
            if destination.exists() and os.path.samefile(link,destination):os.unlink(link)
            else:os.replace(link,destination)
        atomic_json(directory/'metrics.json',metrics)
        for n in ROLES:
            with (directory/n).open('rb') as f:os.fsync(f.fileno())
        fsync_directory(directory);hashes={n:sha(directory/n) for n in sorted(ROLES)};latest={'schema':'wide_gelu768_fullpopulation_durable_pointer_a_v1','slot':directory.name,'absolute_step':absolute,'counter_origin':0,'since_resume_iterations':completed,'recipe_sha256':recipe_sha,'schedule':schedule,'contract_sha256':sha(args.contract),'files_sha256':hashes,'metrics':metrics};atomic_json(out/'latest_checkpoint.json',latest);active_slot=slot;committed=completed;return latest
    def request_stop(*_):
        nonlocal stopped;stopped=True
    previous=signal.signal(signal.SIGTERM,request_stop);initial_save_started=time.monotonic();checkpoint(initial,'initial_endpoint_NEW_population0',phase_saved_rng);checkpoint_cost=time.monotonic()-initial_save_started
    model.train();layout_rng.bit_generator.state=deepcopy(phase_layout_state);restore_rng_last(phase_saved_rng,rng);del saved_rng,phase_saved_rng # LAST: new declared population states plus exact inherited TorchCPU/CUDA before first NEW draw.
    started=calibration_started;log({'event':'initial_endpoint_validation','step':0,'absolute_step':start,'loss':initial,'inherited_best_loss':best,'inherited_best_step':best_step,'LR_transition':lr_transition})
    while completed<H:
        now=time.monotonic();remaining=args.deadline-now;recent=max([w['seconds'] for w in windows[-3:]] or [30.]);reserve=max(180.,1.5*checkpoint_cost);rate=completed/(now-started) if completed else None;eta=(H-completed)/rate if rate else None;near_finish=eta is not None and eta<=180 and eta+max(30.,1.5*checkpoint_cost)<remaining
        if stopped or (remaining<=reserve+1.25*recent and not near_finish):stop_reason='signal' if stopped else 'window_and_save_budget';break
        progress.update(completed,'training',absolute_step=start+completed,committed_new_iterations=committed,force=True);torch.cuda.synchronize();window_started=time.monotonic();raw,applied,guard=draw_window(rng,layout_rng,arrays,targets,'bucket',recipe);raw_bytes=raw.astype('<i8',copy=False).tobytes();applied_bytes=applied.astype('<i8',copy=False).tobytes()
        useful_source=useful_target=allocated_source=allocated_target=0;loss_sum=0.;before_updates=updates;before_skips=skips;cursor=0;norm=None
        for batch_index in range(64):
            now=time.monotonic();estimated_tail=recent*(64-batch_index)/64*1.25+max(30.,1.5*checkpoint_cost)
            if now+30>=args.deadline or (stopped and now+estimated_tail>=args.deadline):partial=True;break
            indices=applied[batch_index*64:(batch_index+1)*64]
            aa=np.asarray(arrays[indices],dtype=np.int64);bb=np.asarray(targets[indices],dtype=np.int64)
            aa=aa[:,:int((aa!=0).sum(1).max())];bb=bb[:,:int((bb!=0).sum(1).max())]
            a=torch.from_numpy(aa).cuda();b=torch.from_numpy(bb).cuda();absolute=start+completed+1;lr=scheduled_lr(schedule,completed+1)
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
            progress.update(completed,'training',absolute_step=start+completed,window_cursor=cursor,committed_new_iterations=committed)
        torch.cuda.synchronize();window_seconds=time.monotonic()-window_started
        if partial:
            atomic_torch_save({'schema':'wide_gelu768_fullpopulation_partial_NONRESUMABLE_a_v1','config':asdict(cfg),'model':model.state_dict(),'step':start+completed,'nonresumable':True,'reason':'raw/layout RNG ahead ofpartialwindow;usecommittedpointer'},out/'partial_model_NONRESUMABLE.pt')
            atomic_json(out/'partial_progress_NONRESUMABLE.json',{'complete':False,'new_iterations':completed,'last_durable_pointer':latest,'consumed_rows':cursor*64,'raw_rows':raw.tolist(),'applied_rows':applied.tolist(),'ahead_raw_generator_state':rng.bit_generator.state,'ahead_layout_rng_state':layout_rng.bit_generator.state});stop_reason='partial_window_deadline';break
        require(cursor==64 and completed%64==0,'Consumed64windowrequired');raw_segment.update(raw_bytes);applied_segment.update(applied_bytes);global_raw_bytes=np.asarray(row_ids[raw],dtype='<u4').tobytes();global_applied_bytes=np.asarray(row_ids[applied],dtype='<u4').tobytes();global_raw_segment.update(global_raw_bytes);global_applied_segment.update(global_applied_bytes)
        row={'event':'window_completed','step':completed,'absolute_step':start+completed,'seconds':window_seconds,**guard,'raw_window_sha256':hashlib.sha256(raw_bytes).hexdigest(),'applied_window_sha256':hashlib.sha256(applied_bytes).hexdigest(),'new_raw_segment_sha256':raw_segment.hexdigest(),'new_applied_segment_sha256':applied_segment.hexdigest(),'global_raw_rowID_window_sha256':hashlib.sha256(global_raw_bytes).hexdigest(),'global_applied_rowID_window_sha256':hashlib.sha256(global_applied_bytes).hexdigest(),'new_global_raw_rowID_segment_sha256':global_raw_segment.hexdigest(),'new_global_applied_rowID_segment_sha256':global_applied_segment.hexdigest(),'useful_source_tokens':useful_source,'useful_target_tokens':useful_target,'allocated_source_positions':allocated_source,'allocated_target_positions':allocated_target,'token_mean_training_loss':loss_sum/useful_target,'optimizer_updates':updates-before_updates,'AMP_skips':skips-before_skips,'last_grad_norm':norm,'engine_observation':observe_engine(torch)};windows.append(row);log(row)
        if completed%1024==0 or completed==H or stopped:
            cost=time.monotonic();val=validation_loss(model,va,vb);legacy_val=validation_loss(model,*legacy);require(math.isfinite(val) and math.isfinite(legacy_val),'Nonfinite checkpoint CE')
            if val<best:best=val;best_step=start+completed;save_best(val,best_step)
            checkpoint(val,'periodic' if completed<H else 'endpoint');checkpoint_cost=max(checkpoint_cost,time.monotonic()-cost);elapsed=time.monotonic()-started
            if completed>=1024 and admitted is None:
                require(updates-parent_updates>0,'No finite NEW optimizer updates');remaining=args.deadline-time.monotonic();rate=completed/elapsed;reserve=max(180.,1.5*checkpoint_cost);eta=(H-completed)/rate;admitted=eta+reserve<remaining
                admission={'mode':'fixed_horizon_physical_feasibility','root_NEW_horizon':H,'actual_NEW_attempts':completed,'checkpoint_inclusive_seconds':elapsed,'rate':rate,'remaining_seconds':remaining,'central_remaining_training_seconds':eta,'inflated1p25_training_seconds_diagnostic':1.25*eta,'save_reserve_seconds':reserve,'fixed_horizon_admitted':admitted,'horizon_shortened':False};metadata['admission']=admission;atomic_json(out/'run_config.json',metadata);atomic_json(out/'admission.json',admission);checkpoint(val,'admission_metadata');checkpoint_cost=max(checkpoint_cost,time.monotonic()-cost);log({'event':'budget_admission',**admission})
                if not admitted:stop_reason='fixed_H_not_admitted';break
            log({'event':'checkpoint_completed','step':completed,'absolute_step':start+completed,'validation_loss':val,'legacy_validation_loss':legacy_val,'elapsed_seconds':time.monotonic()-started,'checkpoint_and_CE_seconds':checkpoint_cost,'latest_checkpoint':latest['slot']})
    if not partial and completed!=committed:
        val=validation_loss(model,va,vb);require(math.isfinite(val),'Nonfinite controlledstopCE')
        if val<best:best=val;best_step=start+completed;save_best(val,best_step)
        checkpoint(val,'controlled_stop')
    result={'schema':'wide_gelu768_fullpopulation_worker_result_a_v1','training_family':'wide_gelu768_fullpopulation_continuation_a_v1','architecture':c['architecture'],'counter_origin':0,'parent_absolute_step':start,'actual_model_iterations':completed,'durable_new_iterations':committed,'durable_absolute_step':start+committed,'requested_new_iterations':H,'fixed_NEW_horizon':H,'horizon_shortened':False,'fixed_horizon_admitted':admitted,'admitted_horizon_complete':admitted is True and not partial and completed==committed==H,'partial_nonresumable':partial,'optimizer_updates':updates,'skipped_updates':skips,'since_resume_optimizer_updates':updates-parent_updates,'since_resume_skipped_updates':skips-parent_skips,'selected_step':best_step,'best_validation_loss':best,'latest_checkpoint':latest,'stop_reason':stop_reason,'whole_worker_seconds':time.monotonic()-startup,'training_and_checkpoint_seconds':time.monotonic()-started,'parent_binding_sha256':sha(args.parent_contract),'source_archive_sha256':c['runtime_archive_sha256'],'inherited_lineage':True}
    atomic_json(out/'worker_result.json',result);log({'event':'finished',**result});signal.signal(signal.SIGTERM,previous);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--contract',type=Path,required=True);p.add_argument('--parent-contract',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--source-archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--new-steps',type=int,required=True);p.add_argument('--deadline',type=float,required=True);p.add_argument('--resident',action='store_true');a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.is_dir():atomic_json(a.output/'failure.json',{'complete':False,'failure_type':type(error).__name__,'durable_pointer_present':(a.output/'latest_checkpoint.json').is_file()})
        raise
