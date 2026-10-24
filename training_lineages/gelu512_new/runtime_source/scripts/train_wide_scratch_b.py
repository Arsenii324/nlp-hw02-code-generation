"""Own512/4+4 fresh encoder-decoder; fixed targets, origin0, finite window commits."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib,json,math,os
from pathlib import Path
import shutil,signal,time

CONFIG={'vocab_size':2048,'d_model':512,'heads':8,'encoder_layers':4,'decoder_layers':4,'ff':2048,'dropout':.1,'max_source':256,'max_target':192,'exclusive_self_attention':False,'learnable_xsa':True,'activation':'gelu','loop_repeats':1,'share_loop_weights':True,'residual_scale':1.,'loop_order':'stack','loop_share_scope':'block','gated_repeats':False,'repeat_residual_scale':.5}
CACHE='8ce0939534bdb5f2b791d19d0ab13f6b4c79402b8627fecad4e9902cd08bb13c'
MATH={'src/nlp_hw02/model.py':'07e58176c39eed8ed03134074282d74589be97120e6d6569a15f27eee1538b26','src/nlp_hw02/optim.py':'eed8391c8385b5d09d52384fc20c975cdb1332c45e04bbb36a2b8682f7a140e9','src/nlp_hw02/optim_batched.py':'9161d39b568ba472fe132b13e50bc906537104e61aca30a0f7de89efff8e246f','src/nlp_hw02/train.py':'1c98897afe264db8ff1803f586ce183adc799dd9d4382dd63805aa0f980bd406','src/nlp_hw02/training_arrays.py':'4e5bd548a59693ab16b7fae6048e40688cd6cbbc59b332e4eccaa1606efa3ec8'}
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
    c=json.loads(args.contract.read_text());require(c['schema']=='wide_scratch_training_a_v1' and c['verified_actual_values'] is True,'Bound actual scratch contract required')
    require(c['config']==CONFIG and c['seed']==42 and c['counter_origin']==0 and c['initialization']=='fresh_own_model07e_no_parent_weights' and c['lr_fraction']==1.,'Fresh512 origin0 recipe required')
    require(type(c['new_steps']) is int and 1024<=c['new_steps']<=2000000 and c['new_steps']%64==0 and c['new_steps']==args.new_steps,'Explicit retained1024 finite cap required')
    require(c['engine'] in ('eager','inplace-compile') and c['admission_mode']=='single_measured64_shortening','Explicit engine/actual admission required')
    runtime=Path(__file__).resolve().parents[1];require(sha(args.source_archive)==c['runtime_archive_sha256'],'Source archive differs')
    require(all(c['runtime_source_sha256'].get(n)==h for n,h in MATH.items()),'Own model/optimizer/loss/array source changed')
    require('scripts/train_wide_scratch_b.py' in c['runtime_source_sha256'] and 'src/nlp_hw02/wide_scratch_recipe_a.py' in c['runtime_source_sha256'],'Fresh scratch source identities missing')
    for n,h in c['runtime_source_sha256'].items():
        p=Path(n);require(not p.is_absolute() and '..' not in p.parts and sha(runtime/p)==h,'Runtime member differs:'+n)
    require(sha(args.data/'cache_manifest.json')==c['cache_manifest_sha256']==CACHE,'Exact6M cache required');cache=json.loads((args.data/'cache_manifest.json').read_text());require(cache['complete'] is True and cache['checksums']==c['cache_checksums'],'Exact6M checksums required')
    for n,h in c['cache_checksums'].items():require(sha(args.data/n)==h,'Actual cached input bytes differ:'+n)
    require(math.isfinite(args.deadline) and time.monotonic()<args.deadline,'Finite remaining worker deadline required')
    return c

def run(args):
    startup=time.monotonic();c=verify(args)
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from nlp_hw02.model import Seq2Seq,ModelConfig
    from nlp_hw02.optim import make_optimizers
    from nlp_hw02.wide_scratch_recipe_a import recipe_for,draw_window,fingerprint,absolute_lr
    from nlp_hw02.train import validation_loss,shifted
    from nlp_hw02.inference import padded_tokens
    from nlp_hw02.tokenizer import PAD
    from nlp_hw02.training_arrays import load_training_arrays
    from nlp_hw02.progress import ProgressWriter
    from nlp_hw02.gated_resuming import optimizer_order
    from nlp_hw02.compile_engine_observer_a import register_engine,observe_engine
    require(str(torch.__version__)=='2.10.0+cu128' and torch.version.cuda=='12.8' and torch.cuda.device_count()==1,'One visible exactTorch2.10/cu128 GPU required')
    name=torch.cuda.get_device_name(0);cap=torch.cuda.get_device_capability(0);require(('T4' in name and cap==(7,5)) or ('A100' in name and cap==(8,0)) or ('H100' in name and cap==(9,0)),'Actual T4/A100/H100 supported; never spoof hardware')
    torch.set_num_threads(2);out=args.output;out.mkdir(parents=True,exist_ok=False);progress=ProgressWriter(out/'progress.json');progress.update(0,'loading_arrays',force=True)
    arrays,targets,loading=load_training_arrays(args.data,resident=args.resident,phase=lambda phase:progress.update(0,phase,force=True));require(arrays.shape==(6000000,256) and targets.shape==(6000000,192) and arrays.dtype==targets.dtype==np.uint16,'Exact unchanged6M token arrays required')
    data=json.loads((args.data/'data_stats.json').read_text());tok=Tokenizer.from_file(str(args.data/'tokenizer.json'));validation=json.loads((args.data/'validation.json').read_text());legacyrows=json.loads((args.data/'validation_legacy.json').read_text())
    va=padded_tokens(tok,[r['question'] for r in validation],256);vb=padded_tokens(tok,[r['code'] for r in validation],192);legacy=(padded_tokens(tok,[r['question'] for r in legacyrows],256),padded_tokens(tok,[r['code'] for r in legacyrows],192))
    torch.manual_seed(42);torch.cuda.manual_seed_all(42);rng=np.random.default_rng(42);layout_rng=np.random.default_rng(20261022)
    cfg=ModelConfig(**CONFIG);model=Seq2Seq(cfg).cuda();optimizers,opt_info=make_optimizers(model,'muon_grouped');scaler=torch.amp.GradScaler('cuda');order=optimizer_order(model,optimizers)
    require(sum(p.numel() for p in model.parameters())==30705728,'Actual512 parameter count differs');gates=[p for n,p in model.named_parameters() if n.endswith('.xsa_alpha')];require(sum(p.numel() for p in gates)==64 and all(tuple(p.shape)==(8,) and bool(p.eq(0).all()) for p in gates),'64 zero-initialized learnedXSA scalars required')
    require(bool(model.embedding.weight[PAD].eq(0).all()) and not hasattr(model,'lm_head'),'Exact07e tied output/embedding init required')
    for optimizer in optimizers:
        for group in optimizer.param_groups:group['lr']=0.  # Fresh absolute step0; real warmup begins atstep1.
    evaluation_indices=[int(i) for i in np.random.default_rng(20261001).permutation(len(validation))[:256]]
    recipe,recipe_sha=recipe_for('bucket');require(c['recipe']==recipe and c['recipe_sha256']==recipe_sha,'Scratch raw/layout recipe differs')
    saved_rng={'numpy':deepcopy(rng.bit_generator.state),'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()};raw_start=fingerprint(saved_rng['numpy']);layout_start=fingerprint(layout_rng.bit_generator.state)
    # Preload diagnostics before final RNG restore in both engine policies.
    observe_engine(torch);calibration_started=time.monotonic();progress.update(0,'initial_validation',force=True);initial=validation_loss(model,va,vb);require(math.isfinite(initial),'Nonfinite real initial CE')
    engine=register_engine(model,c['engine']);best=initial;best_step=0
    metadata={'schema':'wide_scratch_run_a_v1','config':asdict(cfg),'seed':42,'batch_size':64,'data':data,'optimizer':opt_info,'optimizer_order':order,'torch':str(torch.__version__),'cuda':torch.version.cuda,'python':__import__('platform').python_version(),'numpy':np.__version__,'hardware':{'name':name,'compute_capability':cap,'visible':1,'VRAM_bytes':torch.cuda.get_device_properties(0).total_memory},'engine':engine,'counter_origin':0,'initialization':c['initialization'],'initializer':'Exact07e:embedding/positionsnormalstd.02,linearPyTorchdefaults,sharedembedding/output,noindependenthead; notmuPcontrolled','initial_validation_loss':initial,'parameters':30705728,'unique_xsa_scalars':64,'array_loading':loading,'recipe':recipe,'recipe_sha256':recipe_sha,'runtime_archive_sha256':c['runtime_archive_sha256'],'runtime_source_sha256':c['runtime_source_sha256'],'contract_sha256':sha(args.contract),'root_bound_NEW_cap':args.new_steps,'admission_mode':c['admission_mode'],'no_parent_model_moments_or_data_lineage':True,'targets':'Unchangedoriginal6Mcode; nofilter/correction/teacher/publictestexamples','future_resume':'Requiresstrictwideorigin0/source/config/savedLR/state/windowloader; old417kloaderunsupported'}
    atomic_json(out/'run_config.json',metadata)
    completed=committed=updates=skips=seen=0;active_slot=None;latest=None;checkpoint_cost=0.;admitted=None;admission=None;target=args.new_steps;partial=False;stopped=False;windows=[];raw_segment=hashlib.sha256();applied_segment=hashlib.sha256()
    def log(row):
        record={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**row}
        with (out/'train.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps(record,allow_nan=False),flush=True)
    def save_best(val,step):atomic_torch_save({'config':asdict(cfg),'model':model.state_dict(),'step':step,'validation_loss':val,'seed':42,'dataset_revision':data['dataset_revision']},out/'model.pt')
    def checkpoint(val,label,rng_override=None):
        nonlocal active_slot,committed,latest
        require(completed%64==0 and not partial,'Nevercommitpartial scratch frontier');require(updates+skips==completed and seen==completed*64,'Origin0attempt counters differ')
        require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),'Nonfinite checkpoint model')
        require(all(bool(torch.isfinite(v).all()) for o in optimizers for x in o.state.values() for v in x.values() if isinstance(v,torch.Tensor)),'Nonfinite optimizer moments')
        require(all(not isinstance(v,(int,float)) or math.isfinite(v) for v in scaler.state_dict().values()),'Nonfinite scaler state')
        raw=deepcopy(rng_override['numpy'] if rng_override else rng.bit_generator.state);cpu=rng_override['cpu'] if rng_override else torch.get_rng_state();cuda=rng_override['cuda'] if rng_override else torch.cuda.get_rng_state_all()
        sampler={'schema':'wide_scratch_sampler_a_v1','recipe':recipe,'recipe_sha256':recipe_sha,'counter_origin':0,'absolute_step':completed,'pending_rows':0,'at_consumed_window_boundary':True,'windows':completed//64,'raw_segment_sha256':raw_segment.hexdigest(),'applied_segment_sha256':applied_segment.hexdigest(),'raw_generator_start_sha256':raw_start,'raw_generator_end_sha256':fingerprint(raw),'layout_rng_start_sha256':layout_start,'layout_rng_end_sha256':fingerprint(layout_rng.bit_generator.state),'layout_rng_state':deepcopy(layout_rng.bit_generator.state),'admission':admission}
        metrics={'schema':'wide_scratch_metrics_a_v1','selected_step':best_step,'best_validation_loss':best,'final_validation_loss':val,'counter_origin':0,'absolute_completed_step':completed,'completed_steps':completed,'sampled_examples':seen,'optimizer_updates':updates,'skipped_updates':skips,'lr_absolute_step':completed,'sampler_checkpoint':sampler,'checkpoint_label':label,'root_bound_NEW_cap':args.new_steps,'admitted_NEW_horizon':target if admission else None,'validation_generation_indices':evaluation_indices,'no_historical_optimizer_or_sample_lineage_claim':True}
        slot=1-active_slot if active_slot is not None else 0;directory=out/('durable-slot'+str(slot));directory.mkdir(exist_ok=True)
        for n in ('tokenizer.json','validation.json','validation_legacy.json','data_stats.json'):
            if not (directory/n).exists():shutil.copyfile(args.data/n,directory/n)
        atomic_json(directory/'run_config.json',metadata)
        atomic_torch_save({'schema':'wide_scratch_training_state_a_v1','config':asdict(cfg),'model':model.state_dict(),'optimizers':[o.state_dict() for o in optimizers],'optimizer_order':order,'scaler':scaler.state_dict(),'step':completed,'lr_absolute_step':completed,'validation_loss':val,'best_validation_loss':best,'best_step':best_step,'rng_numpy_generator':raw,'rng_torch_cpu':cpu,'rng_torch_cuda':cuda,'sampled_examples':seen,'optimizer_updates':updates,'skipped_updates':skips,'seed':42,'dataset_revision':data['dataset_revision'],'counter_origin':0,'sampler_checkpoint':sampler,'runtime_archive_sha256':c['runtime_archive_sha256'],'runtime_source_sha256':c['runtime_source_sha256']},directory/'last_training.pt')
        destination=directory/'model.pt'
        if not destination.exists() or not os.path.samefile(out/'model.pt',destination):
            link=directory/('model.pt.link-'+str(completed)+'-'+str(os.getpid()));require(not link.exists(),'Ownedmodellinkexists');os.link(out/'model.pt',link)
            if destination.exists() and os.path.samefile(link,destination):os.unlink(link)
            else:os.replace(link,destination)
        atomic_json(directory/'metrics.json',metrics)
        for n in ROLES:
            with (directory/n).open('rb') as f:os.fsync(f.fileno())
        fsync_directory(directory);hashes={n:sha(directory/n) for n in sorted(ROLES)}
        latest={'schema':'wide_scratch_durable_pointer_a_v1','slot':directory.name,'absolute_step':completed,'counter_origin':0,'recipe_sha256':recipe_sha,'contract_sha256':sha(args.contract),'files_sha256':hashes,'metrics':metrics};atomic_json(out/'latest_checkpoint.json',latest);active_slot=slot;committed=completed;return latest
    def request_stop(*_):
        nonlocal stopped;stopped=True
    previous=signal.signal(signal.SIGTERM,request_stop);initial_save_started=time.monotonic();save_best(initial,0);checkpoint(initial,'initial',saved_rng);checkpoint_cost=time.monotonic()-initial_save_started
    model.train();rng.bit_generator.state=deepcopy(saved_rng['numpy']);torch.set_rng_state(saved_rng['cpu']);torch.cuda.set_rng_state_all(saved_rng['cuda']);del saved_rng # LAST before firstactual draw.
    started=calibration_started;log({'event':'initial_validation','step':0,'loss':initial,'metadata':metadata})
    while completed<target:
        now=time.monotonic();remaining=args.deadline-now;recent=max([w['seconds'] for w in windows[-3:]] or [30.]);reserve=max(180.,1.5*checkpoint_cost);rate=completed/(now-started) if completed else None;eta=(target-completed)/rate if rate else None;near_finish=eta is not None and eta<=180 and eta+max(30.,1.5*checkpoint_cost)<remaining
        if stopped or (remaining<=reserve+1.25*recent and not near_finish):break
        progress.update(completed,'training',force=True);torch.cuda.synchronize();window_started=time.monotonic();raw,applied,guard=draw_window(rng,layout_rng,arrays,targets,'bucket');raw_bytes=raw.astype('<i8',copy=False).tobytes();applied_bytes=applied.astype('<i8',copy=False).tobytes()
        useful_source=useful_target=allocated_source=allocated_target=0;loss_sum=0.;before_updates=updates;before_skips=skips;cursor=0;norm=None
        for batch_index in range(64):
            now=time.monotonic();estimated_tail=recent*(64-batch_index)/64*1.25+max(30.,1.5*checkpoint_cost)
            if now+30>=args.deadline or (stopped and now+estimated_tail>=args.deadline):partial=True;break
            indices=applied[batch_index*64:(batch_index+1)*64]
            aa=np.asarray(arrays[indices],dtype=np.int64);bb=np.asarray(targets[indices],dtype=np.int64)
            aa=aa[:,:int((aa!=0).sum(1).max())];bb=bb[:,:int((bb!=0).sum(1).max())]
            a=torch.from_numpy(aa).cuda();b=torch.from_numpy(bb).cuda();absolute=completed+1;lr=c['lr_fraction']*absolute_lr(absolute)
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
            progress.update(completed,'training',absolute_step=completed,window_cursor=cursor,committed_new_iterations=committed)
        torch.cuda.synchronize();window_seconds=time.monotonic()-window_started
        if partial:
            atomic_torch_save({'schema':'wide_scratch_partial_NONRESUMABLE_a_v1','config':asdict(cfg),'model':model.state_dict(),'step':completed,'nonresumable':True,'reason':'raw/layout RNG ahead ofpartialwindow; usecommittedpointer'},out/'partial_model_NONRESUMABLE.pt')
            atomic_json(out/'partial_progress_NONRESUMABLE.json',{'complete':False,'step':completed,'last_durable_pointer':latest,'consumed_rows':cursor*64,'raw_rows':raw.tolist(),'applied_rows':applied.tolist(),'ahead_raw_generator_state':rng.bit_generator.state,'ahead_layout_rng_state':layout_rng.bit_generator.state});break
        require(cursor==64 and completed%64==0,'Consumed64windowrequired');raw_segment.update(raw_bytes);applied_segment.update(applied_bytes)
        row={'event':'window_completed','step':completed,'seconds':window_seconds,**guard,'raw_window_sha256':hashlib.sha256(raw_bytes).hexdigest(),'applied_window_sha256':hashlib.sha256(applied_bytes).hexdigest(),'raw_segment_sha256':raw_segment.hexdigest(),'applied_segment_sha256':applied_segment.hexdigest(),'useful_source_tokens':useful_source,'useful_target_tokens':useful_target,'allocated_source_positions':allocated_source,'allocated_target_positions':allocated_target,'token_mean_training_loss':loss_sum/useful_target,'optimizer_updates':updates-before_updates,'AMP_skips':skips-before_skips,'last_grad_norm':norm,'engine_observation':observe_engine(torch)};windows.append(row);log(row)
        if completed%1024==0 or completed==target or stopped:
            cost=time.monotonic();val=validation_loss(model,va,vb);legacy_val=validation_loss(model,*legacy);require(math.isfinite(val) and math.isfinite(legacy_val),'Nonfinite checkpoint CE')
            if val<best:best=val;best_step=completed;save_best(val,completed)
            checkpoint(val,'periodic' if completed<target else 'endpoint');checkpoint_cost=max(checkpoint_cost,time.monotonic()-cost)
            elapsed=time.monotonic()-started
            if completed>=1024 and admitted is None:
                require(updates>0,'No finiteoptimizer updates byretained1024');remaining=args.deadline-time.monotonic();rate=completed/elapsed;reserve=max(180.,1.5*checkpoint_cost);fit=completed+64*math.floor(max(0.,remaining-reserve-60.)*rate/1.25/64);target=min(args.new_steps,fit);require(target>=completed,'Retained1024cannotfit')
                require(remaining>reserve+60.,'No actual reserve foradmissionmetadata save');admitted=True;admission={'mode':'single_measured64_shortening','root_bound_NEW_cap':args.new_steps,'admitted_NEW_horizon':target,'actual_NEW_attempts':completed,'checkpoint_inclusive_seconds':elapsed,'rate':rate,'remaining_seconds':remaining,'save_reserve_seconds':reserve,'binding_reserve_seconds':60.,'uncertainty_factor':1.25};metadata['admission']=admission;metadata['max_steps']=target;atomic_json(out/'run_config.json',metadata);atomic_json(out/'admission.json',admission);checkpoint(val,'admission_metadata');checkpoint_cost=max(checkpoint_cost,time.monotonic()-cost);require(time.monotonic()+max(180.,1.5*checkpoint_cost)<args.deadline,'Actual admissionmetadata save exhausteddeadline reserve');log({'event':'budget_admission',**admission})
            log({'event':'checkpoint_completed','step':completed,'validation_loss':val,'legacy_validation_loss':legacy_val,'elapsed_seconds':time.monotonic()-started,'checkpoint_and_CE_seconds':checkpoint_cost,'latest_checkpoint':latest['slot']})
    if not partial and completed!=committed:
        val=validation_loss(model,va,vb);require(math.isfinite(val),'Nonfinite controlledstopCE')
        if val<best:best=val;best_step=completed;save_best(val,completed)
        checkpoint(val,'controlled_stop')
    result={'schema':'wide_scratch_worker_result_a_v1','training_family':'wide_scratch_window_a_v1','counter_origin':0,'model_parameters':30705728,'unique_xsa_scalars':64,'actual_model_iterations':completed,'durable_new_iterations':committed,'durable_absolute_step':committed,'requested_new_iterations':args.new_steps,'admitted_NEW_horizon':target if admission else None,'admitted_horizon_complete':admitted is True and not partial and completed==committed==target,'partial_nonresumable':partial,'optimizer_updates':updates,'skipped_updates':skips,'selected_step':best_step,'best_validation_loss':best,'latest_checkpoint':latest,'whole_worker_seconds':time.monotonic()-startup,'training_and_checkpoint_seconds':time.monotonic()-started,'source_archive_sha256':c['runtime_archive_sha256'],'no_parent_lineage':True}
    atomic_json(out/'worker_result.json',result);log({'event':'finished',**result});signal.signal(signal.SIGTERM,previous);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--contract',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--source-archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--new-steps',type=int,required=True);p.add_argument('--deadline',type=float,required=True);p.add_argument('--resident',action='store_true');a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.is_dir():atomic_json(a.output/'failure.json',{'complete':False,'failure_type':type(error).__name__,'durable_pointer_present':(a.output/'latest_checkpoint.json').is_file()})
        raise
