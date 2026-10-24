"""Strict wide B/C origin0 endpoint restore; no legacy300k loader or schedule coercion."""
from copy import deepcopy
from dataclasses import asdict
import hashlib,json,math,time,zipfile
from pathlib import Path

MODEL_SOURCE='07e58176c39eed8ed03134074282d74589be97120e6d6569a15f27eee1538b26'
PARENTS={'gelu512':{'archive':'97494fb88adab12d244716a782b079b74f3072ae11f5f17166c5f22159c857bd','worker':'scripts/train_wide_scratch_b.py','worker_sha':'ee0deb1b81a80e09da3b97e018df9f13a3de516a90b71b50144f844b0e8132a1','schema':'wide_scratch_training_state_a_v1','metadata':'wide_scratch_run_a_v1','metrics':'wide_scratch_metrics_a_v1','activation':'gelu','ff':2048,'parameters':30705728},'swiglu512':{'archive':'36a7a679993a4ec0f19d9351850afe1649e7f46c528a7ff8ff5ceff828b54ca9','worker':'scripts/train_wide_swiglu_c.py','worker_sha':'7c4eadb0d9d4251b9da79a45444814ec573baffc3c069d4449cd9bfdd1843596','schema':'wide_swiglu_training_state_c_v1','metadata':'wide_swiglu_run_c_v1','metrics':'wide_swiglu_metrics_c_v1','activation':'swiglu','ff':1360,'parameters':30645568}}
ROLES={'last_training.pt','model.pt','metrics.json','run_config.json','tokenizer.json','validation.json','validation_legacy.json','data_stats.json'}
STATE_FIELDS={'schema','config','model','optimizers','optimizer_order','scaler','step','lr_absolute_step','validation_loss','best_validation_loss','best_step','rng_numpy_generator','rng_torch_cpu','rng_torch_cuda','sampled_examples','optimizer_updates','skipped_updates','seed','dataset_revision','counter_origin','sampler_checkpoint','runtime_archive_sha256','runtime_source_sha256'}

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def safe(root,name):
    p=Path(name);require(not p.is_absolute() and '..' not in p.parts,'Safe relative role required');q=Path(root)/p;require(q.resolve().is_relative_to(Path(root).resolve()),'Role escaped root');return q
def bound(root,entry):
    p=safe(root,entry['path']);require(sha(p)==entry['sha256'],'Actual bound bytes differ');return p

def verify_parent(contract):
    require(contract.get('schema')=='wide_origin0_resume_parent_a_v1' and contract.get('verified_actual_values') is True,'Unbound wide resume parent; stop before Torch')
    require(contract['architecture'] in PARENTS,'Unsupported wide parent');descriptor=PARENTS[contract['architecture']];root=Path(contract['artifact_root'])
    acceptance_path=bound(root,contract['root_parent_acceptance']);acceptance=json.loads(acceptance_path.read_text());require(acceptance['schema']=='wide_origin0_resume_parent_acceptance_a_v1' and acceptance['root_accepted'] is True and acceptance['complete'] is True,'Root accepted actual family/committed pointer required')
    require(acceptance['architecture']==contract['architecture'] and acceptance['counter_origin']==0,'Accepted wide family differs')
    pointer_path=bound(root,contract['pointer']);pointer=json.loads(pointer_path.read_text());require(pointer['schema']=='wide_scratch_durable_pointer_a_v1' and pointer['counter_origin']==0 and pointer['slot'] in ('durable-slot0','durable-slot1'),'Actual complete wide pointer required')
    require(set(contract['projection'])==set(pointer['files_sha256'])==ROLES,'Exact8 durable roles required')
    files={}
    for n,e in contract['projection'].items():
        require(e['sha256']==pointer['files_sha256'][n]==acceptance['files_sha256'][n],'Accepted pointer role differs:'+n);files[n]=bound(root,e)
    require(acceptance['pointer_sha256']==sha(pointer_path),'Accepted pointer bytes differ')
    parent_contract_path=bound(root,contract['parent_contract']);pc=json.loads(parent_contract_path.read_text());require(pc['verified_actual_values'] is True and pc['counter_origin']==0 and pc['seed']==42 and pc['lr_fraction']==1. and pc['cache_manifest_sha256']=='8ce0939534bdb5f2b791d19d0ab13f6b4c79402b8627fecad4e9902cd08bb13c','Actual fresh6M parent recipe required')
    require(pointer['contract_sha256']==sha(parent_contract_path)==acceptance['contract_sha256'],'Parent contract/pointer/root differs')
    archive=bound(root,contract['parent_archive']);require(sha(archive)==descriptor['archive']==pc['runtime_archive_sha256']==acceptance['runtime_archive_sha256'],'Exact native B/C source archive required')
    with zipfile.ZipFile(archive) as z:
        require(len(z.namelist())==len(set(z.namelist()))==64 and set(z.namelist())==set(pc['runtime_source_sha256']),'Exact runtime member surface required')
        for n,h in pc['runtime_source_sha256'].items():require(not Path(n).is_absolute() and '..' not in Path(n).parts and hashlib.sha256(z.read(n)).hexdigest()==h,'Parent source member differs')
    require(pc['runtime_source_sha256']['src/nlp_hw02/model.py']==MODEL_SOURCE and pc['runtime_source_sha256'][descriptor['worker']]==descriptor['worker_sha'],'Parent own model/worker source differs')
    return descriptor,files,pointer,pc,acceptance

def restore_wide_origin0(model,optimizers,scaler,rng,contract,*,optimizer_info,deadline,progress=None):
    descriptor,files,pointer,pc,acceptance=verify_parent(contract)
    import numpy as np
    import torch
    from .gated_resuming import optimizer_order
    from .wide_scratch_recipe_a import recipe_for,absolute_lr
    require(math.isfinite(deadline) and time.monotonic()<deadline,'Finite future restore/replay deadline required')
    require(str(torch.__version__)=='2.10.0+cu128' and torch.version.cuda=='12.8' and torch.cuda.device_count()==1,'Exact training software/one visible GPU required')
    require(type(model).__name__=='Seq2Seq' and sha(__import__('sys').modules[type(model).__module__].__file__)==MODEL_SOURCE,'Actual model implementation differs')
    require(type(optimizers[0]).__name__=='BatchedMuon' and type(optimizers[1]).__name__=='AdamW','Actual groupedMuon/AdamW implementation required')
    require(sha(__import__('sys').modules[type(optimizers[0]).__module__].__file__)=='9161d39b568ba472fe132b13e50bc906537104e61aca30a0f7de89efff8e246f','Actual Muon implementation differs')
    metadata=json.loads(files['run_config.json'].read_text());metrics=json.loads(files['metrics.json'].read_text());data=json.loads(files['data_stats.json'].read_text());order=optimizer_order(model,optimizers)
    require(metadata['schema']==descriptor['metadata'] and metrics['schema']==descriptor['metrics'] and metadata['torch']==str(torch.__version__) and metadata['cuda']==torch.version.cuda,'Typed native wide metadata/software required')
    require(metadata['config']==pc['config']==asdict(model.cfg) and model.cfg.d_model==512 and model.cfg.heads==8 and model.cfg.encoder_layers==model.cfg.decoder_layers==4 and model.cfg.activation==descriptor['activation'] and model.cfg.ff==descriptor['ff'] and not model.cfg.gated_repeats and model.cfg.loop_repeats==1,'Exact wide topology/config required')
    require(sum(p.numel() for p in model.parameters())==descriptor['parameters'] and sum(p.numel() for n,p in model.named_parameters() if n.endswith('.xsa_alpha'))==64,'Wide parameter/XSA count differs')
    require(metadata['optimizer']==optimizer_info and metadata['numpy']==np.__version__,'Exact optimizer settings/NumPy provenance required')
    require(metadata['optimizer_order']==order and metadata['seed']==42 and metadata['batch_size']==64 and metadata['data']==data and metadata['counter_origin']==0 and metadata['no_parent_model_moments_or_data_lineage'] is True,'Origin0/data/optimizer order differs')
    recipe,recipe_sha=recipe_for('bucket');require(pc['recipe']==metadata['recipe']==recipe and pc['recipe_sha256']==metadata['recipe_sha256']==pointer['recipe_sha256']==recipe_sha,'Original wide bucket recipe differs')
    state=torch.load(files['last_training.pt'],map_location='cpu',weights_only=True);best=torch.load(files['model.pt'],map_location='cpu',weights_only=True)
    require(set(state)==STATE_FIELDS and state['schema']==descriptor['schema'],'Exact typed fullstate fields required')
    require(state['config']==asdict(model.cfg) and state['seed']==42 and state['counter_origin']==0 and state['dataset_revision']==data['dataset_revision'] and state['optimizer_order']==order,'Fullstate own identity differs')
    require(state['runtime_archive_sha256']==descriptor['archive'] and state['runtime_source_sha256']==pc['runtime_source_sha256']==metadata['runtime_source_sha256'],'State/source map differs')
    step=state['step'];updates=state['optimizer_updates'];skips=state['skipped_updates'];seen=state['sampled_examples']
    require(all(type(x) is int and x>=0 for x in [step,updates,skips,seen]) and step>=1024 and step%64==0 and updates>0 and updates+skips==step and seen==step*64 and state['lr_absolute_step']==step==pointer['absolute_step'],'Origin0 durable counters differ')
    require(all(math.isfinite(state[k]) for k in ['validation_loss','best_validation_loss']),'Nonfinite savedCE')
    require(metrics==pointer['metrics'] and metrics['absolute_completed_step']==metrics['completed_steps']==metrics['lr_absolute_step']==step and metrics['optimizer_updates']==updates and metrics['skipped_updates']==skips and metrics['sampled_examples']==seen,'Pointer/metrics/counters differ')
    require(len(optimizers)==len(state['optimizers'])==2,'Exact Muon/AdamW pair required')
    for oi,(optimizer,saved) in enumerate(zip(optimizers,state['optimizers'])):
        fresh=optimizer.state_dict();require(len(saved['param_groups'])==len(fresh['param_groups']),'Optimizer group count differs')
        for old,new,actual in zip(saved['param_groups'],fresh['param_groups'],optimizer.param_groups):
            require(len(old['params'])==len(new['params'])==len(actual['params']),'Optimizer ordered partition differs')
            for key in ['betas','eps','weight_decay','momentum','nesterov','ns_steps','ns_dtype','adjust_lr_fn','lr_multiplier','fused','amsgrad','maximize']:require(old.get(key)==new.get(key),'Optimizer setting differs:'+key)
            require(old['lr']==absolute_lr(step)*old['lr_multiplier'],'Strict original wide absoluteISQRT saved LR differs')
            for key,p in zip(old['params'],actual['params']):
                require(key in saved['state'],'Missing trained parameter state');entry=saved['state'][key];require(set(entry)==({'momentum_buffer'} if oi==0 else {'step','exp_avg','exp_avg_sq'}),'Optimizer state fields differ')
                if oi==1:require(isinstance(entry['step'],torch.Tensor) and entry['step'].numel()==1 and float(entry['step'])==updates,'Adam counter differs')
                for field,v in entry.items():
                    if isinstance(v,torch.Tensor):
                        require(bool(torch.isfinite(v).all()),'Nonfinite optimizer state')
                        if field!='step':require(v.shape==p.shape and v.dtype==p.dtype,'Moment shape/dtype differs')
        ids=[p for g in saved['param_groups'] for p in g['params']];require(len(ids)==len(set(ids)) and set(ids)==set(saved['state']),'Optimizer state coverage differs')
    require(set(best)=={'config','model','step','validation_loss','seed','dataset_revision'} and best['config']==asdict(model.cfg) and best['seed']==42 and best['dataset_revision']==state['dataset_revision'],'Exact six-field CEbest required')
    require(type(best['step']) is int and 0<=best['step']<=step and best['step']==state['best_step']==metrics['selected_step'] and best['validation_loss']==state['best_validation_loss']==metrics['best_validation_loss'] and state['validation_loss']==metrics['final_validation_loss'],'CEbest vs endpoint roles differ')
    for saved in [state['model'],best['model']]:
        fresh=model.state_dict();require(set(saved)==set(fresh),'Model state names differ')
        for n,v in fresh.items():require(saved[n].shape==v.shape and saved[n].dtype==v.dtype and bool(torch.isfinite(saved[n]).all()),'Model shape/dtype/finite differs:'+n)
        for names in order['aliases'].values():require(all(torch.equal(saved[names[0]],saved[n]) for n in names[1:]),'Aliased checkpoint tensors disagree')
    sampler=state['sampler_checkpoint'];require(sampler==metrics['sampler_checkpoint'] and sampler['schema']=='wide_scratch_sampler_a_v1' and sampler['recipe']==recipe and sampler['recipe_sha256']==recipe_sha and sampler['counter_origin']==0 and sampler['absolute_step']==step and sampler['pending_rows']==0 and sampler['at_consumed_window_boundary'] is True and sampler['windows']==step//64,'Resumable complete window sampler required')
    replay=np.random.default_rng(42);layout=np.random.default_rng(20261022);digest=hashlib.sha256();require(fingerprint(replay.bit_generator.state)==sampler['raw_generator_start_sha256'] and fingerprint(layout.bit_generator.state)==sampler['layout_rng_start_sha256'],'Origin0 generator fingerprints differ')
    for window in range(step//64):
        require(time.monotonic()<deadline,'Bounded prefix replay deadline');rows=replay.integers(0,6000000,size=4096);digest.update(rows.astype('<i8',copy=False).tobytes());layout.permutation(64)
        if progress is not None and window%1024==0:progress(window*64)
    require(digest.hexdigest()==sampler['raw_segment_sha256'] and replay.bit_generator.state==state['rng_numpy_generator'] and fingerprint(replay.bit_generator.state)==sampler['raw_generator_end_sha256'],'Actual raw prefix/state replay differs')
    require(layout.bit_generator.state==sampler['layout_rng_state'] and fingerprint(layout.bit_generator.state)==sampler['layout_rng_end_sha256'],'Actual layout RNG replay differs')
    cuda=state['rng_torch_cuda'];require(len(cuda)==1,'One saved CUDA RNG required')
    for v in [state['rng_torch_cpu'],*cuda]:require(v.dtype==torch.uint8 and v.ndim==1 and v.device.type=='cpu','Invalid saved Torch RNG')
    require(set(state['scaler'])==set(scaler.state_dict()) and all(not isinstance(v,(int,float)) or math.isfinite(v) for v in state['scaler'].values()) and state['scaler']['scale']>0,'Finite enabled AMP scaler fields required')
    saved_rng={'numpy':deepcopy(state['rng_numpy_generator']),'cpu':state['rng_torch_cpu'].clone(),'cuda':[v.clone() for v in cuda]}
    model.load_state_dict(state['model'],strict=True) # NEVER CEbest for optimizer continuation.
    for optimizer,saved in zip(optimizers,state['optimizers']):optimizer.load_state_dict(saved)
    scaler.load_state_dict(state['scaler']);require(scaler.state_dict()==state['scaler'],'AMP scaler roundtrip differs')
    result={'absolute_step':step,'lr_absolute_step':step,'counter_origin':0,'sampled_examples':seen,'optimizer_updates':updates,'skipped_updates':skips,'best_step':best['step'],'best_validation_loss':best['validation_loss'],'last_validation_loss':state['validation_loss'],'saved_lr_schedule':'fresh200warmup_absoluteISQRT_fraction1','parent_raw_digest':sampler['raw_segment_sha256'],'parent_applied_row_digest':sampler['applied_segment_sha256'],'applied_digest_scope':'SHA-bound historical declaration; not recomputed withoutarrays','new_segment_digest_policy':'Freshlocalhashers;parenthexdigestsretained,nofabricatedhashlibstate','parent_sampler':deepcopy(sampler),'optimizer_order':order,'parent_hardware':metadata['hardware'],'current_hardware':{'name':torch.cuda.get_device_name(0),'capability':torch.cuda.get_device_capability(0)},'initial_CE_same_hardware_reference_tolerance':{'rtol':1e-6,'atol':1e-7},'cross_hardware_CE_policy':'Explicittransitionmeasurementrequired;notfalseexactequality','root_acceptance':acceptance}
    del state,best
    return result,saved_rng,deepcopy(layout.bit_generator.state)

def restore_rng_last(saved,rng):
    import torch
    rng.bit_generator.state=deepcopy(saved['numpy']);torch.set_rng_state(saved['cpu']);torch.cuda.set_rng_state_all(saved['cuda'])
