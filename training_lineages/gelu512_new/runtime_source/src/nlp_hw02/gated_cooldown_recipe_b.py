"""Schedule-aware strict restoration with unchanged bucket/provenance helpers.

All original order/alias/moment/scaler/RNG guards are retained. The only restore
changes are the strict saved-schedule LR oracle and cooldown metadata equality.
No second fullstate load and no original source mutation.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from .gated_resuming import (ORIGIN,PARENT_B034_SHA,require,absolute_lr,
    optimizer_order,replay_sample_prefix,restore_rng)

from .gated_cooldown_schedule_a import saved_lr_oracle

RECIPES={
    "bucket":{"schema":"gated_window64_v1","layout":"bucket","rows":6000000,"batch_size":64,
        "window_rows":4096,"window_updates":64,"target_length_bin":16,
        "sort_keys":["target_nonpad_length//16","source_nonpad_length","original_window_index"],
        "batch_order":"separate_PCG64_permutation64","layout_seed":20261022,
        "layout_rng_policy":"permutation64_each_consumed_window",
        "raw_generator":"original_seed42_uniform_with_replacement_since300000",
        "loss":"unchanged_token_mean_CE_per_update","schedule":"common1024_ISQRT_then_bound_remaining_envelope_a_v1",
        "checkpoint_cadence_NEW_updates":1024,"first_admission_NEW_updates":1024},
    "uniform":{"schema":"gated_window64_v1","layout":"uniform","rows":6000000,"batch_size":64,
        "window_rows":4096,"window_updates":64,"target_length_bin":None,"sort_keys":[],
        "batch_order":"original_raw_draw_order","layout_seed":20261022,
        "layout_rng_policy":"saved_inactive_unchanged_no_permutation",
        "raw_generator":"original_seed42_uniform_with_replacement_since300000",
        "loss":"unchanged_token_mean_CE_per_update","schedule":"common1024_ISQRT_then_bound_remaining_envelope_a_v1",
        "checkpoint_cadence_NEW_updates":1024,"first_admission_NEW_updates":1024}}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def recipe_for(layout):
    require(layout in RECIPES,'Explicit uniform or bucket layout required')
    recipe=deepcopy(RECIPES[layout]);return recipe,fingerprint(recipe)


def valid_sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)


def restore_window_training(model,optimizers,scaler,rng,parent,*,seed,data,optimizer_info,lane,deadline,progress=None):
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
            require(old['lr']==saved_lr_oracle(state,metadata)*old['lr_multiplier'],'Saved group LR differs from strict saved schedule/phase oracle')
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
    if 'cooldown_checkpoint' in state:require(metrics['cooldown_checkpoint']==state['cooldown_checkpoint'],'Saved metrics/fullstate cooldown binding differs')
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
    sampler_checkpoint=state.get("sampler_checkpoint")
    del state,best
    return result,saved_rng,digest,sampler_checkpoint


def validate_window_parent(saved,metadata,metrics,contract,restored,saved_rng):
    """Require explicit original intervention OR same-recipe durable descendant."""
    recipe,recipe_sha=recipe_for(contract['layout']);start=restored['absolute_step'];kind=contract['parent_kind']
    require(contract['recipe']==recipe and contract['recipe_sha256']==recipe_sha,'Explicit unchanged layout recipe required')
    if kind=='original_uniform_gated':
        require(saved is None and 'window_intervention' not in metadata,'Original-parent role cannot hide prior window state')
        require(start==(345000 if contract['lane']=='control' else 330000),'Original top-level endpoint required, not matched snapshot')
        layout_rng=np.random.default_rng(recipe['layout_seed']);history=[];origin=start
    elif kind=='same_recipe_window':
        require(isinstance(saved,dict),'Window descendant lacks saved recipe state')
        static=metadata['window_intervention']
        require(saved['schema']=='gated_window_checkpoint_v1' and saved['recipe']==recipe and saved['recipe_sha256']==recipe_sha,'Saved recipe identity differs')
        require(saved['absolute_step']==start and saved['at_consumed_window_boundary'] is True and saved['pending_rows']==0,'Parent is not a durable consumed-window state')
        origin=saved['window_origin_step'];require(type(origin) is int and ORIGIN<origin<=start and (start-origin)%64==0,'Window origin/frontier differs')
        require(static['recipe']==recipe and static['recipe_sha256']==recipe_sha and static['window_origin_step']==origin,'Metadata recipe origin differs')
        require(static['runtime_archive_sha256']==contract['runtime_archive_sha256'] and static['runtime_source_sha256']==contract['runtime_source_sha256'],'Descendant restore requires exact same reviewed source archive, no generic fallback')
        require(saved['runtime_archive_sha256']==static['runtime_archive_sha256'] and saved['runtime_source_sha256']==static['runtime_source_sha256'],'Fullstate/metadata source binding differs')
        require(metrics['sampler_checkpoint']==saved and metrics['absolute_completed_step']==start,'Durable metrics/fullstate recipe differs')
        history=saved['segments'];require(isinstance(history,list) and bool(history),'Missing applied-order segment provenance')
        initial_layout=np.random.default_rng(recipe['layout_seed'])
        require(history[0]['layout_rng_start_sha256']==fingerprint(initial_layout.bit_generator.state),'First layout RNG does not match explicit recipe seed')
        frontier=origin;previous=None
        for segment in history:
            require(segment['schema']=='gated_window_segment_v1' and segment['recipe_sha256']==recipe_sha,'Segment recipe differs')
            require(segment['start_step']==frontier and type(segment['end_step']) is int and segment['end_step']>=frontier and (segment['end_step']-frontier)%64==0,'Segment chain has gap/overlap/partial window')
            require(segment['windows']==(segment['end_step']-frontier)//64,'Segment window counter differs')
            for key in ('parent_terminal_sha256','parent_last_training_sha256','runtime_archive_sha256','raw_segment_sha256','applied_segment_sha256','whole_raw_sha256_end','raw_generator_start_sha256','raw_generator_end_sha256','layout_rng_start_sha256','layout_rng_end_sha256','worker_contract_sha256'):
                require(valid_sha(segment[key]),'Invalid segment provenance:'+key)
            require(segment['runtime_archive_sha256']==contract['runtime_archive_sha256'],'Segment source archive differs')
            require(segment['layout']==contract['layout'] and segment['lane']==contract['lane'] and isinstance(segment['run_id'],str) and bool(segment['run_id']),'Segment architecture/layout/run identity differs')
            cap=segment['root_bound_NEW_cap'];target=segment['admitted_NEW_horizon']
            require(type(cap) is int and cap>=1024 and cap%64==0 and segment['end_step']-segment['start_step']<=cap and segment['admission_mode'] in ('fixed','measured64_shortening'),'Segment root cap/admission mode differs')
            if target is not None:
                require(type(target) is int and 1024<=target<=cap and target%64==0 and segment['end_step']-segment['start_step']<=target,'Segment admitted duration differs')
                require(segment['admission_mode']!='fixed' or target==cap,'Fixed segment silently shortened')
            if contract['layout']=='uniform':require(segment['raw_segment_sha256']==segment['applied_segment_sha256'] and segment['layout_rng_start_sha256']==segment['layout_rng_end_sha256'],'Uniform recipe changed applied order or inactive RNG')
            if previous is not None:
                require(segment['raw_generator_start_sha256']==previous['raw_generator_end_sha256'] and segment['layout_rng_start_sha256']==previous['layout_rng_end_sha256'],'Segment RNG chain is discontinuous')
            if segment['windows']==0:
                empty=hashlib.sha256().hexdigest()
                require(segment['raw_segment_sha256']==segment['applied_segment_sha256']==empty and segment['raw_generator_start_sha256']==segment['raw_generator_end_sha256'] and segment['layout_rng_start_sha256']==segment['layout_rng_end_sha256'],'Zero-update checkpoint fabricates applied history')
            frontier=segment['end_step']
            previous=segment
        require(frontier==start and history[-1]['whole_raw_sha256_end']==restored['sampled_indices_sha256'],'Raw prefix/segment end differs')
        decision=saved['admission'];last=history[-1];continuation=metadata['continuation']
        require(last['run_id']==metrics['run_id']==continuation['run_id'] and last['lane']==metrics['lane']==continuation['lane']==contract['lane'] and last['layout']==metrics['layout']==continuation['layout']==contract['layout'],'Last segment run/architecture/layout differs')
        require(last['worker_contract_sha256']==static['contract_sha256'] and last['root_bound_NEW_cap']==continuation['root_bound_NEW_cap'] and last['admission_mode']==continuation['admission_mode'],'Latest segment/current worker contract differs')
        require(decision==continuation.get('admission') and metrics['admitted_NEW_horizon']==last['admitted_NEW_horizon'] and metrics['root_bound_NEW_cap']==last['root_bound_NEW_cap'] and metrics['admission_mode']==last['admission_mode'],'Durable admission/fullstate/metadata differs')
        if decision is None:require(last['admitted_NEW_horizon'] is None,'Missing duration decision')
        else:
            require(decision['mode']==last['admission_mode'] and decision['root_bound_NEW_cap']==last['root_bound_NEW_cap'] and decision['admitted_NEW_horizon']==last['admitted_NEW_horizon'],'Saved duration decision differs')
        require(history[-1]['raw_generator_end_sha256']==fingerprint(saved_rng['numpy']),'Raw Generator end differs')
        require(history[-1]['layout_rng_end_sha256']==fingerprint(saved['layout_rng_state']),'Saved layout RNG end differs')
        layout_rng=np.random.default_rng();layout_rng.bit_generator.state=deepcopy(saved['layout_rng_state'])
        require(fingerprint(layout_rng.bit_generator.state)==history[-1]['layout_rng_end_sha256'],'Layout RNG restore differs')
        history=deepcopy(history)
    else:raise ValueError('Unknown explicit parent recipe role')
    return layout_rng,history,origin


def draw_window(rng,layout_rng,source,target,layout):
    """Actual D raw/frontier guard; uniform does no lookup/sort/permutation."""
    import time
    recipe,_=recipe_for(layout)
    require(len(source)==len(target)==recipe['rows'],'Exact original6M rows required')
    before=deepcopy(rng.bit_generator.state);flat=np.random.default_rng();flat.bit_generator.state=deepcopy(before)
    reference=np.random.default_rng();reference.bit_generator.state=deepcopy(before)
    raw=flat.integers(0,len(source),size=4096)
    original=np.concatenate([reference.integers(0,len(source),size=64) for _ in range(64)])
    require(np.array_equal(raw,original) and flat.bit_generator.state==reference.bit_generator.state,'Actual raw draw/frontier guard failed')
    rng.bit_generator.state=deepcopy(reference.bit_generator.state)
    layout_before=fingerprint(layout_rng.bit_generator.state);lookup_seconds=0.
    if layout=='bucket':
        lookup=time.monotonic();sl=(source[raw]!=0).sum(1);tl=(target[raw]!=0).sum(1)
        order=np.lexsort((np.arange(4096),sl,tl//16)).reshape(64,64)
        order=order[layout_rng.permutation(64)].reshape(-1);applied=raw[order]
        lookup_seconds=time.monotonic()-lookup
    else:
        applied=raw.copy()
        require(np.array_equal(applied,raw) and fingerprint(layout_rng.bit_generator.state)==layout_before,'Uniform order/inactive layout RNG changed')
    require(np.array_equal(np.sort(raw),np.sort(applied)),'Actual row multiset changed')
    return raw,applied,{'raw_generator_start_sha256':fingerprint(before),
        'raw_generator_end_sha256':fingerprint(rng.bit_generator.state),
        'length_lookup_sort_seconds':lookup_seconds,'multiset_verified':True,'layout':layout,
        'layout_rng_start_sha256':layout_before,'layout_rng_end_sha256':fingerprint(layout_rng.bit_generator.state),
        'layout_rng_consumed':layout=='bucket','draw_and_frontier_equivalence_verified':True}


def segment_record(contract,start,end,raw_digest,applied_digest,whole_raw_digest,raw_start,raw_end,layout_start,layout_end):
    recipe,recipe_sha=recipe_for(contract['layout'])
    require(end>=start and (end-start)%64==0,'No resumable segment at partial window')
    return {'schema':'gated_window_segment_v1','recipe_sha256':recipe_sha,'run_id':contract['run_id'],'lane':contract['lane'],'layout':contract['layout'],'start_step':start,'end_step':end,
        'windows':(end-start)//64,'parent_terminal_sha256':contract['parent_terminal_sha256'],
        'parent_last_training_sha256':contract['parent_artifact_sha256']['last_training.pt'],
        'runtime_archive_sha256':contract['runtime_archive_sha256'],
        'raw_segment_sha256':raw_digest.hexdigest(),'applied_segment_sha256':applied_digest.hexdigest(),
        'whole_raw_sha256_end':whole_raw_digest.hexdigest(),'raw_generator_start_sha256':raw_start,
        'raw_generator_end_sha256':raw_end,'layout_rng_start_sha256':layout_start,'layout_rng_end_sha256':layout_end,
        'applied_history_scope':'This resume segment only; earlier segment hashes retained, no hashlib internal state invented'}
