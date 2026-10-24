"""Typed initial cooldownD→fixedfraction ISQRT bridge; no generic fallback.

Strict D endpoint loader and saved-schedule oracle are imported unchanged.
New durable states preserve full computational state; future resume must use an
explicit fixedfraction saved-LR oracle, not the historical cooldown loader.
"""
from copy import deepcopy
import hashlib,json
import numpy as np
from .gated_resuming import require,absolute_lr,restore_rng
from .gated_cooldown_recipe_b import (restore_window_training,recipe_for as parent_recipe_for,
    validate_window_parent as validate_parent_window,fingerprint)
PARENT_ARCHIVE='a8a41094ab5128d33b0b7b5eb95054fa08bf56f8b8c2dcaf80344985fb60e201'
PARENT_STEP=417384

def recipe_for(layout):
    require(layout=='bucket','Only inherited6M bucket grouping supported')
    recipe,_=parent_recipe_for(layout)
    recipe['schedule']='fixed_fraction_absolute_ISQRT_low_lr_a_v1'
    recipe['layout_initialization']='exact_committed_cooldown_D_parent_PCG64_state'
    return recipe,fingerprint(recipe)

def validate_window_parent(saved,metadata,metrics,contract,restored,saved_rng):
    require(contract['parent_kind']=='cooldown_D_half_endpoint' and restored['absolute_step']==PARENT_STEP,
            'Only typed actualD half endpoint417384 supported')
    static=metadata['window_intervention']
    require(static['runtime_archive_sha256']==PARENT_ARCHIVE and
            metadata['continuation']['run_id']=='cooldown-half','Wrong actualD parent source/arm')
    oldrecipe,oldsha=parent_recipe_for('bucket')
    view={**contract,'parent_kind':'same_recipe_window','recipe':oldrecipe,'recipe_sha256':oldsha,
          'runtime_archive_sha256':PARENT_ARCHIVE,'runtime_source_sha256':static['runtime_source_sha256']}
    layout,history,origin=validate_parent_window(saved,metadata,metrics,view,restored,saved_rng)
    require(origin==345000 and history[-1]['end_step']==PARENT_STEP,'Historical D segment frontier differs')
    # Complete old sampler/history remains explicitly nested as parent lineage
    # in every newcheckpoint; newsegments start at the newbranch boundary.
    return layout,[],PARENT_STEP

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
