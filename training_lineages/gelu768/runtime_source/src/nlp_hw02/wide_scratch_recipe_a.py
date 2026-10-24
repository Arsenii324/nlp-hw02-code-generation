"""Fresh origin0 six-million-row bucket recipe, no inherited training lineage."""
from copy import deepcopy
import hashlib,json,math
import numpy as np

def require(ok,message):
    if not ok:raise ValueError(message)

def absolute_lr(step):return 6e-4*min(step/200,math.sqrt(200/max(step,200)))

def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def recipe_for(layout):
    require(layout=='bucket','Exact fresh6M bucket layout required')
    recipe={'schema':'wide_scratch_window_a_v1','layout':'bucket','rows':6000000,'batch_size':64,
        'window_rows':4096,'window_updates':64,'target_length_bin':16,
        'sort_keys':['target_nonpad_length//16','source_nonpad_length','original_window_index'],
        'batch_order':'separate_PCG64_permutation64','layout_seed':20261022,
        'layout_rng_policy':'permutation64_each_consumed_window','raw_generator':'fresh_seed42_uniform_with_replacement_origin0',
        'loss':'unchanged_token_mean_CE_per_update','schedule':'fresh200warmup_absolute_ISQRT_origin0',
        'checkpoint_cadence_NEW_attempts':1024,'first_admission_NEW_attempts':1024,'counter_origin':0}
    return recipe,fingerprint(recipe)

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
