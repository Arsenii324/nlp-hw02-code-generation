"""Explicit own512 population transition; unchanged targets and bucket math."""
from copy import deepcopy
import hashlib,json,time
from pathlib import Path
import numpy as np
from .wide_scratch_recipe_a import fingerprint

ROWS=11811850
CACHE='695ad7d4acf0a9ff3b95249439ada6033fdc6db8693eaca33fbad1959fe9d9f9'
OLD_CACHE='8ce0939534bdb5f2b791d19d0ab13f6b4c79402b8627fecad4e9902cd08bb13c'
CHECKSUMS={'data_stats.json':'7db78ca8f775bfd82180be6fc6e735705cc853817479aed79706e0c2cfb36aa4','tokenizer.json':'0fa06854a98e82cd520c13f4c670040c772fec38f2cd2cacb5de11130bc42d74','train_row_ids.npy':'ada74dc42b9e1ad08b9735b1ed79a326ae5d044f5e047427df3afbe1771c2a52','train_source.npy':'e8824d9b9d500ffd865207b9458effb4e90059f7a3fcb5f0a58e6c1409738953','train_target.npy':'b1947e5eee95a40d2791a42e7ab1393a5095006b9b5db326ea72b2ddc3a5c6a8','validation.json':'73ac2a8b8ea0c16c2eec38c219b62f46857cdfe3de7fab3cab64754aa8b4c24e','validation_legacy.json':'d6a922705f41486208f8f68199bd18ebb4dea74c3b04ab106aed48d4c5d0c7dd'}
ORIGIN={'kernel':'arsen4ikvar/nlp-hw02-full-cache-20261023a','numeric_id':140306514,'output_version':1,'driver_sha256':'2b0f4ac81662096de60904a4f78cbfe1bf1286b170589bab0855cc6733c5e172','terminal_sha256':'b412869f8f90b3e6d704a4f29e60c123b6e45f022587c43f1f59309ce5783655','coverage_sha256':'cb1bf0055deac938665d729b16b217a941bb15befb0adce9fb2be9dd67726e3b'}
def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path,deadline):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):
            require(time.monotonic()<deadline,'Absolute input hashing deadline');h.update(b)
    return h.hexdigest()
def bound(root,entry,deadline):
    p=Path(entry['path']);require(not p.is_absolute() and '..' not in p.parts,'Explicit safe relative population evidence required');q=(Path(root)/p).resolve();require(q.is_relative_to(Path(root).resolve()) and sha(q,deadline)==entry['sha256'],'Population evidence differs');return q

def verify_cache(root,c,deadline):
    root=Path(root);require(c['cache_manifest_sha256']==CACHE and c['cache_checksums']==CHECKSUMS and sha(root/'cache_manifest.json',deadline)==CACHE,'Exact full eligible manifest required')
    manifest=json.loads((root/'cache_manifest.json').read_text());require(manifest['complete'] is True and manifest['checksums']==CHECKSUMS,'Exact complete7 full-cache roles required')
    binding=c['population_binding'];require(binding['origin']==ORIGIN,'Actual pinned full-cache origin required')
    terminal_path=bound(binding['artifact_root'],binding['terminal'],deadline);coverage_path=bound(binding['artifact_root'],binding['coverage'],deadline)
    require(binding['terminal']['sha256']==ORIGIN['terminal_sha256'] and binding['coverage']['sha256']==ORIGIN['coverage_sha256'],'Actual accepted CPU evidence required')
    terminal=json.loads(terminal_path.read_text());coverage=json.loads(coverage_path.read_text());require(terminal['complete'] is True and terminal['cache_ready'] is True and terminal['GPU_requested'] is False and terminal['driver_sha256']==ORIGIN['driver_sha256'] and terminal['artifact_hashes_complete'] is True,'CPU full-cache preparation not ready')
    amap=terminal['artifact_sha256'];require(amap['cache/cache_manifest.json']==CACHE and all(amap['cache/'+n]==h for n,h in CHECKSUMS.items()) and amap['cache/coverage_audit.json']==ORIGIN['coverage_sha256'],'All7 explicit full-cache terminal-map bindings required')
    require(coverage==terminal['coverage'] and coverage['complete'] and coverage['all_eligible_counts_verified'] and coverage['old_rows']==coverage['intersection_rows']==6000000 and coverage['new_rows']==ROWS and coverage['new_rows_not_in_old']==5811850 and coverage['old_rows_absent_from_new']==0 and coverage['fixed_validation_tokenizer_legacy_bytes'] and coverage['proportional_sampling'] is False,'Actual all-eligible coverage differs')
    for n,h in CHECKSUMS.items():require(sha(root/n,deadline)==h,'Actual full-cache bytes differ:'+n)
    return {'origin':ORIGIN,'manifest_sha256':CACHE,'checksums':CHECKSUMS,'coverage':coverage,'old_arrays_mounted_or_read':False,'loading':'read_only_mmap','target_policy':'Original supplied code/EOS/caps/tokenizer unchanged; no filtering/newtargets'}

def compare_parent_data(old,new):
    changed={'train_rows','shards','sampling','train_source_truncation_fraction','train_target_truncation_fraction','sampled_print_fraction','prepare_elapsed_seconds'}
    require(old['train_rows']==6000000 and new['train_rows']==ROWS and new['parent_cache_manifest_sha256']==OLD_CACHE,'Explicit old6M/newfull ranges required')
    require(set(new)==set(old)|{'eligibility_policy','parent_cache_manifest_sha256'} and all(new[k]==v for k,v in old.items() if k not in changed),'Unapproved data metadata drift')
    require(new['sampling']=='all eligible supplied raw rows in sorted shard/row order; duplicate question weighting retained' and len(old['shards'])==len(new['shards'])==60 and sum(x['eligible'] for x in new['shards'])==ROWS,'All-eligible sampling/shard counts differ')
    for a,b in zip(old['shards'],new['shards']):require(all(a[k]==b[k] for k in ('path','rows','validation_selected','hash_bucket_excluded','legacy_matches')) and b['sampled']==b['eligible']==b['rows']-b['hash_bucket_excluded']-b['legacy_matches']+b['bucket_legacy_overlap'],'Original holdout/shard union differs')
    policy=new['eligibility_policy'];require(policy['test_data_used'] is False and policy['proportional_sampling'] is False,'Unchanged supplied eligibility policy required')

def recipe_for(parent_step,raw_seed,layout_seed):
    require(type(parent_step) is int and parent_step>=1024 and parent_step%64==0 and all(type(x) is int and 0<=x<2**64 for x in (raw_seed,layout_seed)),'Actual parent and explicit new PCG64 seeds required')
    recipe={'schema':'wide_fullpopulation_recipe_a_v1','layout':'bucket','rows':ROWS,'batch_size':64,'window_rows':4096,'window_updates':64,'target_length_bin':16,'sort_keys':['target_nonpad_length//16','source_nonpad_length','original_window_index'],'batch_order':'separate_PCG64_permutation64','raw_generator':'NEW independent PCG64 seed at population NEW0; not continued6M example stream','raw_seed':raw_seed,'layout_seed':layout_seed,'parent_absolute_step':parent_step,'parent_cache_manifest_sha256':OLD_CACHE,'cache_manifest_sha256':CACHE,'row_ids_sha256':CHECKSUMS['train_row_ids.npy'],'global_row_ID_mapping':'uint32[shard_index,raw_row_index] from pinned fullcache index','counter_origin':0,'loss':'unchanged_token_mean_CE_per_update','checkpoint_cadence_NEW_attempts':1024,'first_admission_NEW_attempts':1024}
    return recipe,fingerprint(recipe)

def draw_window(rng,layout_rng,source,target,layout,recipe):
    """Original bucket arithmetic; only explicit population/recipe dispatch changes."""
    import time
    require(recipe['schema']=='wide_fullpopulation_recipe_a_v1' and layout==recipe['layout']=='bucket' and recipe['rows']==ROWS,'Exact explicit full-population recipe required')
    require(len(source)==len(target)==recipe['rows'],'Exact full-population rows required')
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
