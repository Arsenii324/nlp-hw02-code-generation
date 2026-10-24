"""One declared2M->6M intervention; strict exact-state restoration stays unchanged."""
import hashlib
import json
import math
from pathlib import Path
from .data_v2 import quotas


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def validate_stats(old,new):
    changed={'train_rows','shards','train_source_truncation_fraction',
             'train_target_truncation_fraction','sampled_print_fraction','prepare_elapsed_seconds'}
    if set(old)!=set(new) or any(old[k]!=new[k] for k in old if k not in changed):
        raise ValueError('Unapproved data metadata difference')
    if old['train_rows']!=2000000 or new['train_rows']!=6000000:
        raise ValueError('Only declared2M->6M transition allowed')
    if new['validation_rows']!=2048 or new['seed']!=42 or len(new['shards'])!=60:
        raise ValueError('Unexpected split/seed/shards')
    for key in changed-{'train_rows','shards'}:
        value=new[key]
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError('Invalid changed statistic: '+key)
        if key!='prepare_elapsed_seconds' and value>1:raise ValueError('Fraction out of bounds')
    expected=quotas(6000000,[s['rows'] for s in old['shards']])
    for a,b,count in zip(old['shards'],new['shards'],expected):
        if set(a)!=set(b) or any(a[k]!=b[k] for k in a if k not in {'sampled','sampled_print_rows'}):
            raise ValueError('Unapproved shard metadata difference')
        if b['sampled']!=count or type(b['sampled_print_rows']) is not int or not 0<=b['sampled_print_rows']<=count:
            raise ValueError('Invalid proportional count/print statistic')


def validate_transition(path,parent_dir,new_cache):
    path,parent_dir,new_cache=Path(path),Path(parent_dir),Path(new_cache)
    m=json.loads(path.read_text())
    if m['schema']!='nlp_hw02_controlled_cache_transition_v1' or m['step']!=240000:
        raise ValueError('Unknown controlled transition or step')
    if m['seed']!=42 or m['batch_size']!=64:raise ValueError('Unexpected transition recipe')
    for name,expected in m['parent_file_sha256'].items():
        if name not in {'last_training.pt','model.pt','run_config.json','metrics.json','data_stats.json'}:
            raise ValueError('Unexpected parent file')
        if sha(parent_dir/name)!=expected:raise ValueError('Parent transition file mismatch: '+name)
    if set(m['parent_file_sha256'])!={'last_training.pt','model.pt','run_config.json','metrics.json','data_stats.json'}:
        raise ValueError('Incomplete parent binding')
    meta=json.loads((parent_dir/'run_config.json').read_text());old=json.loads((parent_dir/'data_stats.json').read_text())
    if meta['data']!=old or meta['seed']!=m['seed'] or meta['batch_size']!=m['batch_size']:
        raise ValueError('Original parent data/recipe mismatch')
    metrics=json.loads((parent_dir/'metrics.json').read_text())
    if metrics['completed_steps']!=m['step']:raise ValueError('Wrong parent horizon')
    prepared=new_cache.parent/'prepare_result.json'
    if sha(prepared)!=m['CPU_terminal_sha256']:raise ValueError('Wrong prepared-cache terminal')
    terminal=json.loads(prepared.read_text())
    if (terminal['complete'] is not True or terminal['cache_ready'] is not True or
        terminal['GPU_requested'] is not False or terminal['source_sha256']!=m['CPU_source_sha256']):
        raise ValueError('Prepared cache not provenance-verified ready')
    if sha(new_cache/'cache_manifest.json')!=m['new_cache_manifest_sha256']:
        raise ValueError('New cache manifest binding mismatch')
    cache_manifest=json.loads((new_cache/'cache_manifest.json').read_text())
    if not cache_manifest['complete'] or cache_manifest['checksums']!=m['new_cache_checksums']:
        raise ValueError('Incomplete/unbound new cache checksums')
    for name,expected in m['new_small_file_sha256'].items():
        if name not in {'data_stats.json','coverage_audit.json','validation.json','validation_legacy.json','tokenizer.json'}:
            raise ValueError('Unexpected transition cache file')
        if sha(new_cache/name)!=expected or terminal['artifact_sha256'].get('cache/'+name)!=expected:
            raise ValueError('Small cache file provenance mismatch: '+name)
    if set(m['new_small_file_sha256'])!={'data_stats.json','coverage_audit.json','validation.json','validation_legacy.json','tokenizer.json'}:
        raise ValueError('Incomplete small-file binding')
    for name in ('validation.json','validation_legacy.json','tokenizer.json'):
        if sha(new_cache/name)!=sha(parent_dir/name):raise ValueError('Fixed holdout/tokenizer changed')
    new=json.loads((new_cache/'data_stats.json').read_text());validate_stats(old,new)
    coverage=json.loads((new_cache/'coverage_audit.json').read_text())
    if (coverage!=terminal['coverage'] or not coverage['complete'] or
        coverage['old_rows']!=2000000 or coverage['new_rows']!=6000000 or
        coverage['intersection_rows']!=2000000 or coverage['new_rows_not_in_old']!=4000000 or
        coverage['old_rows_absent_from_new']!=0 or not coverage['new_cache_nested'] or
        not all(coverage[k] for k in ('unique_row_ids','fixed_validation_tokenizer_legacy_bytes',
                  'new_cache_hashes_verified','parent_row_ids_hash_verified','proportional_quotas_verified'))):
        raise ValueError('Unexpected actual coverage audit')
    record={'kind':m['schema'],'absolute_transition_step':m['step'],'manifest_sha256':sha(path),
            'parent_file_sha256':m['parent_file_sha256'],'old_cache_manifest_sha256':m['old_cache_manifest_sha256'],
            'new_cache_manifest_sha256':m['new_cache_manifest_sha256'],'CPU_terminal_sha256':m['CPU_terminal_sha256'],
            'old_train_rows':2000000,'new_train_rows':6000000,'new_raw_row_ids':4000000,
            'exact_training_example_continuation':False,'full_optimizer_model_RNG_state_restoration':True,
            'note':'Original data validates strict state restoration; actual new data stays in run metadata. RNG state retained, future row-index mapping differs.'}
    return old,record
