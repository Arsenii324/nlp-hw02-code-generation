"""Runtime row-coverage and immutable holdout audit; arrays remain remote."""
import hashlib
import json
from pathlib import Path
import numpy as np
from nlp_hw02.data_v2 import quotas


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def row_keys(cache,stats,pinned_shards):
    rows=np.load(Path(cache)/'train_row_ids.npy',mmap_mode='r')
    if rows.dtype!=np.uint32 or rows.shape!=(stats['train_rows'],2):
        raise ValueError('Row IDs must be uint32 Nx2')
    keys=(rows[:,0].astype(np.uint64)<<32)|rows[:,1].astype(np.uint64)
    if not np.all(keys[1:]>keys[:-1]):raise ValueError('Row IDs must be globally sorted and unique')
    shards=stats['shards']
    if len(shards)!=60 or len(stats['train_shards'])!=60:raise ValueError('All60 supplied train shards required')
    if not np.all(rows[:,0]<len(shards)):raise ValueError('Unknown shard ID')
    expected=quotas(stats['train_rows'],[s['rows'] for s in pinned_shards])
    for index,(info,pinned,count) in enumerate(zip(shards,pinned_shards,expected)):
        positions=np.flatnonzero(rows[:,0]==index)
        if info['sampled']!=count or len(positions)!=count:raise ValueError('Proportional shard quota mismatch')
        if (info['path']!=stats['train_shards'][index] or info['path']!=pinned['path'] or
            info['rows']!=pinned['rows'] or not np.all(rows[positions,1]<pinned['rows'])):
            raise ValueError('Out-of-bounds row or changed pinned shard metadata')
    for name,width in [('train_source.npy',256),('train_target.npy',192)]:
        a=np.load(Path(cache)/name,mmap_mode='r')
        if a.dtype!=np.uint16 or a.shape!=(stats['train_rows'],width):raise ValueError('Token array shape/dtype mismatch')
    return keys


def audit(old,new,manifest):
    old,new=Path(old),Path(new)
    for name,expected in manifest['parent_cache_sha256'].items():
        if sha(old/name)!=expected:raise ValueError('Mounted parent cache identity mismatch: '+name)
    old_stats=json.loads((old/'data_stats.json').read_text());new_stats=json.loads((new/'data_stats.json').read_text())
    if old_stats['train_rows']!=2000000 or new_stats['train_rows']!=manifest['train_rows']:
        raise ValueError('Training row count mismatch')
    for key in ('validation_rows','seed','dataset_revision','train_shards','split_salt','split',
                'tokenizer_training_rows','primary_excludes_tokenizer_training_question_hashes',
                'legacy_validation_rows','vocab_size','sampling'):
        if old_stats[key]!=new_stats[key]:raise ValueError('Split/recipe drift: '+key)
    for name in ('validation.json','validation_legacy.json','tokenizer.json'):
        if sha(old/name)!=sha(new/name):raise ValueError('Fixed holdout/tokenizer bytes changed: '+name)
    cm=json.loads((new/'cache_manifest.json').read_text())
    expected_files={'data_stats.json','validation_legacy.json','validation.json','tokenizer.json',
                    'train_source.npy','train_target.npy','train_row_ids.npy'}
    if not cm['complete'] or set(cm['checksums'])!=expected_files:raise ValueError('Incomplete cache manifest')
    for name,expected in cm['checksums'].items():
        if sha(new/name)!=expected:raise ValueError('Prepared cache hash mismatch: '+name)
    pinned_shards=old_stats['shards']
    a=row_keys(old,old_stats,pinned_shards);b=row_keys(new,new_stats,pinned_shards)
    positions=np.searchsorted(b,a);inbounds=positions<len(b)
    intersection=int(np.count_nonzero(b[positions[inbounds]]==a[inbounds]))
    report={'complete':True,'old_rows':len(a),'new_rows':len(b),'intersection_rows':intersection,
            'new_rows_not_in_old':len(b)-intersection,'old_rows_absent_from_new':len(a)-intersection,
            'new_cache_nested':intersection==len(a),'unique_row_ids':True,
            'fixed_validation_tokenizer_legacy_bytes':True,'new_cache_hashes_verified':True,'parent_row_ids_hash_verified':True,'proportional_quotas_verified':True,
            'dataset_revision':new_stats['dataset_revision'],
            'note':'Unique raw row IDs, not unique question/templates. Sampling is not assumed nested; duplicate question weighting retained.'}
    if report['new_rows_not_in_old']<4000000:raise ValueError('Insufficient new row coverage')
    (new/'coverage_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
