"""Representative supplied-data cache with question-group leakage controls."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from .data import fetch, question_hash
from .tokenizer import EOS

SPLIT_SALT=b'nlp-hw02-v2-20261001:'


def heldout_bucket(question):
    return int.from_bytes(hashlib.sha256(SPLIT_SALT+question.encode()).digest()[:8],'big')%500==0


def quotas(total, sizes):
    values=np.asarray(sizes,dtype=np.float64)*total/sum(sizes)
    counts=np.floor(values).astype(int)
    order=np.argsort(-(values-counts),kind='stable')
    counts[order[:total-int(counts.sum())]]+=1
    assert int(counts.sum())==total
    return counts.tolist()


def prepare_representative(manifest, root, output, cfg, tokenizer_path,
                           legacy_validation_path, train_rows=2_000_000,
                           validation_rows=2048, seed=42, tokenizer_training_rows=100_000):
    output=Path(output); root=Path(root)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Refusing to overwrite a nonempty prepared-cache directory')
    output.mkdir(parents=True,exist_ok=True)
    names=sorted(f['path'] for f in manifest['files'] if f['path'].startswith('data/train/'))
    fetch(manifest,root,names)
    paths=[root/name for name in names]
    sizes=[pq.ParquetFile(p).metadata.num_rows for p in paths]
    train_quota=quotas(train_rows,sizes); val_quota=quotas(validation_rows,sizes)
    if shutil.disk_usage(output).free < train_rows*(cfg.max_source+cfg.max_target)*2+2*1024**3:
        raise RuntimeError('Insufficient cache disk headroom')
    tok=Tokenizer.from_file(str(tokenizer_path))
    assert tok.get_vocab_size()<=65535
    legacy=json.loads(Path(legacy_validation_path).read_text())
    legacy_hashes={question_hash(r['question']) for r in legacy}
    first=pq.read_table(paths[0],columns=['question']).slice(0,tokenizer_training_rows)
    tokenizer_hashes=set(map(question_hash,first['question'].to_pylist()))
    del first
    src=np.lib.format.open_memmap(output/'train_source.npy',mode='w+',dtype=np.uint16,shape=(train_rows,cfg.max_source))
    tgt=np.lib.format.open_memmap(output/'train_target.npy',mode='w+',dtype=np.uint16,shape=(train_rows,cfg.max_target))
    ids=np.lib.format.open_memmap(output/'train_row_ids.npy',mode='w+',dtype=np.uint32,shape=(train_rows,2))
    src[:]=0; tgt[:]=0
    val=[]; val_hashes=set(); offset=0; trunc_src=trunc_tgt=0; shard_stats=[]
    start=time.monotonic()
    for shard,(path,n,nt,nv) in enumerate(zip(paths,sizes,train_quota,val_quota)):
        table=pq.read_table(path,columns=['question','code'])
        questions=table['question'].to_pylist()
        hashes=list(map(question_hash,questions))
        bucket=np.fromiter(map(heldout_bucket,questions),dtype=bool,count=n)
        excluded=np.fromiter((h in legacy_hashes for h in hashes),dtype=bool,count=n)
        eligible=np.flatnonzero(~bucket & ~excluded)
        rng=np.random.default_rng(np.random.SeedSequence([seed,shard]))
        selected=np.sort(rng.choice(eligible,size=nt,replace=False))
        added=0
        val_order=np.random.default_rng(np.random.SeedSequence([seed,shard,1])).permutation(np.flatnonzero(bucket))
        for j in val_order if nv else []:
            h=hashes[j]
            if h in legacy_hashes or h in tokenizer_hashes or h in val_hashes: continue
            val.append({'question':questions[j],'code':table['code'][int(j)].as_py(),
                        'shard':names[shard],'row_index':int(j),'question_sha256':h.hex()})
            val_hashes.add(h); added+=1
            if added==nv: break
        if added!=nv: raise RuntimeError(f'Not enough distinct holdout candidates in {path.name}: {added}/{nv}')
        sampled=table.take(selected).to_pylist()
        print_rows=sum(bool(re.search(r'\bprint\s*\(',r['code'])) for r in sampled)
        for j in range(0,len(sampled),4096):
            chunk=sampled[j:j+4096]
            aa=tok.encode_batch([r['question'] for r in chunk]); bb=tok.encode_batch([r['code'] for r in chunk])
            for k,(a,b) in enumerate(zip(aa,bb)):
                trunc_src+=len(a.ids)>=cfg.max_source; trunc_tgt+=len(b.ids)>=cfg.max_target
                a=a.ids[:cfg.max_source-1]+[EOS]; b=b.ids[:cfg.max_target-1]+[EOS]
                idx=offset+j+k; src[idx,:len(a)]=a; tgt[idx,:len(b)]=b
        ids[offset:offset+nt,0]=shard; ids[offset:offset+nt,1]=selected
        offset+=nt
        shard_stats.append({'path':names[shard],'rows':n,'sampled':nt,'validation_selected':added,
                            'hash_bucket_excluded':int(bucket.sum()),'legacy_matches':int(excluded.sum()),
                            'sampled_print_rows':print_rows})
        print(json.dumps({'stage':'cache','shard':shard,'sampled':offset,'validation':len(val),
                          'elapsed_seconds':time.monotonic()-start}),flush=True)
        del table,questions,hashes,bucket,excluded,eligible,selected,sampled
    assert offset==train_rows and len(val)==validation_rows
    assert not val_hashes.intersection(legacy_hashes|tokenizer_hashes)
    src.flush(); tgt.flush(); ids.flush()
    shutil.copyfile(tokenizer_path,output/'tokenizer.json')
    shutil.copyfile(legacy_validation_path,output/'validation_legacy.json')
    (output/'validation.json').write_text(json.dumps(val,ensure_ascii=False))
    stats={'train_rows':train_rows,'validation_rows':validation_rows,'seed':seed,
           'dataset_revision':manifest['revision'],'train_shards':names,'shards':shard_stats,
           'split_salt':SPLIT_SALT.decode(),'split':'salted question SHA256 first64 bits mod500 ==0 held out; every bucket member excluded',
           'tokenizer_training_rows':tokenizer_training_rows,'primary_excludes_tokenizer_training_question_hashes':True,
           'legacy_validation_rows':len(legacy),'train_source_truncation_fraction':trunc_src/train_rows,
           'train_target_truncation_fraction':trunc_tgt/train_rows,
           'sampled_print_fraction':sum(s['sampled_print_rows'] for s in shard_stats)/train_rows,
           'vocab_size':tok.get_vocab_size(),'sampling':'uniform without replacement within each shard, proportional quotas; duplicate question weighting retained',
           'prepare_elapsed_seconds':time.monotonic()-start}
    (output/'data_stats.json').write_text(json.dumps(stats,indent=2))
    checksums={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.suffix=='.json'}
    # Large arrays are streamed to avoid allocating their full byte contents.
    for path in [output/'train_source.npy',output/'train_target.npy',output/'train_row_ids.npy']:
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
        checksums[path.name]=h.hexdigest()
    (output/'cache_manifest.json').write_text(json.dumps({'checksums':checksums,'complete':True},indent=2))
    print('REPRESENTATIVE_CACHE_OK',flush=True)
    return stats
