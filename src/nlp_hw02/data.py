"""Bounded, pinned raw Parquet download and compact token arrays."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import numpy as np
import pyarrow.parquet as pq
from .tokenizer import train_tokenizer, EOS


def fetch(manifest, root, paths):
    root=Path(root)
    entries={f['path']:f for f in manifest['files']}
    for name in paths:
        path=root/name; path.parent.mkdir(parents=True,exist_ok=True)
        entry=entries[name]
        def valid():
            return path.exists() and path.stat().st_size==entry['size'] and hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        if not valid():
            url=f"https://huggingface.co/datasets/{manifest['repo_id']}/resolve/{manifest['revision']}/{name}"
            temp=path.with_suffix('.download')
            urllib.request.urlretrieve(url,temp)
            if temp.stat().st_size!=entry['size'] or hashlib.sha256(temp.read_bytes()).hexdigest()!=entry['sha256']:
                raise RuntimeError(f'Checksum mismatch: {name}')
            os.replace(temp,path)
        print(f'Verified {name}',flush=True)


def rows(paths):
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096,columns=['question','code']):
            yield from batch.to_pylist()


def question_hash(question):
    return hashlib.sha256(question.encode()).digest()


def prepare(manifest, root, output, cfg):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    names=[f'data/train/part-{i:05d}.parquet' for i in range(4)]
    val_name='data/train/part-00059.parquet'
    fetch(manifest,root,names+[val_name])
    train_paths=[Path(root)/n for n in names]
    def tokenizer_texts():
        for i,row in enumerate(rows(train_paths)):
            if i>=100_000: break
            yield row['question']; yield row['code']
    tok=train_tokenizer(tokenizer_texts(),output/'tokenizer.json',cfg.vocab_size)
    n=sum(pq.ParquetFile(p).metadata.num_rows for p in train_paths)
    src=np.lib.format.open_memmap(output/'train_source.npy',mode='w+',dtype=np.uint16,shape=(n,cfg.max_source))
    tgt=np.lib.format.open_memmap(output/'train_target.npy',mode='w+',dtype=np.uint16,shape=(n,cfg.max_target))
    src[:]=0; tgt[:]=0
    hashes=set(); trunc_src=trunc_tgt=0; offset=0
    for path in train_paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096,columns=['question','code']):
            batch_rows=batch.to_pylist()
            q=[r['question'] for r in batch_rows]; c=[r['code'] for r in batch_rows]
            hashes.update(map(question_hash,q))
            a=tok.encode_batch(q); b=tok.encode_batch(c)
            for j,(aa,bb) in enumerate(zip(a,b)):
                trunc_src+=len(aa.ids)>=cfg.max_source; trunc_tgt+=len(bb.ids)>=cfg.max_target
                ai=aa.ids[:cfg.max_source-1]+[EOS]; bi=bb.ids[:cfg.max_target-1]+[EOS]
                src[offset+j,:len(ai)]=ai; tgt[offset+j,:len(bi)]=bi
            offset+=len(batch_rows)
        print(f'Tokenized {offset}/{n}',flush=True)
    src.flush(); tgt.flush()
    val=[]; val_hashes=set(); excluded=0
    for row in rows([Path(root)/val_name]):
        h=question_hash(row['question'])
        if h in hashes or h in val_hashes: excluded+=1; continue
        val_hashes.add(h); val.append(row)
        if len(val)>=2048: break
    if len(val)<256: raise RuntimeError(f'Only {len(val)} nonoverlapping validation questions')
    (output/'validation.json').write_text(json.dumps(val,ensure_ascii=False))
    stats={'train_rows':n,'validation_rows':len(val),'validation_excluded_duplicate_questions':excluded,
           'train_source_truncation_fraction':trunc_src/n,'train_target_truncation_fraction':trunc_tgt/n,
           'train_shards':names,'validation_shard':val_name,'tokenizer_training_rows':100000,
           'dataset_revision':manifest['revision'],'vocab_size':tok.get_vocab_size()}
    (output/'data_stats.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps(stats),flush=True)
    return tok
