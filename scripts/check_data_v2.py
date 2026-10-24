"""Tiny deterministic cache/leakage checks; no remote compute or real data."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from nlp_hw02.data_v2 import heldout_bucket,quotas,prepare_representative
from nlp_hw02.model import ModelConfig
from nlp_hw02.tokenizer import train_tokenizer,EOS


def main():
    assert quotas(17,[10,10,3])==[8,7,2]
    with tempfile.TemporaryDirectory() as d:
        d=Path(d); raw=d/'raw'; raw.mkdir()
        groups=[]; entries=[]
        for shard in range(2):
            ordinary=[]; held=[]
            i=0
            while len(ordinary)<200 or len(held)<8:
                q=f'question {shard} {i} Привет'; i+=1
                if heldout_bucket(q):
                    if len(held)<8: held.append(q)
                elif len(ordinary)<200: ordinary.append(q)
            qs=held[:2]+ordinary+held[2:]
            # Repeated question across shards must remain excluded as a group.
            if shard==1: qs[-1]=groups[0][-1]['question']
            rows=[{'question':q,'code':f'def solve():\n    return {j}\n'} for j,q in enumerate(qs)]
            groups.append(rows)
            path=raw/f'data/train/part-{shard:05d}.parquet'; path.parent.mkdir(parents=True,exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows),path)
            entries.append({'path':str(path.relative_to(raw)),'size':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        manifest={'repo_id':'never-download','revision':'fixture','files':entries}
        tokpath=d/'tokenizer.json'; train_tokenizer(['x=1\n','question 1'],tokpath,300)
        legacy=d/'legacy.json'; legacy.write_text(json.dumps([groups[1][0],groups[1][20]]))
        cfg=ModelConfig(vocab_size=300,max_source=20,max_target=20)
        for run in ['a','b']:
            out=d/run
            stats=prepare_representative(manifest,raw,out,cfg,tokpath,legacy,
                                        train_rows=100,validation_rows=12,tokenizer_training_rows=10)
            assert stats['train_rows']==100 and stats['validation_rows']==12
            val=json.loads((out/'validation.json').read_text())
            assert len({r['question'] for r in val})==12
            assert all(heldout_bucket(r['question']) for r in val)
            assert not {r['question'] for r in val}&{r['question'] for r in groups[0][:10]}
            assert not {r['question'] for r in val}&{r['question'] for r in [groups[1][0],groups[1][20]]}
            ids=np.load(out/'train_row_ids.npy')
            selected=[groups[int(s)][int(i)] for s,i in ids]
            assert all(not heldout_bucket(r['question']) for r in selected)
            assert groups[1][20] not in selected
            assert len(set(map(tuple,ids.tolist())))==100
            assert np.load(out/'train_source.npy').dtype==np.uint16
            assert all(EOS in row for row in np.load(out/'train_source.npy'))
            assert json.loads((out/'cache_manifest.json').read_text())['complete']
        for name in ['train_source.npy','train_target.npy','train_row_ids.npy','validation.json','tokenizer.json']:
            assert (d/'a'/name).read_bytes()==(d/'b'/name).read_bytes()
        try:
            prepare_representative(manifest,raw,d/'a',cfg,tokpath,legacy,100,4)
        except RuntimeError: pass
        else: raise AssertionError('nonempty cache overwrite was not rejected')
    print('PASS_DATA_V2: proportional quotas, deterministic sampling, bucket-wide group exclusion, legacy/tokenizer holdout isolation, EOS and cache manifest, overwrite refusal')


if __name__=='__main__': main()
