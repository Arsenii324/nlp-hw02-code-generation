"""Standalone loss plots from measured JSONL logs; run where matplotlib exists."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='/content/nlp-hw02')
    parser.add_argument('--out',default='/content/experiment_curves')
    parser.add_argument('--run',action='append',help='DIR=LABEL, all must share validation.json SHA256')
    args=parser.parse_args()
    root=Path(args.root); out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    runs=[s.split('=',1) for s in args.run] if args.run else [('run','AdamW baseline'),('run-xsa','XSA'),('run-muon','Hybrid Muon'),('run-relu2','ReLU²')]
    split_hashes=set()
    import hashlib
    for dirname,label in runs:
        path=root/dirname/'train.jsonl'
        if not path.exists(): continue
        split_hashes.add(hashlib.sha256((root/dirname/'validation.json').read_bytes()).hexdigest())
        assert len(split_hashes)==1,'Do not compare CE on different validation splits'
        records=[json.loads(s) for s in path.read_text().splitlines()]
        val=[r for r in records if r['event']=='validation']
        axes[0].plot([r['step'] for r in val],[r['loss'] for r in val],label=label,lw=1.8)
        axes[1].plot([r['elapsed_seconds']/60 for r in val],[r['loss'] for r in val],label=label,lw=1.8)
    for ax in axes:
        ax.set_ylabel('Held-out token CE (nats)'); ax.grid(alpha=.2); ax.legend(frameon=False)
    axes[0].set_xlabel('Training iterations'); axes[1].set_xlabel('Training + loss-validation minutes')
    fig.suptitle('TinySMS code generation · 2,048 held-out questions · one seed')
    fig.savefig(str(out)+'.png',dpi=180); fig.savefig(str(out)+'.pdf')
    print(str(out)+'.png'); print(str(out)+'.pdf')


if __name__=='__main__': main()
