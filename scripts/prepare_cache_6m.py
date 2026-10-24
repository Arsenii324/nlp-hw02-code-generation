"""CPU-only unchanged pinned-data preparation plus fixed-holdout coverage audit."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nlp_hw02.data_v2 import prepare_representative
from nlp_hw02.model import ModelConfig
from check_cache_coverage import audit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--output',required=True)
    p.add_argument('--parent-cache',required=True);p.add_argument('--manifest',required=True);args=p.parse_args()
    manifest=json.loads(Path(args.manifest).read_text());parent=Path(args.parent_cache)
    if manifest['train_rows']!=6000000 or manifest['seed']!=42:raise ValueError('Only declared6M/seed42 supported')
    prepare_representative(json.loads((ROOT/'assignment/dataset_manifest.json').read_text()),
        args.raw,args.output,ModelConfig(),parent/'tokenizer.json',parent/'validation_legacy.json',
        train_rows=manifest['train_rows'],validation_rows=2048,seed=42,tokenizer_training_rows=100000)
    print(json.dumps(audit(parent,args.output,manifest)),flush=True)
