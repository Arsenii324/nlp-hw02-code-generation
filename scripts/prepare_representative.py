"""CPU-only pinned cache preparation; run from the unpacked source package."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--raw',required=True); p.add_argument('--output',required=True)
    p.add_argument('--tokenizer',required=True); p.add_argument('--legacy-validation',required=True)
    args=p.parse_args()
    from nlp_hw02.data_v2 import prepare_representative
    from nlp_hw02.model import ModelConfig
    prepare_representative(json.loads((ROOT/'assignment/dataset_manifest.json').read_text()),
                           args.raw,args.output,ModelConfig(),args.tokenizer,args.legacy_validation)
