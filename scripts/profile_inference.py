"""Benchmark batching on validation only; never pick settings on test timings."""
import argparse
import json
from pathlib import Path
import sys
import torch
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nlp_hw02.model import load_model
from nlp_hw02.inference import generate_questions


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',required=True)
    args=parser.parse_args(); run=Path(args.run)
    torch.set_num_threads(2)
    model,_=load_model(run/'model.pt')
    tok=Tokenizer.from_file(str(run/'tokenizer.json'))
    questions=[r['question'] for r in json.loads((run/'validation.json').read_text())[:256]]
    generate_questions(model,tok,questions[:1])
    # Predeclared comparison; retain all measurements and ordering differences.
    records=[]; reference=None
    for batch in [128,32,64,256]:
        times=[]
        for repeat in range(2):
            predictions,timing=generate_questions(model,tok,questions,batch)
            times.append(timing['seconds'])
        if reference is None: reference=predictions
        records.append({'batch':batch,'seconds':times,'mean_seconds':sum(times)/len(times),
            'different_text_rows_vs_batch128':sum(a!=b for a,b in zip(reference,predictions))})
    output={'rows':len(questions),'split':'validation only','gpu':torch.cuda.get_device_name(0),
            'measurements':records,'note':'Microbenchmark; not a new full-test timing claim. Cached inference is already comfortably within budget.'}
    (run/'batch_profile.json').write_text(json.dumps(output,indent=2)); print(json.dumps(output,indent=2))


if __name__=='__main__': main()
