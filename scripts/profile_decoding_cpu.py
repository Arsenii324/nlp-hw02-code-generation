"""CPU-only validation decoding comparison; never reads test questions."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import torch
from tokenizers import Tokenizer
from nlp_hw02.model import load_model
from nlp_hw02.inference import generate_questions,padded_tokens
from nlp_hw02.beam import generate_beam
from nlp_hw02.tokenizer import decode,EOS


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',default='artifacts/beam-b005-cpu')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--scope',choices=['pilot','remaining'],default='pilot')
    args=parser.parse_args()
    run=Path(args.run); output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    validation=json.loads((run/'validation.json').read_text())
    metrics=json.loads((run/'metrics.json').read_text())
    pilot_indices=set(metrics['validation_generation_indices'])
    selected=[validation[i] for i in metrics['validation_generation_indices']] if args.scope=='pilot' else [r for i,r in enumerate(validation) if i not in pilot_indices]
    assert len(selected)==(256 if args.scope=='pilot' else 1792)
    model,checkpoint=load_model(run/'model.pt',device='cpu')
    tok=Tokenizer.from_file(str(run/'tokenizer.json'))
    original=[json.loads(s)['prediction'] for s in (run/'validation_predictions.jsonl').read_text().splitlines()] if args.scope=='pilot' else None
    result={'device':'CPU','dtype':'FP32','torch':torch.__version__,'selected_step':checkpoint['step'],
        'checkpoint_sha256':hashlib.sha256((run/'model.pt').read_bytes()).hexdigest(),
        'beam_source_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'src/nlp_hw02/beam.py').read_bytes()).hexdigest(),
        'validation_sha256':hashlib.sha256((run/'validation.json').read_bytes()).hexdigest(),
        'note':'CPU validation-only search diagnostic. CUDA/fulltest timing unmeasured; source/evaluator fixed, no test tuning.',
        'scope':args.scope,'paths':{}}
    greedy=None
    cases=[('greedy',1,0.),('beam4-alpha0',4,0.),('beam4-alpha06',4,.6)] if args.scope=='pilot' else [('greedy',1,0.),('beam4-alpha06',4,.6)]
    for label,width,alpha in cases:
        print(json.dumps({'event':'started','path':label,'rows':len(selected),'device':'CPU'}),flush=True)
        directory=output/label; directory.mkdir(exist_ok=True)
        shutil.copyfile(run/'validation.json',directory/'validation.json')
        start=time.perf_counter()
        if width==1:
            predictions,timing=generate_questions(model,tok,[r['question'] for r in selected],batch_size=args.batch_size)
            greedy=predictions
        else:
            predictions=[]; lengths=[]; capped=0
            for offset in range(0,len(selected),args.batch_size):
                rows=selected[offset:offset+args.batch_size]
                src=padded_tokens(tok,[r['question'] for r in rows],model.cfg.max_source)
                tokens=generate_beam(model,src,beam_size=width,length_penalty=alpha)
                for ids in tokens.tolist():
                    lengths.append(ids.index(EOS) if EOS in ids else len(ids)); capped+=EOS not in ids
                    predictions.append(decode(tok,ids))
                print(json.dumps({'event':'batch','path':label,'completed':len(predictions)}),flush=True)
            timing={'seconds':time.perf_counter()-start,'rows':len(predictions),'capped_rows':capped,
                    'mean_generated_tokens':sum(lengths)/len(lengths),'batch_size':args.batch_size}
        assert len(predictions)==len(selected)
        with (directory/'validation_predictions.jsonl').open('w') as stream:
            for row,pred in zip(selected,predictions): stream.write(json.dumps({**row,'prediction':pred},ensure_ascii=False)+'\n')
        result['paths'][label]={'beam_width':width,'length_penalty':alpha,**timing,
            'changed_vs_cpu_greedy':sum(a!=b for a,b in zip(predictions,greedy)),
            'changed_vs_original_gpu_greedy':sum(a!=b for a,b in zip(predictions,original)) if original is not None else None}
        print(json.dumps({'event':'finished','path':label,**result['paths'][label]}),flush=True)
        (output/'decoding_profile.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__': main()
