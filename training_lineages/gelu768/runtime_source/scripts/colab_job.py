"""Run from an unpacked source bundle; all paths are explicit and VM-local."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','train','test'])
    parser.add_argument('--root',default='/content/nlp-hw02')
    parser.add_argument('--seconds',type=int,default=3600)
    parser.add_argument('--variant',choices=['baseline','xsa','muon','relu2'],default='baseline')
    args=parser.parse_args()
    if args.stage=='prepare' and args.variant!='baseline':
        parser.error('prepare builds shared data: use --variant baseline, then train with your chosen --variant')
    root=Path(args.root); out=root/('run' if args.variant=='baseline' else f'run-{args.variant}'); out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'assignment/dataset_manifest.json').read_text())
    from nlp_hw02.model import ModelConfig,load_model
    from nlp_hw02.data import prepare,fetch
    if args.stage=='prepare': prepare(manifest,root/'data',out,ModelConfig())
    elif args.stage=='train':
        from nlp_hw02.train import train
        train(out,cfg=ModelConfig(exclusive_self_attention=args.variant=='xsa',activation='relu2' if args.variant=='relu2' else 'gelu'),seconds=args.seconds,
              optimizer_kind='muon' if args.variant=='muon' else 'adamw',data_dir=root/'run')
    else:
        import pyarrow.parquet as pq
        import torch
        from tokenizers import Tokenizer
        from nlp_hw02.inference import generate_questions,syntax_valid
        torch.set_num_threads(2)
        name='data/test/test.parquet'; fetch(manifest,root/'data',[name])
        questions=pq.read_table(root/'data'/name,columns=['question'])['question'].to_pylist()
        model,checkpoint=load_model(out/'model.pt')
        tok=Tokenizer.from_file(str(out/'tokenizer.json'))
        # Warm-up uses a training question, never a test prompt.
        row=json.loads((out/'validation.json').read_text())[0]
        generate_questions(model,tok,[row['question']])
        predictions,timing=generate_questions(model,tok,questions)
        assert len(predictions)==len(questions)
        timing.update({'gpu':torch.cuda.get_device_name(0),'gpu_count':torch.cuda.device_count(),
            'selected_step':checkpoint['step'],'syntax_valid_fraction':sum(map(syntax_valid,predictions))/len(predictions),
            'meets_180_seconds_on_one_T4':len(predictions)==1919 and torch.cuda.device_count()==1 and 'T4' in torch.cuda.get_device_name(0) and timing['seconds']<=180})
        (out/'test_timing.json').write_text(json.dumps(timing,indent=2))
        with (out/'test_predictions.jsonl').open('w') as f:
            for i,p in enumerate(predictions): f.write(json.dumps({'row_index':i,'code':p},ensure_ascii=False)+'\n')
        print(json.dumps(timing),flush=True)


if __name__=='__main__': main()
