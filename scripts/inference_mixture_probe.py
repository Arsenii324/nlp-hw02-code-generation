"""Two declared frozen-weight validation decoders and fixed full-test timing."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import torch
from tokenizers import Tokenizer
from inference_probe_protocol import sha
from nlp_hw02.model import load_model,ModelConfig
from nlp_hw02.ensemble import ProbabilityEnsemble
from nlp_hw02.beam_inference import generate_beam_questions
from nlp_hw02.inference import syntax_valid


def select_probe_model(models,spec):
 names=spec['parents'];mechanism=spec['mechanism']
 if mechanism=='equal_token_probability_ensemble':
  if len(names)!=2:raise ValueError('Two models required for declared ensemble')
  return ProbabilityEnsemble([models[name] for name in names],spec['weights']).eval()
 if mechanism in ('ordinary_wider_beam','learned_wider_beam'):
  if len(names)!=1:raise ValueError('One model required for wider beam')
  return models[names[0]].eval()
 raise ValueError('Unknown decoder mechanism')


def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--parents',required=True);p.add_argument('--tools',required=True);p.add_argument('--output',required=True);p.add_argument('--lane',choices=['b025','b026','b029','b030'],required=True);a=p.parse_args()
 manifest=json.loads(Path(a.manifest).read_text());parents={k:Path(v) for k,v in json.loads(a.parents).items()};tools=Path(a.tools);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 assert torch.cuda.device_count()==1 and 'T4' in torch.cuda.get_device_name(0)
 assert str(torch.__version__)==manifest['torch'] and torch.version.cuda==manifest['cuda']
 torch.set_num_threads(2)
 import check_probability_ensemble as oracle
 import sys
 old_argv=sys.argv
 try:sys.argv=['check_probability_ensemble','--cuda'];oracle.main()
 finally:sys.argv=old_argv
 models={};checkpoints={}
 spec=manifest['lanes'][a.lane];names=spec['parents']
 for name in names:
  run=parents[name];pins=manifest['parents'][name]
  assert sha(run/'model.pt')==pins['files'][f'run-{name}/model.pt']
  model,checkpoint=load_model(run/'model.pt')
  assert checkpoint['step']==pins['selected_step'] and asdict(model.cfg)==asdict(ModelConfig(**pins['config']))
  models[name]=model;checkpoints[name]=pins['files'][f'run-{name}/model.pt']
 baseline=parents[manifest.get('validation_parent','b019')];validation=json.loads((baseline/'validation.json').read_text());pilot=set(json.loads((baseline/'metrics.json').read_text())['validation_generation_indices'])
 assert pilot==set(manifest['validation_generation_indices'])
 assert all(pilot==set(json.loads((path/'metrics.json').read_text())['validation_generation_indices']) for path in parents.values())
 selected=[row for i,row in enumerate(validation) if i not in pilot];assert len(selected)==1792
 tok=Tokenizer.from_file(str(baseline/'tokenizer.json'))
 model=select_probe_model(models,spec)
 report={'complete':False,'lane':a.lane,'torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu_count':1,'gpu':torch.cuda.get_device_name(0),'decoder':spec,'checkpoint_sha256':checkpoints,'validation_sha256':sha(baseline/'validation.json'),'tokenizer_sha256':sha(baseline/'tokenizer.json'),'source_sha256':manifest['tools_sha256']['inference_source.bin'],'selection_uses_reference_or_numeric_interpreter':False}
 def save():
  temp=out/'probe_result.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(out/'probe_result.json')
 def generate(questions):return generate_beam_questions(model,tok,questions,batch_size=spec['batch_size'],beam_size=spec['beam_width'],length_penalty=spec['length_penalty'])
 generate([validation[0]['question']]);torch.cuda.reset_peak_memory_stats()
 predictions,timing=generate([row['question'] for row in selected]);timing['peak_allocated_bytes']=torch.cuda.max_memory_allocated();timing['syntax_valid']=sum(map(syntax_valid,predictions))
 with (out/'validation_predictions.jsonl').open('w') as f:
  for row,pred in zip(selected,predictions):f.write(json.dumps({**row,'prediction':pred},ensure_ascii=False)+'\n')
 shutil.copyfile(baseline/'validation.json',out/'validation.json');report['validation_timing']=timing;save();print(json.dumps({'event':'validation_complete','lane':a.lane,**timing}),flush=True)
 # Fixed test questions only; no score or decoder/model adjustment after validation.
 import pyarrow.parquet as pq
 test=tools/'test.parquet';assert sha(test)==manifest['tools_sha256']['test.parquet']
 questions=pq.read_table(test,columns=['question'])['question'].to_pylist();assert len(questions)==1919
 torch.cuda.reset_peak_memory_stats();predictions,timing=generate(questions);timing.update(gpu_count=1,gpu=report['gpu'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),test_parquet_sha256=sha(test),meets_180_seconds_on_one_T4=timing['seconds']<=180)
 with (out/'test_predictions.jsonl').open('w') as f:
  for index,pred in enumerate(predictions):f.write(json.dumps({'row_index':index,'code':pred},ensure_ascii=False)+'\n')
 (out/'test_timing.json').write_text(json.dumps(timing,indent=2)+'\n');report.update(complete=True,test_timing=timing);save();print(json.dumps({'event':'fixed_test_complete','lane':a.lane,**timing}),flush=True)


if __name__=='__main__':main()
