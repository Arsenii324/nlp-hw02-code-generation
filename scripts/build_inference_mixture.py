"""Freeze two inference probes using mounted trained parents; no launch/upload."""
import argparse
import hashlib,json
from pathlib import Path
import shutil,zipfile
from inference_probe_protocol import sha
ROOT=Path(__file__).resolve().parents[1]
MEMBERS=('src/nlp_hw02/__init__.py','src/nlp_hw02/model.py','src/nlp_hw02/tokenizer.py','src/nlp_hw02/inference.py','src/nlp_hw02/beam.py','src/nlp_hw02/beam_inference.py','src/nlp_hw02/ensemble.py','scripts/inference_probe_protocol.py','scripts/inference_mixture_probe.py','scripts/check_probability_ensemble.py')


def stage(tools,kernel,tag,trained_180k=False):
 tools=Path(tools);kernel=Path(kernel)
 if not tag.isalnum():raise ValueError('Unique alphanumeric tag required')
 if tools.resolve()==kernel.resolve():raise ValueError('Distinct directories required')
 for directory in (tools,kernel):
  if directory.exists():raise ValueError('Refuse to overwrite a frozen stage')
 parents={}
 parent_specs=([('b027','b027-b028','arsen4ikvar/nlp-hw02-training-180k-20261003'),('b028','b027-b028','arsen4ikvar/nlp-hw02-training-180k-20261003')] if trained_180k else [('b019','b019-b020','arsen4ikvar/nlp-hw02-polar-resume-20261002'),('b024','b023-b024','arsen4ikvar/nlp-hw02-xsa-continuation-20261003')])
 completed=180000 if trained_180k else 120000
 for run,pair,slug in parent_specs:
  root=ROOT/'artifacts/kaggle'/(pair+'-v1');terminal=json.loads((root/'session_result.json').read_text());directory=root/f'run-{run}'
  assert terminal['complete'] and terminal['training_complete']
  files={f'run-{run}/{name}':terminal['artifact_sha256'][f'run-{run}/{name}'] for name in ['model.pt','run_config.json','metrics.json','tokenizer.json','validation.json']}
  for filename,digest in files.items():assert sha(root/filename)==digest
  metadata=json.loads((directory/'run_config.json').read_text());metrics=json.loads((directory/'metrics.json').read_text())
  assert metrics['completed_steps']==completed and metadata['torch']=='2.10.0+cu128' and metadata['cuda']=='12.8'
  selected={'b027':180000,'b028':179000}.get(run,120000)
  assert metrics['selected_step']==selected
  assert metadata['config']['learnable_xsa']==(run in ('b024','b028'))
  parents[run]={'run':run,'kernel':slug,'version':1,'terminal_sha256':sha(root/'session_result.json'),'source_sha256':terminal['training_source_sha256'],'files':files,'config':metadata['config'],'selected_step':selected,'completed_steps':completed,'cache_manifest_sha256':terminal['cache_manifest_sha256']}
 ordinary,learned=parents
 for name in ['tokenizer.json','validation.json']:
  assert parents[ordinary]['files'][f'run-{ordinary}/{name}']==parents[learned]['files'][f'run-{learned}/{name}']
 assert parents[ordinary]['cache_manifest_sha256']==parents[learned]['cache_manifest_sha256']
 source_files={name:(ROOT/name).read_bytes() for name in MEMBERS}
 for name,value in source_files.items():compile(value,name,'exec')
 driver=(ROOT/'scripts/kaggle_inference_mixture.py').read_bytes();compile(driver,'driver','exec')
 test=ROOT/'artifacts/kaggle/xsa-retry-tools/test.parquet';assert sha(test)=='8bde59c8d8bb7edc4e8a503be22fc132c4a47773025d333432e78c5489e6522a'
 tools.mkdir(parents=True);kernel.mkdir(parents=True)
 with zipfile.ZipFile(tools/'inference_source.bin','w',zipfile.ZIP_DEFLATED) as z:
  for name,value in source_files.items():
   info=zipfile.ZipInfo(name,date_time=(2026,10,3,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,value)
 shutil.copyfile(test,tools/'test.parquet');(kernel/'run.py').write_bytes(driver)
 anchor=ROOT/'artifacts/kaggle'/(parent_specs[0][1]+'-v1')/f'run-{ordinary}'
 wide_parent=learned if trained_180k else ordinary
 ensemble_lane,beam_lane=('b029','b030') if trained_180k else ('b025','b026')
 record={'kind':'inference-mixture-probe','parents':parents,'validation_parent':ordinary,'torch':'2.10.0+cu128','cuda':'12.8','cache_manifest_sha256':parents[ordinary]['cache_manifest_sha256'],'driver_sha256':sha(kernel/'run.py'),'tools_sha256':{name:sha(tools/name) for name in ['inference_source.bin','test.parquet']},'source_member_sha256':{name:hashlib.sha256(value).hexdigest() for name,value in source_files.items()},
 'validation_generation_indices':json.loads((anchor/'metrics.json').read_text())['validation_generation_indices'],
 'lanes':{ensemble_lane:{'device':0,'mechanism':'equal_token_probability_ensemble','parents':[ordinary,learned],'weights':[.5,.5],'beam_width':4,'length_penalty':.6,'batch_size':256},beam_lane:{'device':1,'mechanism':'learned_wider_beam' if trained_180k else 'ordinary_wider_beam','parents':[wide_parent],'beam_width':8,'length_penalty':.6,'batch_size':128}},
 'outer_timeout_seconds':600,'parent_seconds':540,'startup_seconds':90,'GPU_sessions_concurrent_limit':1,'validation_rows':1792,'test_rows':1919,'no_training_extra_targets_numeric_selection_or_submission':True,'selected_bundle_changed':False,'hypothesis':('Trained180k ordinary/learned-XSA errors may improve arithmetic probability-mixture decoding; beam8 tests additional search on selected learned179k model. Measured quality/time unknown.' if trained_180k else 'Complementary trained ordinary/learned-XSA errors may improve probability-mixture decoding; beam8 tests additional search on selected stronger ordinary model. Measured quality/time unknown.')}
 for directory in (tools,kernel):(directory/'inference_probe_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
 dataset='arsen4ikvar/nlp-hw02-tools-inference-mixture-'+tag;identity='arsen4ikvar/nlp-hw02-inference-mixture-'+tag
 (tools/'dataset-metadata.json').write_text(json.dumps({'id':dataset,'title':'NLP HW02 inference mixture tools '+tag,'licenses':[{'name':'other'}]},indent=2)+'\n')
 (kernel/'kernel-metadata.json').write_text(json.dumps({'id':identity,'title':identity.split('/')[1].replace('-',' '),'code_file':'run.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'machine_shape':'NvidiaTeslaT4','dataset_sources':[dataset],'competition_sources':[],'kernel_sources':list(dict.fromkeys(x['kernel'] for x in parents.values())),
 'docker_image':'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461','docker_image_pinning_type':'original'},indent=2)+'\n')
 return record


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tools',required=True);p.add_argument('--kernel',required=True);p.add_argument('--tag',required=True);p.add_argument('--trained-180k',action='store_true');a=p.parse_args();print(json.dumps(stage(a.tools,a.kernel,a.tag,a.trained_180k),indent=2))
