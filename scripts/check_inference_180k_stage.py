"""Catch inherited120k/ordinary routing and duplicate mounts in real new stage."""
import json,tempfile,zipfile,hashlib
from pathlib import Path
from build_inference_mixture import stage
from inference_probe_protocol import verified_inference_parents
ROOT=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='hw02-inference180k-') as td:
 tools=Path(td)/'tools';kernel=Path(td)/'kernel'
 m=stage(tools,kernel,'fixture180k',trained_180k=True)
 assert set(m['parents'])=={'b027','b028'} and set(m['lanes'])=={'b029','b030'}
 assert m['parents']['b027']['selected_step']==180000
 assert m['parents']['b028']['selected_step']==179000
 assert all(p['completed_steps']==180000 for p in m['parents'].values())
 assert m['validation_parent']=='b027'
 assert m['lanes']['b029']['parents']==['b027','b028'] and m['lanes']['b029']['weights']==[.5,.5]
 assert m['lanes']['b030']['parents']==['b028'] and m['lanes']['b030']['beam_width']==8 and m['lanes']['b030']['batch_size']==128
 meta=json.loads((kernel/'kernel-metadata.json').read_text())
 assert meta['kernel_sources']==['arsen4ikvar/nlp-hw02-training-180k-20261003']
 assert meta['title'].replace(' ','-')==meta['id'].split('/')[1]
 assert meta['is_private'] and not meta['enable_internet'] and meta['machine_shape']=='NvidiaTeslaT4'
 with zipfile.ZipFile(tools/'inference_source.bin') as z:
  assert z.testzip() is None and set(z.namelist())==set(m['source_member_sha256'])
  for name,h in m['source_member_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==h
 paths=verified_inference_parents(ROOT/'artifacts/kaggle/b027-b028-v1',m)
 assert set(paths)=={'b027','b028'}
 original=json.dumps(m)
 for run,key,value in [('b028','selected_step',180000),('b027','completed_steps',120000),('b028','source_sha256','wrong'),('b027','files',{}),('b028','config',{'learnable_xsa':False})]:
  bad=json.loads(original);bad['parents'][run][key]=value
  try:verified_inference_parents(ROOT/'artifacts/kaggle/b027-b028-v1',bad)
  except (ValueError,AssertionError):pass
  else:raise AssertionError('Invalid newparentaccepted '+run+'/'+key)
 try:stage(tools,kernel,'fixture180k',trained_180k=True)
 except ValueError:pass
 else:raise AssertionError('Frozen stage overwritten')
print('PASS new180k stage / selected179k vs completed180k / one sharedmount / wrongparent rejection')
