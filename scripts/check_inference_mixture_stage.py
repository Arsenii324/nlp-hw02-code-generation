"""Minimal frozen-source/parent/metadata staging contract; no remote actions."""
import json,tempfile,zipfile
from pathlib import Path
from build_inference_mixture import stage
from inference_probe_protocol import sha
with tempfile.TemporaryDirectory(prefix='hw02-mixture-stage-') as name:
 tools=Path(name)/'tools';kernel=Path(name)/'kernel';record=stage(tools,kernel,'fixture20261003')
 assert set(record['lanes'])=={'b025','b026'} and record['lanes']['b025']['weights']==[.5,.5]
 assert record['lanes']['b026']['beam_width']==8 and record['lanes']['b026']['batch_size']==128
 assert set(record['parents'])=={'b019','b024'}
 with zipfile.ZipFile(tools/'inference_source.bin') as z:
  assert z.testzip() is None and set(z.namelist())==set(record['source_member_sha256'])
  for filename,digest in record['source_member_sha256'].items():
   import hashlib
   assert hashlib.sha256(z.read(filename)).hexdigest()==digest
 for filename,digest in record['tools_sha256'].items():assert sha(tools/filename)==digest
 assert sha(kernel/'run.py')==record['driver_sha256']
 metadata=json.loads((kernel/'kernel-metadata.json').read_text());assert len(metadata['kernel_sources'])==2 and metadata['is_private'] and metadata['enable_gpu']
 assert all(p.suffix not in ['.pt','.npy','.npz'] for p in tools.iterdir())
 try:stage(tools,kernel,'fixture20261003')
 except ValueError:pass
 else:raise AssertionError('frozen stage overwritten')
 print(json.dumps({'event':'inference_mixture_stage_checks_passed','members':len(record['source_member_sha256']),'tool_bytes':sum(p.stat().st_size for p in tools.iterdir()),'no_weights_or_arrays_uploaded':True}))
