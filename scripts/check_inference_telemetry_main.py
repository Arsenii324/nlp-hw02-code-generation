"""Run real parent main with real children: old timeout aborts, new preserves."""
import json,hashlib,subprocess,sys,tempfile,types,time
from pathlib import Path
from unittest.mock import patch
import inference_probe_protocol
ROOT=Path(__file__).resolve().parents[1]
real_run=subprocess.run;real_popen=subprocess.Popen

def check(driver_path,want_complete):
 module=types.ModuleType('fixture_parent');module.__file__=str(driver_path)
 exec(compile(driver_path.read_text(),str(driver_path),'exec'),module.__dict__)
 with tempfile.TemporaryDirectory(prefix='hw02-parent-telemetry-') as td:
  base=Path(td);working=base/'working';working.mkdir();tools=base/'input';tools.mkdir()
  original=ROOT/'artifacts/kaggle/tools-inference-mixture-20261003b'
  manifest=json.loads((original/'inference_probe_manifest.json').read_text());manifest['driver_sha256']=hashlib.sha256(driver_path.read_bytes()).hexdigest()
  (tools/'inference_probe_manifest.json').write_text(json.dumps(manifest))
  for name in ['inference_source.bin','test.parquet']:(tools/name).write_bytes((original/name).read_bytes())
  parents={name:ROOT/'artifacts/kaggle/b027-b028-v1'/f'run-{name}' for name in ['b027','b028']}
  aliases={'/kaggle/working':working,'/kaggle/input':tools,'/kaggle/temp/inference-mixture':base/'temp'}
  def paths(value,*parts):return aliases.get(str(value),Path(value)).joinpath(*parts)
  def parent_boundary(input_root,record):
   assert input_root=='/kaggle/input' and set(record['parents'])=={'b027','b028'}
   return parents
  def child_boundary(argv,**kwargs):
   assert '--lane' in argv and '--output' in argv
   lane=argv[argv.index('--lane')+1];out=Path(argv[argv.index('--output')+1]);assert lane in ['b029','b030']
   assert kwargs['env']['CUDA_VISIBLE_DEVICES']==str(manifest['lanes'][lane]['device'])
   # Actual subprocesses with real exit/reaping; CUDA work is a declared double.
   code='import time,json;from pathlib import Path;time.sleep(.5);p=Path('+repr(str(out))+');p.mkdir();(p/"probe_result.json").write_text(json.dumps({"complete":True}))'
   return real_popen([sys.executable,'-c',code],**kwargs)
  def query_boundary(argv,**kwargs):
   assert argv[0]=='nvidia-smi'
   if '--query-gpu=name' in argv:return subprocess.CompletedProcess(argv,0,'Tesla T4\nTesla T4\n','')
   assert kwargs['timeout']==5
   return real_run([sys.executable,'-c','import time;time.sleep(1)'],timeout=.02,capture_output=True,text=True)
  first_clock=True
  def clock():
   nonlocal first_clock
   value=time.monotonic()
   if first_clock:first_clock=False;return value
   return value+31.
  failure=None
  with patch.object(module,'time',types.SimpleNamespace(monotonic=clock,sleep=time.sleep,strftime=time.strftime,gmtime=time.gmtime)),patch.object(module,'Path',side_effect=paths),patch.object(inference_probe_protocol,'verified_inference_parents',side_effect=parent_boundary),patch.object(module,'subprocess',types.SimpleNamespace(Popen=child_boundary,run=query_boundary,TimeoutExpired=subprocess.TimeoutExpired,STDOUT=subprocess.STDOUT)):
   try:module.main()
   except subprocess.TimeoutExpired as exc:failure=exc
  terminal=json.loads((working/'session_result.json').read_text())
  assert terminal['complete'] is want_complete
  if want_complete:
   assert failure is None and all(v['exit']==0 for v in terminal['lanes'].values())
   events=[json.loads(s) for s in (working/'session_events.jsonl').read_text().splitlines()]
   assert any(e['event']=='telemetry_unavailable' for e in events)
  else:assert failure is not None
check((ROOT/'artifacts').resolve().parent/'kaggle/inference-mixture-20261003b/run.py',False)
check(ROOT/'scripts/kaggle_inference_mixture.py',True)
print('PASS original main abort reproduced / corrected main real twochildren complete after telemetrytimeout; mockedhardware/CUDA not a GPUproof')
