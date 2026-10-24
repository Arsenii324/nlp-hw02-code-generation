"""Finite two-T4 inference comparison; no training, installs or submission."""
import hashlib,json,os
from pathlib import Path
import shutil,subprocess,sys,time,zipfile


def sample_gpu_progress(emit,run=None):
 if run is None:run=subprocess.run
 try:
  sample=run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
 except (subprocess.TimeoutExpired,OSError) as exc:
  emit('telemetry_unavailable',reason=repr(exc));return None
 if sample.returncode:
  emit('telemetry_unavailable',reason=sample.stderr.strip(),exit=sample.returncode);return None
 return sample.stdout.strip()


def main():
 started=time.monotonic();working=Path('/kaggle/working');temp=Path('/kaggle/temp/inference-mixture');temp.mkdir(parents=True,exist_ok=False)
 children=[];terminal={'complete':False,'phase':'startup'}
 def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
 def emit(event,**fields):
  row={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'session_seconds':time.monotonic()-started,'event':event,**fields}
  with (working/'session_events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print(json.dumps(row),flush=True)
 try:
  manifests=list(Path('/kaggle/input').rglob('inference_probe_manifest.json'));assert len(manifests)==1
  tools=manifests[0].parent;manifest=json.loads(manifests[0].read_text());assert sha(__file__)==manifest['driver_sha256']
  for name,digest in manifest['tools_sha256'].items():assert Path(name).name==name and sha(tools/name)==digest
  for name in ['inference_probe_manifest.json','inference_source.bin']:shutil.copyfile(tools/name,working/name)
  shutil.copyfile(__file__,working/'run_driver.py')
  source=temp/'source'
  with zipfile.ZipFile(tools/'inference_source.bin') as z:
   assert z.testzip() is None and set(z.namelist())==set(manifest['source_member_sha256'])
   for name in z.namelist():
    assert not Path(name).is_absolute() and '..' not in Path(name).parts
    assert hashlib.sha256(z.read(name)).hexdigest()==manifest['source_member_sha256'][name]
   z.extractall(source)
  sys.path.insert(0,str(source/'scripts'));from inference_probe_protocol import verified_inference_parents,finish_children
  parents=verified_inference_parents('/kaggle/input',manifest)
  assert len({sha(path/'tokenizer.json') for path in parents.values()})==1 and len({sha(path/'validation.json') for path in parents.values()})==1
  hardware=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=15).stdout.splitlines();assert len(hardware)==2 and all('T4' in x for x in hardware)
  emit('startup_complete',hardware=hardware,parent_runs={k:str(v) for k,v in parents.items()});assert time.monotonic()-started<90
  terminal['phase']='inference'
  for name,spec in manifest['lanes'].items():
   log=(working/f'{name}.log').open('w')
   process=subprocess.Popen([sys.executable,'-u',str(source/'scripts/inference_mixture_probe.py'),'--manifest',str(manifests[0]),'--parents',json.dumps({k:str(v) for k,v in parents.items()}),'--tools',str(tools),'--output',str(working/name),'--lane',name],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'PYTHONPATH':str(source/'src'),'CUDA_VISIBLE_DEVICES':str(spec['device']),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'})
   children.append((name,process,log));emit('lane_started',lane=name,device=spec['device'],pid=process.pid)
  announced=set();last=0
  while any(p.poll() is None for _,p,_ in children):
   if time.monotonic()-started>=540:raise TimeoutError('Parent540s deadline; no platform-timeout race')
   for name,p,_ in children:
    if p.poll() is not None and name not in announced:
     emit('lane_exit',lane=name,exit=p.returncode);announced.add(name)
     # Preserve the useful peer; a failed lane does not cancel it.
   if time.monotonic()-started-last>=30:
    sample=sample_gpu_progress(emit)
    emit('progress',lanes={name:p.poll() for name,p,_ in children},gpu_sample=sample);last=time.monotonic()-started
   time.sleep(1)
  results={}
  for name,p,_ in children:
   result=working/name/'probe_result.json'
   results[name]={'exit':p.returncode,'report':json.loads(result.read_text()) if result.exists() else None}
  terminal.update(complete=all(x['exit']==0 and x['report'] and x['report']['complete'] for x in results.values()),phase='finished',lanes=results,manifest_sha256=sha(manifests[0]))
  emit('probes_finished',complete=terminal['complete'],lanes={k:v['exit'] for k,v in results.items()})
 except BaseException as exc:terminal.update(failure_type=type(exc).__name__,failure=str(exc));raise
 finally:
  if children:finish_children([p for _,p,_ in children])
  for _,_,log in children:log.close()
  terminal['session_script_seconds']=time.monotonic()-started
  terminal['artifact_sha256']={str(p.relative_to(working)):sha(p) for p in working.rglob('*') if p.is_file() and not p.is_symlink() and p.name!='session_result.json'}
  (working/'session_result.json').write_text(json.dumps(terminal,indent=2)+'\n');print(json.dumps({'event':'session_terminal','complete':terminal['complete'],'seconds':terminal['session_script_seconds']}),flush=True)
 if not terminal['complete']:raise RuntimeError('Incomplete probe; inspect preserved results')


if __name__=='__main__':main()
