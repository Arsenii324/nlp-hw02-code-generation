"""Build a root-bound, local-only Kaggle T4 proof packet for corrected v8."""
import hashlib, json, shutil, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'artifacts/kaggle/antigravity-mixture1166-proof-stage-20261024a-v8/tools'
STAGE=ROOT/'artifacts/kaggle/mixture1166-t4-proof-v9-20261024a'
TOOLS=STAGE/'tools'; KERNEL=STAGE/'kernel'
DATASET='arsen4ikvar/nlp-hw02-mixture1166-proof-20261024a-tools'
KERNEL_ID='arsen4ikvar/nlp-hw02-mixture1166-proof-20261024a'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def need(v,m):
 if not v: raise ValueError(m)
def main():
 need(not STAGE.exists(),'Fresh proof stage required; preserve all prior attempts')
 sr=json.loads((ROOT/'artifacts/kaggle/antigravity-mixture1166-proof-stage-20261024a-v8/build_receipt_v8.json').read_text())
 review=json.loads((ROOT/'reports/evidence/antigravity_mixture1166_stage_20261024a_v8_root_static_review.json').read_text())
 need(sr['complete'] and review['passed'],'v8 build/static review required')
 need(sr['manifest_sha256']==review['manifest_sha256'] and sr['bundle_sha256']==review['bundle_sha256'],'v8 review identity mismatch')
 bundle=SRC/'candidate_bundle_v8.zip'; nb=json.loads((SRC/'inference_v8.ipynb').read_text())
 code=next(c for c in nb['cells'] if c['cell_type']=='code')
 lines=code['source']; joined=''.join(lines)
 need('ROOT_BOUND=False' in joined and 'SOURCE_ONLY_UNBOUND' in joined,'Expected source-only gate absent')
 joined=joined.replace('ROOT_BOUND=False','ROOT_BOUND=True').replace('EXPECTED_PACKAGE_SHA="SOURCE_ONLY_UNBOUND"',f'EXPECTED_PACKAGE_SHA="{sr["bundle_sha256"]}"')
 code['source']=joined.splitlines(True);nb['metadata']['proof_status']='ROOT_BOUND_ONE_T4_PROOF_ONLY'
 need(sha(bundle)==sr['bundle_sha256'],'Exact v8 package changed')
 TOOLS.mkdir(parents=True);KERNEL.mkdir(parents=True)
 shutil.copyfile(bundle,TOOLS/'candidate_bundle_v8.zip')
 (TOOLS/'inference_v8.ipynb').write_text(json.dumps(nb,indent=1)+'\n')
 shutil.copyfile(ROOT/'scripts/execute_inference_notebook.py',TOOLS/'execute_inference_notebook.py')
 (TOOLS/'stage_manifest_v8.json').write_bytes((SRC/'stage_manifest_v8.json').read_bytes())
 (TOOLS/'dataset-metadata.json').write_text(json.dumps({'id':DATASET,'title':'NLP HW02 corrected mixture 1166 one T4 proof','licenses':[{'name':'CC0-1.0'}]},indent=2)+'\n')
 # Parent verifies exact dataset contents and notebook/bundle bindings, uses GPU0 only,
 # runs one fresh executor child, records the four required clocks, then reaps it.
 run=r'''#!/usr/bin/env python3
import hashlib,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
START=time.monotonic(); ROOT=Path('/kaggle/working/mixture1166-proof'); ROOT.mkdir(exist_ok=False)
DATASET='arsen4ikvar/nlp-hw02-mixture1166-proof-20261024a-tools'
EXPECTED={'candidate_bundle_v8.zip':'BUNDLE_SHA','inference_v8.ipynb':'NOTEBOOK_SHA','execute_inference_notebook.py':'EXECUTOR_SHA','stage_manifest_v8.json':'MANIFEST_SHA'}
RESULT={'schema':'mixture1166_t4_proof_parent_v1','complete':False,'phase':'startup','device_policy':'physical T4x2; child sees GPU0 only','cleanup_verified':False,'test_labels_read':False,'validation_generated':False,'canonical_promotion':False}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def save():
 p=ROOT/'session_result.json';p.write_text(json.dumps(RESULT,indent=2,allow_nan=False)+'\n')
def fail(msg): raise RuntimeError(msg)
child=None; log=None; clean={'verified':False,'reaped':False,'group_exited':False,'signals':[]}
try:
 import subprocess as sp
 gpu=sp.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True,timeout=15,check=True).stdout.splitlines()
 if len(gpu)!=2 or any('T4' not in x for x in gpu):fail('Need physical T4x2 host')
 paths=list(Path('/kaggle/input').rglob('candidate_bundle_v8.zip'))
 if len(paths)!=1:fail('Exact single private proof dataset bundle mount required')
 src=paths[0].parent
 RESULT['dataset_mount']=str(src);RESULT['physical_gpus']=gpu
 for name,template in EXPECTED.items():
  dest=ROOT/name;shutil.copyfile(src/name,dest)
  if sha(dest)!=template:fail('Input SHA mismatch: '+name)
 RESULT['input_sha256']={n:sha(ROOT/n) for n in EXPECTED};RESULT['phase']='inputs_verified';save()
 exe=ROOT/'execute_inference_notebook.py';out=ROOT/'executed_v8.ipynb';log=ROOT/'child.log';lf=log.open('wb')
 env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'}
 child=subprocess.Popen([sys.executable,'-B',str(exe),'--notebook',str(ROOT/'inference_v8.ipynb'),'--output',str(out)],cwd=ROOT,env=env,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
 RESULT.update(phase='inference_child_running',child_pid=child.pid,child_started_elapsed= time.monotonic()-START);save()
 try: rc=child.wait(timeout=720)
 except subprocess.TimeoutExpired:
  os.killpg(child.pid,signal.SIGTERM);clean['signals'].append('SIGTERM')
  try:child.wait(timeout=8)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);clean['signals'].append('SIGKILL');child.wait(timeout=8)
  rc=child.returncode
 lf.flush();os.fsync(lf.fileno());lf.close()
 try:os.killpg(child.pid,0);alive=True
 except ProcessLookupError:alive=False
 clean.update(reaped=child.poll() is not None,group_exited=not alive,verified=child.poll() is not None and not alive)
 RESULT.update(child_exit=rc,cleanup=clean,cleanup_verified=clean['verified'],executor_seconds=time.monotonic()-RESULT['child_started_elapsed']-START,child_log_tail=(log.read_text(errors='replace')[-8000:] if log.exists() else ''))
 if not clean['verified']:fail('Inference child cleanup unverified')
 ex=ROOT/'executed_v8.ipynb.json'
 if rc!=0 or not ex.exists():fail('Exact notebook executor failed')
 execution=json.loads(ex.read_text());timing=json.loads((ROOT/'candidate_output/candidate_timing.json').read_text())
 pred=ROOT/'candidate_output/predictions.jsonl';lines=pred.read_text().splitlines()
 RESULT.update(executor_report=execution,core_seconds=timing['generation_selection_serialization_seconds'],notebook_seconds=timing['notebook_start_through_hash_seconds'],output_sha256=sha(pred),expected_output_sha256=timing['expected_output_sha256'],output_rows=len(lines),output_matches=timing['output_reproduced'],candidates=timing['candidate_count'],backend=timing['backend'],timing=timing)
 RESULT['whole_child_seconds']=time.monotonic()-START
 clocks=[RESULT['core_seconds'],RESULT['notebook_seconds'],execution['elapsed_seconds'],RESULT['whole_child_seconds']]
 RESULT['clocks_seconds']={'generation_selection_serialization':clocks[0],'notebook':clocks[1],'executor':clocks[2],'whole_child':clocks[3]}
 if len(lines)!=1919 or RESULT['output_matches'] is not True or any(not isinstance(x,(int,float)) or x<=0 or x>180 for x in clocks):fail('Output parity/row count/four-clock gate failed')
 RESULT['proof_qualified']=True;RESULT['complete']=True;RESULT['phase']='finished'
except BaseException as e:
 RESULT.update(complete=False,proof_qualified=False,phase='failed',failure_type=type(e).__name__,safe_failure=str(e)[:500])
 if child is not None and child.poll() is None:
  try:os.killpg(child.pid,signal.SIGTERM);clean['signals'].append('SIGTERM');child.wait(timeout=8)
  except Exception:
   try:os.killpg(child.pid,signal.SIGKILL);clean['signals'].append('SIGKILL');child.wait(timeout=8)
   except Exception:pass
  try:os.killpg(child.pid,0);alive=True
  except ProcessLookupError:alive=False
  clean.update(reaped=child.poll() is not None,group_exited=not alive,verified=child.poll() is not None and not alive);RESULT.update(cleanup=clean,cleanup_verified=clean['verified'])
finally:
 files={}
 for p in ROOT.rglob('*'):
  if p.is_file() and p.name!='session_result.json':
   try: files[str(p.relative_to(ROOT))]={'size_bytes':p.stat().st_size,'sha256':sha(p)}
   except Exception:pass
 RESULT['artifact_sha256']=files;RESULT['artifact_total_bytes']=sum(x['size_bytes'] for x in files.values())
 RESULT['whole_child_seconds']=time.monotonic()-START
 if RESULT.get('proof_qualified') and RESULT['whole_child_seconds']>180:
  RESULT.update(complete=False,proof_qualified=False,phase='failed',safe_failure='Final inclusive whole-child clock exceeded 180s')
  if isinstance(RESULT.get('clocks_seconds'),dict):RESULT['clocks_seconds']['whole_child']=RESULT['whole_child_seconds']
 save()
 print(json.dumps({'complete':RESULT['complete'],'proof_qualified':RESULT.get('proof_qualified',False),'failure':RESULT.get('safe_failure'),'clocks':RESULT.get('clocks_seconds')},flush=True))
'''
 run=run.replace('BUNDLE_SHA',sr['bundle_sha256']).replace('NOTEBOOK_SHA',sha(TOOLS/'inference_v8.ipynb')).replace('EXECUTOR_SHA',sha(TOOLS/'execute_inference_notebook.py')).replace('MANIFEST_SHA',sr['manifest_sha256'])
 (KERNEL/'run.py').write_text(run)
 (KERNEL/'kernel-metadata.json').write_text(json.dumps({'id':KERNEL_ID,'title':'nlp-hw02-mixture1166-proof-20261024a','code_file':'run.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'machine_shape':'NvidiaTeslaT4','dataset_sources':[DATASET],'kernel_sources':[],'competition_sources':[],'model_sources':[],'docker_image':'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461','docker_image_pinning_type':'original'},indent=2)+'\n')
 files={str(p.relative_to(STAGE)):sha(p) for p in sorted(STAGE.rglob('*')) if p.is_file()}
 rec={'schema':'mixture1166_t4_proof_stage_v8_root_bound_v1','complete':True,'stage':str(STAGE.relative_to(ROOT)),'dataset':DATASET,'kernel':KERNEL_ID,'files_sha256':files,'v8_manifest_sha256':sr['manifest_sha256'],'candidate_bundle_sha256':sr['bundle_sha256'],'notebook_sha256':sha(TOOLS/'inference_v8.ipynb'),'executor_sha256':sha(TOOLS/'execute_inference_notebook.py'),'driver_sha256':sha(KERNEL/'run.py'),'kernel_metadata_sha256':sha(KERNEL/'kernel-metadata.json'),'local_model_deserialization':False,'provider_action':False,'gpu_run':False,'test_labels_read':False}
 (STAGE/'root_build_receipt.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec,indent=2))
if __name__=='__main__':main()
