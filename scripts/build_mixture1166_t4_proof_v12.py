"""Build a root-bound, local-only Kaggle T4 proof packet for corrected v8."""
import hashlib, json, shutil, zipfile, io
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'artifacts/kaggle/antigravity-mixture1166-proof-stage-20261024a-v8/tools'
STAGE=ROOT/'artifacts/kaggle/mixture1166-t4-proof-v12-20261024a'
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
 source_bundle=SRC/'candidate_bundle_v8.zip'; nb=json.loads((SRC/'inference_v8.ipynb').read_text())
 need(sha(source_bundle)==sr['bundle_sha256'],'Exact reviewed v8 source bundle required')
 source_manifest=json.loads((SRC/'stage_manifest_v8.json').read_text())
 need(source_manifest['status']=='LOCAL_STAGE_AWAITING_INDEPENDENT_REVIEW_AND_ROOT_BINDING' and source_manifest['verified_actual_values'] is False,'Expected inert source manifest')
 source_manifest['status']='ROOT_BOUND_READY';source_manifest['verified_actual_values']=True
 source_manifest['root_binding']={'schema':'mixture1166_t4_proof_root_binding_v1','stage_static_review_sha256':sha(ROOT/'reports/evidence/antigravity_mixture1166_stage_20261024a_v8_root_static_review.json'),'source_bundle_sha256':sr['bundle_sha256'],'selection_basis':'corrected exact-lineage 1166/1529 validation grid; test predictions are not used for tuning'}
 rootbundle=io.BytesIO()
 with zipfile.ZipFile(source_bundle,'r') as zin,zipfile.ZipFile(rootbundle,'w',compression=zipfile.ZIP_STORED) as zout:
  need('stage_manifest_v8.json' in zin.namelist(),'Source manifest absent from package')
  need(set(zin.namelist())==set(source_manifest['payload_sha256'])|{'stage_manifest_v8.json'},'Exact source bundle member set required')
  for info in zin.infolist():
   content=json.dumps(source_manifest,indent=2,sort_keys=True).encode()+b'\n' if info.filename=='stage_manifest_v8.json' else zin.read(info.filename)
   zout.writestr(info,content)
 bundle_sha=hashlib.sha256(rootbundle.getvalue()).hexdigest()
 nb=json.loads((SRC/'inference_v8.ipynb').read_text())
 code=next(c for c in nb['cells'] if c['cell_type']=='code')
 lines=code['source']; joined=''.join(lines)
 need('ROOT_BOUND=False' in joined and 'SOURCE_ONLY_UNBOUND' in joined,'Expected source-only gate absent')
 joined=joined.replace('ROOT_BOUND=False','ROOT_BOUND=True').replace('EXPECTED_PACKAGE_SHA="SOURCE_ONLY_UNBOUND"',f'EXPECTED_PACKAGE_SHA="{bundle_sha}"')
 joined=joined.replace('MANIFEST=json.loads(z.read("stage_manifest_v8.json"))\n for info in z.infolist():', 'MANIFEST=json.loads(z.read("stage_manifest_v8.json"))\n if set(names)!=set(MANIFEST["payload_sha256"])|{"stage_manifest_v8.json"}: raise RuntimeError("Exact root-bound package member set required")\n for info in z.infolist():')
 code['source']=joined.splitlines(True);nb['metadata']['proof_status']='ROOT_BOUND_ONE_T4_PROOF_ONLY'
 TOOLS.mkdir(parents=True);KERNEL.mkdir(parents=True)
 (TOOLS/'candidate_bundle_v8.zip').write_bytes(rootbundle.getvalue())
 (TOOLS/'inference_v8.ipynb').write_text(json.dumps(nb,indent=1)+'\n')
 shutil.copyfile(ROOT/'scripts/execute_inference_notebook.py',TOOLS/'execute_inference_notebook.py')
 (TOOLS/'stage_manifest_v8.json').write_text(json.dumps(source_manifest,indent=2,sort_keys=True)+'\n')
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
 if execution.get('complete') is not True or execution.get('executed_code_cells')!=3 or execution.get('input_sha256')!=EXPECTED['inference_v8.ipynb'] or execution.get('output_sha256')!=sha(out):fail('Executed notebook hash/cell receipt mismatch')
 if sha(pred)!=timing.get('expected_output_sha256') or timing.get('output_reproduced') is not True:fail('Exact expected output hash mismatch')
 for line in lines:
  row=json.loads(line)
  if set(row)!={'code'} or not isinstance(row['code'],str):fail('Output schema is not code-only JSONL')
  compile(row['code'],'<generated-answer>','exec')
 RESULT.update(executor_report=execution,core_seconds=timing['generation_selection_serialization_seconds'],notebook_seconds=timing['notebook_start_through_hash_seconds'],output_sha256=sha(pred),expected_output_sha256=timing['expected_output_sha256'],output_rows=len(lines),output_matches=timing['output_reproduced'],candidates=timing['candidate_count'],backend=timing['backend'],timing=timing)
 if timing.get('candidate_count')!=1919*11 or timing.get('backend',{}).get('visible_gpu_count')!=1 or 'T4' not in timing.get('backend',{}).get('gpu',''):fail('Candidate count or one-visible-T4 backend mismatch')
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
 total=sum(x['size_bytes'] for x in files.values())
 RESULT['artifact_sha256']=files;RESULT['artifact_total_bytes']=total
 if total>256*1024**2 or any(x['size_bytes']>128*1024**2 for x in files.values()):
  RESULT.update(complete=False,proof_qualified=False,phase='failed',safe_failure='Terminal artifact cap exceeded; no output qualification')
 RESULT['whole_child_seconds']=time.monotonic()-START
 if RESULT.get('proof_qualified') and RESULT['whole_child_seconds']>180:
  RESULT.update(complete=False,proof_qualified=False,phase='failed',safe_failure='Final inclusive whole-child clock exceeded 180s')
  if isinstance(RESULT.get('clocks_seconds'),dict):RESULT['clocks_seconds']['whole_child']=RESULT['whole_child_seconds']
 save()
 print(json.dumps({'complete':RESULT['complete'],'proof_qualified':RESULT.get('proof_qualified',False),'failure':RESULT.get('safe_failure'),'clocks':RESULT.get('clocks_seconds')},flush=True))
'''
 manifest_sha=sha(TOOLS/'stage_manifest_v8.json')
 need(sha(TOOLS/'candidate_bundle_v8.zip')==bundle_sha,'Root-bound candidate bundle SHA mismatch')
 run=run.replace('BUNDLE_SHA',bundle_sha).replace('NOTEBOOK_SHA',sha(TOOLS/'inference_v8.ipynb')).replace('EXECUTOR_SHA',sha(TOOLS/'execute_inference_notebook.py')).replace('MANIFEST_SHA',manifest_sha)
 (KERNEL/'run.py').write_text(run)
 (KERNEL/'kernel-metadata.json').write_text(json.dumps({'id':KERNEL_ID,'title':'nlp-hw02-mixture1166-proof-20261024a','code_file':'run.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'machine_shape':'NvidiaTeslaT4','dataset_sources':[DATASET],'kernel_sources':[],'competition_sources':[],'model_sources':[],'docker_image':'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461','docker_image_pinning_type':'original'},indent=2)+'\n')
 files={str(p.relative_to(STAGE)):sha(p) for p in sorted(STAGE.rglob('*')) if p.is_file()}
 rec={'schema':'mixture1166_t4_proof_stage_v12_root_bound_v1','complete':True,'stage':str(STAGE.relative_to(ROOT)),'dataset':DATASET,'kernel':KERNEL_ID,'files_sha256':files,'source_manifest_sha256':sr['manifest_sha256'],'manifest_sha256':manifest_sha,'source_bundle_sha256':sr['bundle_sha256'],'candidate_bundle_sha256':bundle_sha,'notebook_sha256':sha(TOOLS/'inference_v8.ipynb'),'executor_sha256':sha(TOOLS/'execute_inference_notebook.py'),'driver_sha256':sha(KERNEL/'run.py'),'kernel_metadata_sha256':sha(KERNEL/'kernel-metadata.json'),'local_model_deserialization':False,'provider_action':False,'gpu_run':False,'test_labels_read':False}
 (STAGE/'root_build_receipt.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec,indent=2))
if __name__=='__main__':main()
