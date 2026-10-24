"""Bounded two-selected-model recovery only; no optimizer/arrays/provider mutation."""
import time
START=time.monotonic()
from pathlib import Path
import hashlib,json,shutil,signal,sys
R=Path(__file__).resolve().parents[1];D=R/'artifacts/kaggle/fullrange-cooldown-v1-c-diagnostics-retry1';O=R/'artifacts/kaggle/fullrange-cooldown-v1-c-selected-models';E=R/'reports/evidence/fullrange_bucket_cooldown_launch_20261023c/selected_model_recovery.json'
DRIVER='fb7b52c0ed3d823b2710db8dcef918cf807ef169770ed7f56b3be4183de83f08';TERMINAL='18647c3cb0bb0fe263cdb8b3e741573c0b2d758babda00d2807999f3bf830afc';MANIFEST='a596fb5228854491c1781ca668c87d9b95cd2c235a35b753cec6f606eb81529f';OWNER='arsen4ikvar';SLUG='nlp-hw02-fullrange-cooldown-20261023c'
S={'schema':'fullrange_C_selected_models_recovery_v1','complete':False,'diagnostic_root':str(D),'diagnostic_root_relative':str(D.relative_to(R)),'output_root':str(O),'output_root_relative':str(O.relative_to(R)),'kernel_ref':OWNER+'/'+SLUG,'version':1,'files':{},'total_bytes':0,'remote_mutation':False,'fullstate_or_arrays':False,'signed_URLs_logged':False}
class Bound(BaseException):pass
class Failure(ValueError):pass
def need(x,m):
 if not x:raise Failure(m)
def guard():
 if time.monotonic()-START>=300:raise Bound()
def alarm(s,f):raise Bound()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):guard();h.update(b)
 return h.hexdigest()
def save():E.parent.mkdir(parents=True,exist_ok=True);E.write_text(json.dumps(S,indent=2)+'\n')
def main():
 need(not O.exists() and not E.exists(),'Fresh recovery paths required');S['free_before']=shutil.disk_usage(R).free;need(S['free_before']>=2*1024**3,'Require2GiB free before recovery');need(sha(D/'session_result.json')==TERMINAL,'Actual terminal changed');t=json.loads((D/'session_result.json').read_text());need(t['complete'] and t['paired_complete'] and t['artifact_hashes_complete'] and t['driver_sha256']==DRIVER and t['manifest_sha256']==MANIFEST,'Actual terminal incomplete/identity mismatch')
 allowed={}
 for role in ('full-remaining','half-remaining'):
  pp=D/('run-'+role+'/latest_checkpoint.json');need(sha(pp)==t['artifact_sha256']['run-'+role+'/latest_checkpoint.json'],'Pointer digest differs');p=json.loads(pp.read_text());need(p['slot']=='durable-slot0' and p['absolute_step']==373480,'Actual slot/frontier differs');name='run-'+role+'/'+p['slot']+'/model.pt';need(t['artifact_sha256'][name]==p['files_sha256']['model.pt'],'Actual model role differs');allowed[name]=p['files_sha256']['model.pt']
 import kaggle,requests
 from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest,ApiListKernelSessionOutputRequest
 api=kaggle.api
 def identity():
  guard()
  with api.build_kaggle_client() as c:
   q=ApiGetKernelRequest();q.user_name=OWNER;q.kernel_slug=SLUG;x=c.kernels.kernels_api_client.get_kernel(q)
  m=x.metadata;v={'numeric_id':m.id,'ref':m.ref,'version':m.current_version_number,'driver_sha256':hashlib.sha256(x.blob.source.encode()).hexdigest(),'private':m.is_private,'internet':m.enable_internet,'gpu':m.enable_gpu};need(v['numeric_id']==140361812 and v['ref']==OWNER+'/'+SLUG and v['version']==1 and v['driver_sha256']==DRIVER and v['private'] and not v['internet'] and v['gpu'],'Current GetKernel identity changed');return v
 S['before']=identity();save();items={};token=None
 for page in range(4):
  guard()
  with api.build_kaggle_client() as c:
   q=ApiListKernelSessionOutputRequest();q.user_name=OWNER;q.kernel_slug=SLUG;q.version_label='v1';q.page_size=200
   if token:q.page_token=token
   x=c.kernels.kernels_api_client.list_kernel_session_output(q)
  for v in x.files or []:
   if v.file_name in allowed:need(v.file_name not in items,'Duplicate selected provider path');items[v.file_name]=v.url
  token=x.next_page_token
  if not token:break
 need(not token and set(items)==set(allowed),'Boundedv1 output listing lacks exact2roles');O.mkdir(parents=True)
 for n,v in allowed.items():
  guard();p=O/n;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.pt.partial');h=hashlib.sha256();size=0
  with requests.get(items[n],stream=True,timeout=(10,20)) as response:
   need(response.status_code==200,'Selected stream HTTPnot200');length=response.headers.get('Content-Length');need(length is None or 0<int(length)<=80*1024**2,'Content-Length bound')
   with tmp.open('xb') as f:
    for b in response.iter_content(1048576):
     guard()
     if not b:continue
     size+=len(b);need(size<=80*1024**2 and S['total_bytes']+size<=180*1024**2,'Selected byte bounds');f.write(b);h.update(b)
  need(h.hexdigest()==v,'Selected terminalSHA mismatch');tmp.replace(p);S['files'][n]={'sha256':v,'bytes':size,'path':str(p.relative_to(R))};S['total_bytes']+=size;save()
 S['after']=identity();need(S['before']==S['after'],'Provider identity changed during transfer');S.update(complete=True,terminal_sha256=TERMINAL,training_source_sha256=t['source_sha256'],manifest_sha256=MANIFEST,free_after=shutil.disk_usage(R).free)
if __name__=='__main__':
 signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,310)
 try:main()
 except BaseException as e:
  S['failure_type']=type(e).__name__
  if isinstance(e,Failure):S['safe_failure']=str(e)
 finally:S['seconds']=time.monotonic()-START;save()
 print(json.dumps({'complete':S['complete'],'files':len(S['files']),'total_bytes':S['total_bytes'],'failure_type':S.get('failure_type')}));sys.exit(0 if S['complete'] else 1)
