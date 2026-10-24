"""Dormant until explicit root terminal notification; bounded terminal-first recovery."""
import time
START=time.monotonic()
import argparse,hashlib,json,shutil,signal,sys,logging
logging.disable(logging.CRITICAL)
from pathlib import Path,PurePosixPath
R=Path(__file__).resolve().parents[1];DEST=R/'artifacts/kaggle/half-lowlr-v1-a-diagnostics/recovery-current-v1';E=R/'reports/evidence/half_low_lr_launch_20261023a/recovery-current-v1.json';WATCH=R/'reports/evidence/half_low_lr_20261023a_watch.json'
OWNER='arsen4ikvar';SLUG='nlp-hw02-half-lowlr-20261023a';KERNEL=OWNER+'/'+SLUG+'/1';DRIVER='5535189eb744d68788485c45e3a6473c3642b992808f8850b2064614f787fb62';MANIFEST='cdfc2341ba3e708fe5186c7d2df3a4a0aa91bece9fea9f22a400223d33d3296e';SOURCE='2f83d9d2b1561b796a32c71b7cb878a10f6380d553a11c97fec6617826c1c2a6'
FILE=32*1024**2;TOTAL=64*1024**2;WORK=285;HARD=300
S={'schema':'half_low_lr_terminal_first_text_recovery_v1','complete':False,'kernel':KERNEL,'numeric_id':140432606,'destination':str(DEST),'destination_relative':str(DEST.relative_to(R)),'files':{},'total_bytes':0,'per_file_limit':FILE,'aggregate_limit':TOTAL,'work_seconds':WORK,'hard_seconds':HARD,'provider_mutation':False,'PT_BIN_NPY_inputs_downloaded':False,'signed_URLs_logged':False}
class Deadline(BaseException):pass
class Controlled(ValueError):pass
def require(v,m):
 if not v:raise Controlled(m)
def guard():
 if time.monotonic()-START>=WORK:raise Deadline()
def alarm(s,f):raise Deadline()
def save():E.parent.mkdir(parents=True,exist_ok=True);tmp=E.with_suffix('.partial');tmp.write_text(json.dumps(S,indent=2)+'\n');tmp.replace(E)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):guard();h.update(b)
 return h.hexdigest()
def safe(n):
 p=PurePosixPath(n);require(isinstance(n,str) and n and str(p)==n and not p.is_absolute() and '..' not in p.parts and '\\' not in n and not any(x.startswith('.') for x in p.parts),'Unsafe relative output path');return p
def allowed(n):
 p=safe(n)
 if n in ('session_events.jsonl','low-lr-eighth.log','low-lr-quarter.log'):return True
 if p.parts[0]=='coordination':return len(p.parts)==2 and p.name in ('common_horizon.json','low-lr-eighth-request.json','low-lr-quarter-request.json')
 return p.parts[0] in ('run-low-lr-eighth','run-low-lr-quarter') and p.suffix in ('.json','.jsonl','.log') and not any(x in p.parts for x in ('inputs','runtime','transport','__pycache__','tmp','_tmp')) and not any('.link' in x for x in p.parts)
def main(root_notified):
 require(root_notified,'Root terminal notification is required; no requests made')
 w=json.loads(WATCH.read_text());require(w['kernel']==KERNEL and w.get('status') in ('COMPLETE','ERROR','CANCELED','CANCELLED'),'Existing sole observer must record terminal status; no new polling')
 require(not DEST.exists() and not E.exists(),'Fresh recovery paths only');require(shutil.disk_usage(R).free>=2*1024**3,'Require2GiB free');S['free_before']=shutil.disk_usage(R).free;S['root_terminal_notification_acknowledged']=True;S['local_watch']={'path':str(WATCH.relative_to(R)),'sha256':sha(WATCH),'status':w['status']}
 import kaggle,requests
 from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest,ApiListKernelSessionOutputRequest
 api=kaggle.api
 def identity():
  guard()
  with api.build_kaggle_client() as c:
   q=ApiGetKernelRequest();q.user_name=OWNER;q.kernel_slug=SLUG;x=c.kernels.kernels_api_client.get_kernel(q)
  m=x.metadata;v={'numeric_id':m.id,'ref':m.ref,'version':m.current_version_number,'driver_sha256':hashlib.sha256(x.blob.source.encode()).hexdigest(),'private':m.is_private,'internet':m.enable_internet,'gpu':m.enable_gpu}
  require(v['numeric_id']==140432606 and v['ref']==OWNER+'/'+SLUG and v['version']==1 and v['driver_sha256']==DRIVER and v['private'] and not v['internet'] and v['gpu'],'Current GetKernel/version/source changed');return v
 S['before']=identity();save();items={};token=None
 for page in range(4):
  guard()
  with api.build_kaggle_client() as c:
   q=ApiListKernelSessionOutputRequest();q.user_name=OWNER;q.kernel_slug=SLUG;q.version_label='v1';q.page_size=200
   if token:q.page_token=token
   x=c.kernels.kernels_api_client.list_kernel_session_output(q)
  for item in x.files or []:
   try:safe(item.file_name)
   except Controlled:S['unsafe_provider_names_skipped']=S.get('unsafe_provider_names_skipped',0)+1;continue
   require(item.file_name not in items,'Duplicate provider output name');items[item.file_name]=item.url
  token=x.next_page_token
  if not token:break
 require(not token,'Output pagination bound');S['provider_file_count']=len(items);S['pages']=page+1;DEST.mkdir(parents=True,exist_ok=False)
 def download(n,expected=None):
  guard();require(n in items and (n=='session_result.json' or allowed(n)),'Missing/nonallowlisted output');p=DEST.joinpath(*safe(n).parts);require(p.resolve().is_relative_to(DEST.resolve()) and not p.exists() and not p.is_symlink() and not any(v.is_symlink() for v in p.parents),'Fresh safe nonsymlink output required');p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial');h=hashlib.sha256();size=0
  with requests.get(items[n],stream=True,timeout=(10,20)) as response:
   require(response.status_code==200,'Output HTTPnot200');length=response.headers.get('Content-Length');require(length is None or 0<=int(length)<=FILE and S['total_bytes']+int(length)<=TOTAL,'Content-Length cap')
   with tmp.open('xb') as stream:
    for b in response.iter_content(1048576):
     guard()
     if not b:continue
     size+=len(b);require(size<=FILE and S['total_bytes']+size<=TOTAL,'Recovery byte cap');stream.write(b);h.update(b)
  digest=h.hexdigest();require(expected is None or expected==digest,'Terminal-bound SHA mismatch');tmp.replace(p);S['files'][n]={'sha256':digest,'bytes':size,'terminal_mapped':expected is not None};S['total_bytes']+=size;save();return p
 tp=download('session_result.json');t=json.loads(tp.read_text());S['terminal_sha256']=sha(tp)
 require(t['schema']=='half_low_lr_parent_result_a_v1' and t['driver_sha256']==DRIVER and t['manifest_sha256']==MANIFEST and t['source_sha256']==SOURCE,'Actual terminal writer source/manifest mismatch')
 S['terminal_declared_complete']=t['complete'];S['artifact_hashes_declared_complete']=t['artifact_hashes_complete'];amap=t['artifact_sha256'];require(isinstance(amap,dict),'Actual terminal map required')
 chosen={};skipped={}
 for n,v in amap.items():
  try:safe(n)
  except Controlled:S['unsafe_terminal_names_skipped']=S.get('unsafe_terminal_names_skipped',0)+1;continue
  require(isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v),'Actual terminal digest required')
  if allowed(n):chosen[n]=v
  else:skipped[n]=v
 require(set(chosen)<=set(items),'Provider missing selected terminal-mapped text outputs');S.update(terminal_map_count=len(amap),selected_map_count=len(chosen),excluded_map_count=len(skipped),excluded_declared_sha256=skipped,remote_symlink_type_exposed_by_API=False,local_symlinks_rejected=True)
 for n,v in chosen.items():download(n,v)
 S['after']=identity();require(S['before']==S['after'],'Provider identity changed during recovery');S.update(complete=True,free_after=shutil.disk_usage(R).free,terminal_children=t.get('runs'),terminal_cleanup=t.get('cleanup'),route='Explicit outputversion_label=v1; current unversioned GetKernel identity/source beforeafter; exact selected terminal-mapped text-only streams; binary/runtime declarations retained but never downloaded')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--terminal-notified-by-root',action='store_true');a=p.parse_args()
 if not a.terminal_notified_by_root:print(json.dumps({'status':'DORMANT_ROOT_TERMINAL_FLAG_REQUIRED','provider_requests':False}));sys.exit(0)
 # No SDK import or request before both explicit root notification and local terminal watch gate.
 signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,HARD)
 try:main(a.terminal_notified_by_root)
 except BaseException as e:
  S['failure_type']=type(e).__name__
  if isinstance(e,Controlled):S['safe_failure']=str(e)
 finally:S['elapsed_seconds']=time.monotonic()-START;save()
 print(json.dumps({'complete':S['complete'],'files':len(S['files']),'bytes':S['total_bytes'],'failure_type':S.get('failure_type')}));sys.exit(0 if S['complete'] else 1)
