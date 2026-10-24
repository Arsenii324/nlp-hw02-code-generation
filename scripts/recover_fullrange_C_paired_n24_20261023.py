"""Dormant until explicit root terminal notification; bounded terminal-first recovery."""
import time
START=time.monotonic()
import argparse,hashlib,json,shutil,signal,sys
from pathlib import Path,PurePosixPath
R=Path(__file__).resolve().parents[1];DEST=R/'artifacts/kaggle/fullrange-paired-n24-v1-c';E=R/'reports/evidence/fullrange_paired_n24_launch_20261023c/recovery-current-v1.json';WATCH=R/'reports/evidence/fullrange_paired_n24_20261023c_watch.json'
OWNER='arsen4ikvar';SLUG='nlp-hw02-fullrange-n24-paired-20261023c';KERNEL=OWNER+'/'+SLUG+'/1';DRIVER='937bc22c66ff629374d39ab5cef81b19ba50882dd0236d95bacf438c083afa74';MANIFEST='3b0e41de4ff36e98dda7a73dfcddc680166461f9be3d582d198f6a1d38585eaa';SOURCE='49768fc7374b5896d2c3caea7edf9c2a42865cc5072454acbb2cc4452a0d6eab'
FILE=32*1024**2;TOTAL=64*1024**2;WORK=285;HARD=300
S={'schema':'fullrange_C_N24_terminal_first_recovery_v1','complete':False,'kernel':KERNEL,'numeric_id':140375738,'destination':str(DEST),'destination_relative':str(DEST.relative_to(R)),'files':{},'total_bytes':0,'per_file_limit':FILE,'aggregate_limit':TOTAL,'work_seconds':WORK,'hard_seconds':HARD,'provider_mutation':False,'PT_BIN_NPY_inputs_downloaded':False,'signed_URLs_logged':False}
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
 if n in ('session_events.jsonl','full-remaining.log','half-remaining.log'):return True
 return p.parts[0] in ('run-full-remaining','run-half-remaining') and p.suffix in ('.json','.jsonl','.log','.py') and not any(x in p.parts for x in ('inputs','training_projection','training_tools','__pycache__','runtime','transport','_tmp','tmp'))
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
  require(v['numeric_id']==140375738 and v['ref']==OWNER+'/'+SLUG and v['version']==1 and v['driver_sha256']==DRIVER and v['private'] and not v['internet'] and v['gpu'],'Current GetKernel/version/source changed');return v
 S['before']=identity();save();items={};token=None
 for page in range(4):
  guard()
  with api.build_kaggle_client() as c:
   q=ApiListKernelSessionOutputRequest();q.user_name=OWNER;q.kernel_slug=SLUG;q.version_label='v1';q.page_size=200
   if token:q.page_token=token
   x=c.kernels.kernels_api_client.list_kernel_session_output(q)
  for item in x.files or []:
   safe(item.file_name);require(item.file_name not in items,'Duplicate provider output name');items[item.file_name]=item.url
  token=x.next_page_token
  if not token:break
 require(not token,'Output pagination bound');S['provider_file_count']=len(items);S['pages']=page+1;DEST.mkdir(parents=True)
 def download(n,expected=None):
  guard();require(n in items and (n=='session_result.json' or allowed(n)),'Missing/nonallowlisted output');p=DEST.joinpath(*safe(n).parts);require(p.resolve().is_relative_to(DEST.resolve()) and not p.exists(),'Fresh safe output required');p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial');h=hashlib.sha256();size=0
  with requests.get(items[n],stream=True,timeout=(10,20)) as response:
   require(response.status_code==200,'Output HTTPnot200');length=response.headers.get('Content-Length');require(length is None or 0<int(length)<=FILE and S['total_bytes']+int(length)<=TOTAL,'Content-Length cap')
   with tmp.open('xb') as stream:
    for b in response.iter_content(1048576):
     guard()
     if not b:continue
     size+=len(b);require(size<=FILE and S['total_bytes']+size<=TOTAL,'Recovery byte cap');stream.write(b);h.update(b)
  digest=h.hexdigest();require(expected is None or expected==digest,'Terminal-bound SHA mismatch');tmp.replace(p);S['files'][n]={'sha256':digest,'bytes':size,'terminal_mapped':expected is not None};S['total_bytes']+=size;save();return p
 tp=download('session_result.json');t=json.loads(tp.read_text());S['terminal_sha256']=sha(tp)
 require(t['parent_sha256']==DRIVER and t['tools_manifest_sha256']==MANIFEST and t['source_freeze_sha256']==SOURCE,'Actual terminal writer source/manifest/freeze mismatch');S['terminal_declared_complete']=t['complete'];S['artifact_hashes_declared_complete']=t['artifact_hashes_complete'];amap=t['artifact_sha256'];require(isinstance(amap,dict) and all(allowed(n) for n in amap),'Terminal contains input/binary/transport or unexpected mapped paths');require(set(amap)<=set(items),'Provider missing mapped outputs');S['terminal_map_count']=len(amap)
 for n,v in amap.items():require(isinstance(v,str) and len(v)==64,'Actual terminal digest required');download(n,v)
 S['after']=identity();require(S['before']==S['after'],'Provider identity changed during recovery');S.update(complete=True,free_after=shutil.disk_usage(R).free,terminal_children=t.get('children'),terminal_cleanup=t.get('cleanup'),route='Explicit outputversion_label=v1; current unversioned GetKernel identity/source beforeafter; exact terminal mapped text-only streams')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--terminal-notified-by-root',action='store_true');a=p.parse_args()
 # No SDK import or request before both explicit root notification and local terminal watch gate.
 signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,HARD)
 try:main(a.terminal_notified_by_root)
 except BaseException as e:
  S['failure_type']=type(e).__name__
  if isinstance(e,Controlled):S['safe_failure']=str(e)
 finally:S['elapsed_seconds']=time.monotonic()-START;save()
 print(json.dumps({'complete':S['complete'],'files':len(S['files']),'bytes':S['total_bytes'],'failure_type':S.get('failure_type')}));sys.exit(0 if S['complete'] else 1)
