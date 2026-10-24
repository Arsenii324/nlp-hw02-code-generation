"""Authorized bounded small-output recovery; URLs stay in memory and are never logged."""
import time
STARTED=time.monotonic()
import hashlib
import json
from pathlib import Path,PurePosixPath
import signal
import sys
from datetime import datetime,timezone

PROJECT=Path(__file__).resolve().parents[1]
DEST=PROJECT/'artifacts/kaggle/generic-fixed24-proof-v1-a'
EVIDENCE=PROJECT/'reports/evidence/generic_fixed24_proof_launch_20261023a/recovery-current-v1.json'
KERNEL='arsen4ikvar/nlp-hw02-generic-fixed24-proof-20261023a/1';OWNER,SLUG,_=KERNEL.split('/')
DRIVER='9ab7abfceee7d00f4abc16a243fa28a82a43d08bf88652e5b4ddda98ab737d9a';MANIFEST='98adf6d2895c542515ec6b27ec780a9579ae36b4e4ec2d7ec998df262d6033cc'
LIMIT_FILE=64*1024*1024;LIMIT_TOTAL=180*1024*1024;WORK=175;HARD=180
STATE={'started_utc':datetime.now(timezone.utc).isoformat(),'kernel':KERNEL,'complete':False,'files':[],'total_bytes':0,'max_file_bytes':LIMIT_FILE,'max_total_bytes':LIMIT_TOTAL,'absolute_seconds':HARD,'work_seconds':WORK,'signed_URLs_logged':False,'no_duplicates':True}
class Deadline(BaseException):pass

def alarm(signum,frame):
 if time.monotonic()-STARTED>=HARD:raise SystemExit(124)
 signal.setitimer(signal.ITIMER_REAL,max(.001,STARTED+HARD-time.monotonic()))
 raise Deadline()

def guard():
 if time.monotonic()-STARTED>=WORK:raise Deadline()

def require(ok,message):
 if not ok:raise ValueError(message)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):guard();h.update(b)
 return h.hexdigest()

def save():
 EVIDENCE.parent.mkdir(parents=True,exist_ok=True);tmp=EVIDENCE.with_suffix('.tmp');tmp.write_text(json.dumps(STATE,indent=2)+'\n');tmp.replace(EVIDENCE)

def safe(name):
 p=PurePosixPath(name)
 require(isinstance(name,str) and str(p)==name and not p.is_absolute() and '..' not in p.parts and '\\' not in name and name and not any(x.startswith('.') for x in p.parts),'Unsafe output path')
 require(not any(x in p.parts for x in ('_tmp','tmp','runtime','_crossscore_runtime')),'Temporary runtime output forbidden')
 return p

def allowed(name):
 p=safe(name)
 return p.suffix in ('.json','.jsonl','.log','.py') or name in ('notebook-proof/original_notebook.ipynb.bin','notebook-proof/executed_notebook.ipynb.bin')

def main():
 import kaggle,requests
 from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest,ApiListKernelSessionOutputRequest
 require(not DEST.exists(),'Fresh recovery directory required; duplicates forbidden');DEST.mkdir(parents=True)
 api=kaggle.api
 def identity():
  guard()
  with api.build_kaggle_client() as client:
   req=ApiGetKernelRequest();req.user_name=OWNER;req.kernel_slug=SLUG;response=client.kernels.kernels_api_client.get_kernel(req)
  meta=response.metadata;digest=hashlib.sha256(response.blob.source.encode()).hexdigest()
  require(meta.current_version_number==1 and digest==DRIVER,'Currentversion/source identity differs')
  require(meta.ref==OWNER+'/'+SLUG and meta.is_private and not meta.enable_internet and meta.enable_gpu,'Private/offline GPU metadata differs')
  return {'current_version_number':meta.current_version_number,'source_sha256':digest,'private':meta.is_private,'internet':meta.enable_internet,'ref':meta.ref}
 STATE['before']=identity();save();items={};token=None;provider_log='';pages=0
 while True:
  guard();pages+=1;require(pages<=4,'Output pagination bound exceeded')
  with api.build_kaggle_client() as client:
   req=ApiListKernelSessionOutputRequest();req.user_name=OWNER;req.kernel_slug=SLUG;req.page_size=200
   if token:req.page_token=token
   response=client.kernels.kernels_api_client.list_kernel_session_output(req)
  if response.log:provider_log=response.log
  for item in response.files or []:
   name=item.file_name;safe(name);require(name not in items,'Duplicate provider output path');items[name]=item.url
  token=response.next_page_token
  if not token:break
 STATE['provider_file_names']=sorted(items);STATE['pages']=pages;save()
 def download(name,expected=None):
  guard();require(name in items and allowed(name),'Output missing/not allowlisted');path=DEST.joinpath(*safe(name).parts);require(path.resolve().is_relative_to(DEST.resolve()),'Destination escaped root');require(not path.exists(),'Duplicate local output forbidden');path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.partial');h=hashlib.sha256();size=0
  remaining=STARTED+WORK-time.monotonic()
  with requests.get(items[name],stream=True,timeout=(min(15,remaining),min(15,remaining))) as response:
   require(response.status_code==200,'Output stream HTTP status not200')
   length=response.headers.get('Content-Length')
   if length:require(0<=int(length)<=LIMIT_FILE and STATE['total_bytes']+int(length)<=LIMIT_TOTAL,'Output Content-Length budget exceeded')
   with temporary.open('xb') as f:
    for block in response.iter_content(1024*1024):
     guard()
     if not block:continue
     size+=len(block);require(size<=LIMIT_FILE and STATE['total_bytes']+size<=LIMIT_TOTAL,'Streamed byte budget exceeded');f.write(block);h.update(block)
  digest=h.hexdigest();require(expected is None or digest==expected,'Terminal-bound output SHA differs')
  temporary.replace(path);STATE['total_bytes']+=size;STATE['files'].append({'name':name,'bytes':size,'sha256':digest});save();return path
 terminal_path=download('session_result.json');terminal=json.loads(terminal_path.read_text());STATE['terminal_sha256']=sha(terminal_path)
 require(terminal['driver_sha256']==DRIVER and terminal['manifest_sha256']==MANIFEST,'Terminal source/manifest identity differs')
 expected=terminal['artifact_sha256'];require(terminal['artifact_hashes_complete'],'Terminal artifact hashes incomplete')
 require(all(allowed(name) for name in expected),'Terminal includes forbidden large/runtime outputs')
 require(set(expected).issubset(items),'Provider missing terminal-mapped outputs')
 STATE['terminal_map_files']=len(expected);STATE['terminal_declared_complete']=terminal['complete'];save()
 for name,digest in expected.items():
  require(isinstance(digest,str) and len(digest)==64,'Invalid terminal artifact SHA');download(name,digest)
 # Provider session log is not terminal-mapped. Preserve it separately, never URLs.
 if provider_log:
  data=provider_log.encode();require(len(data)<=LIMIT_FILE and STATE['total_bytes']+len(data)<=LIMIT_TOTAL,'Provider log budget exceeded')
  name=SLUG+'.log';require(name not in expected and not (DEST/name).exists(),'Duplicate provider log');(DEST/name).write_bytes(data);STATE['files'].append({'name':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'terminal_mapped':False});STATE['total_bytes']+=len(data)
 STATE['after']=identity();require(STATE['before']==STATE['after'],'Current version/source changed during recovery')
 STATE['complete']=True;STATE['route']='Installed CLI compatible current-session API; currentv1 and exactsource before/after, every mapped output streamed against actual terminal SHA. No signed URLs persisted.'

if __name__=='__main__':
 signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(.001,STARTED+WORK-time.monotonic()))
 code=0
 try:main()
 except BaseException as error:
  STATE['failure_type']=type(error).__name__
  # SDK/requests exceptions can contain signed URLs; never stringify them.
  if isinstance(error,ValueError):STATE['controlled_failure_message']=str(error)[:256]
  code=1
 finally:
  STATE['elapsed_seconds']=time.monotonic()-STARTED;STATE['finished_utc']=datetime.now(timezone.utc).isoformat();save()
 print(json.dumps({'complete':STATE['complete'],'files':len(STATE['files']),'total_bytes':STATE['total_bytes'],'elapsed_seconds':STATE['elapsed_seconds'],'terminal_sha256':STATE.get('terminal_sha256'),'failure_type':STATE.get('failure_type'),'evidence':str(EVIDENCE)}),flush=True)
 sys.exit(code)
