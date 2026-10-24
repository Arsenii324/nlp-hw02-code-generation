"""Explicit actual terminal diagnostic recovery only; no provider mutation/models."""
import logging
logging.disable(logging.CRITICAL)
import datetime,hashlib,json,os,pathlib,shutil,signal,time
R=pathlib.Path(__file__).resolve().parents[1];O=R/'artifacts/kaggle/tpu-pilot-policy3b-refresh3-diagnostics';E=R/'reports/evidence/tpu_compile_failure_20261023/recovery.json';START=time.monotonic();END=START+180;TOTAL=16*1024**2;FILE=4*1024**2
S={'schema':'tpu_compile_failure_diagnostics_recovery_v1','complete':False,'read_only':True,'kernel':'arsen4ikvar/nlp-hw02-tpu-pilot-20261022c','version':1,'output_root':str(O),'output_root_relative':str(O.relative_to(R)),'files':{},'total_bytes':0,'max_file_bytes':FILE,'max_total_bytes':TOTAL,'absolute_seconds':180,'signed_URLs_persisted':False,'no_weights_arrays_fullstates':True}
def need(x,m):
 if not x:raise ValueError(m)
def guard():
 if time.monotonic()>=END:raise TimeoutError()
def save():
 E.parent.mkdir(parents=True,exist_ok=True);S['elapsed_seconds']=time.monotonic()-START;E.write_text(json.dumps(S,indent=2)+'\n')
def alarm(*_):raise TimeoutError()
def main():
 need(not O.exists() and not E.exists(),'Fresh recovery required');need(shutil.disk_usage(R).free>1024**3,'Free1GiB required')
 import kaggle,requests
 from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest,ApiListKernelSessionOutputRequest
 def identity():
  guard()
  with kaggle.api.build_kaggle_client() as c:
   q=ApiGetKernelRequest();q.user_name='arsen4ikvar';q.kernel_slug='nlp-hw02-tpu-pilot-20261022c';x=c.kernels.kernels_api_client.get_kernel(q)
  m=x.metadata;v={'id':m.id,'ref':m.ref,'version':m.current_version_number,'source_sha256':hashlib.sha256(x.blob.source.encode()).hexdigest(),'private':m.is_private,'TPU':m.enable_tpu,'GPU':m.enable_gpu,'internet':m.enable_internet}
  need(v['id']==140257929 and v['version']==1 and v['ref']==S['kernel'] and v['source_sha256']=='78066e47b9c0c7f1535bdfdee4835f929480d5e1fbf05d01eca82291fcea114d' and v['private'] and v['TPU'] and not v['GPU'] and not v['internet'],'Current identity differs');return v
 S['before']=identity();save();items={};token=None
 for _ in range(4):
  guard()
  with kaggle.api.build_kaggle_client() as c:
   q=ApiListKernelSessionOutputRequest();q.user_name='arsen4ikvar';q.kernel_slug='nlp-hw02-tpu-pilot-20261022c';q.version_label='v1';q.page_size=200
   if token:q.page_token=token
   x=c.kernels.kernels_api_client.list_kernel_session_output(q)
  for f in x.files or []:
   if f.file_name.startswith('tpu-pilot/'):
    need(f.file_name not in items,'Duplicate output name');items[f.file_name]=f.url
  token=x.next_page_token
  if not token:break
 need(not token,'Bounded listing incomplete');O.mkdir();S['listed_names']=sorted(items)
 def fetch(name,expected=None):
  guard();rel=pathlib.PurePosixPath(name);need(not rel.is_absolute() and '..' not in rel.parts and name in items,'Unsafe/missing diagnostic name');p=O/pathlib.Path(name);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial');h=hashlib.sha256();n=0
  with requests.get(items[name],stream=True,timeout=(15,45)) as response:
   response.raise_for_status()
   with tmp.open('xb') as out:
    for b in response.iter_content(1024*1024):
     guard()
     if b:n+=len(b);need(n<=FILE and S['total_bytes']+n<=TOTAL,'Diagnostic bytes exceed bound');out.write(b);h.update(b)
  digest=h.hexdigest();need(expected is None or digest==expected,'Terminal diagnostic SHA mismatch');os.replace(tmp,p);S['files'][name]={'sha256':digest,'bytes':n};S['total_bytes']+=n;save();return p
 terminal=fetch('tpu-pilot/terminal.json');t=json.loads(terminal.read_text());need(t['complete'] is False and t['cleanup']['reaped'] and t['cleanup']['group_exited'],'Actual failure/cleanup differs');S['terminal_sha256']=S['files']['tpu-pilot/terminal.json']['sha256']
 checks=t['checksums'];S['terminal_checksums']=checks
 for rel,h in checks.items():
  p=pathlib.PurePosixPath(rel);need(not p.is_absolute() and '..' not in p.parts,'Unsafe terminal path')
  if p.suffix in ('.json','.jsonl','.log','.txt') and not any(s in p.parts for s in ('first','latest','runtime','source')):fetch('tpu-pilot/'+rel,h)
 S['excluded_terminal_files']=[n for n in checks if 'tpu-pilot/'+n not in S['files']];S['after']=identity();need(S['after']==S['before'],'Current identity changed');S['complete']=True
signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,180)
try:main()
except BaseException as e:S['failure_type']=type(e).__name__
finally:save();print(json.dumps({'complete':S['complete'],'files':len(S['files']),'bytes':S['total_bytes'],'receipt':str(E.relative_to(R)),'failure_type':S.get('failure_type')}))
