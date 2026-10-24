"""Owned VM only; signed URLs remain in RAM, history disabled, no credentials sent."""
import subprocess,json,hashlib,pathlib,sys,time
ROOT=pathlib.Path('/Users/a2mogus/build-projs/dl2-hse/nlp-hw02-code-generation')
SESSION='nlp-hw02-continuation-20261023h'
GEN=r'''
import kaggle,json,hashlib,pathlib
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest,ApiListKernelSessionOutputRequest
R=pathlib.Path('/Users/a2mogus/build-projs/dl2-hse/nlp-hw02-code-generation');out={'files':[],'identities':[]}
cache=json.loads((R/'artifacts/kaggle/cache-6m-v1/cache/cache_manifest.json').read_text());term=json.loads((R/'artifacts/kaggle/gated-b034-final-v1/session_result.json').read_text())
for slug,id,driver,prefix,hashes,names in [('nlp-hw02-cache-6m-20261004c',136971567,'c7bd00baa1822cb9fb6230fef310f77499f8c07314d18f3a98d3c8382ac6b054','cache',cache['checksums'],['train_source.npy','train_target.npy','train_row_ids.npy','data_stats.json','tokenizer.json','validation.json','validation_legacy.json','cache_manifest.json']),('nlp-hw02-gated-b034-20261022b',140259193,'2dd4ebd12cdea6ea670cd7967fc092f7bf3ad2f82a89606d7c9574d22db8caf3','run-control',term['artifact_sha256'],['last_training.pt','model.pt','metrics.json','run_config.json','tokenizer.json','data_stats.json','validation.json','validation_legacy.json'])]:
 with kaggle.api.build_kaggle_client() as c:
  q=ApiGetKernelRequest();q.user_name='arsen4ikvar';q.kernel_slug=slug;x=c.kernels.kernels_api_client.get_kernel(q)
  ident=dict(id=x.metadata.id,ref=x.metadata.ref,version=x.metadata.current_version_number,driver_sha256=hashlib.sha256(x.blob.source.encode()).hexdigest());assert ident['id']==id and ident['version']==1 and ident['driver_sha256']==driver;out['identities'].append(ident)
  q=ApiListKernelSessionOutputRequest();q.user_name='arsen4ikvar';q.kernel_slug=slug;q.version_label='v1';q.page_size=200;x=c.kernels.kernels_api_client.list_kernel_session_output(q);items={a.file_name:a.url for a in x.files};assert not x.next_page_token
  for name in names:
   remote=prefix+'/'+name;dest=('cache/' if prefix=='cache' else 'parent/')+name
   h=hashes.get(remote,hashes.get(name))
   if name=='cache_manifest.json':h=hashlib.sha256((R/'artifacts/kaggle/cache-6m-v1/cache/cache_manifest.json').read_bytes()).hexdigest()
   assert h and remote in items
   out['files'].append(dict(path=dest,sha256=h,url=items[remote]))
print('PRIVATE_PACKET='+json.dumps(out))
'''
WORKER=r'''
import sys,json,time,pathlib,hashlib,os,requests,concurrent.futures,shutil
P=pathlib.Path('/content/continuation-h');s=time.monotonic();packet=json.load(sys.stdin);deadline=s+2400;r={'schema':'colab_h_direct_input_transport_v1','complete':False,'identities':packet['identities'],'files':{},'signed_urls_persisted':False,'credentials_uploaded':False,'history_stored':False,'free_before':shutil.disk_usage(P).free}
def save():
 t=P/'transport-result.json.partial';t.write_text(json.dumps(r,indent=2));os.replace(t,P/'transport-result.json')
def one(item):
 p=P/item['path'];p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial');h=hashlib.sha256();n=0;start=time.monotonic()
 try:
  with requests.get(item['url'],stream=True,timeout=(15,45)) as x:
   x.raise_for_status()
   with tmp.open('wb') as f:
    for b in x.iter_content(4*1024**2):
     if time.monotonic()>deadline:raise TimeoutError()
     if b:h.update(b);n+=len(b);f.write(b)
  if h.hexdigest()!=item['sha256']:raise ValueError('SHA mismatch')
  os.replace(tmp,p);return item['path'],dict(complete=True,bytes=n,sha256=h.hexdigest(),seconds=time.monotonic()-start)
 except BaseException as e:return item['path'],dict(complete=False,error_type=type(e).__name__,bytes=n,seconds=time.monotonic()-start)
try:
 assert r['free_before']>12*1024**3
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for f in concurrent.futures.as_completed([pool.submit(one,x) for x in packet['files']]):
   n,v=f.result();r['files'][n]=v;save();print(json.dumps({'file':n,**v}),flush=True)
 r['complete']=all(v['complete'] for v in r['files'].values()) and len(r['files'])==len(packet['files'])
except BaseException as e:r['error_type']=type(e).__name__
r.update(seconds=time.monotonic()-s,free_after=shutil.disk_usage(P).free);save()
'''
try:
 p=subprocess.run(['/Users/a2mogus/anaconda3/bin/python','-c',GEN],capture_output=True,text=True,check=True,timeout=90)
 line=next(x for x in p.stdout.splitlines() if x.startswith('PRIVATE_PACKET='));packet=json.loads(line.split('=',1)[1]);del p,line
 from colab_cli.common import state
 from colab_cli.runtime import ColabRuntime
 st=state.get_session(SESSION);runtime=ColabRuntime(st.url,st.token,kernel_id=st.kernel_id,session_id=st.session_id)
 code="import subprocess,json,pathlib\n_p=pathlib.Path('/content/continuation-h');_p.mkdir(exist_ok=True)\n_f=(_p/'transport.log').open('w')\n_x=subprocess.Popen([__import__('sys').executable,'-u','-c',"+repr(WORKER)+"],stdin=subprocess.PIPE,stdout=_f,stderr=subprocess.STDOUT,start_new_session=True)\n_x.stdin.write("+repr(json.dumps(packet).encode())+");_x.stdin.close()\nprint(json.dumps({'transport_started':True,'pid':_x.pid,'files':"+str(len(packet['files']))+"}))\ndel _x,_f\n"
 del packet
 reply=runtime.kernel_client.execute(code,store_history=False,timeout=30);del code
 outputs=reply.get('outputs',[])
 for o in outputs:
  if o.get('output_type')=='stream':print(o.get('text','')[:800])
 print(json.dumps({'request_complete':True,'history_disabled':True}))
except BaseException as e:
 print(json.dumps({'request_complete':False,'error_type':type(e).__name__}));sys.exit(1)
