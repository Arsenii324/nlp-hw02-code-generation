"""Freeze two cold one-T4 notebook arms plus held-out diagnostics, reusing exact private v1 weights."""
import ast,base64,hashlib,json,zipfile,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];S=R/'artifacts/kaggle/final-inference-probe-20261024b';S.mkdir(exist_ok=True);K=S/'kernel';K.mkdir(exist_ok=True)
def sha(b):return hashlib.sha256(b).hexdigest()
def enc(x):return base64.b64encode(zlib.compress(json.dumps(x,separators=(',',':')).encode(),9)).decode()
with zipfile.ZipFile(R/'artifacts/kaggle/mixture1166-t4-proof-v13-20261024a/tools/candidate_bundle_v8.zip') as z:
 old=json.loads(z.read('stage_manifest_v8.json'));sources={n:z.read(n).decode() for n in z.namelist() if n.endswith('.py')}
sources['src/nlp_hw02/inference.py']=(R/'src/nlp_hw02/inference.py').read_text()
runtime=sources['src/nlp_hw02/mixture1166_runtime.py']
runtime=runtime.replace("len(models)==2 and [m['samples_per_question'] for m in manifest['models']]==[8,3]", "len(models)==len(manifest['models']) and [m['samples_per_question'] for m in manifest['models']] in ([8],[8,3],[12],[8,8])")
runtime=runtime.replace("len(questions)==1919", "len(questions)>0").replace("len(rows)!=1919", "len(rows)!=len(questions)")
for field,name in [('candidates','codes'),('scores','scores'),('capped','flags')]:
 oldline=f"{name}=pools[0][i]['{field}'][:8]+pools[1][i]['{field}'][:3]"
 newline=f"{name}=[v for pool,spec in zip(pools,manifest['models']) for v in pool[i]['{field}'][:spec.get('vote_samples',spec['samples_per_question'])]]"
 assert oldline in runtime;runtime=runtime.replace(oldline,newline)
runtime=runtime.replace("len(codes)!=11", "len(codes)!=sum(s.get('vote_samples',s['samples_per_question']) for s in manifest['models'])").replace("'rows':1919", "'rows':len(questions)").replace("'samples':11", "'samples':len(codes)").replace("'candidates_per_question':11", "'candidates_per_question':sum(s.get('vote_samples',s['samples_per_question']) for s in manifest['models'])")
runtime=runtime.replace("atomic_json(dest/(label+'_progress.json'),", "\n                if done%256==0: print(json.dumps({'event':'progress','label':label,'rows':done,'total':len(qs),'seconds':time.perf_counter()-start}),flush=True)\n                atomic_json(dest/(label+'_progress.json'),")
sources['src/nlp_hw02/mixture1166_runtime.py']=runtime
for n,b in sources.items():ast.parse(b,filename=n)
validation=json.loads((R/'artifacts/paid-gelu768-validation-20261024a/final400k/inputs/validation_questions.json').read_text());assert len(validation)==1792
cell1='''import os,sys,time,json,hashlib,zlib,base64,ast
from pathlib import Path
NOTEBOOK_START=time.perf_counter();ROOT=Path.cwd()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def unpack(s):return json.loads(zlib.decompress(base64.b64decode(s)))
SOURCES=unpack(SOURCE_B64);MANIFEST=unpack(MANIFEST_B64)
paths=list(Path('/kaggle/input').rglob('candidate_bundle_v8/stage_manifest_v8.json'))
if len(paths)!=1 or sha(paths[0])!='462773c8f5e62a5c72f4dd56963d339870b82f6975bec5dc58a4c23d652290b7':raise RuntimeError('Exact private v1 package mount required')
INPUT=paths[0].parent
WORK=ROOT/'source';WORK.mkdir(exist_ok=False)
for name,content in SOURCES.items():
 p=WORK/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
sys.path.insert(0,str(WORK/'src'))
QUESTIONS=unpack(VALIDATION_B64) if SPLIT=='validation' else json.loads((INPUT/'test_questions.json').read_text())
if SPLIT=='test' and (len(QUESTIONS)!=1919 or sha(INPUT/'test_questions.json')!='6dd86866b2ed86a09781b7f471fc2fce5c00975503c4a85f1ec959dae0b93279'):raise RuntimeError('Test projection mismatch')
MANIFEST['ordered_test_questions_sha256']=hashlib.sha256(json.dumps(QUESTIONS,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
'''
cell2='''import torch
from tokenizers import Tokenizer
from nlp_hw02.model import load_model
if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'T4' not in torch.cuda.get_device_name(0):raise RuntimeError('Exactly one visible T4 required')
torch.set_num_threads(2);LOAD_START=time.perf_counter();MODELS=[]
for spec in MANIFEST['models']:
 p=INPUT/spec['file']
 if sha(p)!=spec['sha256']:raise RuntimeError('Weight hash mismatch')
 model,checkpoint=load_model(p)
 if any(checkpoint.get(k)!=spec[k] for k in ('config','step','validation_loss')) or checkpoint.get('seed')!=42 or checkpoint.get('dataset_revision')!='b2724689ff079e9c22efd548fe134bfb29036dbb':raise RuntimeError('Checkpoint metadata mismatch')
 MODELS.append(model);del checkpoint
if sha(INPUT/'tokenizer.json')!=MANIFEST['tokenizer_sha256']:raise RuntimeError('Tokenizer hash mismatch')
TOKENIZER=Tokenizer.from_file(str(INPUT/'tokenizer.json'));torch.cuda.synchronize();LOAD_SECONDS=time.perf_counter()-LOAD_START
'''
cell3='''from nlp_hw02.mixture1166_runtime import run
OUT=ROOT/'candidate_output';RESULT=run(MODELS,TOKENIZER,QUESTIONS,OUT,MANIFEST)
RESULT['notebook_seconds']=time.perf_counter()-NOTEBOOK_START;RESULT['model_load_seconds']=LOAD_SECONDS
RESULT['syntax_errors']=[]
for i,line in enumerate((OUT/'predictions.jsonl').read_text().splitlines()):
 try:ast.parse(json.loads(line)['code'])
 except (SyntaxError,ValueError):RESULT['syntax_errors'].append(i)
RESULT['split']=SPLIT;RESULT['prediction_sha256']=sha(OUT/'predictions.jsonl');RESULT['test_labels_read']=False
(OUT/'timing.json').write_text(json.dumps(RESULT,indent=2)+'\\n')
print(json.dumps({'event':'complete','rows':RESULT['rows'],'seconds':RESULT['notebook_seconds'],'output_reproduced':RESULT['output_reproduced'],'syntax_errors':len(RESULT['syntax_errors'])}),flush=True)
'''
notebooks={}
for arm,counts,ref in [('standalone',[12],None),('mixture',[8,8],'99e3d45158752387d277f2613c5f0d963f68d957813d59c3fd06c075687ac125')]:
 m=dict(old);m['models']=[dict(s,samples_per_question=n) for s,n in zip(old['models'],counts)];m['expected_output_sha256']=ref
 if arm=='mixture':m['models'][1]['vote_samples']=3
 for split in ('test','validation'):
  pre='SPLIT='+repr(split)+'\nSOURCE_B64='+repr(enc(sources))+'\nMANIFEST_B64='+repr(enc(m))+'\nVALIDATION_B64='+repr(enc(validation) if split=='validation' else enc([]))+'\n'
  cells=[pre+cell1,cell2,cell3]
  for c in cells:ast.parse(c)
  nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':c.splitlines(True),'id':f'cell-{i}'} for i,c in enumerate(cells)]}
  notebooks[arm+'_'+split]=nb;(S/(arm+'_'+split+'.ipynb')).write_text(json.dumps(nb,indent=1)+'\n')
executor=(R/'scripts/execute_inference_notebook.py').read_text()
parent='''import os,json,time,subprocess,signal,hashlib
from pathlib import Path
START=time.monotonic();ROOT=Path('/kaggle/working/final_probe');ROOT.mkdir(exist_ok=False)
NOTEBOOKS=NOTEBOOKS_LITERAL
EXECUTOR=EXECUTOR_LITERAL
REC={'complete':False,'arms':{},'test_labels_read':False,'physical_device_policy':'T4x2, one visible T4 per child'}
def save(): (ROOT/'parent_result.json').write_text(json.dumps(REC,indent=2)+'\\n')
children=[]
try:
 names=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True,timeout=15,check=True).stdout.splitlines()
 if len(names)!=2 or any('T4' not in x for x in names):raise RuntimeError('Need physical T4x2')
 for split in ('test','validation'):
  children=[]
  for gpu,arm in enumerate(('standalone','mixture')):
   key=arm+'_'+split;d=ROOT/key;d.mkdir();nb=d/'inference.ipynb';nb.write_text(json.dumps(NOTEBOOKS[key],indent=1)+'\\n');(d/'executor.py').write_text(EXECUTOR)
   env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'}
   log=(d/'child.log').open('wb');started=time.monotonic()
   p=subprocess.Popen(['python','-u','executor.py','--notebook','inference.ipynb','--output','executed.ipynb'],cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   children.append((key,p,log,started,d));REC['arms'][key]={'pid':p.pid,'status':'running'};save()
  ended={};last_progress=0
  while any(key not in ended for key,p,_,_,_ in children):
   if time.monotonic()-START>700:raise TimeoutError('Whole session work deadline')
   for key,p,log,started,d in children:
    if p.poll() is not None and key not in ended:ended[key]=time.monotonic()-started
   if time.monotonic()-last_progress>30:
    progress={}
    for key,p,log,started,d in children:
     progress[key]={'alive':p.poll() is None,'elapsed':time.monotonic()-started}
     for f in (d/'candidate_output').glob('*/*_progress.json'):
      try:progress[key][f.parent.name]=json.loads(f.read_text()).get('rows')
      except Exception:pass
    print(json.dumps({'event':'progress','arms':progress}),flush=True);last_progress=time.monotonic()
   time.sleep(2)
  for key,p,log,started,d in children:
   log.close();elapsed=ended[key]
   rec={'exit':p.returncode,'parent_inclusive_seconds':elapsed,'status':'terminal'}
   t=d/'candidate_output/timing.json';e=d/'executed.json'
   if t.exists():rec['timing']=json.loads(t.read_text())
   if e.exists():rec['executor']=json.loads(e.read_text())
   rec['runtime_qualified']=bool(key.endswith('_test') and p.returncode==0 and elapsed<=180 and rec.get('timing',{}).get('notebook_seconds',999)<=180 and rec.get('executor',{}).get('elapsed_seconds',999)<=180 and rec.get('timing',{}).get('rows')==1919)
   REC['arms'][key]=rec;save();print(json.dumps({'event':'arm_terminal','arm':key,'exit':p.returncode,'seconds':elapsed,'qualified':rec['runtime_qualified']}),flush=True)
 REC['complete']=all(a.get('exit')==0 for a in REC['arms'].values());REC['elapsed_seconds']=time.monotonic()-START
except BaseException as exc:REC['failure_type']=type(exc).__name__;REC['failure']=str(exc)[:500]
finally:
 for key,p,log,_,_ in children:
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
  if not log.closed:log.close()
 REC['children_reaped']=all(p.poll() is not None for _,p,_,_,_ in children);save()
 print(json.dumps({'event':'parent_terminal','complete':REC['complete'],'elapsed_seconds':time.monotonic()-START}),flush=True)
'''
parent=parent.replace('NOTEBOOKS_LITERAL',repr(notebooks)).replace('EXECUTOR_LITERAL',repr(executor));ast.parse(parent);(K/'run.py').write_text(parent)
meta={'id':'arsen4ikvar/nlp-hw02-final-inference-probe-20261024b','title':'nlp-hw02-final-inference-probe-20261024b','code_file':'run.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'machine_shape':'NvidiaTeslaT4','dataset_sources':['arsen4ikvar/nlp-hw02-mixture1166-proof-20261024a-tools'],'kernel_sources':[],'competition_sources':[],'model_sources':[],'docker_image':'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461','docker_image_pinning_type':'original'}
(K/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
rec={'schema':'final_inference_probe_20261024a','hypothesis':'Fullpop768 N12 tests sample scaling; mixture generatesN8 on both lineages and uses first8+3 votes to test exact prefix method and cold runtime','files':{str(p.relative_to(S)):sha(p.read_bytes()) for p in S.rglob('*') if p.is_file() and p.name!='manifest.json'},'dataset_version':1,'dataset_private_readback_receipt':'reports/evidence/mixture1166_t4_proof_v13/private_readback_v2.json','validation_questions':len(validation),'test_questions':1919,'compute_seconds':900,'final_gpu_reserve_seconds':7200,'test_labels_read':False,'provider_action':False}
(S/'manifest.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps({'stage':str(S),'driver_bytes':(K/'run.py').stat().st_size,'driver_sha256':sha(parent.encode())}))
