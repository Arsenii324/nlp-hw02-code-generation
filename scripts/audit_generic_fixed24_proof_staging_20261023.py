"""Static actual-byte stage audit only; never imports worker/model or executes notebook."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1];W=ROOT/'.worktrees/strong-inference-fallback-20261022'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def ordered(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def node(t,name):return next(n for n in t.body if getattr(n,'name',None)==name)
def dump(n):return ast.dump(n,include_attributes=False)

def main():
 build=json.loads((ROOT/'reports/evidence/generic_fixed24_proof_build_20261023.json').read_text());tools=Path(build['tools']);kernel=Path(build['kernel']);m=json.loads((tools/'generic_fixed24_proof_manifest.json').read_text());checks={}
 def check(key,ok):
  checks[key]=bool(ok)
  if not ok:raise ValueError(key)
 check('actual_manifest_driver_build_pins',sha(tools/'generic_fixed24_proof_manifest.json')==build['manifest_sha256'] and sha(kernel/'run.py')==m['driver_sha256']==build['driver_sha256'])
 check('all5_inputs_actual_hashes',len(m['inputs_sha256'])==5 and all(sha(tools/n)==v for n,v in m['inputs_sha256'].items()))
 check('minimal_goldfree_tool_allowlist',set(p.name for p in tools.iterdir())==set(m['inputs_sha256'])|{'generic_fixed24_proof_manifest.json','dataset-metadata.json'})
 for p in (kernel/'run.py',tools/'generic_fixed24_validation_parity.py',tools/'execute_inference_notebook.py',W/'scripts/build_kaggle_generic_fixed24_proof.py'):ast.parse(p.read_text())
 check('allsource_AST_parse',True)
 old=ast.parse((W/'scripts/kaggle_k32_budget_probe.py').read_text());new=ast.parse((kernel/'run.py').read_text())
 names=('AbsoluteDeadline','arm_absolute','absolute_alarm','sha','emit','start','stop','terminate')
 check('all8_reviewed_lifecycle_helpers_exact_AST',all(dump(node(old,n))==dump(node(new,n)) for n in names))
 check('whole_main_guard_exact_AST',dump(old.body[-1])==dump(new.body[-1]))
 final=node(new,'finalize');arch=[n for n in final.body if isinstance(n,ast.Try) and any(isinstance(c,ast.Call) and isinstance(c.func,ast.Name) and c.func.id=='before_hash' for c in ast.walk(n))];check('exactly_one_bounded_postcleanup_archive_hook',len(arch)==1)
 final.body.remove(arch[0]);check('remaining_whole_finalizer_exact_AST',dump(final)==dump(node(old,'finalize')))
 check('absolute_budget_manifest',m['bounds']=={'platform_seconds':900,'parent_seconds':840,'work_cutoff_seconds':740,'whole_worker_deadline_seconds':735,'aggregate_cleanup_end_seconds':785,'artifact_hash_end_seconds':835,'terminal_hard_exit_seconds':840,'startup_min_useful_seconds':220})
 check('proof180_parity300_children',m['lanes']['notebook-proof']['child_seconds']==180 and m['lanes']['validation-parity']['child_seconds']==300)
 source=(kernel/'run.py').read_text();val=(tools/'generic_fixed24_validation_parity.py').read_text()
 check('core_AND_strongest_AND_wholechild_AND_executor180',"passed=core<=180 and strongest<=180 and child_wall<=180 and execution['elapsed_seconds']<=180" in source and "'complete':passed" in source)
 check('proof_summary_independent_and_incremental',"STATE['lanes'][lane['name']]=result" in source and "STATE['proof_complete']=result['complete']" in source and "complete=STATE['proof_complete'] and STATE['validation_parity_complete']" in source)
 check('postcleanup_archive_no_models_in_artifactroot',"tempfile.mkdtemp(prefix='generic-fixed24-proof-')" in source and "shutil.rmtree(PROOF)" in source and "if STATE.get('cleanup_incomplete'):raise RuntimeError" in source)
 check('parity_imports_exact_module_not_new_sampler',"from nlp_hw02.selected_sampling import fixed24_samples,GENERATION,validate_deployment" in val and "timing=fixed24_samples(model,tok,questions,out,'validation1792',args.deadline,lambda:STOP)" in val and 'torch.multinomial' not in val)
 calls=[n for n in ast.walk(ast.parse(val)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='fixed24_samples'];check('parity_only_one1792_call_no_warmup',len(calls)==1 and "'warmup_performed':False" in val)
 notebook=json.loads((tools/'notebook.bin').read_text());cells=[c for c in notebook['cells'] if c['cell_type']=='code'];check('unchanged_fresh3_plain_cells',len(cells)==3 and all(c['execution_count'] is None and c['outputs']==[] for c in cells))
 for c in cells:ast.parse(''.join(c['source']))
 check('exact_proven_executor_pin',sha(tools/'execute_inference_notebook.py')=='195029b10921308202714f357bbf0a25b7731d455b526931d724a8a4616d5fce')
 with zipfile.ZipFile(tools/'selected_bundle.bin') as z:
  c=json.loads(z.read('candidate_manifest.json'));check('actual_fixed24_package_manifest',c['deployment']==m['deployment'] and c['deployment']['mode']=='fixed24' and hashlib.sha256(z.read('candidate_manifest.json')).hexdigest()==m['candidate_manifest_sha256'])
  check('all12_unique_package_allowlisted',len(z.namelist())==12 and len(set(z.namelist()))==12 and set(z.namelist())==set(m['runtime_member_sha256'])|{'candidate_manifest.json'})
  for name,value in m['runtime_member_sha256'].items():
   h=hashlib.sha256()
   with z.open(name) as f:
    for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
   check('member:'+name,h.hexdigest()==value)
 q=json.loads((tools/'validation_questions.json').read_text());pred=ROOT/'artifacts/kaggle/sample-scaling-v1/samples24/sample-plurality24_predictions.jsonl';rows=[json.loads(s) for s in pred.read_text().splitlines()];codes=[r['prediction'] for r in rows];expected=m['validation_reproduction']
 check('actual1792_question_projection_order',len(q)==len(rows)==1792 and all(hashlib.sha256(question.encode()).hexdigest()==row['question_sha256'] for question,row in zip(q,rows)) and ordered(q)==expected['ordered_questions_sha256'])
 check('actual_oldval_hash_only_expected_projection',sha(pred)==expected['source_selected_artifact_sha256'] and ordered(codes)==expected['ordered_codes_sha256'] and hashlib.sha256(''.join(json.dumps({'code':code},ensure_ascii=False)+'\n' for code in codes).encode()).hexdigest()==expected['code_only_jsonl_sha256'])
 metadata=json.loads((kernel/'kernel-metadata.json').read_text());dataset=json.loads((tools/'dataset-metadata.json').read_text())
 check('private_offline_exact_owner_title_slug_image',metadata['id']=='arsen4ikvar/nlp-hw02-generic-fixed24-proof-20261023a' and metadata['title']==metadata['id'].split('/')[1] and metadata['is_private'] and not metadata['enable_internet'] and metadata['dataset_sources']==[dataset['id']] and not metadata['kernel_sources'] and metadata['docker_image']=='gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461')
 check('canonical_B031_and_backup_unchanged',sha(ROOT/'artifacts/selected_bundle.zip')==sha(ROOT/'artifacts/selected-b031-preserved-20261022/selected_bundle.zip')=='e5e9b6650e530c1015a03dcee6fa30ae0e9ed215ed01b0c1bff9ff7c8b80c323' and sha(ROOT/'submit/inference.ipynb')==sha(ROOT/'artifacts/selected-b031-preserved-20261022/inference.ipynb')=='1b79bae3331774f8cc0fe5c9ca935a85b9fb7f2f6633b81dde9ed33109a2b6a3')
 report={'status':'PASS static actual-byte author audit; independent review and fresh actual execution pending','checks':checks,'all_checks_pass':all(checks.values()),'stage':build,'scope':'AST/static actual-byte hashing and projection only; no worker/model/notebook/tests/simulation/cloud execution','script_sha256':sha(__file__),'actual_proof':None,'canonical_promotion':False}
 out=ROOT/'reports/evidence/generic_fixed24_proof_static_20261023.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checks':len(checks),'all_checks_pass':True,'evidence':str(out)}))

if __name__=='__main__':main()
