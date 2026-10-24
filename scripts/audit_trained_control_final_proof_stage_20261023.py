"""Frozen stage B actual-byte/source inspection only; no workers/model/tests/cloud."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'.worktrees/strong-inference-fallback-20261022'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def node(t,name):return next(n for n in t.body if getattr(n,'name',None)==name)
def dump(n):return ast.dump(n,include_attributes=False)

def main():
 build=json.loads((ROOT/'reports/evidence/trained_control_final_proof_b_build_20261023.json').read_text());tools=Path(build['tools']);kernel=Path(build['kernel']);manifest=tools/'trained_control_final_proof_manifest.json';m=json.loads(manifest.read_text());checks={}
 def check(key,ok):
  checks[key]=bool(ok)
  if not ok:raise ValueError(key)
 check('manifest_driver_worker_builder_actualpins',sha(manifest)==build['manifest_sha256'] and sha(kernel/'run.py')==build['driver_sha256'] and sha(tools/'trained_control_final_validation_parity.py')==build['worker_sha256'] and sha(W/'scripts/build_kaggle_trained_control_final_proof_b.py')==build['builder_sha256'])
 check('all5_actualinputs_streamhash',len(m['inputs_sha256'])==5 and all(sha(tools/name)==value for name,value in m['inputs_sha256'].items()))
 check('minimal_goldfree_tool_allowlist',{p.name for p in tools.iterdir()}==set(m['inputs_sha256'])|{'trained_control_final_proof_manifest.json','dataset-metadata.json'})
 old=ast.parse((W/'scripts/kaggle_generic_fixed24_proof.py').read_text());new=ast.parse((kernel/'run.py').read_text());worker=(tools/'trained_control_final_validation_parity.py').read_text();parent=(kernel/'run.py').read_text();ast.parse(worker);ast.parse((W/'scripts/build_kaggle_trained_control_final_proof_b.py').read_text());check('allsource_AST_parse',True)
 for name in ('AbsoluteDeadline','arm_absolute','absolute_alarm','sha','emit','start','stop','terminate','finalize','before_hash','archive_proof'):check('exact_reviewed_generic_AST:'+name,dump(node(old,name))==dump(node(new,name)))
 check('exact_reviewed_main_guard',dump(old.body[-1])==dump(new.body[-1]))
 a=ast.parse((W/'scripts/kaggle_trained_control_final_proof.py').read_text());b=ast.parse((W/'scripts/kaggle_trained_control_final_proof_b.py').read_text())
 def assignment(tree,name):return next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id==name for x in n.targets))
 oldfiles=ast.literal_eval(assignment(a,'PROOF_FILES').value);newfiles=ast.literal_eval(assignment(b,'PROOF_FILES').value);check('B_archives_unrenamed_partial_prediction',set(newfiles)==set(oldfiles)|{'test_n24_predictions.jsonl'} and len(newfiles)==len(oldfiles)+1)
 assignment(b,'PROOF_FILES').value=assignment(a,'PROOF_FILES').value;check('whole_parent_B_only_archive_filename_delta',dump(a)==dump(b))
 initial=json.loads((ROOT/'reports/evidence/trained_control_final_proof_build_20261023.json').read_text());at=Path(initial['tools']);ak=Path(initial['kernel']);check('stageA_preserved_allinputs_manifest_driver',sha(at/'trained_control_final_proof_manifest.json')==initial['manifest_sha256'] and sha(ak/'run.py')==initial['driver_sha256'] and all(sha(at/name)==value for name,value in initial['inputs_sha256'].items()) and initial['inputs_sha256']==m['inputs_sha256'])
 check('all4_runtime_180gates_required',"passed=core<=180 and strongest<=180 and child_wall<=180 and execution['elapsed_seconds']<=180" in parent and "'complete':passed" in parent)
 check('trained_E_bare_samples_pool_schema',"require(row['samples']==24" in parent and "unsupported+=row['supported_candidates']==0" in parent and "'warmup_n24'" in parent and "'test_n24'" in parent and "'validation_n24'" in parent and 'actual24_rows' not in parent)
 check('validationworker_only_actual_trained_Cmodule','from nlp_hw02.trained_selected_sampling_c import sample_fixed,GENERATION,validate_deployment,verify_topology' in worker and "timing=sample_fixed(model,tok,questions,24,out,'validation_n24'" in worker and 'nlp_hw02.selected_sampling' not in worker and 'torch.multinomial' not in worker)
 check('strict_actual344k_source_checkpoint_topology','checkpoint_step\']==344000' in worker and "verify_topology(model,'control')==manifest['deployment']['actual_parity']['topology']" in worker and "model_source_sha256']=='07e58176c39eed8ed03134074282d74589be97120e6d6569a15f27eee1538b26'" in worker)
 check('no_validation_warmup_and_onecall',worker.count('timing=sample_fixed(')==1 and "'warmup_performed':False" in worker)
 check('absolute_bounds_and_lane_childlimits',m['bounds']=={'platform_seconds':900,'parent_seconds':840,'work_cutoff_seconds':740,'whole_worker_deadline_seconds':735,'aggregate_cleanup_end_seconds':785,'artifact_hash_end_seconds':835,'terminal_hard_exit_seconds':840,'startup_min_useful_seconds':220} and m['lanes']['notebook-proof']['child_seconds']==180 and m['lanes']['validation-parity']['child_seconds']==300)
 check('peer_summary_and_tempCWD_archival',"STATE['lanes'][lane['name']]=result" in parent and "os.chdir(cwd)" in parent and "finally:os.chdir(previous_cwd)" in parent and "tempfile.mkdtemp(prefix='trained-control-final-proof-')" in parent)
 nb=json.loads((tools/'notebook.bin').read_text());cells=[c for c in nb['cells'] if c['cell_type']=='code'];check('exact_unexecuted3cells_newnotebook',len(cells)==3 and all(c['execution_count'] is None and c['outputs']==[] for c in cells) and sha(tools/'notebook.bin')=='2b1f2d877cb89d861f8a360a67fa108aa3dce332b2c593064f1940fd3307f977')
 for c in cells:ast.parse(''.join(c['source']))
 with zipfile.ZipFile(tools/'selected_bundle.bin') as z:
  manifest_bytes=z.read('candidate_manifest.json');candidate=json.loads(manifest_bytes);check('actual_package_manifest_deployment_pinned',sha(tools/'selected_bundle.bin')=='7c9c7885912e867cbf2bd8eba9ee583cc6424265a2f8f3032299c2a1411c6083' and hashlib.sha256(manifest_bytes).hexdigest()==m['candidate_manifest_sha256'] and candidate['deployment']==m['deployment'] and candidate['member_sha256']==m['runtime_member_sha256'])
  check('exact11_safe_unique_members',len(z.namelist())==len(set(z.namelist()))==11 and set(z.namelist())==set(candidate['member_sha256'])|{'candidate_manifest.json'})
  for name,digest in candidate['member_sha256'].items():
   h=hashlib.sha256()
   with z.open(name) as stream:
    for block in iter(lambda:stream.read(1048576),b''):h.update(block)
   check('payload:'+name,h.hexdigest()==digest)
 d=m['deployment'];check('actual_root_receipt_role_model_frozen',d['root_quality_decision_sha256']==m['root_quality_selection_receipt_sha256']=='0b33aa20cef35e2fe484ba1c93f54e9375b72cf13528bb83e439244adfd90ba5' and d['role_id']=='control-final' and d['mode']=='fixed24' and d['samples_per_question']==24 and d['checkpoint_step']==344000 and d['model_sha256']=='eed1fdbb4f7e056536b322c08fad8660f5df24460f716a4d8e5382eb58ce8dc9')
 q=json.loads((tools/'validation_questions.json').read_text());source=ROOT/'artifacts/kaggle/trained-gated-sampling-v1-b/control-final/validation_n24_predictions.jsonl';pred=[json.loads(s)['code'] for s in source.read_text().splitlines()];r=m['validation_reproduction'];check('actual1792_projection_hashes',len(q)==len(pred)==1792 and r['code_only_jsonl_sha256']==sha(source)=='1de89c6c0437b3ff3a653f8605386848aadf5a55ecf503dfac99aadebd65a427' and r['ordered_codes_sha256']==hashlib.sha256(json.dumps(pred,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()=='7769e3414a209c912e83c98ac77aa9eea2e0236083fd1c79d3d256159c1a0b61')
 metadata=json.loads((kernel/'kernel-metadata.json').read_text());dataset=json.loads((tools/'dataset-metadata.json').read_text());check('private_offline_exact_owner_slug_image',metadata['id']=='arsen4ikvar/nlp-hw02-trained-control-final-proof-20261023b' and metadata['title']==metadata['id'].split('/')[1] and metadata['is_private'] and not metadata['enable_internet'] and dataset['isPrivate'] and metadata['dataset_sources']==[dataset['id']] and metadata['kernel_sources']==[] and metadata['docker_image']=='gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461')
 check('current_B034_canonical_and_backup_unchanged',sha(ROOT/'artifacts/selected_bundle.zip')==sha(ROOT/'artifacts/selected-b034-fixed24-preserved-20261023/selected_bundle.zip')=='aba6112e3d3d3c9d80c742a71d146d88bccf65946ba7ef6edded6dce2bd581fb' and sha(ROOT/'submit/inference.ipynb')==sha(ROOT/'artifacts/selected-b034-fixed24-preserved-20261023/inference.ipynb')=='c940570f7b24903ceade666e4f56b0441a8cdfd0e538480a56bf251daa322270')
 report={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS local bound proofstage B static readback; independent whole-stage review and actual newbyte proof pending','checks':checks,'all_checks_pass':all(checks.values()),'stage':build,'scope':'Actual-byte/AST/static metadata only; no model/tests/worker/notebook/cloud execution','script_sha256':sha(__file__),'actual_fresh_proof':None,'canonical_changed':False};p=ROOT/'reports/evidence/trained_control_final_proof_b_static_20261023.json';p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checks':len(checks),'all_checks_pass':True,'evidence':str(p)}))

if __name__=='__main__':main()
