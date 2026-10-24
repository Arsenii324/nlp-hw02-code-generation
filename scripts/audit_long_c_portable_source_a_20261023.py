"""Static source/hash audit only: no import/execution of newly implemented modules."""
import ast
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'.worktrees/strong-inference-fallback-20261022'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def fn(t,n):return next(v for v in t.body if isinstance(v,ast.FunctionDef) and v.name==n)
def norm(v):return ast.dump(v,include_attributes=False)
def main():
 files=['scripts/long_c_sampling_contract_a.py','scripts/probe_long_c_selected_sampling_a.py','scripts/package_trained_window_selected_a.py','src/nlp_hw02/trained_window_selected_sampling_a.py','scripts/build_trained_window_selected_notebook_a.py']
 files+=['configs/'+n for n in ['long_c_sampling_producer_a.template.json','long_c_selected_deployment_a.template.json','long_c_root_integrity_a.template.json','long_c_root_selection_a.template.json','long_c_runtime_a.template.json','long_c_notebook_a.template.json']]
 checks={}
 def check(k,v):
  checks[k]=bool(v)
  if not v:raise ValueError(k)
 trees={n:ast.parse((W/n).read_text()) for n in files if n.endswith('.py')};check('all5_new_python_sources_parse',len(trees)==5)
 e=ast.parse((W/'scripts/probe_trained_gated_sampling_e.py').read_text());c=ast.parse((W/'src/nlp_hw02/trained_selected_sampling_c.py').read_text());p=trees['scripts/probe_long_c_selected_sampling_a.py'];r=trees['src/nlp_hw02/trained_window_selected_sampling_a.py'];h=trees['scripts/long_c_sampling_contract_a.py'];b=trees['scripts/package_trained_window_selected_a.py']
 for name in ('actual_prefix_parity','verify_topology','generate_candidates','sample_fixed'):check('producer_exact_frozen_whole_AST:'+name,norm(fn(e,name))==norm(fn(p,name)))
 for name in ('verify_topology','generate_candidates','sample_fixed','verify_parity_evidence'):check('runtime_exact_frozen_whole_AST:'+name,norm(fn(c,name))==norm(fn(r,name)))
 for tree,label in [(p,'producer'),(r,'runtime'),(h,'shared_validator'),(b,'packager')]:
  check(label+':no_top_level_torch_import',not any(isinstance(n,ast.Import) and any(v.name.startswith('torch') for v in n.names) or isinstance(n,ast.ImportFrom) and (n.module or '').startswith('torch') for n in tree.body))
 check('shared_validator_first_statement_unbound_guard',isinstance(fn(h,'verify_training').body[0],ast.Expr) and 'verified_actual_values' in ast.unparse(fn(h,'verify_training').body[0]))
 check('runtime_first_statement_unbound_guard','verified_actual_values' in ast.unparse(fn(r,'validate_deployment').body[0]))
 check('packager_training_validation_precedes_runtime_import',ast.unparse(fn(b,'bind_reproduction').body[0]).endswith('verify_training(spec)'))
 prod=ast.unparse(fn(p,'main'));check('producer_verifies_before_torch_import',prod.index('verify_training(spec)')<prod.index('import torch'))
 check('actual_parity_before_any_sampling',prod.index("actual_prefix_parity(model, tok, pairs, 'control', out)")<prod.index('sample_fixed(model, tok, ordered'))
 check('producer_receipt_preserves_failure_and_whole_time',all(x in prod for x in ['finally:',"report['whole_worker_main_seconds']","report['complete'] = True",'failure_type']))
 hs=(W/'scripts/long_c_sampling_contract_a.py').read_text();check('inherited344_reuse_before_input_load',hs.index("if selected['model_sha256']==CURRENT_MODEL_SHA")<hs.index('inputs={}'))
 check('no_fullstate_local_read',"if name!='last_training.pt':files[name]=artifact" in hs)
 check('source_uses_window_identity_not_E_budget',all(x in hs for x in ['345000+n','since_resume_iterations','admitted_NEW_horizon','at_consumed_window_boundary','root_cpu_integrity_sha256']) and 'E_MANIFEST_SHA' not in hs)
 for name in files:
  if name.endswith('.json'):
   v=json.loads((W/name).read_text());check(name+':unbound',v.get('verified_actual_values') is False or v.get('complete') is False or v.get('selected_after_actual_quality_audit') is False)
   for k in ('samples_per_question','model_sha256','checkpoint_step','package_sha256'):
    if k in v:check(name+':NULL_'+k,v[k] is None)
 nb=(W/'scripts/build_trained_window_selected_notebook_a.py').read_text();check('notebook_source_unbound_and_exact_API',all(x in nb for x in ['digest=stage[\'package_sha256\'] if bound else None','BOUND_CANDIDATE = {bound!r}','long_c_selected_bundle_a_v1','from nlp_hw02.trained_window_selected_sampling_a import run_long_c_selected_candidate','notebook_start_through_output_and_initial_sidecar','strongest <= 180']))
 protected={'scripts/probe_trained_gated_sampling_e.py':'c2074502f2eeb9305011e0b7bb78654bc0ce4f0538c159acce0716f0817f42d0','src/nlp_hw02/trained_selected_sampling_c.py':'aa362c163c2ed1fa9de29dd4299c5874438c68ff245d8dc0c9203bdfd8385b73','scripts/package_trained_selected_d.py':'f96ad26981984c90b45437d18fcd5f99d151da8323ec09b52234b042f1e48c73','scripts/build_trained_selected_notebook_c.py':'14ffd50fa99a5e3fa8e995d6e9bd55ff46f8e75784dca01e52107b6b2602493f'}
 for name,digest in protected.items():check('protected:'+name,sha(W/name)==digest)
 check('canonical_pair_still_proved344',sha(ROOT/'artifacts/selected_bundle.zip')=='7c9c7885912e867cbf2bd8eba9ee583cc6424265a2f8f3032299c2a1411c6083' and sha(ROOT/'submit/inference.ipynb')=='2b1f2d877cb89d861f8a360a67fa108aa3dce332b2c593064f1940fd3307f977')
 evidence={'schema':'long_c_portable_source_a_static_v1','utc':datetime.now(timezone.utc).isoformat(),'complete':True,'checks':checks,'all_checks_pass':all(checks.values()),'source_sha256':{n:sha(W/n) for n in files},'audit_script_sha256':sha(__file__),'model_or_new_source_execution':False,'tests_or_fixtures':False,'actual_build':False,'cloud_operations':False,'future_actual_binding':None,'independent_review_pending':True,'limitations':['Static source/hash/AST audit, no runtime proof.','Future parent interface only; no finite cloud stage built.','Future actual checkpoints, parity, sampling and root quality receipts remain absent/unbound.']}
 out=ROOT/'reports/evidence/long_c_portable_source_a_static_20261023.json';out.write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps({'checks':len(checks),'all_pass':all(checks.values()),'evidence':str(out)}))
if __name__=='__main__':main()
