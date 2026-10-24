"""Source-only C static checks; never imports portable/model code or binds/builds ZIP."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];W=ROOT/'.worktrees/strong-inference-fallback-20261022'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def node(t,name):return next(n for n in t.body if getattr(n,'name',None)==name)
def dump(x):return ast.dump(x,include_attributes=False)

def main():
 paths={'module':W/'src/nlp_hw02/trained_selected_sampling_c.py','packager':W/'scripts/package_trained_selected_c.py','notebook_builder':W/'scripts/build_trained_selected_notebook_c.py','contract':W/'configs/trained_selected_sampling_c_20261023.template.json','notebook_template':W/'submit/candidates/trained_selected_c_template_20261023.ipynb'}
 checks={}
 def check(key,ok):
  checks[key]=bool(ok)
  if not ok:raise ValueError(key)
 trees={key:ast.parse(paths[key].read_text()) for key in ('module','packager','notebook_builder')};check('all3_source_AST_parse',True)
 old=ast.parse((W/'scripts/probe_trained_gated_sampling_e.py').read_text());new=trees['module'];generic=ast.parse((W/'scripts/package_selected_b034_b.py').read_text())
 for name in ('generate_candidates','sample_fixed','verify_topology'):check('entire_frozen_E_AST:'+name,dump(node(old,name))==dump(node(new,name)))
 check('stream_copy_member_exact_reviewed_B',dump(node(trees['packager'],'copy_member'))==dump(node(generic,'copy_member')))
 def assign(tree,name):return next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
 check('exact_parity_tolerances',dump(assign(old,'TOLERANCES'))==dump(assign(new,'TOLERANCES')))
 top_imports=[n for n in new.body if isinstance(n,(ast.Import,ast.ImportFrom))];check('portable_module_no_top_Torch_model_import',not any(any(x.name=='torch' or x.name.startswith('nlp_hw02') for x in n.names) if isinstance(n,ast.Import) else (n.module or '').startswith(('torch','nlp_hw02')) for n in top_imports))
 spec=json.loads(paths['contract'].read_text());check('unbound_contract_no_winner_or_receipts',spec['verified_actual_values'] is False and all(spec[k] is None for k in ('source_receipt_format','mode','role_id','samples_per_question','model_sha256','model_config','parameters','checkpoint_step','checkpoint_validation_loss','dataset_revision','actual_E_tools_root','actual_E_output_root')) and all(v is None for v in spec['receipts'].values()) and all(v is None for v in spec['reproduction'].values()) and all(v is None for v in spec['root_quality_decision'].values()))
 module=paths['module'].read_text();package=paths['packager'].read_text();check('only_explicit_3Eroles_8_16_24',set(ast.literal_eval(assign(new,'ROLE_DEFINITIONS').value))=={'control-matched','control-final','treatment-matched'} and "mode']==f'fixed{count}'" in module and "count in definition['allowed_samples']" in module)
 binder=node(trees['packager'],'bind_reproduction');check('unbound_guard_before_module_or_paths',isinstance(binder.body[0],ast.Expr) and isinstance(binder.body[0].value,ast.Call) and binder.body[0].value.func.id=='require' and "spec.get('verified_actual_values') is True" in ast.get_source_segment(package,binder.body[0]))
 runner=node(new,'run_trained_selected_candidate');check('unbound_runner_before_Torch',ast.get_source_segment(module,runner).index('validate_deployment')<ast.get_source_segment(module,runner).index('import torch'))
 check('unsupported_future_continuation_fails_closed',"Unsupported continuation/other receipt schema" in module and "Unsupported continuation/other receipt schema" in package and "source_receipt_format']=='trained-gated-stage-e-v1'" in package)
 check('actual_terminal_role_parity_source_required',all(text in package for text in ("terminal['roles'][role]['complete'] is True","terminal['artifact_hashes_complete'] is True","report['parity_passed_before_sampling'] is True","terminal['inputs_sha256']==m['inputs_sha256']","module.verify_parity_evidence(parity,selected['training_lane'])")))
 check('actual_root_quality_decision_required',"choice['selected_after_actual_quality_audit'] is True" in package and "choice['E_terminal_sha256']==receipts['terminal']" in package and "decision['sha256']" in package)
 check('exact_model_source_noconversion',"MODEL_SOURCE_SHA='07e58176c39eed8ed03134074282d74589be97120e6d6569a15f27eee1538b26'" in module and "vars(model.cfg)==deployment['model_config']" in module and "topology==deployment['actual_parity']['topology']" in module)
 check('goldfree1919_and_warmup4_then_samecall_reset',"sample_fixed(model,tok,questions[:4],count" in module and "sample_fixed(model,tok,questions,count" in module and "len(questions)==1919" in module and "torch.manual_seed(20261002);torch.cuda.manual_seed_all(20261002)" in ast.get_source_segment(module,node(new,'sample_fixed')))
 check('raw_codeonly_order_readback_and_reproduction',all(s in module for s in ("set(row)=={'code'}","row['question_sha256']==hashlib.sha256(question.encode()).hexdigest()","row['candidates'][row['selected_index']]==code","output_sha==expected['code_only_jsonl_sha256'] and code_sha==expected['ordered_codes_sha256']")))
 check('minimal10_payloads_streaming_no_optimizer_history',len(ast.literal_eval(assign(trees['packager'],'RUNTIME').value))==5 and "files={'model.pt'" in package and "'tokenizer.json':" in package and "'test.parquet':" in package and "'src/nlp_hw02/sampling_candidate.py':" in package and "'src/nlp_hw02/trained_selected_sampling_c.py':" in package and "copy_member(package,name,incoming)" in package and "require(not output.exists()" in package)
 nb=json.loads(paths['notebook_template'].read_text());cells=[c for c in nb['cells'] if c['cell_type']=='code'];check('unexecuted3_plain_code_cells',len(cells)==3 and all(c['execution_count'] is None and c['outputs']==[] for c in cells))
 sources=[''.join(c['source']) for c in cells]
 for s in sources:ast.parse(s)
 check('template_NULLhash_false_bound_beforeTorch',"ZIP_SHA256 = None" in sources[0] and "BOUND_CANDIDATE = False" in sources[0] and "if not BOUND_CANDIDATE" in sources[0] and 'import torch' not in sources[0] and sources[1].startswith('import torch'))
 first=ast.parse(sources[0]);check('strongest_clock_first_nonbootstrap_boundary',isinstance(first.body[0],ast.Import) and first.body[0].names[0].name=='time' and isinstance(first.body[1],ast.Assign) and first.body[1].targets[0].id=='NOTEBOOK_STARTED')
 check('strict6field_checkpoint_config_step_loss_seed_revision',all(s in sources[1] for s in ('"config", "model", "step", "validation_loss", "seed", "dataset_revision"','deployment["checkpoint_step"]','deployment["checkpoint_validation_loss"]','deployment["dataset_revision"]','checkpoint["seed"] != 42','sha_file(model_module.__file__)')))
 check('new_notebook_strongest_runtime_gates_retained',all(s in sources[2] for s in ('strongest = time.perf_counter() - NOTEBOOK_STARTED','result["generation_selection_serialization"]["complete_path_seconds"] <= 180 and strongest <= 180','if not result["reproduction"]["matches"]:','if not result["runtime_gate_passed"]:','candidate_failure.json')))
 check('one_visibleT4_and_generic_ZIP_no_runtime_override','os.environ["CUDA_VISIBLE_DEVICES"] = "0"' in sources[0] and 'ZIP_NAME = "selected_bundle.zip"' in sources[0] and 'torch.cuda.device_count() != 1' in sources[1] and 'MODE = validate_deployment(manifest["deployment"])' in sources[1])
 e=W/'artifacts/kaggle/trained-gated-sampling-tools-20261023b';m=json.loads((e/'trained_gated_sampling_manifest.json').read_text());check('E_READY_payload_unchanged',sha(e/'trained_gated_sampling_manifest.json')=='75432d90bb4969a7e4ddc823f845f90112522240804801d3a1dbe081dcfac2e2' and all(sha(e/name)==value for name,value in m['inputs_sha256'].items()))
 check('current_B034_canonical_backup_proof_unchanged',sha(ROOT/'artifacts/selected_bundle.zip')==sha(ROOT/'artifacts/selected-b034-fixed24-preserved-20261023/selected_bundle.zip')=='aba6112e3d3d3c9d80c742a71d146d88bccf65946ba7ef6edded6dce2bd581fb' and sha(ROOT/'submit/inference.ipynb')==sha(ROOT/'artifacts/selected-b034-fixed24-preserved-20261023/inference.ipynb')=='c940570f7b24903ceade666e4f56b0441a8cdfd0e538480a56bf251daa322270' and sha(ROOT/'artifacts/kaggle/generic-fixed24-proof-v1-a/session_result.json')=='64dabb8bfe11f3f7672e146b4415a30a039db681ac75d67acbcc50646cdb2885')
 report={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS source-only C static identity checks; unbound, no actual weight ZIP build/model/tests/cloud','all_checks_pass':all(checks.values()),'checks':checks,'source_paths':{k:str(v.relative_to(W)) for k,v in paths.items()},'pins':{k:sha(v) for k,v in paths.items()},'script_sha256':sha(__file__),'future_actual_terminal_sha256':None,'future_selected_role':None,'future_selected_samples':None,'future_model_sha256':None,'future_package_sha256':None,'actual_fresh_proof':None,'unsupported_receipt_adapters':['Any non-E continuation or other schema until separately reviewed adapter'],'canonical_changed':False}
 out=ROOT/'reports/evidence/trained_selected_portable_c_static_20261023.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'static_checks':len(checks),'all_checks_pass':True,'pins':report['pins'],'evidence':str(out)}))

if __name__=='__main__':main()
