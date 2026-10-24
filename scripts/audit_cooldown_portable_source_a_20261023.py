"""Static readback only: hashes/JSON/source AST; never import or run candidate code."""
from pathlib import Path
import ast
import datetime
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
W=ROOT/'.worktrees/strong-inference-fallback-20261022'
C=ROOT/'.worktrees/gated-cooldown-prefix-20261023'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def load(p):return json.loads(p.read_text())
def fn(p,name):return next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
def eq(a,b):return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
checks={}
def check(name,ok):
    checks[name]=bool(ok)
    if not ok:raise AssertionError(name)
fpath=W/'configs/cooldown_portable_source_a_20261023.json';f=load(fpath);oldpath=W/'configs/long_c_portable_source_b_20261023.json';old=load(oldpath)
check('accepted_B_freeze_unchanged',sha(oldpath)=='8e78d6ae3b373660358075017cf26627bcf4ec51deaebf4faae34f5712e5ed8a')
check('all11_B_source_pins_unchanged',all(sha(W/n)==v for n,v in old['source_sha256'].items()))
check('fresh_exact11_source_files',len(f['source_sha256'])==11 and not(set(f['source_sha256'])&set(old['source_sha256'])))
check('all11_fresh_source_pins',all(sha(W/n)==v for n,v in f['source_sha256'].items()))
for n in f['source_sha256']:
    p=W/n
    if n.endswith('.py'):ast.parse(p.read_text())
    else:load(p)
check('all_fresh_source_AST_and_JSON_parse',True)
producer=W/'scripts/probe_cooldown_selected_sampling_a.py';prior=W/'scripts/probe_long_c_selected_sampling_b.py';runtime=W/'src/nlp_hw02/cooldown_selected_sampling_a.py';priorruntime=W/'src/nlp_hw02/trained_window_selected_sampling_a.py';helper=W/'scripts/cooldown_sampling_contract_a.py'
for name in ('verify_topology','actual_prefix_parity','generate_candidates','sample_fixed','atomic_json'):check('exact_producer_'+name,eq(fn(producer,name),fn(prior,name)))
for name in ('verify_topology','generate_candidates','sample_fixed','atomic_json','verify_parity_evidence'):check('exact_runtime_'+name,eq(fn(runtime,name),fn(priorruntime,name)))
for name in ('digest','prospective_plan','validate_plan','validate_binding','base_lr','scheduled_lr','checkpoint_schedule'):check('exact_reviewed_schedule_'+name,eq(fn(helper,name),fn(C/'src/nlp_hw02/gated_cooldown_schedule_a.py',name)))
check('exact_packager_pool_replay_validator',eq(fn(W/'scripts/package_cooldown_selected_a.py','verify_path'),fn(W/'scripts/package_trained_window_selected_b.py','verify_path')))
check('exact_packager_streaming_copy',eq(fn(W/'scripts/package_cooldown_selected_a.py','copy_member'),fn(W/'scripts/package_trained_window_selected_b.py','copy_member')))
# The entire producer/notebook differs only in version/schema/path prose routing.
def reroute(s):
    return s.replace('long_c_sampling_contract_a','cooldown_sampling_contract_a').replace('trained_window_selected_sampling_a','cooldown_selected_sampling_a').replace('run_long_c_selected_candidate','run_cooldown_selected_candidate').replace('long_c','cooldown').replace('long-c','cooldown').replace('Long-C','Cooldown').replace('long-C','cooldown')
check('whole_producer_metadata_routing_only',eq(ast.parse(producer.read_text()),ast.parse(reroute(prior.read_text()))))
notebook=W/'scripts/build_cooldown_selected_notebook_a.py';oldnotebook=W/'scripts/build_trained_window_selected_notebook_a.py'
check('whole_notebook_clock_and_failure_flow_route_only',eq(ast.parse(notebook.read_text()),ast.parse(reroute(oldnotebook.read_text()))))
main=fn(producer,'main');success=next(n for n in ast.walk(main) if isinstance(n,ast.Try) and any(isinstance(x,ast.With) for x in n.body))
check('success_after_TemporaryDirectory_exit_and_exception_false','report[\'complete\']=True' in ast.get_source_segment(producer.read_text(),success) and any(isinstance(n,ast.keyword) and n.arg=='complete' and isinstance(n.value,ast.Constant) and n.value.value is False for h in success.handlers for n in ast.walk(h)))
training=fn(helper,'verify_training');check('unbound_guard_first_then_source_freeze',isinstance(training.body[0],ast.Expr) and ast.get_source_segment(helper.read_text(),training.body[0]).startswith('require(spec.get') and isinstance(training.body[1].value,ast.Call) and training.body[1].value.func.id=='verify_source_freeze')
check('shared_adapter_has_no_Torch_numpy_model_import',not any(isinstance(n,ast.Import) and any(a.name.split('.')[0] in ('torch','numpy','nlp_hw02') for a in n.names) or isinstance(n,ast.ImportFrom) and (n.module or '').split('.')[0] in ('torch','numpy','nlp_hw02') for n in ast.walk(ast.parse(helper.read_text()))))
text=helper.read_text();check('explicit_cooldown_protocol_record_schemas',all(x in text for x in ('gated_cooldown_common_decision_a_v1','gated_cooldown_common_release_a_v1','gated_cooldown_prefix_request_a_v1','gated_cooldown_decision_ack_a_v1','gated_cooldown_continuation_a_v1','gated_cooldown_parent_a_v1')))
check('actual_endpoint_parent_not_selectedbest_restore',all(x in text for x in ('f9d4f20ca116b4d71089cbafeecab9c6cfb210752bd572e8da5bcca9627cfb0d','original_uniform_gated','gated_window_durable_pointer_v1','last_training.pt','selected_ce_best')))
check('actual_recovery_beforeafter_version_required',all(x in text for x in ('cooldown_training_recovery_v1','current_version_before','current_version_after','source_before_sha256','source_after_sha256')))
check('no_historical_prefix_pointer_read_claim','Historical' in text or 'historical alternating-slot' in text)
check('peer_failure_requires_explicit_rootreceipt_and_no_equalclaim',all(x in text for x in ('allow_completed_lane_with_incomplete_peer','equal_draw_comparison_supported','same_whole_raw','peer_outcomes')))
check('inherited_exact344k_reuse_preserved',"raise ReuseCurrentCanonical" in text and "CURRENT_MODEL_SHA='eed1fdbb" in text)
for kind in ('sampling_producer','selected_deployment','root_integrity','root_selection','runtime','notebook'):
    d=load(W/f'configs/cooldown_{kind}_a.template.json');check('template_'+kind+'_unbound',d.get('verified_actual_values',d.get('complete',d.get('selected_after_actual_quality_audit'))) is False)
    if kind in ('sampling_producer','selected_deployment'):check('template_'+kind+'_actual_fields_NULL',d['training']['terminal']['sha256'] is None and d['training']['run_id'] is None and d['training']['recovery']['sha256'] is None and d['selected']['model_sha256'] is None and d['samples_per_question'] is None and d['reviewed_source']['sha256'] is None)
check('all_future_freeze_values_NULL',all(f[k] is None for k in ('actual_training_terminal','actual_common_H','actual_decision','actual_release','selected_model','selected_run','selected_count','actual_sampling_terminal','root_quality_decision','package','fresh_notebook_proof')) and f['verified_actual_values'] is False and f['canonical_promotion'] is False)
manifest=C/'artifacts/kaggle/cooldown-prefix-tools-20261023a/gated_cooldown_continuation_manifest.json';m=load(manifest)
check('accepted_cooldown_bound_manifest_unchanged',sha(manifest)=='21b3da9f31c76cbd5aa7a7e9afe585715b0dd02540687af62faaed2390d09b59')
check('all4_cooldown_stage_inputs_unchanged',all(sha(manifest.parent/n)==v for n,v in m['input_sha256'].items()))
check('accepted_C_source_freeze_unchanged',sha(C/'artifacts/gated-cooldown-source-20261023c/source_manifest.json')=='36bd37a3e481f9c5835884047fa13ceb68b7daf1dcaf44524607782e182208fe')
check('accepted_C_driver_and_worker_unchanged',sha(C/'scripts/kaggle_gated_cooldown_prefix_b.py')=='1836b30f498cf46f7e693ea5a327811c18e9f1124b415f3244b6b27bd0a7aad9' and sha(C/'scripts/train_gated_cooldown_prefix_b.py')=='f3ec97cd2cda2e97935a9d3311694b67c7414f640e690ce84e6da54a93447296')
check('current_canonical_pair_unchanged',sha(ROOT/'artifacts/selected_bundle.zip')=='7c9c7885912e867cbf2bd8eba9ee583cc6424265a2f8f3032299c2a1411c6083' and sha(ROOT/'submit/inference.ipynb')=='2b1f2d877cb89d861f8a360a67fa108aa3dce332b2c593064f1940fd3307f977')
evidence={'schema':'cooldown_portable_source_a_static_v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'decision':'SOURCE_ONLY_STATIC_PASS_INDEPENDENT_REVIEW_PENDING','freeze_sha256':sha(fpath),'source_sha256':f['source_sha256'],'checks':checks,'all_checks_pass':all(checks.values()),'actual_training_terminal':None,'actual_selected_weight':None,'limits':['Only JSON/hash/AST source inspection; no module import/execution, tests/model/PT/raw/cloud/binding/package execution','Protocol/runtime interfaces are unbound; actual receipts/H/parity/pools/quality/freshproof still absent','Actual future stage/postbuild must pin this reviewed whole-source freeze, not only dynamic local identities']}
out=ROOT/'reports/evidence/cooldown_portable_source_a_static_20261023.json';out.write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps({'checks':len(checks),'all_pass':True,'freeze_sha256':sha(fpath),'evidence':str(out.relative_to(ROOT)),'evidence_sha256':sha(out)}))
