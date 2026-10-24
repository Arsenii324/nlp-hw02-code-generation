"""Static source/JSON/hash readback only; no candidate import or execution."""
from pathlib import Path
import ast,copy,datetime,hashlib,json
R=Path(__file__).resolve().parents[1];W=R/'.worktrees/strong-inference-fallback-20261022';C=R/'.worktrees/gated-cooldown-prefix-20261023'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(p.read_text())
def tree(p):return ast.parse(p.read_text())
def fn(p,name):return next(n for n in tree(p).body if isinstance(n,ast.FunctionDef) and n.name==name)
def eq(a,b):return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
checks={}
def check(k,v):
 checks[k]=bool(v)
 if not v:raise ValueError(k)
afile=W/'configs/cooldown_portable_source_a_20261023.json';afile_sha='b1cf048ddeedbde9258c2f9299ab66fbebd8562e93555923ca7a8187bf91d386';a=load(afile);bfile=W/'configs/cooldown_portable_source_b_20261023.json';b=load(bfile)
check('all11A_and_freeze_unchanged',sha(afile)==afile_sha and all(sha(W/n)==v for n,v in a['source_sha256'].items()))
check('all11B_distinct_exact_SHA',len(b['source_sha256'])==11 and not(set(b['source_sha256'])&set(a['source_sha256'])) and all(sha(W/n)==v for n,v in b['source_sha256'].items()))
for n in b['source_sha256']:
 if n.endswith('.py'):tree(W/n)
 else:load(W/n)
check('all_B_Python_JSON_parse',True)
pa=W/'scripts/probe_cooldown_selected_sampling_a.py';pb=W/'scripts/probe_cooldown_selected_sampling_b.py';ra=W/'src/nlp_hw02/cooldown_selected_sampling_a.py';rb=W/'src/nlp_hw02/cooldown_selected_sampling_b.py';ha=W/'scripts/cooldown_sampling_contract_a.py';hb=W/'scripts/cooldown_sampling_contract_b.py';ka=W/'scripts/package_cooldown_selected_a.py';kb=W/'scripts/package_cooldown_selected_b.py';na=W/'scripts/build_cooldown_selected_notebook_a.py';nb=W/'scripts/build_cooldown_selected_notebook_b.py'
for n in ('verify_topology','actual_prefix_parity','generate_candidates','sample_fixed','atomic_json'):check('exact_producer_math_'+n,eq(fn(pa,n),fn(pb,n)))
for n in ('verify_topology','generate_candidates','sample_fixed','verify_parity_evidence','atomic_json'):check('exact_portable_math_'+n,eq(fn(ra,n),fn(rb,n)))
for n in ('digest','prospective_plan','validate_plan','validate_binding','base_lr','scheduled_lr','checkpoint_schedule'):check('exact_schedule_JSON_oracle_'+n,eq(fn(ha,n),fn(hb,n)))
for n in ('verify_path','copy_member'):check('exact_package_'+n,eq(fn(ka,n),fn(kb,n)))
check('same_bounded_backend_observer_producer_runtime',eq(fn(pb,'observe_backend'),fn(rb,'observe_backend')))
# Every producer change reconstructed from explicit version routing + metadata only.
def route(s):
 for x,y in [('cooldown_sampling_contract_a','cooldown_sampling_contract_b'),('probe_cooldown_selected_sampling_a','probe_cooldown_selected_sampling_b'),('package_cooldown_selected_a','package_cooldown_selected_b'),('cooldown_selected_sampling_a','cooldown_selected_sampling_b'),('build_cooldown_selected_notebook_a','build_cooldown_selected_notebook_b'),('cooldown_portable_source_a','cooldown_portable_source_b'),('cooldown_sampling_producer_a_v1','cooldown_sampling_producer_b_v1'),('cooldown_selected_deployment_spec_a_v1','cooldown_selected_deployment_spec_b_v1'),('cooldown_sampling_result_a_v1','cooldown_sampling_result_b_v1'),('cooldown_sampling_parent_a_v1','cooldown_sampling_parent_b_v1'),('cooldown_reuse_current_canonical_a_v1','cooldown_reuse_current_canonical_b_v1'),('cooldown_selected_bundle_a_v1','cooldown_selected_bundle_b_v1'),('cooldown_selected_candidate_staging_a_v1','cooldown_selected_candidate_staging_b_v1'),('cooldown_selected_sampling_execution_a_v1','cooldown_selected_sampling_execution_b_v1'),('cooldown_notebook_contract_a_v1','cooldown_notebook_contract_b_v1'),('cooldown_selected_checkpoint_integrity_v1','cooldown_selected_checkpoint_integrity_b_v1'),('cooldown_sampling_selection_v1','cooldown_sampling_selection_b_v1'),('cooldown-actual-sampling-a-v1','cooldown-actual-sampling-b-v1')]:s=s.replace(x,y)
 return s
expected=route(pa.read_text()).replace("prefix='cooldown-sampling-a-'","prefix='cooldown-sampling-b-'").replace("'sampling':spec['generation'],'samples_per_question'","'sampling':spec['generation'],'backend_policy':BACKEND_POLICY,'samples_per_question'").replace("report['software']={'torch':str(torch.__version__),'cuda':torch.version.cuda,'python':platform.python_version()}","report['software']={'torch':str(torch.__version__),'cuda':torch.version.cuda,'python':platform.python_version()};report['observed_backend']=observe_backend(torch);require(report['observed_backend']['matches_training_origin'],'Reference qualification producer must match actual training origin')")
check('whole_producer_main_only_declared_policy_metadata_routing',eq(fn(pb,'main'),next(n for n in ast.parse(expected).body if isinstance(n,ast.FunctionDef) and n.name=='main')))
check('producer_strict_trainingorigin_requirement_unchanged',"require({'torch':str(torch.__version__),'cuda':torch.version.cuda}==identity['required_software'],'Exact trained software required')" in pb.read_text())
check('successaftercleanup_and_exceptionsfalse_preserved',"report['complete']=True;report['phase']='finished'" in pb.read_text() and 'report.update(complete=False,failure_phase=' in pb.read_text())
ht=hb.read_text();rt=rb.read_text();pt=pb.read_text();kt=kb.read_text();nt=nb.read_text()
check('exact_D_source_driver_worker_manifest_freeze_pins',all(s in ht for s in ['a8a41094ab5128d33b0b7b5eb95054fa08bf56f8b8c2dcaf80344985fb60e201','fee6590e61ca5d5fee1eb8996abca87a74686b3dbc07a9df45dff18f1f4ea4d4','16ca2f5347a8bf3718695e85f2dc671111421198d72e552e5a0ab5ac10cac3c2','6d2898f293e4bd54536ecfd793c8fd3beb32cee0aff17905c8a4d9440f110b7c','c7337fdbdf209ba7452de09b43fe2bb0094f93ad97523396748348710e8ee081']))
check('actual_D_rechecks_selected_continue_and_nofabricatedpeer',all(s in ht for s in ['gated_cooldown_postprefix_recheck_d_v1','failed_peer_without_recheck','not_required_H_le2048','Completed/selected H>2048 requires actual D recheck receipt',"if name==selected_run:require(outcome=='continue'",'len(present)<=1']))
check('D_recheck_exactclock_rate_reserve_forecasts',all(s in ht for s in ["r['guard_monotonic']-r['postrelease_anchor_monotonic']","cost=max(r['prefix_checkpoint_inclusive_seconds']/1024,r['postrelease_checkpoint_inclusive_seconds']/1024)",'save=max(180.,1.5*',"(central>remaining)==(outcome=='reject')",'D recheck/common worker deadline differs']))
check('D_evaluator_skip_has_no_fabricated_child',"peer.get('evaluation',{})=={}" in ht and 'Only exact inherited selected weights may skip complete-horizon profiling' in ht)
check('sourcefreeze_before_Torch_and_policy_before_artifacts',ht.index('    verify_source_freeze(spec)')<ht.index("train=spec['training']") and ht.index("spec['backend_policy']==BACKEND_POLICY")<ht.index("train=spec['training']"))
check('shared_adapter_no_tensor_import',not any(isinstance(n,ast.Import) and any(x.name.split('.')[0] in ('torch','numpy','nlp_hw02') for x in n.names) or isinstance(n,ast.ImportFrom) and (n.module or '').split('.')[0] in ('torch','numpy','nlp_hw02') for n in ast.walk(tree(hb))))
policy=b['backend_policy'];policy_r=ast.literal_eval(next(n.value for n in tree(rb).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BACKEND_POLICY' for t in n.targets)));policy_h=ast.literal_eval(next(n.value for n in tree(hb).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BACKEND_POLICY' for t in n.targets)))
check('same_explicit_backendpolicy_freeze_adapter_runtime',policy==policy_h==policy_r and policy['training_origin']=={'torch':'2.10.0+cu128','cuda':'12.8'} and policy['production_reference_reproduction']=='record_only')
check('backendpolicy_bound_rootchoice_and_deployment',"expected['backend_policy']=BACKEND_POLICY" in kt and "'backend_policy':BACKEND_POLICY" in kt and "deployment['backend_policy']==BACKEND_POLICY" in rt)
run=fn(rb,'run_cooldown_selected_candidate');version_require=[n for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='require' and ('torch.__version__' in ast.unparse(n) or 'torch.version.cuda' in ast.unparse(n))]
check('portable_no_observed_versionstring_hardreject',not version_require and 'observed_backend=observe_backend(torch)' in rt)
check('backend_probe_bounded_and_failure_metadata',"timeout=2,check=False" in rt and "except (OSError,subprocess.TimeoutExpired)" in rt and "driver_observation_error" in rt)
check('trainingorigin_exact_receipt_retained',"deployment['required_software']==identity['required_software']=={'torch':'2.10.0+cu128','cuda':'12.8'}" in rt)
check('all_model_source_T4_order_projection_count_timing_gates_retained',all(s in rt for s in ["sha(model_module.__file__)==MODEL_SOURCE_SHA","torch.cuda.device_count()==1",'vars(model.cfg)',"sha(test_parquet)==TEST_SHA",'len(questions)==1919',"row['question_sha256']",'len(codes)==1919',"Finite complete candidate vectors required"]))
# Read embedded notebook code as literal source, without invoking its builder.
ntree=tree(nb);nf=fn(nb,'notebook');infer=next(n.value.value for n in nf.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='infer' for t in n.targets));it=ast.parse(infer)
check('embedded_infer_source_AST_parse',True)
check('production_reference_alwaysrecorded_not_hashthrow','production_reference_match_record_only' in infer and 'external_proof_requires_reference_match_and_four_clocks' in infer and not any(isinstance(n,ast.If) and 'reproduction' in ast.unparse(n.test) for n in ast.walk(it)))
check('notebook_strongest_and_runtime_gate_hard',all(s in infer for s in ['strongest = time.perf_counter() - NOTEBOOK_STARTED','strongest <= 180','complete_path_seconds','if not result["runtime_gate_passed"]:', 'Measured generation or strongest notebook clock exceeds180s']))
check('backend_not_qualified_by_production_metadata',"Not established by production runtime" in rt and "not_established_by_production_metadata" in rt and 'actual_parity_scope' in rt)
for kind in ('sampling_producer','selected_deployment','root_integrity','root_selection','runtime','notebook'):
 d=load(W/f'configs/cooldown_{kind}_b.template.json');check('template_'+kind+'_unbound',d.get('verified_actual_values',d.get('complete',d.get('selected_after_actual_quality_audit'))) is False)
 if kind in ('sampling_producer','selected_deployment'):check('template_'+kind+'_futurefieldsNULL',d['training']['terminal']['sha256'] is None and d['training']['run_id'] is None and d['selected']['model_sha256'] is None and d['samples_per_question'] is None and d['reviewed_source']['sha256'] is None and d['backend_policy']==policy)
check('freeze_actual_futurefields_all_NULL',all(b[k] is None for k in ('actual_training_terminal','actual_common_H','actual_decision','actual_release','actual_rechecks','selected_model','selected_run','selected_count','actual_sampling_terminal','root_quality_decision','package','fresh_notebook_proof')))
e=load(R/'reports/evidence/gated_cooldown_bound_stage_b_20261023.json');tools=R/e['tools'];kernel=R/e['kernel'];check('boundtraining_stageB_6tools_driver_metadata_unchanged',all(sha(tools/n)==v for n,v in e['tools_sha256'].items()) and sha(kernel/'run.py')==e['driver_sha256'] and sha(kernel/'kernel-metadata.json')==e['kernel_metadata_sha256'])
df=C/'artifacts/gated-cooldown-source-20261023d/source_manifest.json';d=load(df);check('accepted_D_archive_freeze_all58physical_unchanged',sha(df)=='c7337fdbdf209ba7452de09b43fe2bb0094f93ad97523396748348710e8ee081' and sha(df.parent/'cooldown_continuation_source.bin')=='a8a41094ab5128d33b0b7b5eb95054fa08bf56f8b8c2dcaf80344985fb60e201' and all(sha(C/n)==v for n,v in d['source_members_sha256'].items()))
result={'schema':'cooldown_portable_source_b_static_v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'freeze_sha256':sha(bfile),'source_sha256':b['source_sha256'],'checks':checks,'all_checks_pass':all(checks.values()),'canonical_observed_sha256':{'zip':sha(R/'artifacts/selected_bundle.zip'),'notebook':sha(R/'submit/inference.ipynb')},'actual_selected_model':None,'actual_terminal':None,'actual_package':None,'scope':'Static hashes/JSON/AST/source inspection only; no candidate imports/model/PT/raw/tests/cloud/actualbinding/package execution. Separate current344k portability owner may change canonical under root authority; this task performs no canonical writes.','independent_review_pending':True}
p=R/'reports/evidence/cooldown_portable_source_b_static_20261023.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'checks':len(checks),'all_pass':True,'freeze_sha256':sha(bfile),'evidence_sha256':sha(p)}))
