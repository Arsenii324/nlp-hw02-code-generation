"""Source/hash/AST/JSON inspection only; no model/test/candidate execution."""
from pathlib import Path
import ast,hashlib,json,datetime
R=Path(__file__).resolve().parents[1];W=R/'.worktrees/strong-inference-fallback-20261022'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(p.read_text())
def tree(p):return ast.parse(p.read_text())
def fn(p,n):return next(x for x in tree(p).body if isinstance(x,ast.FunctionDef) and x.name==n)
def eq(a,b):return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
def route(s):
 for x,y in [('long_c_sampling_contract_a','long_c_sampling_contract_c'),('probe_long_c_selected_sampling_b','probe_long_c_selected_sampling_c'),('package_trained_window_selected_b','package_trained_window_selected_c'),('trained_window_selected_sampling_a','trained_window_selected_sampling_c'),('build_trained_window_selected_notebook_a','build_trained_window_selected_notebook_c'),('long_c_portable_source_a','long_c_portable_source_c'),('long_c_sampling_producer_a_v1','long_c_sampling_producer_c_v1'),('long_c_selected_deployment_spec_a_v1','long_c_selected_deployment_spec_c_v1'),('long_c_sampling_result_a_v1','long_c_sampling_result_c_v1'),('long_c_sampling_parent_a_v1','long_c_sampling_parent_c_v1'),('long_c_reuse_current_canonical_a_v1','long_c_reuse_current_canonical_c_v1'),('long_c_selected_bundle_a_v1','long_c_selected_bundle_c_v1'),('long_c_selected_candidate_staging_a_v1','long_c_selected_candidate_staging_c_v1'),('long_c_selected_sampling_execution_a_v1','long_c_selected_sampling_execution_c_v1'),('long_c_notebook_contract_a_v1','long_c_notebook_contract_c_v1'),('long_c_selected_checkpoint_integrity_v1','long_c_selected_checkpoint_integrity_c_v1'),('long_c_sampling_selection_v1','long_c_sampling_selection_c_v1'),('long-c-actual-sampling-a-v1','long-c-actual-sampling-c-v1')]:s=s.replace(x,y)
 return s
checks={}
def check(k,v):
 checks[k]=bool(v)
 if not v:raise ValueError(k)
fpath=W/'configs/long_c_portable_source_c_20261023.json';f=load(fpath);priorpath=W/'configs/long_c_portable_source_b_20261023.json';prior=load(priorpath);bp=W/'configs/cooldown_portable_source_b_20261023.json';backend=load(bp)
check('accepted_longCBfreeze_all11source_unchanged',sha(priorpath)=='8e78d6ae3b373660358075017cf26627bcf4ec51deaebf4faae34f5712e5ed8a' and all(sha(W/n)==v for n,v in prior['source_sha256'].items()))
check('reviewed_cooldownBfreeze_all11source_unchanged',sha(bp)=='02694489cef9f9c085768d83bd6d8e97a54a7377a0dad557612d3e6da62d22c5' and all(sha(W/n)==v for n,v in backend['source_sha256'].items()))
check('fresh11Csource_SHA_and_disjoint_names',len(f['source_sha256'])==11 and not(set(f['source_sha256'])&set(prior['source_sha256'])) and all(sha(W/n)==v for n,v in f['source_sha256'].items()))
for n in f['source_sha256']:
 if n.endswith('.py'):tree(W/n)
 else:load(W/n)
check('allCsource_AST_JSON_parse',True)
pa=W/'scripts/probe_long_c_selected_sampling_b.py';pc=W/'scripts/probe_long_c_selected_sampling_c.py';ra=W/'src/nlp_hw02/trained_window_selected_sampling_a.py';rc=W/'src/nlp_hw02/trained_window_selected_sampling_c.py';ha=W/'scripts/long_c_sampling_contract_a.py';hc=W/'scripts/long_c_sampling_contract_c.py';ka=W/'scripts/package_trained_window_selected_b.py';kc=W/'scripts/package_trained_window_selected_c.py';nb=W/'scripts/build_cooldown_selected_notebook_b.py';nc=W/'scripts/build_trained_window_selected_notebook_c.py';cb=W/'src/nlp_hw02/cooldown_selected_sampling_b.py'
for name in ('verify_topology','actual_prefix_parity','generate_candidates','sample_fixed','atomic_json'):check('exact_reference_helper_'+name,eq(fn(pa,name),fn(pc,name)))
for name in ('verify_topology','generate_candidates','sample_fixed','verify_parity_evidence','atomic_json'):check('exact_portable_helper_'+name,eq(fn(ra,name),fn(rc,name)))
for name in ('require','sha','valid_sha','rooted','strings_sha'):check('exact_shared_identity_helper_'+name,eq(fn(ha,name),fn(hc,name)))
for name in ('verify_path','copy_member'):check('exact_package_helper_'+name,eq(fn(ka,name),fn(kc,name)))
expected=route(ha.read_text()).replace("    require(spec['generation']","    verify_source_freeze(spec)\n    require(spec['backend_policy']==BACKEND_POLICY,'Explicit reviewed backend policy required before Torch')\n    require(spec['generation']",1)
check('whole_verify_training_same_lineage_only_Croute_preflight_policy',eq(fn(hc,'verify_training'),next(x for x in ast.parse(expected).body if isinstance(x,ast.FunctionDef) and x.name=='verify_training')))
expected=route(pa.read_text()).replace("prefix='long-c-sampling-a-'","prefix='long-c-sampling-c-'").replace("'sampling':spec['generation'],'samples_per_question'","'sampling':spec['generation'],'backend_policy':BACKEND_POLICY,'samples_per_question'").replace("report['software']={'torch':str(torch.__version__),'cuda':torch.version.cuda,'python':platform.python_version()}","report['software']={'torch':str(torch.__version__),'cuda':torch.version.cuda,'python':platform.python_version()};report['observed_backend']=observe_backend(torch);require(report['observed_backend']['matches_training_origin'],'Reference qualification producer must match actual training origin')")
check('whole_producer_main_only_Croute_backendmetadata',eq(fn(pc,'main'),next(x for x in ast.parse(expected).body if isinstance(x,ast.FunctionDef) and x.name=='main')))
expected=route(ra.read_text()).replace("    identity=deployment['training_identity'];", "    require(deployment['backend_policy']==BACKEND_POLICY,'Explicit reviewed backend policy must be bound; no runtime override')\n    identity=deployment['training_identity'];",1)
check('whole_runtime_validator_same_lineage_only_Croute_policy',eq(fn(rc,'validate_deployment'),next(x for x in ast.parse(expected).body if isinstance(x,ast.FunctionDef) and x.name=='validate_deployment')))
check('exact_backend_observer_shared3sources',eq(fn(pc,'observe_backend'),fn(rc,'observe_backend')) and eq(fn(rc,'observe_backend'),fn(cb,'observe_backend')))
policy=backend['backend_policy']
for p,label in [(hc,'shared'),(rc,'runtime')]:
 value=ast.literal_eval(next(x.value for x in tree(p).body if isinstance(x,ast.Assign) and any(isinstance(n,ast.Name) and n.id=='BACKEND_POLICY' for n in x.targets)));check('exact_reviewed_backendpolicy_'+label,value==policy==f['backend_policy'])
ht=hc.read_text();rt=rc.read_text();pt=pc.read_text();kt=kc.read_text()
check('longC_exact_source_pins_not_cooldown_coercion',all(x in ht for x in ['4d50a0b45e39f7be2e56828122c35ae3aab97bd40d6a8390ae74392bb756a64e','e3f859c446e202b4e655b38db27ec5c9ebfe1bad2fd31bf68d75a1b157b6336f','17e077dca618ebedfa10a42fa27edd6ebca9f9a9eacd365cb7e6e85e64f375f6','3d23c10fdb3c1105194cb66ecaf198dd33d19ea3126d430525feac9a1b65a5a0','uniform-control','bucket-control']) and all(x not in ht for x in ('D_recheck','common_NEW_horizon','cooldown_checkpoint')))
check('preflight_freeze_and_policy_before_model_reads',ht.index('    verify_source_freeze(spec)')<ht.index("train=spec['training']") and ht.index("spec['backend_policy']==BACKEND_POLICY")<ht.index("train=spec['training']"))
check('producer_actualtrainingorigin_strict',"require({'torch':str(torch.__version__),'cuda':torch.version.cuda}==identity['required_software'],'Exact trained software required')" in pt and "require(report['observed_backend']['matches_training_origin']" in pt)
check('success_aftercleanup_exceptionfalse_preserved',"report['complete']=True;report['phase']='finished'" in pt and 'report.update(complete=False,failure_phase=' in pt)
run=fn(rc,'run_long_c_selected_candidate');check('portable_no_observed_version_hardreject',not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='require' and ('torch.__version__' in ast.unparse(n) or 'torch.version.cuda' in ast.unparse(n)) for n in ast.walk(run)))
check('trainingorigin_receipt_still_exact',"deployment['required_software']==identity['required_software']=={'torch':'2.10.0+cu128','cuda':'12.8'}" in rt)
check('policy_bound_rootselection_deployment',"expected['backend_policy']=BACKEND_POLICY" in kt and "'backend_policy':BACKEND_POLICY" in kt and "deployment['backend_policy']==BACKEND_POLICY" in rt)
check('metadata_probe_bounded_noqualification',"timeout=2,check=False" in rt and "except (OSError,subprocess.TimeoutExpired)" in rt and 'Not established by production runtime' in rt)
check('hard_model_T4_config_input_order_count_projection_finite_gates',all(x in rt for x in ["sha(model_module.__file__)==MODEL_SOURCE_SHA",'torch.cuda.device_count()==1','vars(model.cfg)',"sha(test_parquet)==TEST_SHA",'len(questions)==1919','len(codes)==1919',"row['question_sha256']",'Finite complete candidate vectors required']))
def infer(p):
 return next(n.value.value for n in fn(p,'notebook').body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='infer' for t in n.targets))
ci=infer(nc);bi=infer(nb).replace('run_cooldown_selected_candidate','run_long_c_selected_candidate');check('whole_inference_cell_exact_reviewed_backendB_clock_policy',eq(ast.parse(ci),ast.parse(bi)))
check('reference_recordonly_notthrow_and_timings_stillhard','production_reference_match_record_only' in ci and 'external_proof_requires_reference_match_and_four_clocks' in ci and 'if not result["runtime_gate_passed"]:' in ci and not any(isinstance(n,ast.If) and 'reproduction' in ast.unparse(n.test) for n in ast.walk(ast.parse(ci))))
for kind in ('sampling_producer','selected_deployment','root_integrity','root_selection','runtime','notebook'):
 d=load(W/f'configs/long_c_{kind}_c.template.json');check('template_'+kind+'_unbound',d.get('verified_actual_values',d.get('complete',d.get('selected_after_actual_quality_audit'))) is False)
 if kind in ('sampling_producer','selected_deployment'):check('template_'+kind+'_futureNULL_reviewedpolicy',d['training']['terminal']['sha256'] is None and d['training']['run_id'] is None and d['selected']['model_sha256'] is None and d['samples_per_question'] is None and d['reviewed_source']['sha256'] is None and d['backend_policy']==policy)
check('all_future_freeze_fieldsNULL',all(f[k] is None for k in ('actual_training_terminal','selected_run','selected_model','selected_count','actual_sampling_terminal','root_quality_decision','package','fresh_notebook_proof')) and f['verified_actual_values'] is False and f['canonical_promotion'] is False)
for evidence_name in ('gated_cooldown_bound_stage_a_20261023.json','gated_cooldown_bound_stage_b_20261023.json'):
 e=load(R/'reports/evidence'/evidence_name);tools=R/e['tools'];kernel=R/e['kernel'];check('prior_stage_unchanged_'+evidence_name,all(sha(tools/n)==v for n,v in e['tools_sha256'].items()) and sha(kernel/'run.py')==e['driver_sha256'] and sha(kernel/'kernel-metadata.json')==e['kernel_metadata_sha256'])
result={'schema':'long_c_portable_source_c_static_v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'freeze_sha256':sha(fpath),'source_sha256':f['source_sha256'],'checks':checks,'all_checks_pass':all(checks.values()),'canonical_observed_sha256':{'zip':sha(R/'artifacts/selected_bundle.zip'),'notebook':sha(R/'submit/inference.ipynb')},'actual_weights_terminal_package_proof':None,'scope':'Staticsource/JSON/AST/hash only; no candidate imports/model/PT/raw/tests/cloud/actualbinding/build/canonicalwrites. Shared policy identifier is reused without cooldown training-role mapping.','independent_review_pending':True}
p=R/'reports/evidence/long_c_portable_source_c_static_20261023.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'checks':len(checks),'all_pass':True,'freeze_sha256':sha(fpath),'evidence_sha256':sha(p)}))
