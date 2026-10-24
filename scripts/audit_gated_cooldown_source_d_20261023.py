"""Source D actual archive/AST/NULL-template readback, no model/tests/imports."""
from pathlib import Path
import ast,copy,datetime,hashlib,json,zipfile
R=Path(__file__).resolve().parents[1];W=R/'.worktrees/gated-cooldown-prefix-20261023';D=W/'artifacts/gated-cooldown-source-20261023d';C=W/'artifacts/gated-cooldown-source-20261023c'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(p.read_text())
def tree(p):return ast.parse(p.read_text())
def norm(n):return ast.dump(n,include_attributes=False)
def fn(t,n):return next(v for v in t.body if isinstance(v,ast.FunctionDef) and v.name==n)
checks={}
def check(k,v):
 checks[k]=bool(v)
 if not v:raise ValueError(k)
f=load(D/'source_manifest.json');prior=load(C/'source_manifest.json')
check('archive_actual_SHA',sha(D/'cooldown_continuation_source.bin')==f['source_sha256'])
with zipfile.ZipFile(D/'cooldown_continuation_source.bin') as z:
 check('exact58_unique_safe_source_SHA',len(z.namelist())==58==len(set(z.namelist())) and set(z.namelist())==set(f['source_members_sha256']) and all(not Path(n).is_absolute() and '..' not in Path(n).parts and hashlib.sha256(z.read(n)).hexdigest()==f['source_members_sha256'][n]==sha(W/n) for n in z.namelist()))
 for n in z.namelist():
  if n.endswith('.py'):ast.parse(z.read(n).decode())
 check('all_archived_Python_AST_parse',True)
check('all58_priorC_physical_source_files_unchanged',all(sha(W/n)==v for n,v in prior['source_members_sha256'].items()))
check('priorC_archive_and_freeze_unchanged',sha(C/'cooldown_continuation_source.bin')=='dd0c3e62c2d87128270ce2e1281015e3b4bd6133e3ccc2467d9f1a7e0175f911' and sha(C/'source_manifest.json')=='36bd37a3e481f9c5835884047fa13ceb68b7daf1dcaf44524607782e182208fe')
check('priorC_builder_unchanged',sha(W/'scripts/build_gated_cooldown_source_b.py')==prior['builder_sha256'])
shared=set(f['source_members_sha256'])&set(prior['source_members_sha256']);check('exact56_shared_Cmembers_and2_fresh_routes',len(shared)==56 and all(f['source_members_sha256'][n]==prior['source_members_sha256'][n] for n in shared) and set(f['source_members_sha256'])-shared=={'scripts/train_gated_cooldown_prefix_c.py','scripts/kaggle_gated_cooldown_prefix_c.py'})
foundation=R/'.worktrees/bucketed-gated-continuation-20261023/artifacts/kaggle/gated-window-tools-20261023c/window_continuation_source.bin'
with zipfile.ZipFile(foundation) as z:check('all53_foundation_exactbytes',len(z.namelist())==53 and all((W/n).read_bytes()==z.read(n) for n in z.namelist()))
oldw=W/'scripts/train_gated_cooldown_prefix_b.py';neww=W/'scripts/train_gated_cooldown_prefix_c.py';oldp=W/'scripts/kaggle_gated_cooldown_prefix_b.py';newp=W/'scripts/kaggle_gated_cooldown_prefix_c.py';ot=tree(oldw);nt=tree(neww);op=tree(oldp);np=tree(newp)
for n in ('require','sha','fsync_directory','atomic_json','atomic_torch_save','safe_relative','safe_run_id'):check('worker_exact_helper_'+n,norm(fn(ot,n))==norm(fn(nt,n)))
for n in ('arm_absolute','absolute_alarm','sha','emit','start','stop','require','snapshot_lanes','terminate','finalize'):check('parent_exact_lifecycle_'+n,norm(fn(op,n))==norm(fn(np,n)))
def batch(t):return next(n for n in ast.walk(fn(t,'run')) if isinstance(n,ast.For) and ast.unparse(n.target)=='batch_index')
check('entire64batch_model_optimizer_scaler_math_exact',norm(batch(ot))==norm(batch(nt)))
# Reconstruct every run delta by reversing the three declared regions only.
oldrun=fn(ot,'run');back=copy.deepcopy(fn(nt,'run'))
oldbranch=next(n for n in ast.walk(oldrun) if isinstance(n,ast.If) and ast.unparse(n.test)=='completed == PREFIX + 1024 and completed < target')
for n in ast.walk(back):
 if isinstance(n,ast.If) and ast.unparse(n.test)=='completed == PREFIX + 1024 and completed < target':n.body=copy.deepcopy(oldbranch.body)
for n in ast.walk(back):
 for field,value in ast.iter_fields(n):
  if isinstance(value,list):setattr(n,field,[v for v in value if not(isinstance(v,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('postrelease_anchor','postrelease_anchor_completed','prefix_checkpoint_inclusive_seconds') for t in v.targets))])
check('whole_worker_run_only_anchor_state_and_one_recheck_delta',norm(back)==norm(oldrun))
oldverify=ast.parse(oldw.read_text().replace('train_gated_cooldown_prefix_b.py','train_gated_cooldown_prefix_c.py').replace('kaggle_gated_cooldown_prefix_b.py','kaggle_gated_cooldown_prefix_c.py'))
check('whole_worker_binding_only_fresh_routes',norm(fn(nt,'verify_binding'))==norm(fn(oldverify,'verify_binding')))
expected=oldp.read_text().replace('train_gated_cooldown_prefix_b.py','train_gated_cooldown_prefix_c.py').replace("(not l['spec'].get('post_training_evaluation') or r['evaluation'].get('complete',False))","(not l['spec'].get('post_training_evaluation') or r['evaluation'].get('complete',False) or (r.get('evaluation_deferred') or {}).get('policy_skip',False))")
check('whole_parent_main_only_worker_route_and_explicit_evalpolicy_skip',norm(fn(np,'main'))==norm(fn(ast.parse(expected),'main')))
launch=fn(np,'launch_evaluation');oldlaunch=fn(op,'launch_evaluation');backlaunch=copy.deepcopy(launch);del backlaunch.body[2:4]
check('whole_evaluator_only2_new_skip_guards',norm(backlaunch)==norm(oldlaunch))
check('eval_incomplete_guard_first',ast.unparse(launch.body[2].test)=="not lane.get('horizon_complete', False)")
check('eval_exact_inherited_digest_skip',"eed1fdbb4f7e056536b322c08fad8660f5df24460f716a4d8e5382eb58ce8dc9" in ast.unparse(launch.body[3]))
newguard=next(n for n in np.body if isinstance(n,ast.If) and '__name__' in ast.unparse(n.test));oldguard=next(n for n in op.body if isinstance(n,ast.If) and '__name__' in ast.unparse(n.test));check('whole_parent_entry_alarm_guard_exact',norm(newguard)==norm(oldguard))
ws=neww.read_text();ps=newp.read_text();check('anchor_after_bindingreturn_before_budgetlog',ws.index('cooldown_binding,cooldown_decision=await_binding')<ws.index('postrelease_anchor=time.monotonic()')<ws.index("log({'event':'budget_admission'"))
check('slowest_two_actual1024_blocks',all(s in ws for s in ['completed-postrelease_anchor_completed==1024','seconds_per_iteration=max(prefix_seconds/1024,postrelease_seconds/1024)','guard_now=time.monotonic()','remaining=args.deadline-guard_now','reserve=max(180.,1.5*checkpoint_cost)','reject=central_forecast>remaining']))
check('125_forecast_diagnostic_not_reject',"conservative_forecast=1.25*central_eta+reserve" in ws and 'if 1.25*(target-completed)' not in ws)
check('actual_continue_and_reject_fullinputs_logged',all(s in ws for s in ['-postprefix_reject.json','-postprefix_continue.json','postrelease_anchor_monotonic','central_headroom_seconds','conservative125_headroom_seconds',"log({'event':'postprefix_cost_recheck',**recheck})"]))
check('exact13200_absolute_tuple','PLATFORM=13200;OUTER=13140;WORK=11640;CLEANUP_END=12000;HASH_END=13095;TAIL=1500' in ps and 'STARTED+WORK-5' in ps)
bounds={'platform':13200,'parent':13140,'work':11640,'worker':11635,'aggregate_cleanup':12000,'hash':13095,'tail':1500};check('freeze_bounds_exact',f['bounds_seconds']==bounds)
for n,d in f['templates_sha256'].items():
 t=load(D/n);check('NULL_template_'+n,sha(D/n)==d and t['verified_actual_values'] is False and t.get('actual_H') is None and t.get('actual_common_H') is None)
 if n=='parent_manifest.template.json':check('template_bounds_and_routes',t['bounds_seconds']==bounds and t['platform_seconds']==13200 and t['driver_sha256']==sha(newp))
check('freeze_actual_bindings_H_kernel_NULL',f['actual_state_bindings'] is None and f['actual_H'] is None and f['actual_kernel'] is None and f['canonical_changes'] is False)
audit=load(R/'reports/evidence/cooldown_actual_rejection_audit_20261023.json');check('real_metadata_arithmetic_evidence_PASS',audit['all_checks_pass'] and len(audit['checks'])==15 and audit['terminal_sha256']=='a5086dd7824a8084d9f00c042bda7b8390f7ba8b5bc8db65d32ff3168b48632d')
check('D_policy_counterfactual_fits_both_actualCcentral_not125',all(x['D_slower_prefix_or_incremental']['central_headroom']>1900 and x['D_slower_prefix_or_incremental']['conservative125_headroom']<0 for x in audit['results'].values()))
check('canonical_pair_unchanged',sha(R/'artifacts/selected_bundle.zip')=='7c9c7885912e867cbf2bd8eba9ee583cc6424265a2f8f3032299c2a1411c6083' and sha(R/'submit/inference.ipynb')=='2b1f2d877cb89d861f8a360a67fa108aa3dce332b2c593064f1940fd3307f977')
A=R/'.worktrees/strong-inference-fallback-20261022';af=load(A/'configs/cooldown_portable_source_a_20261023.json');check('inference_adapter_A_all11_freeze_unchanged',sha(A/'configs/cooldown_portable_source_a_20261023.json')=='b1cf048ddeedbde9258c2f9299ab66fbebd8562e93555923ca7a8187bf91d386' and all(sha(A/n)==v for n,v in af['source_sha256'].items()))
result={'schema':'gated_cooldown_source_d_static_v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':f['source_sha256'],'source_manifest_sha256':sha(D/'source_manifest.json'),'worker_sha256':sha(neww),'driver_sha256':sha(newp),'builder_sha256':sha(W/'scripts/build_gated_cooldown_source_c.py'),'checks':checks,'all_checks_pass':all(checks.values()),'archive_members':58,'unchanged_foundation':53,'unchanged_shared_Cmembers':56,'fresh_routes':2,'actual_H':None,'actual_stage':None,'actual_run':None,'limits':['Source/hash/AST and real archived failed-C metadata arithmetic only; no tests/fixtures/models/PT/raw/cloud/actualbinding','DactualH/rate/quality/runtime unknown; all deployment/cloud gates remain separate','OldC preserved and inferenceadapterA unbound; newDlineage needs explicit freshadapter']}
p=R/'reports/evidence/gated_cooldown_source_d_static_20261023.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'checks':len(checks),'all_pass':True,'source_sha256':f['source_sha256'],'freeze_sha256':sha(D/'source_manifest.json'),'evidence_sha256':sha(p)}))
