"""Static actual-artifact/source audit only: no Torch/checkpoint/model execution."""
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
def dump(n):return ast.dump(n,include_attributes=False)

def main():
 build=json.loads((ROOT/'reports/evidence/trained_gated_sampling_e_build_20261023.json').read_text());t=Path(build['tools']);k=Path(build['kernel']);manifest=t/'trained_gated_sampling_manifest.json';m=json.loads(manifest.read_text());contract=t/'bound_contract.json';c=json.loads(contract.read_text());checks={}
 def check(key,ok):
  checks[key]=bool(ok)
  if not ok:raise ValueError(key)
 check('actualstage_manifest_driver_worker_builder_pins',sha(manifest)==build['manifest_sha256'] and sha(k/'run.py')==build['driver_sha256'] and sha(t/'probe_trained_gated_sampling_e.py')==build['worker_sha256'] and sha(W/'scripts/build_kaggle_trained_gated_sampling_e.py')==build['builder_sha256'])
 check('all22_actualpayload_hashes',len(m['inputs_sha256'])==22 and all(sha(t/name)==value for name,value in m['inputs_sha256'].items()))
 check('minimal_complete_payload_allowlist',set(str(p.relative_to(t)) for p in t.rglob('*') if p.is_file())==set(m['inputs_sha256'])|{'trained_gated_sampling_manifest.json','dataset-metadata.json'})
 check('only3_inferenceweights_no_fullstate_arrays',{name for name in m['inputs_sha256'] if name.endswith('.pt')}=={'training/run-control/model.pt','training/run-control/matched_330k/model.pt','training/run-treatment/matched_330k/model.pt'} and not any(s in name for name in m['inputs_sha256'] for s in ('last_training','.npy','.npz','optimizer')))
 source=W/'scripts/probe_trained_gated_sampling_e.py';builder=W/'scripts/build_kaggle_trained_gated_sampling_e.py';driver=W/'scripts/kaggle_trained_gated_sampling_e.py'
 a=ast.parse((W/'scripts/probe_trained_gated_sampling.py').read_text());b=ast.parse(source.read_text());d=ast.parse(builder.read_text());p=ast.parse(driver.read_text());oldp=ast.parse((W/'scripts/kaggle_trained_gated_sampling.py').read_text())
 check('newsource_staged_bytes_equal',sha(source)==sha(t/source.name) and sha(driver)==sha(k/'run.py'))
 for name in ('extract_runtime','verify_topology','actual_prefix_parity','generate_candidates','sample_fixed'):check('frozen_worker_AST:'+name,dump(node(a,name))==dump(node(b,name)))
 check('binder_worker_verifier_AST_identical',dump(node(b,'verify_contract'))==dump(node(d,'verify_contract')))
 for name in ('AbsoluteDeadline','arm_absolute','absolute_alarm','sha','emit','start','stop','terminate','finalize'):check('exact_lifecycle_AST:'+name,dump(node(p,name))==dump(node(oldp,name)))
 check('exact_main_guard_AST',dump(p.body[-1])==dump(oldp.body[-1]))
 # Compare constant AST rather than trusting duplicate printed tolerance metadata.
 def assignment(t,name):return next(n for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id==name for x in n.targets))
 check('same_parity_tolerances_and_runtime_sourcepins',dump(assignment(a,'TOLERANCES'))==dump(assignment(b,'TOLERANCES')) and dump(assignment(a,'KNOWN_RUNTIME'))==dump(assignment(b,'KNOWN_RUNTIME')))
 check('unchanged_contract_sampling_tolerances',c['sampling']=={'temperature':.8,'top_k':50,'question_batch_size':32,'length_penalty':.6,'seed':20261002,'max_new_tokens':192} and c['parity_tolerances']==ast.literal_eval(assignment(a,'TOLERANCES').value))
 check('actualterminal_and_CPU_integrity_bound',c['terminal']['sha256']==m['actual_training_terminal_sha256']==sha(t/'training/session_result.json')=='991448f5aab75e7a7e8dad030f05f2a8af9a6327c75133831fade72e0c13e609' and m['actual_CPU_integrity_sha256']==sha(ROOT/'reports/evidence/gated_b034_weights_integrity_20261023.json'))
 integrity=json.loads((ROOT/'reports/evidence/gated_b034_weights_integrity_20261023.json').read_text());terminal=json.loads((t/'training/session_result.json').read_text());roledefs={'control-matched':('control',30000,330000,329000,'run-control/matched_330k',[24]),'control-final':('control',45000,345000,344000,'run-control',[24]),'treatment-matched':('treatment',30000,330000,329000,'run-treatment/matched_330k',[8,16])}
 check('exact3role_schema',c['schema']=='trained_gated_sampling_contract_e_v1' and set(c['roles'])==set(roledefs) and c['verified_actual_values'])
 for role,(lane,budget,horizon,step,prefix,samples) in roledefs.items():
  r=c['roles'][role];metrics=json.loads((t/'training'/r['metrics']['path']).read_text());config=json.loads((t/'training'/r['run_config']['path']).read_text())
  check('role_actualpaths:'+role,all(r[key]['path']==prefix+'/'+base and r[key]['sha256']==terminal['artifact_sha256'][prefix+'/'+base]==sha(t/'training'/r[key]['path']) for key,base in [('checkpoint','model.pt'),('metrics','metrics.json'),('run_config','run_config.json'),('tokenizer','tokenizer.json')]))
  check('role_actualmetadata:'+role,r['training_lane']==lane and r['training_budget_new_iterations']==metrics['completed_steps']==budget and r['training_absolute_completed_step']==metrics['absolute_completed_step']==horizon and r['checkpoint_step']==metrics['selected_step']==step and r['checkpoint_validation_loss']==metrics['best_validation_loss']==integrity['roles'][role]['validation_loss'] and r['samples']==samples and metrics['sampled_examples']==budget*64 and metrics['optimizer_updates']+metrics['skipped_updates']==budget)
  check('role_6M_resetlineage:'+role,config['data']['train_rows']==6000000 and config['warmstart']=={'kind':'weights only; fresh optimizers/scaler/RNG','parent_model_sha256':'d09a8611bd6ef8ab2d0711d6999aa419efe50454d1bb12252069c59bc20c8a85','parent_step':300000,'lane':lane,'same_numpy_sample_stream_seed':42} and r['expected_config']==config['config'])
  check('role_unequalbudget_label:'+role,r['comparison_scope']==('45k-control-only-not-equal-budget' if role=='control-final' else 'matched30k-architecture') and r['checkpoint_role']==('final_ce_best' if role=='control-final' else 'matched_ce_best'))
 streams=[json.loads((t/'training'/c['roles'][name]['metrics']['path']).read_text())['sampled_indices_sha256'] for name in ('control-matched','treatment-matched')];check('matched30k_streams_identical',streams[0]==streams[1])
 check('schedule_gpu0serial_gpu1treatment',m['schedule']==[{'device':0,'roles':['control-matched','control-final']},{'device':1,'roles':['treatment-matched']}])
 parent=driver.read_text();worker=source.read_text()
 check('single_shared735_no_second_deadline_reset',parent.count("str(STARTED+735)")==1 and "remaining=STARTED+735-time.monotonic()" in parent and "args.deadline,lambda:stopped" in worker)
 check('measured_finalrole_admission_explicit_not_proof',"1.15*observed+10." in parent and "Measured heuristic, not guarantee" in parent and "not_started_insufficient_absolute_budget" in parent)
 check('immediate_collect_then_successor_no_waitall',"collect(lane);del active[device]" in parent and "launch_next(device)" in parent and "STATE['roles'][name]=result" in parent)
 check('all3_pendingroles_survive_alarm',"STATE['roles']={name:{'complete':False" in parent and "complete=all(r['complete'] for r in STATE['roles'].values())" in parent)
 check('source_parity_before_sampling_unchanged',worker.index("report['parity']=actual_prefix_parity")<worker.index("for count in spec['samples']") and "verify_topology(model,spec['training_lane'])" in worker and "actual_prefix_parity(model,tok,pairs,spec['training_lane'],out)" in worker)
 check('unchanged_exact_bounds',m['bounds_seconds']=={'platform':900,'parent':840,'work':740,'worker_absolute':735,'aggregate_cleanup':785,'hash':835,'hard_exit':840})
 oldt=W/'artifacts/kaggle/trained-gated-sampling-tools-20261023a';oldm=json.loads((oldt/'trained_gated_sampling_manifest.json').read_text());oldk=W/'kaggle/trained-gated-sampling-20261023a'
 check('D_source_and_actualstage_preserved',sha(W/'scripts/probe_trained_gated_sampling.py')==oldm['worker_sha256']=='d11ceb18bef929532a7b5c4cba381895e12bc89c22601139ce518132a08994c6' and sha(W/'scripts/kaggle_trained_gated_sampling.py')==sha(oldk/'run.py')==oldm['driver_sha256']=='a48e30074199a6869fd342a5e45f7bae21c8897f73db1837ba8b3f4e284913ec' and sha(W/'scripts/build_kaggle_trained_gated_sampling_d.py')==oldm['builder_sha256']=='cab1677e74e680b6b717ec4fe17c58f03964a0536660fe1b82816a417a39b570' and all(sha(oldt/name)==value for name,value in oldm['inputs_sha256'].items()))
 metadata=json.loads((k/'kernel-metadata.json').read_text());dataset=json.loads((t/'dataset-metadata.json').read_text());check('private_offline_owner_slug_image',metadata['id']=='arsen4ikvar/nlp-hw02-trained-gated-sampling-20261023b' and metadata['title']==metadata['id'].split('/')[1] and metadata['is_private'] and not metadata['enable_internet'] and metadata['dataset_sources']==[dataset['id']] and dataset['isPrivate'] and not metadata['kernel_sources'] and metadata['docker_image']==m['docker_image']=='gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461')
 check('canonicalB031_unchanged',sha(ROOT/'artifacts/selected_bundle.zip')=='e5e9b6650e530c1015a03dcee6fa30ae0e9ed215ed01b0c1bff9ff7c8b80c323' and sha(ROOT/'submit/inference.ipynb')=='1b79bae3331774f8cc0fe5c9ca935a85b9fb7f2f6633b81dde9ed33109a2b6a3')
 preserved={str(p.relative_to(W)):sha(p) for base in (oldt,oldk) for p in base.rglob('*') if p.is_file()}
 report={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS author static actual-artifact audit; independent Luna actual-stage review and actual runtime pending','all_checks_pass':all(checks.values()),'checks':checks,'stage':build,'contract_sha256':sha(contract),'template_sha256':sha(W/'configs/gated_trained_sampling_contract_20261023e.template.json'),'D_preserved_file_sha256':preserved,'script_sha256':sha(__file__),'no_model_tests_cloud_canonical_actions':True,'actual_parity_or_quality_result':None}
 out=ROOT/'reports/evidence/trained_gated_sampling_e_static_20261023.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'static_checks':len(checks),'all_checks_pass':True,'payloads':len(m['inputs_sha256']),'contract_sha256':sha(contract),'evidence':str(out)}))

if __name__=='__main__':main()
