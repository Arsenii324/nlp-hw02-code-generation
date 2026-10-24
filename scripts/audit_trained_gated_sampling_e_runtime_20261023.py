"""Actual E artifact/parity/runtime/selector replay audit; quality labels stay separate."""
import ast
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import time
from audit_k32_budget_quality_20261023 import select_plurality

ROOT=Path(__file__).resolve().parents[1];W=ROOT/'.worktrees/strong-inference-fallback-20261022';OUT=ROOT/'artifacts/kaggle/trained-gated-sampling-v1-b';TOOLS=W/'artifacts/kaggle/trained-gated-sampling-tools-20261023b'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def ordered(values):return hashlib.sha256(json.dumps(values,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def main():
 started=time.monotonic();checks={};roles={};total_rows=total_candidates=0
 def check(key,ok):
  checks[key]=bool(ok)
  if not ok:raise ValueError(key)
 recovery=ROOT/'reports/evidence/trained_gated_sampling_launch_20261023b/recovery.json';r=json.loads(recovery.read_text());terminal=OUT/'session_result.json';t=json.loads(terminal.read_text());m=json.loads((OUT/'trained_gated_sampling_manifest.json').read_text());c=json.loads((OUT/'bound_contract.json').read_text());cpu=json.loads((ROOT/'reports/evidence/gated_b034_weights_integrity_20261023.json').read_text())
 check('Luna_recovery_pass_currentv1_beforeafter',r['status']=='PASS' and r['pre_transfer_identity']==r['post_transfer_identity'] and r['pre_transfer_identity']['current_version_number']==1 and r['remote_paths_match_terminal'] and r['all_terminal_hashes_match'])
 check('actual_terminal_SHA',sha(terminal)==r['terminal_sha256']=='01066a6d40afcb787a8aa5667e956a2f4e32ce22eb3954b8a1f92239e82d4866')
 check('all63_actual_terminal_artifact_hashes',len(t['artifact_sha256'])==63 and all(sha(OUT/name)==digest for name,digest in t['artifact_sha256'].items()))
 check('exact_manifest_driver_worker_contract_sourcepins',sha(OUT/'trained_gated_sampling_manifest.json')==t['manifest_sha256']=='75432d90bb4969a7e4ddc823f845f90112522240804801d3a1dbe081dcfac2e2' and sha(OUT/'run_driver.py')==t['driver_sha256']=='65f8fed4f7da3545e5c7b683fdaefc464a5f00214c0c3aedb719661d319caca5' and sha(OUT/'probe_trained_gated_sampling_e.py')==t['worker_sha256']=='c2074502f2eeb9305011e0b7bb78654bc0ce4f0538c159acce0716f0817f42d0' and sha(OUT/'bound_contract.json')==t['contract_sha256']=='e70715b069c7d8e848755b67acbad1b403342ea340b4f0311734b024067305fa')
 check('actual_training_terminal_CPUidentity',t['actual_training_terminal_sha256']==c['terminal']['sha256']==cpu['terminal_sha256']=='991448f5aab75e7a7e8dad030f05f2a8af9a6327c75133831fade72e0c13e609')
 check('allroles_complete_stable_hashes',t['complete'] and t['artifact_hashes_complete'] and t['phase']=='finished' and set(t['roles'])=={'control-matched','control-final','treatment-matched'} and all(v['complete'] and v['status']=='complete' for v in t['roles'].values()))
 check('all3_owned_groups_clean',len(t['cleanup'])==3 and all(v['reaped'] and v['group_exited'] and not v['errors'] and not v['deadline_exhausted'] for v in t['cleanup']))
 check('absolute_finite_parent_under740',0<t['session_script_seconds']<740 and t['absolute_budget_seconds']=={'work_cutoff':740,'aggregate_cleanup_end':785,'artifact_hash_end':835,'terminal_hard_exit':840,'external_platform':900})
 check('exact_sampling_and_pairprovenance',t['sampling']==c['sampling']=={'temperature':.8,'top_k':50,'question_batch_size':32,'length_penalty':.6,'seed':20261002,'max_new_tokens':192} and sha(OUT/'parity_provenance.json')==c['inputs']['parity_pairs']['projection_provenance_sha256']=='5a82d1a6df7b233adc42fe6d6b291b56721ec9396eba49dbb3b24eb6529bca52')
 import pyarrow.parquet as pq
 questions=json.loads((TOOLS/'inputs/validation_questions.json').read_text());testpath=TOOLS/'inputs/test.parquet';test=pq.read_table(testpath,columns=['question'])['question'].to_pylist();check('actual1792_1919_inputpins',len(questions)==1792 and len(test)==1919 and sha(TOOLS/'inputs/validation_questions.json')=='83ea3ea9f449619e015d98e21b1c35ec2cc6d2edd01db0c026301e718014f04e' and sha(testpath)=='8bde59c8d8bb7edc4e8a503be22fc132c4a47773025d333432e78c5489e6522a')
 selector=ast.parse((ROOT/'scripts/audit_k32_budget_quality_20261023.py').read_text());frozen=ast.parse((TOOLS/'inputs/sampling_candidate.py').read_text())
 def fn(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 check('exact_legacy_selector_arithmetic_source',sha(ROOT/'src/nlp_hw02/arithmetic.py')=='e01078307ec395b41afe03b588e014b7498ee968ed6ed2f3d13f8105c45663d6' and sha(TOOLS/'inputs/sampling_candidate.py')=='250d34bee2dc60847a65a7d5a784fac261a77db153a4d49f0687a82fe5a98fe5' and all(ast.dump(fn(selector,name),include_attributes=False)==ast.dump(fn(frozen,name),include_attributes=False) for name in ('equal','select_plurality')))
 for name in ('control-matched','control-final','treatment-matched'):
  spec=c['roles'][name];directory=OUT/name;report=json.loads((directory/'gated_sampling_result.json').read_text());parity=json.loads((directory/'actual_parity.json').read_text());saved=t['roles'][name];lane=spec['training_lane'];depth=8 if lane=='treatment' else 4
  check(name+':actual_role_receipt_complete',report['complete'] and report['role_id']==name and report['lane']==lane and report['worker_sha256']==t['worker_sha256'] and report['contract_sha256']==t['contract_sha256'] and report['parity_passed_before_sampling'] and not report['training'] and not report['gold_used_for_selection'])
  check(name+':checkpoint_budget_scope_identity',report['checkpoint_sha256']==saved['checkpoint_sha256']==spec['checkpoint']['sha256']==cpu['roles'][name]['sha256'] and report['step']==saved['selected_step']==spec['checkpoint_step']==cpu['roles'][name]['selected_step'] and all(report[key]==spec[key] for key in ('checkpoint_role','training_budget_new_iterations','training_absolute_completed_step','comparison_scope')))
  check(name+':actualsoftware_oneT4',report['visible_gpu_count']==1 and report['gpu']=='Tesla T4' and report['torch']=='2.10.0+cu128' and report['cuda']=='12.8')
  check(name+':exact_training_runtime_source',report['training_manifest_sha256']=='65a2693da6db3d45413e5f7e199d91cc6fe3d1237ecde49edb0af57bc8779256' and report['training_source_sha256']=='eefda7a0c41df4aa992f5cbde4620d7486b58ee86214c9d708698979ea2c5d83' and report['gated_model_source_sha256']=='07e58176c39eed8ed03134074282d74589be97120e6d6569a15f27eee1538b26')
  check(name+':actual4pairs_saved_parity',parity==report['parity']==saved['parity'] and parity['passed'] and parity['actual_rows']==4 and parity['prefix_width']==192 and parity['active_target_tokens']==399 and parity['tolerances']==c['parity_tolerances'])
  topology=parity['topology'];check(name+':actual_topology_depth_XSA_FFN',topology==report['topology'] and topology['decoder_cache_depth']==depth and topology['unique_xsa_scalars']==32 and topology['unique_parameters']==cpu['roles'][name]['parameters'] and all(topology['topology'][stack]['stored_blocks']==depth and topology['topology'][stack]['effective_blocks']==depth and topology['topology'][stack]['unique_ffns']==depth for stack in ('encoder','decoder')))
  if lane=='treatment':check(name+':actual_nonzero_gates_exactCPUvalues',all(topology['gates'][stack]==cpu['roles'][name]['repeat_gates'][stack+'_repeat_gates'] and len(topology['gates'][stack])==4 and all(math.isfinite(g) and g!=0 for g in topology['gates'][stack]) for stack in ('encoder','decoder')))
  else:check(name+':plain_control_no_gates',topology['gates']=={})
  for dtype,tol in c['parity_tolerances'].items():
   p=parity['results'][dtype];check(name+':parity_'+dtype,p['passed'] and p['finite'] and p['elementwise_close'] and math.isfinite(p['max_abs_logit_error']) and p['max_abs_logit_error']<=tol['max_abs'] and p['cache_depth']==p['distinct_self_K_buffers']==p['distinct_self_V_buffers']==depth and p['active_argmax_disagreements']==0)
  expected_labels={f'{label}_n{count}' for count in spec['samples'] for label in ('validation','warmup','test')};check(name+':all_prescribed_paths',set(report['paths'])==expected_labels and report['sampling']==c['sampling'] and report['samplecounts']==spec['samples'])
  paths={}
  for count in spec['samples']:
   for base,ordered_questions in [('validation',questions),('warmup',test[:4]),('test',test)]:
    label=f'{base}_n{count}';timing=report['paths'][label];pool_path=directory/(label+'_candidates.jsonl');prediction=directory/(label+'_predictions.jsonl');check(name+'/'+label+':saved_hash_timing',sha(pool_path)==timing['candidate_sha256'] and sha(prediction)==timing['prediction_sha256'] and json.loads((directory/(label+'_timing.json')).read_text())==timing)
    allcaps=selectedcaps=unsupported=rows=0;codes=[]
    with pool_path.open() as pool,prediction.open() as predictions:
     for i,question in enumerate(ordered_questions):
      row=json.loads(pool.readline());pred=json.loads(predictions.readline())
      if row['row_index']!=i or row['question_sha256']!=hashlib.sha256(question.encode()).hexdigest() or row['samples']!=count:raise ValueError(name+'/'+label+':order')
      if not all(len(row[k])==count for k in ('candidates','scores','capped','generated_lengths')) or not all(isinstance(c,str) for c in row['candidates']) or not all(math.isfinite(s) for s in row['scores']):raise ValueError(name+'/'+label+':actualvectors')
      if not all(type(v) is bool for v in row['capped']) or not all(type(n) is int and 0<=n<=192 for n in row['generated_lengths']):raise ValueError(name+'/'+label+':caplength')
      index,info=select_plurality(row['candidates'],row['scores'])
      if index!=row['selected_index'] or any(row[k]!=v for k,v in info.items()) or set(pred)!={'code'} or pred['code']!=row['candidates'][index]:raise ValueError(name+'/'+label+':selectorrawprojection')
      rows+=1;allcaps+=sum(row['capped']);selectedcaps+=row['capped'][index];unsupported+=info['supported_candidates']==0;codes.append(pred['code'])
     if pool.readline() or predictions.readline():raise ValueError(name+'/'+label+':extrarows')
    expected={'rows':len(ordered_questions),'samples_per_question':count,'total_candidates':len(ordered_questions)*count,'all_candidate_capped_rows':allcaps,'selected_candidate_capped_rows':selectedcaps,'without_supported_candidate':unsupported,'ordered_questions_sha256':ordered(ordered_questions),**c['sampling']}
    seconds=timing['complete_path_seconds'];check(name+'/'+label+':all_counters_policy_order_clock',all(timing[k]==v for k,v in expected.items()) and math.isfinite(seconds) and seconds>0 and timing['complete_path_within180']==(seconds<=180))
    check(name+'/'+label+':terminal_path_agrees',all(saved['paths'][label][k]==v for k,v in expected.items()) and saved['paths'][label]['complete_path_seconds']==seconds and saved['paths'][label]['prediction_sha256']==timing['prediction_sha256'])
    paths[label]={**expected,'complete_path_seconds':seconds,'complete_path_within180':seconds<=180,'peak_allocated_bytes':timing['peak_allocated_bytes'],'prediction_sha256':sha(prediction),'ordered_codes_sha256':ordered(codes),'candidate_sha256':sha(pool_path),'selector_replay_matches':True};total_rows+=rows;total_candidates+=rows*count
  roles[name]={'complete':True,'selected_step':spec['checkpoint_step'],'checkpoint_sha256':spec['checkpoint']['sha256'],'training_budget_new_iterations':spec['training_budget_new_iterations'],'comparison_scope':spec['comparison_scope'],'parity':parity,'paths':paths,'actual_child_wall_seconds':saved['actual_child_wall_seconds'],'admission':saved['admission']}
 events=[json.loads(s) for s in (OUT/'session_events.jsonl').read_text().splitlines()];starts={r['id']:r for r in events if r['event']=='gated_sampling_started'};validated={r['id']:r for r in events if r['event']=='role_validated'}
 check('actual3roles_device_sameabsolute735',set(starts)==set(c['roles']) and starts['control-matched']['device']==starts['control-final']['device']==0 and starts['treatment-matched']['device']==1 and all(e['absolute_deadline_session_seconds']==735 for e in starts.values()))
 check('actual_immediate_successor_before_peer_finish',starts['control-final']['session_seconds']>=validated['control-matched']['session_seconds'] and starts['control-final']['session_seconds']<validated['treatment-matched']['session_seconds'] and starts['control-final']['session_seconds']-validated['control-matched']['session_seconds']<1)
 check('all267480_actual_candidates_replayed',total_rows==14860 and total_candidates==267480)
 report={'utc':datetime.now(timezone.utc).isoformat(),'complete':True,'all_checks_pass':all(checks.values()),'checks':checks,'roles':roles,'actual_replay_rows':total_rows,'actual_replay_candidates':total_candidates,'session_script_seconds':t['session_script_seconds'],'terminal_sha256':sha(terminal),'recovery_receipt_sha256':sha(recovery),'source_identity':{k:t[k] for k in ('manifest_sha256','driver_sha256','worker_sha256','contract_sha256','actual_training_terminal_sha256')},'CPU_audit_seconds':time.monotonic()-started,'quality_scored':False,'model_executed':False,'cloud_operations':False,'canonical_changed':False,'limits':['Parity source/actual preserved numerical reports verified; no new teacher/cache model execution in CPU audit.','Numeric selector replay uses exact unchanged bounded AST interpreter; generated Python not executed.','Quality labels/consensus unused: root independently owns fixed1529 quality analysis.','E test path clocks include generation/selection/serialization, exclude model load/parity/warmup/hash/timingwrite; not cold notebook proof.','Portable C is unbound; actual E terminal has no duplicated inputs_sha256 field, so its narrow next revision must use actual recovered manifest identity/map instead.','No branch/deployment/canonical change by this audit.']}
 out=ROOT/'reports/evidence/trained_gated_sampling_e_runtime_audit_20261023.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'complete':True,'checks':len(checks),'rows':total_rows,'candidates':total_candidates,'CPU_seconds':report['CPU_audit_seconds'],'test_seconds':{role:{label:p['complete_path_seconds'] for label,p in value['paths'].items() if label.startswith('test')} for role,value in roles.items()},'evidence':str(out)}))

if __name__=='__main__':main()
