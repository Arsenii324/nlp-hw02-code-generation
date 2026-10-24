"""Actual preserved-output CPU audit; no model execution or generated Python execution."""
import ast
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import time
from audit_k32_budget_quality_20261023 import select_plurality,equal

ROOT=Path(__file__).resolve().parents[1];W=ROOT/'.worktrees/strong-inference-fallback-20261022';OUT=ROOT/'artifacts/kaggle/control344-proof-v1-c'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def ordered(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def main():
 started=time.monotonic();checks={};paths={}
 def check(key,ok):
  checks[key]=bool(ok)
  if not ok:raise ValueError(key)
 recovery=ROOT/'reports/evidence/control344_proof_launch_20261023c/recovery-current-v1.json';r=json.loads(recovery.read_text());t=json.loads((OUT/'session_result.json').read_text());m=json.loads((OUT/'trained_control_final_proof_manifest.json').read_text());candidate=W/'artifacts/portable-trained-control-final-20261023a';localtools=ROOT/'artifacts/kaggle/control344-proof-tools-20261023c'
 check('bounded_fresh_recovery_complete',r['complete'] and r['elapsed_seconds']<=180 and r['total_bytes']<=180*1024*1024 and all(f['bytes']<=64*1024*1024 for f in r['files']) and r['before']==r['after'] and r['before']['current_version_number']==1)
 check('all26_recovered_file_hashes',len(r['files'])==26 and all(sha(OUT/f['name'])==f['sha256'] and (OUT/f['name']).stat().st_size==f['bytes'] for f in r['files']))
 check('terminal_actual_SHA',sha(OUT/'session_result.json')==r['terminal_sha256']=='226954f3d29970de77c73d2f83064fb4c90688e100cc4fb3947719649cb58fab')
 check('all24_terminal_artifact_hashes',len(t['artifact_sha256'])==24 and all(sha(OUT/name)==value for name,value in t['artifact_sha256'].items()))
 check('actual_completed_finite_session',t['complete'] and t['proof_complete'] and t['validation_parity_complete'] and t['artifact_hashes_complete'] and t['phase']=='finished' and t['session_script_seconds']<=840)
 check('own_groups_reaped_exited_no_errors',len(t['cleanup'])==2 and all(x['reaped'] and x['group_exited'] and not x['errors'] and not x['deadline_exhausted'] for x in t['cleanup']))
 check('actual_driver_manifest_sources',sha(OUT/'run_driver.py')==t['driver_sha256']==r['before']['source_sha256']=='8e311c9ff18b2a91a69b3cc7c198d4dcc6e273b5394250f6c1836b3afa6b8732' and sha(OUT/'trained_control_final_proof_manifest.json')==t['manifest_sha256']=='ebd00f6708a930b336d6ac86a37cd679ce702be895358e0a7fa578f0ad953fb0')
 check('all6_private_mounted_input_readback_pins',all(x['sha256']==(m['inputs_sha256'][x['name']] if x['name']!='trained_control_final_proof_manifest.json' else t['manifest_sha256']) for x in json.loads((ROOT/'reports/evidence/control344_proof_launch_20261023c/private_dataset_readback.json').read_text())['bounded_stream_readback']['files']))
 check('selectedZIP_actual_new_identity',sha(candidate/'selected_bundle.zip')==m['candidate_package_sha256']=='7c9c7885912e867cbf2bd8eba9ee583cc6424265a2f8f3032299c2a1411c6083')
 for name in ('execute_inference_notebook.py','trained_control_final_validation_parity.py'):check('recovered_source:'+name,sha(OUT/name)==m['inputs_sha256'][name]==sha(localtools/name))
 original=OUT/'notebook-proof/original_notebook.ipynb.bin';executed=OUT/'notebook-proof/executed_notebook.ipynb.bin';nb0=json.loads(original.read_text());nb=json.loads(executed.read_text());code=[c for c in nb['cells'] if c['cell_type']=='code'];execution=json.loads((OUT/'notebook-proof/executed.json').read_text())
 check('original_new_notebook_exact_identity',sha(original)==m['notebook_sha256']==sha(candidate/'inference.ipynb')=='2b1f2d877cb89d861f8a360a67fa108aa3dce332b2c593064f1940fd3307f977')
 check('all3_actual_cells_unchanged_no_errors',len(code)==3 and [c['execution_count'] for c in code]==[1,2,3] and [c['source'] for c in nb0['cells']]==[c['source'] for c in nb['cells']] and not any(o.get('output_type')=='error' for c in code for o in c['outputs']))
 for c in code:ast.parse(''.join(c['source']))
 check('executor_actual_input_output_count',execution['complete'] and execution['executed_code_cells']==3 and execution['input_sha256']==sha(original) and execution['output_sha256']==sha(executed))
 source=ast.parse((ROOT/'scripts/audit_k32_budget_quality_20261023.py').read_text());frozen=ast.parse((W/'src/nlp_hw02/sampling_candidate.py').read_text())
 def fn(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 check('CPU_legacy_selector_exact_AST',all(ast.dump(fn(source,name),include_attributes=False)==ast.dump(fn(frozen,name),include_attributes=False) for name in ('equal','select_plurality')))
 check('exact_runtime_arithmetic_selector_model_module_pins',sha(ROOT/'src/nlp_hw02/arithmetic.py')==m['runtime_member_sha256']['src/nlp_hw02/arithmetic.py']=='e01078307ec395b41afe03b588e014b7498ee968ed6ed2f3d13f8105c45663d6' and sha(W/'src/nlp_hw02/sampling_candidate.py')==m['runtime_member_sha256']['src/nlp_hw02/sampling_candidate.py']=='250d34bee2dc60847a65a7d5a784fac261a77db153a4d49f0687a82fe5a98fe5' and sha(W/'src/nlp_hw02/trained_selected_sampling_c.py')==m['runtime_member_sha256']['src/nlp_hw02/trained_selected_sampling_c.py']=='aa362c163c2ed1fa9de29dd4299c5874438c68ff245d8dc0c9203bdfd8385b73' and m['runtime_member_sha256']['src/nlp_hw02/model.py']=='07e58176c39eed8ed03134074282d74589be97120e6d6569a15f27eee1538b26')
 import pyarrow.parquet as pq
 testpath=ROOT/'.worktrees/gated-adjacent-warmstart-20261022/artifacts/kaggle/tools-gated-b034-20261022b/test.parquet';check('actual1919_input_parquet_pin',sha(testpath)==m['runtime_member_sha256']['test.parquet']);test=pq.read_table(testpath,columns=['question'])['question'].to_pylist();validation=json.loads((localtools/'validation_questions.json').read_text());check('actual1792_input_projection_pin',sha(localtools/'validation_questions.json')==m['inputs_sha256']['validation_questions.json'] and len(validation)==1792 and len(test)==1919)
 side=json.loads((OUT/'notebook-proof/selected_predictions/candidate_timing.json').read_text());parity=json.loads((OUT/'validation-parity/validation_parity_result.json').read_text())
 check('single_fixed24_deployment_actualmodel_package',side['deployment']==m['deployment']==parity['policy']==t['deployment'] and side['package_sha256']==m['candidate_package_sha256'] and side['model_sha256']==parity['model_sha256']==m['runtime_member_sha256']['model.pt']=='eed1fdbb4f7e056536b322c08fad8660f5df24460f716a4d8e5382eb58ce8dc9')
 check('actual_one_visible_T4_each_lane',all(s['hardware']=={'gpu':'Tesla T4','visible_gpu_count':1} for s in (side,parity)) and t['hardware']==['Tesla T4','Tesla T4'])
 check('trained_software_actual',side['software']==parity['software']=={'torch':'2.10.0+cu128','cuda':'12.8','python':'3.12.13'})
 for label,directory,questions,timing,predname in [('warmup_n24',OUT/'notebook-proof/selected_predictions',test[:4],side['warmup_first4'],'warmup_n24_predictions.jsonl'),('test_n24',OUT/'notebook-proof/selected_predictions',test,side['generation_selection_serialization'],'predictions.jsonl'),('validation_n24',OUT/'validation-parity',validation,parity['generation_selection_serialization'],'validation_n24_predictions.jsonl')]:
  pool=directory/(label+'_candidates.jsonl');prediction=directory/predname;rows=[json.loads(s) for s in pool.read_text().splitlines()];preds=[json.loads(s) for s in prediction.read_text().splitlines()];check(label+':actual_path_hashes_rows',sha(pool)==timing['candidate_sha256'] and sha(prediction)==timing['prediction_sha256'] and len(rows)==len(preds)==len(questions))
  allcaps=selectedcaps=unsupported=0;codes=[]
  for index,(question,row,pred) in enumerate(zip(questions,rows,preds)):
   if row['row_index']!=index or row['question_sha256']!=hashlib.sha256(question.encode()).hexdigest():raise ValueError(label+':inputorder')
   if row['samples']!=24 or any(timing[k]!=v for k,v in m['deployment']['generation'].items()):raise ValueError(label+':policy')
   if not all(len(row[k])==24 for k in ('candidates','scores','capped','generated_lengths')) or not all(isinstance(c,str) for c in row['candidates']) or not all(math.isfinite(s) for s in row['scores']):raise ValueError(label+':vectors')
   if not all(type(c) is bool for c in row['capped']) or not all(type(n) is int and 0<=n<=192 for n in row['generated_lengths']):raise ValueError(label+':caps')
   chosen,info=select_plurality(row['candidates'],row['scores'])
   if chosen!=row['selected_index'] or info!={k:row[k] for k in ('supported_candidates','winning_votes','numeric_groups')} or set(pred)!={'code'} or pred['code']!=row['candidates'][chosen]:raise ValueError(label+':selector_projection')
   allcaps+=sum(row['capped']);selectedcaps+=row['capped'][chosen];unsupported+=info['supported_candidates']==0;codes.append(pred['code'])
  check(label+':all_counts_caps_and_order',timing['total_candidates']==len(questions)*24 and timing['samples_per_question']==24 and timing['rows']==len(questions) and timing['all_candidate_capped_rows']==allcaps==0 and timing['selected_candidate_capped_rows']==selectedcaps==0 and timing['without_supported_candidate']==unsupported and timing['ordered_questions_sha256']==ordered(questions))
  expected=m['validation_reproduction'] if label=='validation_n24' else (m['deployment']['reproduction'] if label=='test_n24' else None)
  if expected:check(label+':original_exact_code_reproduction',sha(prediction)==expected['code_only_jsonl_sha256'] and ordered(codes)==expected['ordered_codes_sha256'] and ordered(questions)==expected['ordered_questions_sha256'])
  paths[label]={'rows':len(questions),'actual_candidates':len(questions)*24,'all_candidate_caps':allcaps,'selected_caps':selectedcaps,'prediction_sha256':sha(prediction),'candidate_sha256':sha(pool),'ordered_codes_sha256':ordered(codes),'ordered_questions_sha256':ordered(questions),'complete_path_seconds':timing['complete_path_seconds'],'peak_allocated_bytes':timing['peak_allocated_bytes'],'selector_replay_matches':True}
 check('fixed_policy_original_caps_and_no_runtime_override',m['deployment']['mode']=='fixed24' and m['deployment']['generation']=={'temperature':0.8,'top_k':50,'question_batch_size':32,'length_penalty':0.6,'seed':20261002,'max_new_tokens':192} and side['caps']=={'source':256,'target':192,'recommended64_not_enforced':True})
 proof=t['lanes']['notebook-proof'];clocks={'core_generation_selection_serialization':side['generation_selection_serialization']['complete_path_seconds'],'strongest_notebook_through_initial_sidecar':side['notebook_timing_seconds']['notebook_start_through_output_and_initial_sidecar'],'executor_cells':execution['elapsed_seconds'],'parent_observed_whole_child':proof['whole_child_seconds']}
 check('all4_runtime_clocks_positive_finite_le180',all(math.isfinite(v) and 0<v<=180 for v in clocks.values()))
 check('all_runtime_flags_and_terminal_agree',proof['runtime_pass'] and side['runtime_gate_passed'] and side['notebook_total_within180'] and proof['core_seconds']==clocks['core_generation_selection_serialization'] and proof['strongest_notebook_seconds']==clocks['strongest_notebook_through_initial_sidecar'] and proof['executor_cells_seconds']==clocks['executor_cells'])
 check('production_and_validation_pools_exact_actual_E',all(sha(OUT/new)==sha(ROOT/'artifacts/kaggle/trained-gated-sampling-v1-b/control-final'/old) for new,old in [('notebook-proof/selected_predictions/predictions.jsonl','test_n24_predictions.jsonl'),('notebook-proof/selected_predictions/test_n24_candidates.jsonl','test_n24_candidates.jsonl'),('validation-parity/validation_n24_candidates.jsonl','validation_n24_candidates.jsonl')]))
 check('parity_complete_nogold_noscoring_nowarmup',parity['complete'] and parity['reproduction']['matches'] and not parity['gold_used'] and not parity['training'] and not parity['validation_quality_scored'] and not parity['warmup_performed'])
 check('production_no_validation_gold_training',side['execution_complete'] and side['notebook_execution_complete'] and side['reproduction']['matches'] and not side['validation_generated'] and not side['selection_uses_reference'] and not side['training'] and not t['training'] and not t['gold_used'])
 check('actual_root_selection_and_source_binding',m['root_quality_selection_receipt_sha256']==sha(ROOT/'reports/evidence/trained_control_final_portable_selection_20261023.json')=='0b33aa20cef35e2fe484ba1c93f54e9375b72cf13528bb83e439244adfd90ba5' and m['D_source_freeze_sha256']==sha(W/'configs/trained_selected_portable_source_d_20261023.json') and m['deployment']['role_id']=='control-final' and m['deployment']['checkpoint_step']==344000 and m['deployment']['samples_per_question']==24 and m['deployment']['parameters']==17524256 and m['deployment']['model_config']['gated_repeats'] is False)
 check('actual_bound_topology_and_prior_parity',side['topology']==m['deployment']['actual_parity']['topology'] and side['topology']['decoder_cache_depth']==4 and side['topology']['gates']=={} and m['deployment']['actual_parity']['passed'] and all(x['passed'] and x['active_argmax_disagreements']==0 for x in m['deployment']['actual_parity']['results'].values()))
 report={'utc':datetime.now(timezone.utc).isoformat(),'complete':True,'status':'Actual control344k fixed24 exact-byte one-visible-T4 proof passes author CPU audit; root independent verification before promotion','checks':checks,'all_checks_pass':all(checks.values()),'runtime_seconds':clocks,'paths':paths,'replay_rows':sum(p['rows'] for p in paths.values()),'replay_candidates':sum(p['actual_candidates'] for p in paths.values()),'terminal_sha256':sha(OUT/'session_result.json'),'recovery_receipt_sha256':sha(recovery),'package_sha256':m['candidate_package_sha256'],'original_notebook_sha256':sha(original),'executed_notebook_sha256':sha(executed),'manifest_sha256':sha(OUT/'trained_control_final_proof_manifest.json'),'driver_sha256':sha(OUT/'run_driver.py'),'script_sha256':sha(__file__),'actual_session_script_seconds':t['session_script_seconds'],'software':side['software'],'hardware':side['hardware'],'limits':['Root independent actual artifact verification required before canonical promotion.','Callback queued was not delivered; no callback receipt or course submission claimed.','Only pinned Parquet question order proven; bot ordering/official entrypoint/timer/library exact contracts remain unconfirmed.','Both lanes used one visible T4 each concurrently on two physical T4s; this proves one-device inference under observed environment, not arbitrary host contention.','Final timing-sidecar rewrite/print excluded from strongest notebook clock but inside measured wholechild.','No official pass1/testlabel score, data filtering, model runs or new sampling during CPU audit.'],'CPU_audit_seconds':time.monotonic()-started,'canonical_promoted':False,'root_independent_review_pending':True}
 out=ROOT/'reports/evidence/control344_proof_runtime_audit_20261023.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'complete':True,'checks':len(checks),'clocks':clocks,'replayed_rows':report['replay_rows'],'replayed_candidates':report['replay_candidates'],'CPU_seconds':report['CPU_audit_seconds'],'evidence':str(out)}))

if __name__=='__main__':main()
