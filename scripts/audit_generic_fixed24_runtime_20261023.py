"""Actual preserved-output CPU audit; no model execution or generated Python execution."""
import ast
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import time
from audit_k32_budget_quality_20261023 import select_plurality,equal

ROOT=Path(__file__).resolve().parents[1];W=ROOT/'.worktrees/strong-inference-fallback-20261022';OUT=ROOT/'artifacts/kaggle/generic-fixed24-proof-v1-a'
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
 recovery=ROOT/'reports/evidence/generic_fixed24_proof_launch_20261023a/recovery-current-v1.json';r=json.loads(recovery.read_text());t=json.loads((OUT/'session_result.json').read_text());m=json.loads((OUT/'generic_fixed24_proof_manifest.json').read_text());candidate=W/'artifacts/portable-b034-fixed24-20261023a';localtools=W/'artifacts/kaggle/generic-fixed24-proof-tools-20261023a'
 check('bounded_fresh_recovery_complete',r['complete'] and r['elapsed_seconds']<=180 and r['total_bytes']<=180*1024*1024 and all(f['bytes']<=64*1024*1024 for f in r['files']) and r['before']==r['after'] and r['before']['current_version_number']==1)
 check('all26_recovered_file_hashes',len(r['files'])==26 and all(sha(OUT/f['name'])==f['sha256'] and (OUT/f['name']).stat().st_size==f['bytes'] for f in r['files']))
 check('terminal_actual_SHA',sha(OUT/'session_result.json')==r['terminal_sha256']=='64dabb8bfe11f3f7672e146b4415a30a039db681ac75d67acbcc50646cdb2885')
 check('all24_terminal_artifact_hashes',len(t['artifact_sha256'])==24 and all(sha(OUT/name)==value for name,value in t['artifact_sha256'].items()))
 check('actual_completed_finite_session',t['complete'] and t['proof_complete'] and t['validation_parity_complete'] and t['artifact_hashes_complete'] and t['phase']=='finished' and t['session_script_seconds']<=840)
 check('own_groups_reaped_exited_no_errors',len(t['cleanup'])==2 and all(x['reaped'] and x['group_exited'] and not x['errors'] and not x['deadline_exhausted'] for x in t['cleanup']))
 check('actual_driver_manifest_sources',sha(OUT/'run_driver.py')==t['driver_sha256']==r['before']['source_sha256']=='9ab7abfceee7d00f4abc16a243fa28a82a43d08bf88652e5b4ddda98ab737d9a' and sha(OUT/'generic_fixed24_proof_manifest.json')==t['manifest_sha256']=='98adf6d2895c542515ec6b27ec780a9579ae36b4e4ec2d7ec998df262d6033cc')
 check('all6_private_mounted_input_readback_pins',all(x['sha256']==(m['inputs_sha256'][x['name']] if x['name']!='generic_fixed24_proof_manifest.json' else t['manifest_sha256']) for x in json.loads((ROOT/'reports/evidence/generic_fixed24_proof_launch_20261023a/input_readback.json').read_text())['files']))
 check('selectedZIP_actual_new_identity',sha(candidate/'selected_bundle.zip')==m['candidate_package_sha256']=='aba6112e3d3d3c9d80c742a71d146d88bccf65946ba7ef6edded6dce2bd581fb')
 for name in ('execute_inference_notebook.py','generic_fixed24_validation_parity.py'):check('recovered_source:'+name,sha(OUT/name)==m['inputs_sha256'][name]==sha(localtools/name))
 original=OUT/'notebook-proof/original_notebook.ipynb.bin';executed=OUT/'notebook-proof/executed_notebook.ipynb.bin';nb0=json.loads(original.read_text());nb=json.loads(executed.read_text());code=[c for c in nb['cells'] if c['cell_type']=='code'];execution=json.loads((OUT/'notebook-proof/executed.json').read_text())
 check('original_new_notebook_exact_identity',sha(original)==m['notebook_sha256']==sha(candidate/'inference.ipynb')=='c940570f7b24903ceade666e4f56b0441a8cdfd0e538480a56bf251daa322270')
 check('all3_actual_cells_unchanged_no_errors',len(code)==3 and [c['execution_count'] for c in code]==[1,2,3] and [c['source'] for c in nb0['cells']]==[c['source'] for c in nb['cells']] and not any(o.get('output_type')=='error' for c in code for o in c['outputs']))
 for c in code:ast.parse(''.join(c['source']))
 check('executor_actual_input_output_count',execution['complete'] and execution['executed_code_cells']==3 and execution['input_sha256']==sha(original) and execution['output_sha256']==sha(executed))
 source=ast.parse((ROOT/'scripts/audit_k32_budget_quality_20261023.py').read_text());frozen=ast.parse((W/'src/nlp_hw02/sampling_candidate.py').read_text())
 def fn(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 check('CPU_legacy_selector_exact_AST',all(ast.dump(fn(source,name),include_attributes=False)==ast.dump(fn(frozen,name),include_attributes=False) for name in ('equal','select_plurality')))
 import pyarrow.parquet as pq
 testpath=ROOT/'.worktrees/gated-adjacent-warmstart-20261022/artifacts/kaggle/tools-gated-b034-20261022b/test.parquet';check('actual1919_input_parquet_pin',sha(testpath)==m['runtime_member_sha256']['test.parquet']);test=pq.read_table(testpath,columns=['question'])['question'].to_pylist();validation=json.loads((localtools/'validation_questions.json').read_text());check('actual1792_input_projection_pin',sha(localtools/'validation_questions.json')==m['inputs_sha256']['validation_questions.json'] and len(validation)==1792 and len(test)==1919)
 side=json.loads((OUT/'notebook-proof/selected_predictions/candidate_timing.json').read_text());parity=json.loads((OUT/'validation-parity/validation_parity_result.json').read_text())
 check('single_fixed24_deployment_actualmodel_package',side['deployment']==m['deployment']==parity['policy']==t['deployment'] and side['package_sha256']==m['candidate_package_sha256'] and side['model_sha256']==parity['model_sha256']==m['runtime_member_sha256']['model.pt']=='d09a8611bd6ef8ab2d0711d6999aa419efe50454d1bb12252069c59bc20c8a85')
 check('actual_one_visible_T4_each_lane',all(s['hardware']=={'gpu':'Tesla T4','visible_gpu_count':1} for s in (side,parity)) and t['hardware']==['Tesla T4','Tesla T4'])
 check('trained_software_actual',side['software']==parity['software']=={'torch':'2.10.0+cu128','cuda':'12.8','python':'3.12.13'})
 for label,directory,questions,timing,predname in [('test_warmup4',OUT/'notebook-proof/selected_predictions',test[:4],side['warmup_first4'],'test_warmup4_predictions.jsonl'),('test1919',OUT/'notebook-proof/selected_predictions',test,side['generation_selection_serialization'],'predictions.jsonl'),('validation1792',OUT/'validation-parity',validation,parity['generation_selection_serialization'],'validation1792_predictions.jsonl')]:
  pool=directory/(label+'_candidates.jsonl');prediction=directory/predname;rows=[json.loads(s) for s in pool.read_text().splitlines()];preds=[json.loads(s) for s in prediction.read_text().splitlines()];check(label+':actual_path_hashes_rows',sha(pool)==timing['candidate_sha256'] and sha(prediction)==timing['prediction_sha256'] and len(rows)==len(preds)==len(questions))
  allcaps=selectedcaps=unsupported=0;codes=[]
  for index,(question,row,pred) in enumerate(zip(questions,rows,preds)):
   if row['row_index']!=index or row['question_sha256']!=hashlib.sha256(question.encode()).hexdigest():raise ValueError(label+':inputorder')
   if row['mode']!='fixed24' or row['samples_generated']!=24 or any(row[k]!=v for k,v in m['deployment']['generation'].items()):raise ValueError(label+':policy')
   if not all(len(row[k])==24 for k in ('candidates','scores','capped','generated_lengths')) or not all(isinstance(c,str) for c in row['candidates']) or not all(math.isfinite(s) for s in row['scores']):raise ValueError(label+':vectors')
   if not all(type(c) is bool for c in row['capped']) or not all(type(n) is int and 0<=n<=192 for n in row['generated_lengths']):raise ValueError(label+':caps')
   chosen,info=select_plurality(row['candidates'],row['scores'])
   if chosen!=row['selected_index'] or info!=row['final_diagnostics'] or set(pred)!={'code'} or pred['code']!=row['candidates'][chosen]:raise ValueError(label+':selector_projection')
   allcaps+=sum(row['capped']);selectedcaps+=row['capped'][chosen];unsupported+=info['supported_candidates']==0;codes.append(pred['code'])
  check(label+':all_counts_caps_and_order',timing['total_candidates']==len(questions)*24 and timing['actual24_rows']==len(questions) and timing['all_candidate_capped_rows']==allcaps==0 and timing['selected_candidate_capped_rows']==selectedcaps==0 and timing['without_supported_candidate']==unsupported and timing['ordered_questions_sha256']==ordered(questions))
  expected=m['validation_reproduction'] if label=='validation1792' else (m['deployment']['reproduction'] if label=='test1919' else None)
  if expected:check(label+':original_exact_code_reproduction',sha(prediction)==expected['code_only_jsonl_sha256'] and ordered(codes)==expected['ordered_codes_sha256'] and ordered(questions)==expected['ordered_questions_sha256'])
  paths[label]={'rows':len(questions),'actual_candidates':len(questions)*24,'all_candidate_caps':allcaps,'selected_caps':selectedcaps,'prediction_sha256':sha(prediction),'candidate_sha256':sha(pool),'ordered_codes_sha256':ordered(codes),'ordered_questions_sha256':ordered(questions),'complete_path_seconds':timing['complete_path_seconds'],'peak_allocated_bytes':timing['peak_allocated_bytes'],'selector_replay_matches':True}
 proof=t['lanes']['notebook-proof'];clocks={'core_generation_selection_serialization':side['generation_selection_serialization']['complete_path_seconds'],'strongest_notebook_through_initial_sidecar':side['notebook_timing_seconds']['notebook_start_through_output_and_initial_sidecar'],'executor_cells':execution['elapsed_seconds'],'parent_observed_whole_child':proof['whole_child_seconds']}
 check('all4_runtime_clocks_positive_finite_le180',all(math.isfinite(v) and 0<v<=180 for v in clocks.values()))
 check('all_runtime_flags_and_terminal_agree',proof['runtime_pass'] and side['runtime_gate_passed'] and side['notebook_total_within180'] and proof['core_seconds']==clocks['core_generation_selection_serialization'] and proof['strongest_notebook_seconds']==clocks['strongest_notebook_through_initial_sidecar'] and proof['executor_cells_seconds']==clocks['executor_cells'])
 check('production_original822654_exactbytes',(OUT/'notebook-proof/selected_predictions/predictions.jsonl').read_bytes()==(ROOT/'artifacts/kaggle/portable-proof-crossscore-v1/notebook-proof/b034-sampling-draft-temp08-seed20261002/predictions.jsonl').read_bytes())
 check('parity_complete_nogold_noscoring_nowarmup',parity['complete'] and parity['reproduction']['matches'] and not parity['gold_used'] and not parity['training'] and not parity['validation_quality_scored'] and not parity['warmup_performed'])
 check('production_no_validation_gold_training',side['execution_complete'] and side['notebook_execution_complete'] and side['reproduction']['matches'] and not side['validation_generated'] and not side['selection_uses_reference'] and not side['training'] and not t['training'] and not t['gold_used'])
 check('canonicalB031_unchanged',sha(ROOT/'artifacts/selected_bundle.zip')=='e5e9b6650e530c1015a03dcee6fa30ae0e9ed215ed01b0c1bff9ff7c8b80c323' and sha(ROOT/'submit/inference.ipynb')=='1b79bae3331774f8cc0fe5c9ca935a85b9fb7f2f6633b81dde9ed33109a2b6a3')
 report={'utc':datetime.now(timezone.utc).isoformat(),'complete':True,'status':'Actual generic fixed24 exact-byte one-visible-T4 proof passes author CPU audit; root independent verification before promotion','checks':checks,'all_checks_pass':all(checks.values()),'runtime_seconds':clocks,'paths':paths,'replay_rows':sum(p['rows'] for p in paths.values()),'replay_candidates':sum(p['actual_candidates'] for p in paths.values()),'terminal_sha256':sha(OUT/'session_result.json'),'recovery_receipt_sha256':sha(recovery),'package_sha256':m['candidate_package_sha256'],'original_notebook_sha256':sha(original),'executed_notebook_sha256':sha(executed),'manifest_sha256':sha(OUT/'generic_fixed24_proof_manifest.json'),'driver_sha256':sha(OUT/'run_driver.py'),'script_sha256':sha(__file__),'actual_session_script_seconds':t['session_script_seconds'],'software':side['software'],'hardware':side['hardware'],'limits':['Root independent actual artifact verification required before canonical promotion.','Callback queued was not delivered; no callback receipt or course submission claimed.','Only pinned Parquet question order proven; bot ordering/official entrypoint/timer/library exact contracts remain unconfirmed.','Both lanes used one visible T4 each concurrently on two physical T4s; this proves one-device inference under observed environment, not arbitrary host contention.','Final timing-sidecar rewrite/print excluded from strongest notebook clock but inside measured wholechild.','No official pass1/testlabel score, data filtering, model runs or new sampling during CPU audit.'],'CPU_audit_seconds':time.monotonic()-started,'canonical_promoted':False,'root_independent_review_pending':True}
 out=ROOT/'reports/evidence/generic_fixed24_runtime_audit_20261023.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'complete':True,'checks':len(checks),'clocks':clocks,'replayed_rows':report['replay_rows'],'replayed_candidates':report['replay_candidates'],'CPU_seconds':report['CPU_audit_seconds'],'evidence':str(out)}))

if __name__=='__main__':main()
