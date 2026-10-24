"""Actual recovered multi-count pool replay/quality. Bounded AST numeric interpretation only."""
from pathlib import Path
from collections import Counter
import ast,datetime,hashlib,json,math,sys,time
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from audit_wide_n8_validation_actual_b import select_plurality,selector_ast_matches,score_one
from analyze_gated_results_20261023 import paired
B=R/'artifacts/kaggle/wide-t4-budget-v1-a-diagnostics';T=R/'artifacts/kaggle/wide-t4-budget-20261024a/tools';E=R/'reports/evidence/wide_t4_budget_launch_20261024a/actual_result';START=time.monotonic()
METRICS=['legacy','augassign_fixed_output','augassign_additive1e6']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):
 with Path(p).open() as f:return [json.loads(s) for s in f]
def score(codes,refs,qhash,allowed=None):
 result={};outcomes={}
 for m in METRICS:
  count=Counter();o={}
  for q,code in zip(qhash,codes):
   if allowed is not None and q not in allowed:continue
   v=score_one(code,refs[q],m)
   if v is None:continue
   ok,supported=v;o[q]=ok;count['correct' if ok else 'wrong_numeric' if supported else 'unsupported']+=1
  assert len(o)==(1529 if allowed is None else 1141);result[m]=dict(count);outcomes[m]=o
 return result,outcomes

def main():
 terminal=read(B/'session_result.json');recovery=read(E/'recovery.json');manifest=read(T/'wide_t4_budget_manifest_a.json')
 assert recovery['complete'] and recovery['before']==recovery['after'] and recovery['before']['numeric_id']==140490282 and recovery['before']['version']==1
 assert sha(B/'session_result.json')==recovery['terminal_sha256']=='303cf3816e56c3272d9eb7256a3652904c0a7af9e09e79a56cb404264cc3ced4'
 assert terminal['complete'] and terminal['artifact_hashes_complete'] and terminal['parent_sha256']=='c74f7f8ac94670782e5cacc744923e1ea4041fd97e0979eec2ce82a277a909f3' and terminal['tools_manifest_sha256']==sha(T/'wide_t4_budget_manifest_a.json')=='92e00934aa37164b3ef20f96cb7a677233edd555c24436a4a40cdfc470c5ad85'
 assert len(terminal['artifact_sha256'])==79 and set(terminal['artifact_sha256'])==set(recovery['files'])-{'session_result.json'} and all(sha(B/n)==v==recovery['files'][n]['sha256'] for n,v in terminal['artifact_sha256'].items())
 assert set(str(p.relative_to(B)) for p in B.rglob('*') if p.is_file())==set(recovery['files']) and len(recovery['files'])==80
 assert len(terminal['cleanup'])==2 and all(x['reaped'] and x['group_exited'] and not x['errors'] for x in terminal['cleanup'])
 for p in [R/'src/nlp_hw02/arithmetic.py',T/'inputs/sampling_candidate.py']:
  assert sha(p)==('e01078307ec395b41afe03b588e014b7498ee968ed6ed2f3d13f8105c45663d6' if p.name=='arithmetic.py' else '250d34bee2dc60847a65a7d5a784fac261a77db153a4d49f0687a82fe5a98fe5')
 assert selector_ast_matches(T/'inputs/sampling_candidate.py')
 qp=T/'inputs/validation_questions.json';assert sha(qp)=='83ea3ea9f449619e015d98e21b1c35ec2cc6d2edd01db0c026301e718014f04e';questions=read(qp);qhash=[hashlib.sha256(q.encode()).hexdigest() for q in questions];assert len(qhash)==len(set(qhash))==1792
 refp=R/'reports/evidence/primary_reference_consensus.json';assert sha(refp)=='deb96602c88b983ecfbf23738d2a64192057273798548ab61a7cdf7acfc7c119';refs=read(refp)['groups_by_question_sha256']
 maskp=R/'reports/evidence/gemini_reference_screen_20261024/auxiliary_reference_screen_mask.json';assert sha(maskp)=='0e16c82bc9711fa3c71e56a74dac09c7d4bbde1132397dc139d697dbf9b434e2';mask=read(maskp);assert mask['complete_all14_calls'] and mask['question_sha256_order']==qhash;aux={qhash[i] for i in mask['retained_global_indices'] if refs[qhash[i]]['status']=='supported_consensus'};assert len(aux)==1141
 specs={};outcomes={};auxout={}
 baseline=R/'artifacts/kaggle/cooldown-paired-n24-v1-a-diagnostics/run-cooldown-half/validation_n24_predictions.jsonl';assert sha(baseline)=='cc52e3d2c138c00f16676225bf022ce1bb82f32689d164867a6508f477d400b3'
 sources={'selected_half415656_N24':baseline}
 for role in terminal['children']:
  assert role['complete'] and role['actual_exit']==0 and role['done'] and role['final_cleanup']['reaped'] and role['final_cleanup']['group_exited'] and not role['final_cleanup']['errors'];run=role['run_id'];expected=manifest['roles'][run]
  assert role['worker_identity']==expected['expected_identity'] and role['counts_completed']==expected['counts']
  for n in role['counts_completed']:sources[run+'_N'+str(n)]=B/role['output']/f'validation_n{n}_predictions.jsonl'
 for label,path in sources.items():
  predictions=rows(path);assert len(predictions)==1792 and all(set(x)=={'code'} and isinstance(x['code'],str) for x in predictions)
  quality,oc=score([x['code'] for x in predictions],refs,qhash);masked,ao=score([x['code'] for x in predictions],refs,qhash,aux)
  specs[label]={'prediction_sha256':sha(path),'quality_1529':quality,'auxiliary_1141':masked};outcomes[label]=oc;auxout[label]=ao
 assert specs['selected_half415656_N24']['quality_1529']['legacy']['correct']==1117
 comparisons={label:{m:paired(outcomes['selected_half415656_N24'][m],outcomes[label][m]) for m in METRICS} for label in specs if label!='selected_half415656_N24'}
 quick={'schema':'wide_T4_actual_selected_numeric_preliminary_v1','status':'SELECTED_PREDICTIONS_SCORED_POOL_REPLAY_PENDING','script_sha256':sha(__file__),'terminal_sha256':sha(B/'session_result.json'),'quality':specs,'paired_baseline_left_vs_policy_right':comparisons,'mask_sha256':sha(maskp),'limits':['No wholepool replay/parity audit claimed by this preliminary receipt.','No officialpass1 or coldnotebookproof.','Auxiliarymask notcorrectnessgold; primary1529unchanged.']};pre=E/'preliminary_quality.json';assert not pre.exists();pre.write_text(json.dumps(quick,indent=2)+'\n');print(json.dumps({'early_quality':{k:{m:v['correct'] for m,v in x['quality_1529'].items()} for k,x in specs.items()},'auxiliary':{k:{m:v['correct'] for m,v in x['auxiliary_1141'].items()} for k,x in specs.items()},'preliminary_sha256':sha(pre)},indent=2),flush=True)
 # The test file is read question-only for order hashes; no reference/test correctness is scored.
 import pyarrow.parquet as pq
 assert sha(T/'inputs/test.parquet')=='8bde59c8d8bb7edc4e8a503be22fc132c4a47773025d333432e78c5489e6522a';test=pq.read_table(T/'inputs/test.parquet',columns=['question'])['question'].to_pylist();assert len(test)==1919
 orders={'validation':questions,'warmup':test[:4],'test':test};audits={};totalrows=totalcandidates=0
 for role in terminal['children']:
  run=role['run_id'];base=B/role['output'];w=read(base/'wide_t4_budget_result_a.json');assert sha(base/'wide_t4_budget_result_a.json')==role['worker_result_sha256'];expected=manifest['roles'][run]
  assert w['complete'] and w['phase']=='finished' and w['parity_passed_before_sampling'] and w['screen_only_not_notebook_proof'] and not w['gold_used_for_selection'] and not w['training'];assert w['identity']==role['worker_identity'] and w['contract_sha256']==role['contract_sha256'] and w['producer_sha256']==expected['producer_sha256'] and w['donor_producer_sha256']==expected['donor_producer_sha256'] and w['donor_adapter_sha256']==expected['donor_adapter_sha256'];assert w['counts_completed']==role['counts_completed'] and not w['optional_skipped']
  assert 'T4' in w['hardware']['gpu'] and w['hardware']['visible_gpu_count']==1 and w['hardware']['capability']==[7,5] and w['software']['matches_training_origin']
  parity=read(base/'actual_parity.json');assert parity==w['parity']==role['actual_parity'] and parity['passed'] and parity['actual_rows']==4
  assert parity['topology']['decoder_cache_depth']==4 and parity['topology']['unique_parameters']==expected['expected_identity']['parameters']
  for dt in ['fp32','fp16']:
   v=parity['results'][dt];assert v['finite'] and v['elementwise_close'] and v['passed'] and v['max_abs_logit_error']<=parity['tolerances'][dt]['max_abs'] and v['cache_depth']==4 and v['distinct_self_K_buffers']==v['distinct_self_V_buffers']==4
  inputs={k:v['sha256'] for k,v in read(T/expected['contract_path'])['base_contract']['inputs'].items()};inputs['test_parquet']=sha(T/'inputs/test.parquet');assert w['inputs_sha256']==inputs
  pathaudits={}
  for label,timing in w['paths'].items():
   kind,n=label.rsplit('_n',1);n=int(n);ordered=orders[kind];predpath=base/(label+'_predictions.jsonl');poolpath=base/(label+'_candidates.jsonl');side=read(base/(label+'_timing.json'));assert timing==side==role['worker_paths'][label]
   assert sha(predpath)==timing['prediction_sha256'] and sha(poolpath)==timing['candidate_sha256'] and timing['rows']==len(ordered) and timing['samples_per_question']==n and timing['total_candidates']==len(ordered)*n
   assert timing['ordered_questions_sha256']==hashlib.sha256(json.dumps(ordered,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
   assert all(timing[k]==v for k,v in {'temperature':.8,'top_k':50,'question_batch_size':32,'seed':20261002,'length_penalty':.6,'max_new_tokens':192}.items())
   candidates=rows(poolpath);pred=rows(predpath);assert len(candidates)==len(pred)==len(ordered);caps=selectedcaps=unsupported=0;lengths=Counter()
   for i,(row,pr,q) in enumerate(zip(candidates,pred,ordered)):
    assert row['row_index']==i and row['question_sha256']==hashlib.sha256(q.encode()).hexdigest() and row['samples']==n
    assert all(len(row[k])==n for k in ['candidates','scores','capped','generated_lengths']) and all(isinstance(x,str) for x in row['candidates']) and all(type(x) in (int,float) and math.isfinite(x) for x in row['scores'])
    assert all(type(x)is bool for x in row['capped']) and all(type(x)is int and 0<=x<=192 for x in row['generated_lengths']) and all((l==192) if cap else l<192 for cap,l in zip(row['capped'],row['generated_lengths']))
    idx,info=select_plurality(row['candidates'],row['scores']);assert idx==row['selected_index'] and info=={k:row[k] for k in info} and set(pr)=={'code'} and pr['code']==row['candidates'][idx]
    caps+=sum(row['capped']);selectedcaps+=row['capped'][idx];unsupported+=info['supported_candidates']==0;lengths.update(row['generated_lengths'])
   assert (caps,selectedcaps,unsupported)==(timing['all_candidate_capped_rows'],timing['selected_candidate_capped_rows'],timing['without_supported_candidate'])
   seconds=timing['complete_path_seconds'];assert math.isfinite(seconds) and seconds>0 and timing['complete_path_within180'] is (seconds<=180);outer=w['outer_call_clocks'][label];assert outer['call_through_timing_sidecar_seconds']>=seconds
   progress=read(base/(label+'_progress.json'));assert progress['completed_rows']==progress['requested_rows']==len(ordered)
   totalrows+=len(ordered);totalcandidates+=len(ordered)*n;pathaudits[label]={'rows':len(ordered),'candidates':len(ordered)*n,'selector_replay_all_rows':True,'core_seconds':seconds,'outer_call_seconds':outer['call_through_timing_sidecar_seconds'],'core_within180':seconds<=180,'all_caps':caps,'selected_caps':selectedcaps,'unsupported_rows':unsupported,'candidate_sha256':sha(poolpath),'prediction_sha256':sha(predpath),'peak_allocated_bytes':timing['peak_allocated_bytes'],'length_distribution':dict(lengths)}
  assert set(w['paths'])=={f'{kind}_n{n}' for n in expected['counts'] for kind in orders}
  audits[run]={'identity':w['identity'],'parity':parity,'paths':pathaudits,'count_decisions':w['count_decisions'],'counts_completed':w['counts_completed'],'optional_skipped':w['optional_skipped'],'whole_worker_seconds':w['whole_worker_main_seconds'],'whole_child_seconds':role['whole_child_seconds'],'four_clock_proof':role['four_clock_proof']}
 # Same-model A100 controls are provenance/maturity comparisons; no speed normalization claim.
 controls={'gelu512_A100_228352':(R/'artifacts/paid-wide-validation-20261024a/gelu-231424-result/reference/validation_n8_predictions.jsonl','e4f252f39264a470ff5999ce24d4dee285003802fc3d33872a4a588975de5250','gelu512_N8'),'gelu768_A100_52224':(R/'artifacts/paid-gelu768-validation-20261024a/first50k-result/reference/validation_n8_predictions.jsonl','2f7779e2151ef8f974dd1d7e61f3348a4894cd9d5c44242aa491db6feb028c5a','gelu768_N8')}
 a100={}
 for label,(path,pin,target) in controls.items():
  assert sha(path)==pin;codes=[x['code'] for x in rows(path)];s,o=score(codes,refs,qhash);ss,oo=score(codes,refs,qhash,aux);currentcodes=[x['code'] for x in rows(sources[target])];a100[label]={'prediction_sha256':pin,'quality_1529':s,'auxiliary_1141':ss,'exact_selected_code_rows_equal_to_T4':sum(x==y for x,y in zip(codes,currentcodes)),'paired_A100_left_T4_right':{m:paired(o[m],outcomes[target][m]) for m in METRICS},'same_step_model_identity_from_prior_receipt':label.endswith('228352'),'maturity_confounded':not label.endswith('228352')}
 report={'schema':'wide_T4_actual_wholepool_quality_audit_v1','status':'PASS_ACTUAL_HASH_ORDER_PARITY_AND_SELECTOR_REPLAY','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'audit_source_sha256':sha(__file__),'terminal_sha256':sha(B/'session_result.json'),'recovery_sha256':sha(E/'recovery.json'),'manifest_sha256':sha(T/'wide_t4_budget_manifest_a.json'),'recovered_mapped_files':79,'retained_files':80,'retained_bytes':recovery['total_bytes'],'parent_complete':terminal['complete'],'cleanup':terminal['cleanup'],'session_seconds':terminal['session_script_seconds'],'roles':audits,'all_pool_rows':totalrows,'all_actual_candidates_replayed':totalcandidates,'quality':specs,'paired_baseline_left_policy_right':comparisons,'paired_aux_baseline_left_policy_right':{k:{m:paired(auxout['selected_half415656_N24'][m],auxout[k][m]) for m in METRICS} for k in specs if k!='selected_half415656_N24'},'A100_comparisons':a100,'fixed_references_sha256':sha(refp),'auxiliary_mask_sha256':sha(maskp),'fixed_primary_n':1529,'auxiliary_supported_n':1141,'metric_source_SHA':{n:sha(R/'src/nlp_hw02'/n) for n in ['arithmetic.py','arithmetic_augassign_audit.py']},'limits':['ActualT4core/outercalltimings excludecoldmodel/parity/notebooksetup; strongestnotebook/executorclocksNULL. Wholechild includesallcounts and is not perpolicy<180proof.','512N24actualcore>180; othercounts are budgetscreen candidates only, no finalqualification/canonicalpromotion.','Frozen noisy1529consensus unchanged; all3diagnostics and fixedaux1141 are not officialpass1. Gemini mask precisionUNKNOWN.','A100same512model can show backend numericdifferences; 76852224 vs currentT4checkpoint includes maturity confound, no causalarchitectureclaim.','No generatedPythonexecution/model/weights/arraysread or newselectors/countgrids. Testquestiononlyread for order; no testcorrectnessscores.']};out=E/'independent_whole_audit.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'audit_sha256':sha(out),'pool_rows':totalrows,'candidates':totalcandidates,'seconds':time.monotonic()-START,'status':report['status']},indent=2),flush=True)
if __name__=='__main__':main()
