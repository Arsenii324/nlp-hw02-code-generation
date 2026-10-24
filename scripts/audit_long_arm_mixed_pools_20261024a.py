"""Predeclared8pair/16ordering retainedfirst12+12 diagnostic. No model/code execution."""
from pathlib import Path
from collections import Counter
import ast,datetime,hashlib,json,math,sys,time
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
import audit_cooldown_mixed_pools_20261023 as frozen
from analyze_gated_results_20261023 import paired
E=R/'reports/evidence/long_arm_mixed_pools_20261024a';START=time.monotonic();METRICS=['legacy','augassign_fixed_output','augassign_additive1e6']
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def main():
 plan=read(E/'predeclaration.json');assert h(E/'predeclaration.json')=='e4ecb27193518c3bae102ca1cf8112f33a40ebaaa1c55d5b95fba764ffb81efe' and len(plan['pairs'])==8 and plan['candidate_indices_per_parent']==list(range(12))
 selector=R/'artifacts/kaggle/wide-t4-budget-20261024a/tools/inputs/sampling_candidate.py';assert h(selector)=='250d34bee2dc60847a65a7d5a784fac261a77db153a4d49f0687a82fe5a98fe5';old=ast.parse(selector.read_text());new=ast.parse(Path(frozen.__file__).read_text());fn=lambda t,n:next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name==n)
 assert all(ast.dump(fn(old,n),include_attributes=False)==ast.dump(fn(new,n),include_attributes=False) for n in ['equal','select_plurality'])
 assert h(R/'src/nlp_hw02/arithmetic.py')=='e01078307ec395b41afe03b588e014b7498ee968ed6ed2f3d13f8105c45663d6' and h(R/'src/nlp_hw02/arithmetic_augassign_audit.py')=='fa26100d91331f2ffe684f7729ad37870c6ac53a0f8834cf6922f430908ecc16'
 qpath=selector.parent/'validation_questions.json';assert h(qpath)=='83ea3ea9f449619e015d98e21b1c35ec2cc6d2edd01db0c026301e718014f04e';questions=read(qpath);qh=[hashlib.sha256(q.encode()).hexdigest() for q in questions];assert len(qh)==len(set(qh))==1792
 rp=R/'reports/evidence/primary_reference_consensus.json';assert h(rp)=='deb96602c88b983ecfbf23738d2a64192057273798548ab61a7cdf7acfc7c119';refs=read(rp)['groups_by_question_sha256'];supported={q for q in qh if refs[q]['status']=='supported_consensus'};assert len(supported)==1529
 mp=R/'reports/evidence/gemini_reference_screen_20261024/auxiliary_reference_screen_mask.json';assert h(mp)=='0e16c82bc9711fa3c71e56a74dac09c7d4bbde1132397dc139d697dbf9b434e2';mask=read(mp);assert mask['question_sha256_order']==qh and mask['complete_all14_calls'];aux={qh[i] for i in mask['retained_global_indices']}&supported;assert len(aux)==1141
 groups={'six':('six-m-long-n24-v1-a-diagnostics-retry1','f34045407356dcad7b73687a5b19c78c2f3d12e5fa33201908e7a05c57abdcb1',{'six_full':'cooldown-full','six_half':'cooldown-half'},'cooldown_sampling_result.json'),'range':('fullrange-long-n24-v1-a-diagnostics','392696ad5ef00cacc206330adc7a0a511ba7fb807ae2436f91987835f91ce5de',{'range_full':'full-remaining','range_half':'half-remaining'},'fullrange_sampling_result.json'),'selected':('cooldown-paired-n24-v1-a-diagnostics','5accb52bf6b291f1b1d1db677a83bf7086aa37842dc91b205d4d6c2251c2b3b9',{'selected_half415656':'cooldown-half'},'cooldown_sampling_result.json')}
 pools={};identities={};pins={};controlcodes={};controlindices={}
 for _,(directory,tpin,labels,resultname) in groups.items():
  base=R/'artifacts/kaggle'/directory;terminal=read(base/'session_result.json');assert h(base/'session_result.json')==tpin and terminal['complete'] and terminal['artifact_hashes_complete'] and all(x['reaped'] and x['group_exited'] and not x.get('errors') for x in terminal['cleanup']);assert all(h(base/n)==v for n,v in terminal['artifact_sha256'].items());pins[str((base/'session_result.json').relative_to(R))]=tpin
  for label,run in labels.items():
   child=next(c for c in terminal['children'] if c['run_id']==run);assert child['complete'] and child['actual_exit']==0;out=base/child['output'];receipt=read(out/resultname);identity=child['worker_identity'];assert receipt['complete'] and receipt['parity_passed_before_sampling'] and not receipt['training'] and not receipt['gold_used_for_selection']
   p=out/'validation_n24_candidates.jsonl';predp=out/'validation_n24_predictions.jsonl';timingp=out/'validation_n24_timing.json';timing=read(timingp)
   for x in [p,predp,timingp,out/resultname]:assert h(x)==terminal['artifact_sha256'][str(x.relative_to(base))];pins[str(x.relative_to(R))]=h(x)
   assert timing['candidate_sha256']==h(p) and timing['prediction_sha256']==h(predp) and all(timing[k]==v for k,v in {'rows':1792,'samples_per_question':24,'total_candidates':43008,'temperature':.8,'top_k':50,'question_batch_size':32,'seed':20261002,'length_penalty':.6,'max_new_tokens':192}.items())
   assert timing['ordered_questions_sha256']==hashlib.sha256(json.dumps(questions,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
   pool=rows(p);pred=rows(predp);assert len(pool)==len(pred)==1792;codes=[];idxs=[];caps=selectedcaps=0
   for i,(row,pr,q) in enumerate(zip(pool,pred,qh)):
    assert row['row_index']==i and row['question_sha256']==q and row['samples']==24 and all(len(row[k])==24 for k in ['candidates','scores','capped','generated_lengths']) and all(math.isfinite(s) for s in row['scores'])
    assert all(type(c)is bool and type(l)is int and 0<=l<=192 and c==(l==192) for c,l in zip(row['capped'],row['generated_lengths']))
    j,info=frozen.select_plurality(row['candidates'],row['scores']);assert j==row['selected_index'] and info=={k:row[k] for k in info} and set(pr)=={'code'} and pr['code']==row['candidates'][j];codes.append(pr['code']);idxs.append(j);caps+=sum(row['capped']);selectedcaps+=row['capped'][j]
   assert caps==timing['all_candidate_capped_rows'] and selectedcaps==timing['selected_candidate_capped_rows'];pools[label]=pool;controlcodes[label]=codes;controlindices[label]=idxs;identities[label]={k:identity.get(k) for k in ['model_sha256','model_source_sha256','step','validation_loss','absolute_completed_step','admitted_NEW_horizon','training_terminal_sha256','training_source_sha256','training_driver_sha256','layout','counter_origin']};identities[label]['test_clock_not_mix_latency']=child.get('worker_paths',{}).get('test_n24',{}).get('complete_path_seconds')
 def score(codes):
  out={};outcomes={}
  for m in METRICS:
   o={};allcount=Counter();ac=Counter();halves=[Counter(),Counter()];auxhalves=[Counter(),Counter()];support=Counter()
   for i,(q,c) in enumerate(zip(qh,codes)):
    v=frozen.evaluate_code(c) if m=='legacy' else frozen.aug(c);support['supported' if v.supported else 'unsupported']+=1
    if q not in supported:continue
    target=refs[q]['consensus_value'];ok=v.supported and (abs(v.value-target)<=1e-6+1e-6*abs(target) if m=='augassign_additive1e6' else frozen.equal(v.value,target));cat='correct' if ok else 'wrong_numeric' if v.supported else 'unsupported';o[q]=bool(ok);allcount[cat]+=1;halves[i//896][cat]+=1
    if q in aux:ac[cat]+=1;auxhalves[i//896][cat]+=1
   assert sum(allcount.values())==1529 and sum(ac.values())==1141;outcomes[m]=o;out[m]={'primary1529':dict(allcount),'aux1141':dict(ac),'primary_ordered_halves':[dict(x) for x in halves],'aux_ordered_halves':[dict(x) for x in auxhalves],'selected_support_all1792':dict(support)}
  return out,outcomes
 controls={};outcomes={};selection_records={};mixtures={}
 for name,codes in controlcodes.items():controls[name],outcomes[name]=score(codes);selection_records[name]=[{'selected_source_index':i,'code_sha256':hashlib.sha256(c.encode()).hexdigest()} for i,c in zip(controlindices[name],codes)]
 assert all(controls['selected_half415656'][m]['primary1529']['correct']==1117 for m in METRICS)
 print(json.dumps({'controls':{k:{m:v[m]['primary1529'].get('correct',0) for m in METRICS} for k,v in controls.items()}},indent=2),flush=True)
 for a,b in plan['pairs']:
  for left,right in [(a,b),(b,a)]:
   name=left+'__then__'+right;codes=[];meta=[];support=Counter();duplicates=Counter();caps=Counter()
   for i,(x,y) in enumerate(zip(pools[left],pools[right])):
    candidates=x['candidates'][:12]+y['candidates'][:12];scores=x['scores'][:12]+y['scores'][:12];index,info=frozen.select_plurality(candidates,scores);chosen=candidates[index];codes.append(chosen);v=frozen.evaluate_code(chosen)
    votes=[sum(w.supported and frozen.equal(w.value,v.value) for w in map(frozen.evaluate_code,z)) if v.supported else 0 for z in [x['candidates'][:12],y['candidates'][:12]]];assert sum(votes)==info['winning_votes'];support['both_models_vote_for_winner' if all(votes) else 'left_only_winner' if votes[0] else 'right_only_winner' if votes[1] else 'no_supported_winner']+=1;support['supported_candidate_slots']+=info['supported_candidates'];support['numeric_groups_total']+=info['numeric_groups'];duplicates['duplicate_program_slots']=duplicates.get('duplicate_program_slots',0)+24-len(set(candidates));duplicates['cross_model_shared_unique_programs']+=len(set(x['candidates'][:12])&set(y['candidates'][:12]));caps['all_candidate_capped']+=sum(x['capped'][:12])+sum(y['capped'][:12]);caps['selected_capped']+=(x['capped'][:12]+y['capped'][:12])[index]
    meta.append({'selected_mixed_index':index,'source':left if index<12 else right,'source_index':index%12,'code_sha256':hashlib.sha256(chosen.encode()).hexdigest(),'votes_left':votes[0],'votes_right':votes[1],**info})
   quality,outcomes[name]=score(codes);selection_records[name]=meta
   comparisons={control:{m:paired(outcomes[control][m],outcomes[name][m]) for m in METRICS} for control in {left,right,'selected_half415656'}}
   mixtures[name]={'parents':[left,right],'subset':'first12each','quality':quality,'paired_controls_left_mixture_right':comparisons,'winner_support':dict(support),'duplicates':dict(duplicates),'caps':dict(caps),'ordered_selected_code_digest':hashlib.sha256(b''.join((json.dumps({'code':c},ensure_ascii=False)+'\n').encode() for c in codes)).hexdigest(),'actual_N12_stream':False,'latency_measured':False}
   print(json.dumps({'mixture':name,'legacy':quality['legacy']['primary1529'].get('correct',0),'aux':quality['legacy']['aux1141'].get('correct',0),'baseline_pair':comparisons['selected_half415656']['legacy']}),flush=True)
 order_effects={}
 for a,b in plan['pairs']:
  x=a+'__then__'+b;y=b+'__then__'+a;order_effects[a+'+'+b]={'different_selected_code_rows':sum(p['code_sha256']!=q['code_sha256'] for p,q in zip(selection_records[x],selection_records[y])),'different_legacy_outcomes':sum(outcomes[x]['legacy'][q]!=outcomes[y]['legacy'][q] for q in supported)}
 op=E/'ordered_outcomes.jsonl';assert not op.exists()
 with op.open('x') as f:
  for i,q in enumerate(qh):f.write(json.dumps({'row_index':i,'question_sha256':q,'supported_reference':q in supported,'auxiliary':q in aux,'selections':{k:v[i] for k,v in selection_records.items()},'correct':{k:{m:v[m].get(q) for m in METRICS} for k,v in outcomes.items()}},allow_nan=False)+'\n')
 base=controls['selected_half415656'];halves_population=[sum(q in supported for q in qh[:896]),sum(q in supported for q in qh[896:])];auxhalves_population=[sum(q in aux for q in qh[:896]),sum(q in aux for q in qh[896:])]
 # Predeclared qualitative decision aid, not another policy: both orderings/allmetrics and both orderedhalves must avoid a net regression.
 robust=[]
 for a,b in plan['pairs']:
  variants=[mixtures[a+'__then__'+b],mixtures[b+'__then__'+a]]
  if all(v['quality'][m]['primary1529'].get('correct',0)>1117 and all(v['quality'][m]['primary_ordered_halves'][i].get('correct',0)>=base[m]['primary_ordered_halves'][i].get('correct',0) for i in [0,1]) for v in variants for m in METRICS):robust.append([a,b])
 report={'schema':'long_arm_cached_first12plus12_diagnostic_v1','complete':True,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'predeclaration_sha256':h(E/'predeclaration.json'),'audit_source_sha256':h(__file__),'exact_selector_AST':True,'input_SHA':pins,'evaluator_SHA':{n:h(R/'src/nlp_hw02'/n) for n in ['arithmetic.py','arithmetic_augassign_audit.py']},'selector_sha256':h(selector),'questions_sha256':h(qpath),'reference_sha256':h(rp),'aux_mask_sha256':h(mp),'identities':identities,'control_candidates_replayed':5*1792*24,'mixture_candidate_slots':16*1792*24,'pairs':8,'fixed_ordered_variants':16,'controls':controls,'mixtures':mixtures,'order_effects':order_effects,'primary_halves_denominators':halves_population,'aux_halves_denominators':auxhalves_population,'robust_both_orders_allmetrics_no_half_regression_pairs':robust,'ordered_outcomes_sha256':h(op),'elapsed_CPU_seconds':time.monotonic()-START,'limits':['No actualN12stream,newseed,latency,inferencehardwareor180squalification. First12retainedN24 are not an independent modelrun.','Only originalrawscores/duplicatevotes/exactstablefrozenplurality. No last12grid/reweighting/selector tuning.','Fixednoisy1529primary andGemini1141aux not officialpass1/correctnessgold. Orderedhalves diagnostic, no adaptivefilter.','ParentselectedCEbest steps andtrainingbudgets differ. No causalclaim forpopulations/LRenvelopes; rawstreamsnotassumedidentical acrosspopulations.','No model/PT/arrays/provider calls/generatedPython execution/newtargets/maskapplication/canonical changes.','Observedbestscore isselectedfrom16predeclared variants; descriptive pairedp values do not correct repeatedvalidation.']};out=E/'actual_audit.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'audit_sha256':h(out),'elapsed_seconds':report['elapsed_CPU_seconds'],'robust_pairs':robust,'top_legacy':sorted([(k,v['quality']['legacy']['primary1529'].get('correct',0)) for k,v in mixtures.items()],key=lambda x:x[1],reverse=True)},indent=2),flush=True)
if __name__=='__main__':main()
