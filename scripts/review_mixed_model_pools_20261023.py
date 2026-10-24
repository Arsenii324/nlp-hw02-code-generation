"""Independent retained-output replay; only existing bounded arithmetic interpreters."""
from pathlib import Path
from collections import Counter
from functools import lru_cache
from datetime import datetime,timezone
import ast,hashlib,json,math,sys,time
R=Path(__file__).resolve().parents[1];W=R/'.worktrees/strong-inference-fallback-20261022';START=time.monotonic();checks={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def req(v,m):
 if not v:raise ValueError(m)
def ck(n,v):checks[n]=bool(v);req(v,n)
sys.path.insert(0,str(R/'src'))
from nlp_hw02.arithmetic import evaluate_code as rawlegacy
from nlp_hw02.arithmetic_augassign_audit import evaluate_code_augassign as rawaug
@lru_cache(maxsize=None)
def legacy(s):return rawlegacy(s)
@lru_cache(maxsize=None)
def aug(s):
 a=legacy(s);b=rawaug(s);req(not a.supported or b.supported and b.value==a.value,'Aug changed supported value');return b
freezep=R/'reports/evidence/mixed_model_pool_analysis_freeze_20261023.json';F=load(freezep);ep=R/'reports/evidence/mixed_model_retained_pool_diagnostic_v2_20261023.json';E=load(ep)
ck('exact_freeze',sha(freezep)=='17f6cefe23b818439db9b1a0ff6623fdd3dd82525050ac996157a509ceb3c022');ck('all8frozenfiles',len(F['files_sha256'])==8 and all(sha(R/n)==h for n,h in F['files_sha256'].items()));ck('exact_successful_evidence',sha(ep)=='e18b8150d1b3bacecbb7dc14aa6eac0b548d8eb9140f77ef0615f885319c467a' and E['complete'] and E['all_checks_pass'] and not E['failures']);ck('all14actualinputSHA',len(E['input_sha256'])==14 and E['input_sha256']==F['actual_inputs_sha256'] and all(sha(R/n)==h for n,h in E['input_sha256'].items()));ck('all6source_refs_SHA',all(sha(R/n)==h for n,h in F['source_and_reference_sha256'].items()))
selector=W/'src/nlp_hw02/sampling_candidate.py';tree=ast.parse(selector.read_text());fn=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('equal','select_plurality')];author=ast.parse((R/'scripts/audit_mixed_model_pools_20261023b.py').read_text());ck('selector_twofunctions_exactAST',len(fn)==2 and all(ast.dump(n,include_attributes=False)==ast.dump(next(x for x in author.body if isinstance(x,ast.FunctionDef) and x.name==n.name),include_attributes=False) for n in fn));ns={'math':math,'evaluate_code':legacy};exec(compile(ast.Module(body=fn,type_ignores=[]),str(selector),'exec'),ns);select=ns['select_plurality'];equal=ns['equal']
Q=load(W/'artifacts/kaggle/k32-budget-tools-20261023a/validation_questions.json');qh=[hashlib.sha256(q.encode()).hexdigest() for q in Q];refs=load(R/'reports/evidence/primary_reference_consensus.json')['groups_by_question_sha256'];scored=[refs[q]['status']=='supported_consensus' for q in qh];outcomes=lines(R/E['ordered_outcomes_path']);ck('ordered1792fixed1529outcomes',len(qh)==len(set(qh))==len(outcomes)==1792 and sum(scored)==1529 and all(x['row_index']==i and x['question_sha256']==qh[i] and x['scored']==scored[i] for i,x in enumerate(outcomes)))
paths={'old344':(R/'artifacts/kaggle/trained-gated-sampling-v1-b','control-final'),'uniform':(R/'artifacts/kaggle/long-c-paired-v2-c-diagnostics','run-uniform-control'),'bucket':(R/'artifacts/kaggle/long-c-paired-v2-c-diagnostics','run-bucket-control')};pools={};categories={};pairs=[('old344','uniform'),('old344','bucket'),('uniform','bucket')];metrics=('legacy','augassign_fixed_output','augassign_additive1e6');summary={}
def score(name,codes,expected):
 categories[name]={m:[] for m in metrics};support={'all1792':Counter(),'scored1529':Counter()}
 for i,code in enumerate(codes):
  a=legacy(code);b=aug(code);support['all1792']['legacy_supported' if a.supported else 'legacy_unsupported']+=1
  if scored[i]:support['scored1529']['legacy_supported' if a.supported else 'legacy_unsupported']+=1
  for m,v in [('legacy',a),('augassign_fixed_output',b),('augassign_additive1e6',b)]:
   if not scored[i]:category=None
   else:
    target=refs[qh[i]]['consensus_value'];ok=(abs(v.value-target)<=1e-6+1e-6*abs(target) if m==metrics[2] else equal(v.value,target)) if v.supported else False;category='correct' if ok else 'wrong_numeric' if v.supported else 'unsupported'
   categories[name][m].append(category);req(outcomes[i]['categories'][name][m]==category,'Actual perrow diagnostic differs '+name)
 counts={m:dict(Counter(x for x in values if x is not None)) for m,values in categories[name].items()};req(counts==expected['quality'] and {k:dict(v) for k,v in support.items()}==expected['selected_support'],'Quality/support summary differs')
 projection=hashlib.sha256(b''.join((json.dumps({'code':code},ensure_ascii=False)+'\n').encode() for code in codes)).hexdigest();req(projection==expected['ordered_code_only_projection_sha256'],'Selected rawcode projection differs');summary[name]=counts
for name,(base,role) in paths.items():
 terminal=load(base/'session_result.json');d=base/role;pool=lines(d/'validation_n24_candidates.jsonl');pred=lines(d/'validation_n24_predictions.jsonl');codes=[]
 req(len(pool)==len(pred)==1792,'Original1792')
 for p in [d/'validation_n24_candidates.jsonl',d/'validation_n24_predictions.jsonl']:req(sha(p)==terminal['artifact_sha256'][str(p.relative_to(base))],'Terminal map actual input')
 for i,(row,pr) in enumerate(zip(pool,pred)):
  req(row['row_index']==i and row['question_sha256']==qh[i] and row['samples']==24,'Original question/order')
  j,info=select(row['candidates'],row['scores']);req(j==row['selected_index']==outcomes[i]['selections'][name]['selected_index'] and all(row[k]==v for k,v in info.items()) and set(pr)=={'code'} and pr['code']==row['candidates'][j],'Original selector replay');codes.append(pr['code'])
 pools[name]=pool;score(name,codes,E['controls'][name]);ck('original_full24_'+name,True)
for left,right in pairs:
 pair=left+'+'+right;fullunion=sum(categories[left]['legacy'][i]=='correct' or categories[right]['legacy'][i]=='correct' for i in range(1792) if scored[i]);req(fullunion==E['complementarity'][pair]['original_full24_selected_oracle_union_correct'],'Full24oracle union')
 for half,start in [('first12',0),('last12',12)]:
  label=pair+'/'+half;expect=E['mixtures'][label];codes=[];agreement={'all1792':Counter(),'scored1529':Counter()};support=Counter();duplicates=Counter();oracle=Counter();caps=Counter()
  req(expect['models']==[left,right] and expect['subset_indices_in_each_model']==[start,start+11] and expect['candidate_order']=='left12 then right12' and expect['total_candidates']==43008 and expect['actual_N12_generation'] is expect['latency_measured'] is False,'Predeclared mixture')
  for i,(a,b) in enumerate(zip(pools[left],pools[right])):
   la=a['candidates'][start:start+12];rb=b['candidates'][start:start+12];ls=a['scores'][start:start+12];rs=b['scores'][start:start+12];mixed=la+rb;scores=ls+rs;j,info=select(mixed,scores);lj,_=select(la,ls);rj,_=select(rb,rs);lv=legacy(la[lj]);rv=legacy(rb[rj]);winner=legacy(mixed[j]);lvals=[legacy(x) for x in la];rvals=[legacy(x) for x in rb];lcount=sum(v.supported and equal(v.value,winner.value) for v in lvals) if winner.supported else 0;rcount=sum(v.supported and equal(v.value,winner.value) for v in rvals) if winner.supported else 0
   actual={'selected_index':j,'source_model':left if j<12 else right,'source_index':start+j%12,'left_subset_selected_index':start+lj,'right_subset_selected_index':start+rj,'left_winning_group_votes':lcount,'right_winning_group_votes':rcount,**info};req(actual==outcomes[i]['selections'][label] and lcount+rcount==info['winning_votes'],'Perrow source/index/group support differs')
   reason='both_supported_agree' if lv.supported and rv.supported and equal(lv.value,rv.value) else 'both_supported_disagree' if lv.supported and rv.supported else 'left_only_supported' if lv.supported else 'right_only_supported' if rv.supported else 'neither_supported';agreement['all1792'][reason]+=1
   if scored[i]:agreement['scored1529'][reason]+=1
   support['winning_group_supported_by_both_models' if lcount and rcount else 'winning_group_left_only' if lcount else 'winning_group_right_only' if rcount else 'no_supported_winner']+=1;support['winning_votes_left_total']+=lcount;support['winning_votes_right_total']+=rcount;support['supported_candidate_total']+=info['supported_candidates'];support['numeric_group_total']+=info['numeric_groups'];duplicates['duplicate_votes_beyond_unique_programs']+=24-len(set(mixed));duplicates['cross_model_shared_unique_programs']+=len(set(la)&set(rb));duplicates['rows_with_cross_model_identical_program']+=bool(set(la)&set(rb));cap=a['capped'][start:start+12]+b['capped'][start:start+12];caps['all_candidate_capped']+=sum(cap);caps['selected_capped']+=cap[j]
   if scored[i]:
    target=refs[qh[i]]['consensus_value'];lc=lv.supported and equal(lv.value,target);rc=rv.supported and equal(rv.value,target);oracle['left_subset12_selected_correct']+=lc;oracle['right_subset12_selected_correct']+=rc;oracle['either_subset12_selected_correct']+=lc or rc;oracle['both_subset12_selected_correct']+=lc and rc;oracle['any_mixed24_candidate_correct']+=any(v.supported and equal(v.value,target) for v in lvals+rvals)
   codes.append(mixed[j])
  oracle['either_original_full24_selected_correct']=fullunion;req(dict(oracle)==expect['bounded_oracles'] and dict(support)==expect['winner_support'] and dict(duplicates)==expect['duplicate_vote_diagnostics'] and dict(caps)==expect['caps'] and {k:dict(v) for k,v in agreement.items()}==expect['numeric_agreement'],'Mixture diagnostic summary differs '+label);score(label,codes,expect);ck('whole1792_mixture_'+label,True)
for label,controls in E['paired'].items():
 for control,expected in controls.items():
  for m,e in expected.items():
   c=Counter((a=='correct',b=='correct') for a,b in zip(categories[control][m],categories[label][m]) if a is not None);wins=c[False,True];loss=c[True,False];n=wins+loss;p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,loss)+1))/2**n) if n else 1.;req(e=={'rows':1529,'wins':wins,'losses':loss,'net':wins-loss,'both_correct':c[True,True],'both_not_correct':c[False,False],'descriptive_exact_paired_binomial_p':p},'Paired exact counts differ')
ck('all_paired_counts_and_exact_p',True)
qualifying=[]
for a,b in pairs:
 pair=a+'+'+b;strong=max(summary[a]['legacy']['correct'],summary[b]['legacy']['correct']);counts=[summary[pair+'/'+half]['legacy']['correct'] for half in ['first12','last12']]
 if min(counts)>strong:qualifying.append(pair)
ck('predeclared_gate_only_old344_uniform',qualifying==['old344+uniform'] and [x['pair'] for x in E['predeclared_consistency_gate_qualifying_pairs']]==qualifying and len(E['mixtures'])==6)
ck('failed_attempt_and_success_outcomes_identical',sha(R/'reports/evidence/mixed_model_retained_pool_diagnostic_20261023_outcomes.jsonl')==sha(R/E['ordered_outcomes_path'])=='c29a5240860a3db2aed56e4eb8dfaea6133cb77804c150619100b189db7a3894')
ck('source_A13_and_externalB_freezes_preserved',sha(W/'configs/full_cache_transition_source_a_20261023.json')==F['full_cache_A_freeze_unchanged'] and sha(W/'configs/full_cache_external_source_b_20261023.json')==F['external_full_cache_B_freeze_unchanged'])
D={'schema':'mixed_model_retained_pool_independent_review_v1','utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_WHOLE_ACTUAL_SIX_MIXTURE_REPLAY_NO_MATERIAL_FINDING','checks':checks,'check_count':len(checks),'all_evidence_checks_pass':True,'author_freeze_sha256':sha(freezep),'author_evidence_sha256':sha(ep),'author_outcomes_sha256':sha(R/E['ordered_outcomes_path']),'source_audit_path':str(Path(__file__).relative_to(R)),'source_audit_sha256':sha(Path(__file__)),'quality_recomputed':summary,'replayed_original_rows':5376,'replayed_original_candidate_slots':129024,'replayed_mixture_rows':10752,'replayed_mixture_candidate_slots':258048,'predeclared_gate_qualifying_pairs':qualifying,'findings':[],'whole_source_review':['Read entire author audit and predeclared six-combination plan; exact frozen selector AST unchanged, left12+right12 rawscore/rawprogram duplicate votes and stable ties/fallback preserved.','Selector is called before reference-based scoring/oracles. No reference-dependent model pairing/subset/weight/index choice enters the six policies; target references used only offline metrics/oracle bounds.','All nine policies three diagnostic perrow categories and selected source/index/support metadata independently replayed. All numeric agreement, duplicate counts, winner contributions, support/caps, selected rawcode projection, paired counts/p and oracle bounds agree.','First attempt failure is bookkeeping path normalization; preserved outcome bytes equal successor; direct source diff only adds resolve() before relative_to.','Report interpretation correctly calls gains weak and halves non-independent; no actualN12/runtime/180proof or officialpass1 claim.'],'recommendation':'Old344+uniform is the only pair meeting the predeclared two-positive-halves gate, but +5/+3 noisy repeated-validation gains do not independently justify spending GPU quota. Retain current proved canonical; actualN12+N12 generation and complete oneT4 loading/memory/selection/serialization timing and fresh proof would be separate root allocation decision.','limits':['Disjoint first/last12 subsets of one retainedN24 call are not independent seeds or actualN12 RNG/tensor trajectories. Both halves depend on reused models/questions/reference pool.','Raw normalized log scores are not calibrated across models; diagnostic preserves original convention, establishes no probabilistic calibration.','1147 original full24 selected union uses48 candidate slots; other oracle values use labels and are not deployed selection policies.','Three existing diagnostic evaluators processed retained generated text with restricted AST arithmetic; no generated program eval/exec, model imports/weights/generation, trainingdata change, tests/fixtures, cloud/provider/download/canonical/source edits.','Source-freeze/actual recovery input files retained and unmodified; all six outcomes reported, no new grid or policy variants added.'],'audit_seconds':time.monotonic()-START}
out=R/'reports/evidence/mixed_model_retained_pool_independent_review_20261023.json';req(not out.exists(),'Preserve frozen evidence');out.write_text(json.dumps(D,indent=2)+'\n');print(json.dumps({'path':str(out.relative_to(R)),'sha256':sha(out),'checks':len(checks),'legacy':{k:v['legacy']['correct'] for k,v in summary.items()},'audit_seconds':D['audit_seconds']},indent=2))
