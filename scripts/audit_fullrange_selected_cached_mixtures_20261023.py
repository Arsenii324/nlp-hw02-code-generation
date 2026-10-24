"""Exactly4 predeclared cachedN24 subsets; restrictedAST scoring, no generated exec."""
from pathlib import Path
from datetime import datetime,timezone
from functools import lru_cache
from collections import Counter
import json,hashlib,math,ast,sys
R=Path(__file__).resolve().parents[1];W=R/'.worktrees/strong-inference-fallback-20261022';plan=R/'docs/plans/2026-10-23-fullrange-selected-cached-mixture.md';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();j=lambda p:json.loads(p.read_text());lines=lambda p:[json.loads(v) for v in p.read_text().splitlines()];checks={}
def ck(k,v):
 checks[k]=bool(v)
 assert v,k
sys.path.insert(0,str(R/'src'))
from nlp_hw02.arithmetic import evaluate_code as raw
from nlp_hw02.arithmetic_augassign_audit import evaluate_code_augassign as rawaug
@lru_cache(maxsize=None)
def ev(code):return raw(code)
@lru_cache(maxsize=None)
def aug(code):
 a=ev(code);b=rawaug(code);assert not a.supported or b.supported and a.value==b.value;return b
source=W/'src/nlp_hw02/sampling_candidate.py';ck('selector_arithmetic_pins',sha(source)=='250d34bee2dc60847a65a7d5a784fac261a77db153a4d49f0687a82fe5a98fe5' and sha(R/'src/nlp_hw02/arithmetic.py')=='e01078307ec395b41afe03b588e014b7498ee968ed6ed2f3d13f8105c45663d6' and sha(R/'src/nlp_hw02/arithmetic_augassign_audit.py')=='fa26100d91331f2ffe684f7729ad37870c6ac53a0f8834cf6922f430908ecc16')
fn=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('equal','select_plurality')];assert len(fn)==2;ns={'math':math,'evaluate_code':ev};exec(compile(ast.Module(body=fn,type_ignores=[]),str(source),'exec'),ns);select=ns['select_plurality'];equal=ns['equal']
Q=j(W/'artifacts/kaggle/k32-budget-tools-20261023a/validation_questions.json');qh=[hashlib.sha256(v.encode()).hexdigest() for v in Q];rp=R/'reports/evidence/primary_reference_consensus.json';refs=j(rp)['groups_by_question_sha256'];ck('fixed1792_1529',len(Q)==1792 and sum(refs[q]['status']=='supported_consensus' for q in qh)==1529 and sha(rp)=='deb96602c88b983ecfbf23738d2a64192057273798548ab61a7cdf7acfc7c119')
metrics=('legacy','augassign_fixed_output','augassign_additive1e6');pools={};controls={};outcomes={};provenance={};selections={}
def score(codes):
 result={k:[] for k in metrics}
 for code,q in zip(codes,qh):
  a=ev(code);b=aug(code)
  for k,v in [('legacy',a),('augassign_fixed_output',b),('augassign_additive1e6',b)]:
   if refs[q]['status']!='supported_consensus':result[k].append(None);continue
   target=refs[q]['consensus_value'];ok=(abs(v.value-target)<=1e-6+1e-6*abs(target) if k=='augassign_additive1e6' else equal(v.value,target)) if v.supported else False;result[k].append('correct' if ok else 'wrong_numeric' if v.supported else 'unsupported')
 return result,{k:dict(Counter(x for x in a if x is not None)) for k,a in result.items()}
for label,base,role,side,expected in [('selected',R/'artifacts/kaggle/cooldown-paired-n24-v1-a-diagnostics','run-cooldown-half','cooldown_sampling_result.json',1117),('full',R/'artifacts/kaggle/fullrange-paired-n24-v1-c','run-full-remaining','fullrange_sampling_result.json',1107),('half',R/'artifacts/kaggle/fullrange-paired-n24-v1-c','run-half-remaining','fullrange_sampling_result.json',1105)]:
 t=j(base/'session_result.json');d=base/role;cp=d/'validation_n24_candidates.jsonl';pp=d/'validation_n24_predictions.jsonl';sr=d/side;s=j(sr)
 ck(label+'_actualterminal_pool_prediction_sourcehash',t['complete'] and t['artifact_hashes_complete'] and all(sha(v)==t['artifact_sha256'][str(v.relative_to(base))] for v in [cp,pp,sr]))
 pool=lines(cp);pred=lines(pp);ck(label+'_1792order24finite',len(pool)==len(pred)==1792 and all(r['row_index']==i and r['question_sha256']==qh[i] and r['samples']==24 and len(r['candidates'])==len(r['scores'])==24 and all(math.isfinite(v) for v in r['scores']) for i,r in enumerate(pool)))
 chosen=[]
 for row,pr in zip(pool,pred):
  ix,info=select(row['candidates'],row['scores']);assert ix==row['selected_index'] and all(row[k]==v for k,v in info.items()) and pr=={'code':row['candidates'][ix]};chosen.append(pr['code'])
 outcomes[label],summary=score(chosen);ck(label+'_full24_control_score',summary['legacy']['correct']==expected);controls[label]=summary;pools[label]=pool
 provenance[label]={'terminal_sha256':sha(base/'session_result.json'),'candidate_pool_sha256':sha(cp),'predictions_sha256':sha(pp),'producer_result_sha256':sha(sr),'model_sha256':s['identity']['model_sha256'],'model_source_sha256':s['identity']['model_source_sha256'],'producer_source_sha256':s['producer_sha256'],'adapter_source_sha256':s['adapter_sha256'],'selected_step':s['identity']['step'],'training_source_sha256':s['identity']['training_source_sha256']}
mixtures={};paired={}
for other in ('full','half'):
 for tag,start in [('first12',0),('last12',12)]:
  key='selected+'+other+'/'+tag;codes=[];picks=[]
  for i,(a,b) in enumerate(zip(pools['selected'],pools[other])):
   cs=a['candidates'][start:start+12]+b['candidates'][start:start+12];ss=a['scores'][start:start+12]+b['scores'][start:start+12];ix,info=select(cs,ss);codes.append(cs[ix]);picks.append({'index':ix,'source_model':'selected' if ix<12 else other,'source_index':start+ix%12,**info})
  outcomes[key],summary=score(codes);mixtures[key]={'metrics':summary,'subset_indices_each':[start,start+11],'candidate_order':['selected',other],'ordered_code_only_sha256':hashlib.sha256(b''.join((json.dumps({'code':v},ensure_ascii=False)+'\n').encode() for v in codes)).hexdigest(),'actualN12streams':False,'runtime_measured':False};selections[key]=picks;paired[key]={}
  for metric in metrics:
   a=outcomes['selected'][metric];b=outcomes[key][metric];z=Counter((x,y) for x,y in zip(a,b) if x is not None);wins=sum(v for (x,y),v in z.items() if x!='correct' and y=='correct');loss=sum(v for (x,y),v in z.items() if x=='correct' and y!='correct');n=wins+loss;paired[key][metric]={'wins':wins,'losses':loss,'net':wins-loss,'descriptive_exact_discordance_p':min(1.,2*sum(math.comb(n,k) for k in range(min(wins,loss)+1))/2**n) if n else 1.,'category_transitions':{x+'->'+y:v for (x,y),v in z.items()}}
ck('exact4_predeclared_mixtures',len(mixtures)==4)
O=R/'reports/evidence/fullrange_selected_cached_mixture_outcomes_20261023.jsonl';assert not O.exists()
with O.open('x') as stream:
 for i,q in enumerate(qh):stream.write(json.dumps({'row_index':i,'question_sha256':q,'selections':{k:v[i] for k,v in selections.items()},'categories':{k:{m:v[i] for m,v in values.items()} for k,values in outcomes.items()}})+'\n')
gains=[k for k,v in mixtures.items() if v['metrics']['legacy']['correct']>1117];x={'schema':'fullrange_selected_cached_mixture_diagnostic_v1','utc':datetime.now(timezone.utc).isoformat(),'complete':True,'checks':checks,'plan_sha256':sha(plan),'source_sha256':sha(Path(__file__)),'selector_source_sha256':sha(source),'reference_sha256':sha(rp),'provenance':provenance,'controls':controls,'mixtures':mixtures,'paired_vs_selected1117':paired,'gains_over1117':gains,'stop_if_none':not gains,'outcomes_sha256':sha(O),'limits':['Exactly4cachedN24 first/last12+12 policies retained; no actualN12stream generation, newseedreplication, latencyproof orselectorretuning.','Mixed rawsameconvention normalizedLP scores uncalibrated; exactlegacyplurality duplicates/ties unchanged; reference usedonlyafterselection forscoring.','Reusednoisy1529references/repeatedselection meansscores/p areexploratory,notofficialpass1. No filtering/generatedPythonexec/model/PT/array/provider/tests/canonical actions.']};p=R/'reports/evidence/fullrange_selected_cached_mixtures_20261023.json';assert not p.exists();p.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({'sha256':sha(p),'scores':{k:{m:z['correct'] for m,z in v['metrics'].items()} for k,v in mixtures.items()},'gains':gains,'paired':{k:v['legacy'] for k,v in paired.items()}},indent=1))
