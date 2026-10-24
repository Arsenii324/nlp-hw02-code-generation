"""Actual fullrange retained diagnostics; restricted AST arithmetic only, no generated Python execution."""
from pathlib import Path
from collections import Counter
from functools import lru_cache
from datetime import datetime,timezone
import hashlib,json,math,sys,time
R=Path(__file__).resolve().parents[1];B=R/'artifacts/kaggle/fullrange-cooldown-v1-c-diagnostics-retry1';START=time.monotonic();checks={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def require(v,m):
 if not v:raise ValueError(m)
def check(k,v):checks[k]=bool(v);require(v,k)
t=load(B/'session_result.json');amap=t['artifact_sha256']
check('actualterminalexact',sha(B/'session_result.json')=='18647c3cb0bb0fe263cdb8b3e741573c0b2d758babda00d2807999f3bf830afc')
# Frozen actual predictions and existing restricted AST numerical diagnostics only.
sys.path.insert(0,str(R/'src'))
from nlp_hw02.arithmetic import evaluate_code as rawlegacy
from nlp_hw02.arithmetic_augassign_audit import evaluate_code_augassign as rawaug
@lru_cache(maxsize=None)
def legacy(s):return rawlegacy(s)
@lru_cache(maxsize=None)
def aug(s):
 a=legacy(s);b=rawaug(s);require(not a.supported or b.supported and b.value==a.value,'Aug changed legacy supported value');return b
def equal(a,b):return a==b if isinstance(a,int) and isinstance(b,int) else math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9)
refsfile=R/'reports/evidence/primary_reference_consensus.json';refs=load(refsfile)['groups_by_question_sha256'];check('immutable_refs_evaluators',sha(refsfile)=='deb96602c88b983ecfbf23738d2a64192057273798548ab61a7cdf7acfc7c119' and sha(R/'src/nlp_hw02/arithmetic.py')=='e01078307ec395b41afe03b588e014b7498ee968ed6ed2f3d13f8105c45663d6' and sha(R/'src/nlp_hw02/arithmetic_augassign_audit.py')=='fa26100d91331f2ffe684f7729ad37870c6ac53a0f8834cf6922f430908ecc16')
Q=load(R/'.worktrees/strong-inference-fallback-20261022/artifacts/kaggle/k32-budget-tools-20261023a/validation_questions.json');qh=[hashlib.sha256(q.encode()).hexdigest() for q in Q];check('same1792fixed1529question_order',len(Q)==1792 and sum(refs[q]['status']=='supported_consensus' for q in qh)==1529)
quality={};outcomes={};old=R/'artifacts/kaggle/cooldown-prefix-v1-b-diagnostics';ot=load(old/'session_result.json')
for label,base,prefix,terminalmap in [('full',B,'evaluation-full-remaining',amap),('half',B,'evaluation-half-remaining',amap),('selected-half',old,'evaluation-cooldown-half',ot['artifact_sha256'])]:
 for mode in ['greedy','beam4-alpha06']:
  name=label+'-'+mode;p=base/prefix/mode/'validation_predictions.jsonl';pred=rows(p);check(name+'_actualSHA_all1792exactorder',sha(p)==terminalmap[str(p.relative_to(base))] and len(pred)==1792 and [x['question'] for x in pred]==Q and [x['question_sha256'] for x in pred]==qh)
  cats={k:[] for k in ['legacy','augassign_fixed_output','augassign_additive1e6']};support=Counter()
  for row,q in zip(pred,qh):
   a=legacy(row['prediction']);b=aug(row['prediction']);support['legacy_supported']+=a.supported;support['augassign_supported']+=b.supported
   for key,v in [('legacy',a),('augassign_fixed_output',b),('augassign_additive1e6',b)]:
    if refs[q]['status']!='supported_consensus':cats[key].append(None);continue
    target=refs[q]['consensus_value'];ok=(abs(v.value-target)<=1e-6+1e-6*abs(target) if key=='augassign_additive1e6' else equal(v.value,target)) if v.supported else False;cats[key].append('correct' if ok else 'wrong_numeric' if v.supported else 'unsupported')
  counts={k:dict(Counter(v for v in x if v is not None)) for k,x in cats.items()};check(name+'_all1529categories',all(sum(z.values())==1529 for z in counts.values()));quality[name]={'prediction_sha256':sha(p),'metrics':counts,'all1792support':dict(support)};outcomes[name]=cats
paired={}
for mode in ['greedy','beam4-alpha06']:
 for left,right in [('selected-half','full'),('selected-half','half'),('full','half')]:
  a=left+'-'+mode;b=right+'-'+mode;key=a+' vs '+b;paired[key]={}
  for metric in outcomes[a]:
   c=Counter((x,y) for x,y in zip(outcomes[a][metric],outcomes[b][metric]) if x is not None);wins=sum(v for (x,y),v in c.items() if x!='correct' and y=='correct');loss=sum(v for (x,y),v in c.items() if x=='correct' and y!='correct');n=wins+loss;pval=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,loss)+1))/2**n) if n else 1.;comparison={'left_only_correct':loss,'right_only_correct':wins,'descriptive_exact_discordance_p':pval,'rows':1529};require(comparison['rows']==1529,'Paired denominator differs');paired[key][metric]={**comparison,'category_transitions':{x+'->'+y:v for (x,y),v in c.items()}}
check('all18paired_diagnostics_exact',True)

check('selectedhalfbeam1037_notN24_1117',quality['selected-half-beam4-alpha06']['metrics']['legacy']['correct']==1037)
report={'schema':'fullrange_cooldown_pilot_c_quality_independent_v1','utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_ACTUAL_RETAINED_PREDICTIONS_FIXED_NUMERIC_DIAGNOSTICS','terminal_sha256':sha(B/'session_result.json'),'reference_sha256':sha(refsfile),'quality':quality,'paired':paired,'checks':checks,'check_count':len(checks),'audit_source_sha256':sha(Path(__file__)),'limits':['Same fixed original1529 supported/noisy reference-consensus subset of1792; not official pass1. Greedy/beam diagnostics, not N24.','Existing restricted AST interpreters only; generated Python is never run. AugAssign only generated-output support, additive1e-6 formula diagnostic not confirmed official tolerance.','Repeated selection and one modelseed make p descriptive. Pilot pair sharespopulation/draws; selectedhalfbaseline trained6M forlonger72384 ratherthan28480attempts, so no isolatedpopulation orscheduleeffect claim.','No model/PT/array/raw/testoutcome/provider/canonical actions; whole recovery/training/protocol review is separate and stillinprogress.'],'audit_seconds':time.monotonic()-START}
out=R/'reports/evidence/fullrange_cooldown_pilot_c_quality_independent_20261023.json';require(not out.exists(),'Preserve evidence');out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'sha256':sha(out),'checks':len(checks),'quality':{n:{k:v['correct'] for k,v in d['metrics'].items()} for n,d in quality.items()}},indent=2))
