"""Root numeric audit of completed actual pools; generated Python is never executed."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
from functools import lru_cache
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from nlp_hw02.arithmetic import evaluate_code
from nlp_hw02.arithmetic_augassign_audit import evaluate_code_augassign
from analyze_gated_results_20261023 import equal,paired

def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]

def main():
 b=ROOT/'artifacts/kaggle/long-c-paired-v2-c-diagnostics';t=read(b/'session_result.json')
 assert sha(b/'session_result.json')=='3706104d23c5f5eb07ad27d32533df24386530bc892469514c18d738bacae1d2' and t['complete'] and t['artifact_hashes_complete']
 assert all(r['reaped'] and r['group_exited'] and not r['errors'] for r in t['cleanup'])
 qp=ROOT/'.worktrees/strong-inference-fallback-20261022/artifacts/trained-gated-sampling-inputs-20261023a/validation_questions.json';assert sha(qp)=='83ea3ea9f449619e015d98e21b1c35ec2cc6d2edd01db0c026301e718014f04e'
 questions=read(qp);qhash=[hashlib.sha256(q.encode()).hexdigest() for q in questions];assert len(qhash)==len(set(qhash))==1792
 ref=ROOT/'reports/evidence/primary_reference_consensus.json';assert sha(ref)=='deb96602c88b983ecfbf23738d2a64192057273798548ab61a7cdf7acfc7c119';refs=read(ref)['groups_by_question_sha256']
 specifications={}
 for run in t['children']:
  assert run['complete'] and run['actual_exit']==0
  path=b/run['output']/'validation_n24_predictions.jsonl';pool=b/run['output']/'validation_n24_candidates.jsonl';timing=run['worker_paths']['validation_n24']
  assert sha(path)==timing['prediction_sha256']==t['artifact_sha256'][str(path.relative_to(b))] and sha(pool)==timing['candidate_sha256']==t['artifact_sha256'][str(pool.relative_to(b))]
  pred=rows(path);candidates=rows(pool);assert len(pred)==len(candidates)==1792
  for i,(pr,ca,q) in enumerate(zip(pred,candidates,qhash)):
   assert set(pr)=={'code'} and isinstance(pr['code'],str) and ca['row_index']==i and ca['question_sha256']==q and ca['samples']==24 and len(ca['candidates'])==24 and pr['code']==ca['candidates'][ca['selected_index']]
  specifications[run['run_id']]={'codes':[x['code'] for x in pred],'prediction_sha256':sha(path),'model_sha256':run['worker_identity']['model_sha256'],'selected_step':run['worker_identity']['step'],'test_complete_path_seconds':run['worker_paths']['test_n24']['complete_path_seconds']}
 oldbase=ROOT/'artifacts/kaggle/trained-gated-sampling-v1-b';ot=read(oldbase/'session_result.json');p=oldbase/'control-final/validation_n24_predictions.jsonl';assert sha(p)==ot['artifact_sha256'][str(p.relative_to(oldbase))]
 old=ot['roles']['control-final'];assert old['checkpoint_sha256']=='eed1fdbb4f7e056536b322c08fad8660f5df24460f716a4d8e5382eb58ce8dc9' and old['selected_step']==344000
 specifications['old344']={'codes':[x['code'] for x in rows(p)],'prediction_sha256':sha(p),'model_sha256':old['checkpoint_sha256'],'selected_step':344000}
 evaluators={'legacy':(lru_cache(None)(evaluate_code),False),'augassign_fixed_output':(lru_cache(None)(evaluate_code_augassign),False),'augassign_additive1e6':(lru_cache(None)(evaluate_code_augassign),True)}
 scores={};outcomes={}
 for label,spec in specifications.items():
  scores[label]={k:v for k,v in spec.items() if k!='codes'};scores[label]['metrics']={};outcomes[label]={}
  assert len(spec['codes'])==1792
  for metric,(evaluate,additive) in evaluators.items():
   categories=Counter();current={}
   for q,code in zip(qhash,spec['codes']):
    if refs[q]['status']!='supported_consensus':continue
    value=evaluate(code);ok=bool(value.supported and equal(value.value,refs[q]['consensus_value'],additive));current[q]=ok;categories['correct' if ok else 'wrong_numeric' if value.supported else 'unsupported']+=1
   assert len(current)==1529;scores[label]['metrics'][metric]=dict(categories);outcomes[label][metric]=current
 assert scores['old344']['metrics']['legacy']['correct']==1099
 comparisons={}
 for a,c in [('old344','uniform-control'),('old344','bucket-control'),('uniform-control','bucket-control')]:comparisons[a+' vs '+c]={m:paired(outcomes[a][m],outcomes[c][m]) for m in evaluators}
 report={'schema':'long_c_actual_fixed24_root_numeric_audit_v1','utc':datetime.now(timezone.utc).isoformat(),'terminal_sha256':sha(b/'session_result.json'),'script_sha256':sha(__file__),'reference_sha256':sha(ref),'scores':scores,'paired':comparisons,'evaluator_sha256':{n:sha(ROOT/'src/nlp_hw02'/n) for n in ('arithmetic.py','arithmetic_augassign_audit.py')},'limits':['Fixed reused noisy1529 supplied-reference consensus, not officialpass1 or unbiased model-selection significance.','Different NEWdraw budgets; matched decoder/generation policy, no causal bucketing effect.','Generated output AST only, no code execution/reference rewrite/filtering/new targets/test outcomes.','1919 component complete-path timing excludes coldpackage/model/parity/startup and is not final four-clockproof.','Wholepool selector/parity audit is separate independent gate; no promotion/canonical/provider change by this audit.']}
 out=ROOT/'reports/evidence/long_c_fixed24_root_quality_20261023.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'scores':{a:{m:d['correct'] for m,d in v['metrics'].items()} for a,v in scores.items()},'paired':comparisons,'report':str(out.relative_to(ROOT))},indent=2))
if __name__=='__main__':main()
