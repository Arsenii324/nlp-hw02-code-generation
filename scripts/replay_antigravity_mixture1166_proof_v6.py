"""Rebuild source-only 768FP N8 + 512_new N3 plurality references.

Reads existing candidate pools only. No model loading/inference or test labels.
"""
import ast, hashlib, json, math
from pathlib import Path
from nlp_hw02.arithmetic import evaluate_code

ROOT=Path(__file__).resolve().parents[1]
VAL=[ROOT/'artifacts/antigravity_gelu768_fullpop531k_eval/validation_n8_candidates.jsonl',ROOT/'artifacts/antigravity_gelu512_long262k_eval/validation_n8_candidates.jsonl']
TEST=[ROOT/'artifacts/antigravity_gelu768_fullpop531k_eval/test_n8_candidates.jsonl',ROOT/'artifacts/antigravity_gelu512_long262k_eval/test_n8_candidates.jsonl']
REF=ROOT/'reports/evidence/primary_reference_consensus.json'
OUT=ROOT/'artifacts/mixture_768fp_n8_512new_n3_test_predictions_v6_20261024a.jsonl'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ordered_sha(rows):return hashlib.sha256(json.dumps(rows,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def equal(a,b):return a==b if isinstance(a,int) and isinstance(b,int) else math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9)
def select(codes,scores):
    groups=[]
    for i,code in enumerate(codes):
        r=evaluate_code(code)
        if not r.supported:continue
        g=next((x for x in groups if equal(x['value'],r.value)),None)
        if g is None:g={'value':r.value,'indices':[]};groups.append(g)
        g['indices'].append(i)
    if not groups:return 0
    best=max(groups,key=lambda g:(len(g['indices']),max(scores[i] for i in g['indices'])))
    return max(best['indices'],key=lambda i:scores[i])
def load(paths,expected):
    pools=[[json.loads(x) for x in p.read_text().splitlines()] for p in paths]
    if any(len(p)!=expected for p in pools):raise ValueError(f'Expected {expected} rows in every pool')
    for i in range(expected):
        q=pools[0][i]['question_sha256']
        if any(row[i]['row_index']!=i or row[i]['question_sha256']!=q or not isinstance(row[i]['candidates'],list) or not isinstance(row[i]['scores'],list) for row in pools):raise ValueError('Candidate pool identity/order mismatch')
    return pools
def main():
    val=load(VAL,1792);refs=json.loads(REF.read_text())['groups_by_question_sha256']; correct=den=0
    for i in range(1792):
        q=val[0][i]['question_sha256'];ref=refs[q]
        if ref['status']!='supported_consensus':continue
        codes=val[0][i]['candidates'][:8]+val[1][i]['candidates'][:3];scores=val[0][i]['scores'][:8]+val[1][i]['scores'][:3]
        code=codes[select(codes,scores)];r=evaluate_code(code);den+=1;correct+=bool(r.supported and equal(r.value,ref['consensus_value']))
    test=load(TEST,1919);questions=[None]*1919;codes=[]
    for i in range(1919):
        candidates=test[0][i]['candidates'][:8]+test[1][i]['candidates'][:3]
        scores=test[0][i]['scores'][:8]+test[1][i]['scores'][:3]
        if len(candidates)!=11 or len(scores)!=11:raise ValueError('Expected 11 candidates per test question')
        code=candidates[select(candidates,scores)];ast.parse(code);codes.append(code)
    if OUT.exists():raise FileExistsError(f'Preserving existing output: {OUT}')
    with OUT.open('x',encoding='utf-8') as f:
        for code in codes:f.write(json.dumps({'code':code},ensure_ascii=False)+'\n')
    return {'schema':'antigravity_mixture1166_reference_replay_v6_v1','models':['768FP','512_new'],'counts':[8,3],'validation':{'correct':correct,'denominator':den,'candidate_pool_sha256':[sha(p) for p in VAL],'consensus_sha256':sha(REF)},'test':{'rows':len(codes),'candidate_pool_sha256':[sha(p) for p in TEST],'output_path':str(OUT.relative_to(ROOT)),'output_sha256':sha(OUT),'ordered_codes_sha256':ordered_sha(codes),'ordered_question_hashes_sha256':ordered_sha([r['question_sha256'] for r in test[0]])},'selector_sha256':sha(Path(__file__).resolve().parents[1]/'scripts/antigravity_build_champion_mixture.py'),'test_labels_read':False,'model_or_weight_loaded':False,'inference_performed':False}
if __name__=='__main__':print(json.dumps(main(),indent=2))
