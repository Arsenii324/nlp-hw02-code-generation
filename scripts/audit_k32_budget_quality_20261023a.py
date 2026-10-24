"""Actual preserved K32 CPU replay/scoring audit; never execute generated code.

Exact copied frozen selector functions below; arithmetic is bounded AST-only.
No Torch/model import, generation, tests, downloads or training-data modification.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime,timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import time
from nlp_hw02.arithmetic import evaluate_code as raw_legacy
from nlp_hw02.arithmetic_augassign_audit import evaluate_code_augassign as raw_aug
from audit_augassign_metric import select as diagnostic_select

@lru_cache(maxsize=None)
def evaluate_code(code):return raw_legacy(code)

AUG_CHECKS=Counter()
@lru_cache(maxsize=None)
def aug(code):
    old=evaluate_code(code);new=raw_aug(code)
    if old.supported and (not new.supported or old.value!=new.value):raise ValueError('AugAssign changed a legacy accepted value')
    AUG_CHECKS['unique_programs_checked']+=1;AUG_CHECKS['newly_supported']+=not old.supported and new.supported
    return new

def equal(a,b):
    return a==b if isinstance(a,int) and isinstance(b,int) else math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9)

def select_plurality(codes,scores):
    if not codes or len(codes)!=len(scores) or not all(math.isfinite(s) for s in scores):
        raise ValueError('Nonempty candidates and finite scores required')
    groups=[];supported=0
    for i,code in enumerate(codes):
        result=evaluate_code(code)
        if not result.supported:continue
        supported+=1
        group=next((g for g in groups if equal(g['value'],result.value)),None)
        if group is None:group={'value':result.value,'indices':[]};groups.append(group)
        group['indices'].append(i)
    if not groups:return 0,{'supported_candidates':0,'winning_votes':0,'numeric_groups':0}
    best=max(groups,key=lambda g:(len(g['indices']),max(scores[i] for i in g['indices'])))
    index=max(best['indices'],key=lambda i:scores[i])
    return index,{'supported_candidates':supported,'winning_votes':len(best['indices']),'numeric_groups':len(groups)}

def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def load_lines(path):
    with path.open() as f:return [json.loads(line) for line in f]


def paired(control,treatment):
    pairs=Counter((a=='correct',b=='correct') for a,b in zip(control,treatment) if a is not None)
    require(sum(pairs.values())==1529,'Paired fixed reference population differs')
    wins=pairs[False,True];losses=pairs[True,False];n=wins+losses
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
    return {'wins':wins,'losses':losses,'net':wins-losses,'both_correct':pairs[True,True],'both_not_correct':pairs[False,False],'descriptive_exact_paired_binomial_p':p}


def main(args,report,save):
    root=args.root;work=root/'.worktrees/strong-inference-fallback-20261022';base=root/'artifacts/kaggle/k32-budget-v1';old=root/'artifacts/kaggle/sample-scaling-v1'
    receipt_path=root/'reports/evidence/k32_budget_launch_20261023/streamed-recovery-current-v1.json';receipt=json.loads(receipt_path.read_text());terminal=json.loads((base/'session_result.json').read_text())
    require(receipt['complete'] is True and terminal['complete'] is True and terminal['artifact_hashes_complete'] is True,'Incomplete recovered actual run')
    require(len(receipt['files'])==len(terminal['artifact_sha256'])==50,'Exact50 artifact map required')
    for entry in receipt['files']:
        p=base/entry['name'];require(p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']==terminal['artifact_sha256'][entry['name']],'Actual recovered hash differs:'+entry['name'])
    manifest=json.loads((base/'k32_budget_manifest.json').read_text());require(sha(base/'k32_budget_manifest.json')==terminal['manifest_sha256']=='bc962d830a8bdb9276dd0609b5b9f6c7d23a6cab3b67a1e90d4517982f45b32e','Stage manifest identity differs')
    require(terminal['driver_sha256']=='9ff613406d698d9f274c083bf8ffe6ab9e4977176699e7475203619bb47b9020' and sha(base/'strong_model_k32_budget_probe.py')=='c930fa19358d07a1a5ea993d02da9c915026d6941ef1e0e875062eca8446c8d6','Actual source differs')
    require(sha(root/'src/nlp_hw02/arithmetic.py')=='e01078307ec395b41afe03b588e014b7498ee968ed6ed2f3d13f8105c45663d6' and sha(root/'src/nlp_hw02/arithmetic_augassign_audit.py')=='fa26100d91331f2ffe684f7729ad37870c6ac53a0f8834cf6922f430908ecc16','Scoring interpreter source differs')
    selector=work/'src/nlp_hw02/sampling_candidate.py';require(sha(selector)=='250d34bee2dc60847a65a7d5a784fac261a77db153a4d49f0687a82fe5a98fe5','Frozen selector source differs')
    original=ast.parse(selector.read_text());current=ast.parse(Path(__file__).read_text())
    def function(t,n):return next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name==n)
    for name in ('equal','select_plurality'):require(ast.dump(function(original,name),include_attributes=False)==ast.dump(function(current,name),include_attributes=False),'Copied exact selector AST differs:'+name)
    qpath=work/'artifacts/kaggle/k32-budget-tools-20261023a/validation_questions.json';require(sha(qpath)=='83ea3ea9f449619e015d98e21b1c35ec2cc6d2edd01db0c026301e718014f04e','Fixed1792 question source differs')
    questions=json.loads(qpath.read_text());qh=[hashlib.sha256(q.encode()).hexdigest() for q in questions];require(len(qh)==len(set(qh))==1792,'Question population duplicates/count differs')
    refs_path=root/'reports/evidence/primary_reference_consensus.json';require(sha(refs_path)=='deb96602c88b983ecfbf23738d2a64192057273798548ab61a7cdf7acfc7c119','Fixed reference pool changed');refs=json.loads(refs_path.read_text())['groups_by_question_sha256']
    require(sum(refs[q]['status']=='supported_consensus' for q in qh)==1529,'Fixed1529 scoring projection differs')
    test_path=work/'artifacts/trained-gated-sampling-inputs-20261023a/test.parquet';require(sha(test_path)=='8bde59c8d8bb7edc4e8a503be22fc132c4a47773025d333432e78c5489e6522a','Pinned1919 test input differs')
    import pyarrow.parquet as pq
    test=pq.ParquetFile(test_path).read(columns=['question']).column('question').to_pylist();require(len(test)==1919,'Testquestion count differs');th=[hashlib.sha256(q.encode()).hexdigest() for q in test]
    report.update(phase='replay',verified_recovered_artifact_hashes=50,terminal_sha256=sha(base/'session_result.json'),receipt_sha256=sha(receipt_path),manifest_sha256=sha(base/'k32_budget_manifest.json'),
        input_sha256={str(p.relative_to(root)):sha(p) for p in (qpath,refs_path,test_path,selector,root/'src/nlp_hw02/arithmetic.py',root/'src/nlp_hw02/arithmetic_augassign_audit.py',Path(__file__),root/'scripts/audit_augassign_metric.py')},
        reporting_rows=1792,strict_reference_rows=1529,test_question_rows=1919,paths={},quality={},paired={},replay_checks=Counter());save()
    outcomes={};reasons={};changes={};indices={}
    def score(label,codes,scores,chosen):
        legacy=[];augfix=[];augre=[];why=Counter();changed=[];selected=[]
        for i,(c,s,j,q) in enumerate(zip(codes,scores,chosen,qh)):
            k=diagnostic_select(c,s,aug);selected.append(k)
            if k!=j:changed.append({'row':i,'question_sha256':q,'legacy_index':j,'diagnostic_index':k})
            reference=refs[q]
            if reference['status']!='supported_consensus':legacy.append(None);augfix.append(None);augre.append(None);continue
            target=reference['consensus_value']
            for output,evaluator,code,tag in ((legacy,evaluate_code,c[j],'legacy'),(augfix,aug,c[j],'augassign_fixed_output'),(augre,aug,c[k],'augassign_reselected')):
                value=evaluator(code);category=('correct' if equal(value.value,target) else 'wrong_numeric') if value.supported else 'unsupported';output.append(category)
                if not value.supported:why[tag+':'+value.reason]+=1
        for tag,result in [('legacy',legacy),('augassign_fixed_output',augfix),('augassign_reselected',augre)]:
            require(sum(v is not None for v in result)==1529,'Scored rowcount differs');outcomes[label+'/'+tag]=result
        report['quality'][label]={tag:dict(Counter(v for v in result if v is not None)) for tag,result in [('legacy',legacy),('augassign_fixed_output',augfix),('augassign_reselected',augre)]}
        report['quality'][label]['augassign_reselection_changed_rows']=len(changed);report['quality'][label]['unsupported_reasons']=dict(why)
        changes[label]=changed;indices[label]={'legacy':chosen,'augassign_reselected':selected}
    for mode in ('fixed32','adaptive8-32'):
        run=base/mode;result=json.loads((run/'k32_budget_result.json').read_text());require(result['complete'] is True and result['mode']==mode and result['visible_gpu_count']==1 and 'T4' in result['gpu'],'Actual lane source/hardware completion differs')
        report['paths'][mode]={}
        for label,ordered in [('validation1792',qh),('test_warmup4',th[:4]),('test1919',th)]:
            pool=load_lines(run/(label+'_candidates.jsonl'));pred=load_lines(run/(label+'_predictions.jsonl'));timing=json.loads((run/(label+'_timing.json')).read_text())
            require(len(pool)==len(pred)==len(ordered) and timing==result['paths'][label],'Actual ordered rowcount/timing differs')
            require(sha(run/(label+'_candidates.jsonl'))==timing['candidate_sha256'] and sha(run/(label+'_predictions.jsonl'))==timing['prediction_sha256'],'Pool/output timing hash differs')
            total=allcaps=selcaps=unsupported=n8=n32=stopcount=0;cohorts=Counter();chosen=[]
            for i,(row,prediction,q) in enumerate(zip(pool,pred,ordered)):
                require(row['row_index']==i and row['question_sha256']==q and row['mode']==mode,'Question/mode order differs')
                require(all(row[k]==v for k,v in {'seed':20261002,'temperature':.8,'question_batch_size':32,'top_k':50,'length_penalty':.6,'max_new_tokens':192}.items()),'Sampling configuration differs')
                n=row['samples_generated'];require(n in (8,32) and all(len(row[k])==n for k in ('candidates','scores','capped','generated_lengths')),'Actual vector count differs')
                require(all(isinstance(c,str) for c in row['candidates']) and all(math.isfinite(v) for v in row['scores']),'Candidate text/score invalid')
                require(all(type(c) is bool and type(t) is int and 0<=t<=192 and c==(t==192) for c,t in zip(row['capped'],row['generated_lengths'])),'Actual cap/length differs')
                prefix,info=select_plurality(row['candidates'][:8],row['scores'][:8]);stop=info['supported_candidates']==8 and info['numeric_groups']==1
                cohort='unanimous8' if stop else ('no_supported' if info['supported_candidates']==0 else 'majority6_7' if info['winning_votes']>=6 else 'disputed')
                require(prefix==row['prefix8_selected_index'] and info==row['prefix8_diagnostics'] and stop==row['would_stop8'] and cohort==row['prefix8_category'],'Prefix8 actual selector/group replay differs')
                require(row['stopped_at8']==(mode=='adaptive8-32' and stop) and n==(8 if mode=='adaptive8-32' and stop else 32),'Adaptive stop/count accounting differs')
                index,final=select_plurality(row['candidates'],row['scores']);require(index==row['selected_index'] and final==row['final_diagnostics'] and set(prediction)=={'code'} and prediction['code']==row['candidates'][index],'Exact final legacy selector/raw projection differs')
                report['replay_checks']['actual_main_rows']+=1;report['replay_checks']['actual_main_candidates']+=n
                chosen.append(index);total+=n;n8+=n==8;n32+=n==32;stopcount+=stop;cohorts[cohort]+=1;allcaps+=sum(row['capped']);selcaps+=row['capped'][index];unsupported+=final['supported_candidates']==0
            expected={'rows':len(pool),'actual8_rows':n8,'actual32_rows':n32,'first8_would_stop_rows':stopcount,'expanded_after8_rows':n32 if mode=='adaptive8-32' else 0,'total_candidates':total,'mean_candidates':total/len(pool),'prefix8_categories':dict(cohorts),'all_candidate_capped_rows':allcaps,'selected_candidate_capped_rows':selcaps,'without_supported_candidate':unsupported}
            require(all(timing[k]==v for k,v in expected.items()),'Actual candidate/timing counters differ')
            for chunk in timing['chunks']:
                rows=pool[chunk['offset']:chunk['offset']+chunk['rows']];require(chunk['extra24_rows']==(sum(r['samples_generated']==32 for r in rows) if mode=='adaptive8-32' else 0),'Actual compactedchunk count differs')
            require(timing['complete_path_within180']==(timing['complete_path_seconds']<=180),'Timing180 flag differs')
            report['paths'][mode][label]={**expected,'complete_path_seconds':timing['complete_path_seconds'],'sum_chunk_compute_seconds':timing['sum_chunk_compute_seconds'],'peak_allocated_bytes':timing['peak_allocated_bytes'],'prediction_sha256':timing['prediction_sha256'],'ordered_questions_sha256':timing['ordered_questions_sha256']}
            if label=='validation1792':
                score(mode,[r['candidates'] for r in pool],[r['scores'] for r in pool],chosen)
            if mode=='fixed32' and label!='test_warmup4':
                nested=json.loads((run/(label+'_nested_report.json')).read_text());require(nested==result['nested_fixed32'][label] and nested['source_candidates_sha256']==timing['candidate_sha256'],'Nested source/result identity differs')
                for prefix in (8,16,24,32):
                    selections=load_lines(run/(label+f'_nested{prefix}_selection.jsonl'));outputs=load_lines(run/(label+f'_nested{prefix}_predictions.jsonl'));require(len(selections)==len(outputs)==len(pool),'Nested actual rowcount differs')
                    require(sha(run/(label+f'_nested{prefix}_selection.jsonl'))==nested['prefixes'][str(prefix)]['selection_sha256'] and sha(run/(label+f'_nested{prefix}_predictions.jsonl'))==nested['prefixes'][str(prefix)]['prediction_sha256'],'Nested output/selection hash differs')
                    selected=[]
                    for i,(row,record,output,q) in enumerate(zip(pool,selections,outputs,ordered)):
                        index,info=select_plurality(row['candidates'][:prefix],row['scores'][:prefix]);require(record=={'row_index':i,'question_sha256':q,'prefix':prefix,'source_samples':32,'selected_index':index,**info},'Exact nested selector/index/info/order differs')
                        require(set(output)=={'code'} and output['code']==row['candidates'][index],'Nested rawcode projection differs');selected.append(index);report['replay_checks']['nested_rows']+=1
                    if prefix==32:require(outputs==pred,'Nested32/main bytes/order projection differs')
                    if label=='validation1792':score('fixed32_prefix'+str(prefix),[r['candidates'][:prefix] for r in pool],[r['scores'][:prefix] for r in pool],selected)
            print(json.dumps({'audit_path_complete':mode+'/'+label,'legacy_quality':report['quality'].get(mode,{}).get('legacy'),'samples':total}),flush=True);save()
    old_terminal=old/'probe_terminal.json';require(sha(old_terminal)=='190a57c7167329dff11ccabb554285222176570f016fece1a64222b187aedf2c','OriginalK24 terminal identity differs');ot=json.loads(old_terminal.read_text())
    old_pool_path=old/'samples24/validation24_sample_candidates.jsonl';old_pred_path=old/'samples24/sample-plurality24_predictions.jsonl'
    for p in (old_pool_path,old_pred_path):require(sha(p)==ot['artifact_sha256'][str(p.relative_to(old))],'OriginalK24 input artifact bytes differ')
    pool=load_lines(old_pool_path);pred=load_lines(old_pred_path);require(len(pool)==len(pred)==1792,'OriginalK24 rows differ');selected=[]
    for i,(row,prediction,q) in enumerate(zip(pool,pred,qh)):
        require(row['question_sha256']==prediction['question_sha256']==q and hashlib.sha256(prediction['question'].encode()).hexdigest()==q,'OriginalK24 question mapping differs')
        require(len(row['candidates'])==len(row['scores'])==24,'OriginalK24 vectors differ');index,info=select_plurality(row['candidates'],row['scores'])
        require(index==row['selected_index'] and prediction['prediction']==row['candidates'][index] and all(row[k]==v for k,v in info.items()),'OriginalK24 selector/raw projection differs');selected.append(index)
    score('originalK24',[r['candidates'] for r in pool],[r['scores'] for r in pool],selected);require(report['quality']['originalK24']['legacy']['correct']==1087,'OriginalK24 no longer reproduces1087 baseline')
    report['original_K24_input_sha256']={str(p.relative_to(root)):sha(p) for p in (old_terminal,old_pool_path,old_pred_path)}
    comparisons=[('originalK24','fixed32'),('originalK24','adaptive8-32'),('fixed32','adaptive8-32'),('fixed32_prefix8','fixed32_prefix16'),('fixed32_prefix16','fixed32_prefix24'),('fixed32_prefix24','fixed32_prefix32'),('fixed32_prefix8','fixed32_prefix32')]
    for control,treatment in comparisons:
        report['paired'][treatment+'_vs_'+control]={tag:paired(outcomes[control+'/'+tag],outcomes[treatment+'/'+tag]) for tag in ('legacy','augassign_fixed_output','augassign_reselected')}
    outcome_path=args.output.with_name(args.output.stem+'_outcomes.jsonl');require(not outcome_path.exists(),'Fresh per-question outcome path required')
    with outcome_path.open('x') as stream:
        for i,q in enumerate(qh):stream.write(json.dumps({'row_index':i,'question_sha256':q,'scored':refs[q]['status']=='supported_consensus','categories':{k:v[i] for k,v in outcomes.items()},'selected_indices':{k:{t:a[i] for t,a in v.items()} for k,v in indices.items()}},allow_nan=False)+'\n')
    report.update(phase='complete',complete=True,augassign_checks=dict(AUG_CHECKS),augassign_legacy_supported_values_changed=0,augassign_reselection_changes=changes,
        outcome_file=str(outcome_path.relative_to(root)),outcome_sha256=sha(outcome_path),session_script_seconds=terminal['session_script_seconds'],
        model_sha256=manifest['runtime_member_sha256']['model.pt'],scope='Actual preserved real outputs only; fixed1529 labels and1792 ordered questions. No generation/testlabels/filtering.',
        limits=['Not official pass1; restricted legacy/diagnostic interpreter and noisy reference pool remain.','AugAssign fixed-output scoring and cached-pool reselection keep identical reference labels; deployed legacy selector unchanged.','Only within fixed32 actualpool nestedprefixes sharecandidates; originalK24/fixed32/adaptive have distinct tensor/RNG streams despite same seed.','Paired p-values are descriptive after repeated validation/selection; no general seed robustness claim.','Full1919 policy clocks include candidate/progress serialization, unlike historical K24 compute-only clock; not cold notebook proof.','No packaging/policybinding/canonical/cloud changes; fresh chosen-byte oneT4 proof mandatory.'])
    save();print(json.dumps({'quality':report['quality'],'paired':report['paired'],'paths':report['paths'],'checks':report['replay_checks']},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();require(not args.output.exists(),'Fresh evidence output required')
    report={'utc':datetime.now(timezone.utc).isoformat(),'complete':False,'phase':'hashes','failures':[]};started=time.monotonic()
    def save():
        report['audit_CPU_elapsed_seconds']=time.monotonic()-started;args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:main(args,report,save)
    except BaseException as error:
        report['failures'].append({'type':type(error).__name__,'error':str(error)[:8192]});report['complete']=False;save();raise
