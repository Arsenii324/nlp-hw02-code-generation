"""Actual retained candidate subsets only; no model/generated Python execution."""
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

@lru_cache(maxsize=None)
def evaluate_code(code):return raw_legacy(code)

@lru_cache(maxsize=None)
def aug(code):
    old=evaluate_code(code);new=raw_aug(code)
    if old.supported and (not new.supported or old.value!=new.value):raise ValueError('AugAssign changed legacy accepted value')
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
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def lines(path):
    with Path(path).open() as f:return [json.loads(line) for line in f]


def paired(control,treatment):
    c=Counter((a=='correct',b=='correct') for a,b in zip(control,treatment) if a is not None)
    require(sum(c.values())==1529,'Fixed paired population differs');wins=c[False,True];losses=c[True,False];n=wins+losses
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
    return {'rows':1529,'wins':wins,'losses':losses,'net':wins-losses,'both_correct':c[True,True],'both_not_correct':c[False,False],'descriptive_exact_paired_binomial_p':p}


def main(args,report,save):
    root=args.root;work=root/'.worktrees/strong-inference-fallback-20261022';plan=root/'docs/plans/2026-10-23-mixed-model-pool-diagnostic.md'
    require(sha(plan)=='5218e38c8bff8e21394ff02288e809d0c3330101645f571317a31179b7c174f4','Predeclared six-combination plan differs')
    source=work/'src/nlp_hw02/sampling_candidate.py';require(sha(source)=='250d34bee2dc60847a65a7d5a784fac261a77db153a4d49f0687a82fe5a98fe5','Frozen selector source differs')
    original=ast.parse(source.read_text());current=ast.parse(Path(__file__).read_text())
    for name in ('equal','select_plurality'):
        fn=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        require(ast.dump(fn(original),include_attributes=False)==ast.dump(fn(current),include_attributes=False),'Exact copied selector AST differs:'+name)
    arithmetic=root/'src/nlp_hw02/arithmetic.py';augsource=root/'src/nlp_hw02/arithmetic_augassign_audit.py'
    require(sha(arithmetic)=='e01078307ec395b41afe03b588e014b7498ee968ed6ed2f3d13f8105c45663d6' and sha(augsource)=='fa26100d91331f2ffe684f7729ad37870c6ac53a0f8834cf6922f430908ecc16','Existing bounded evaluators differ')
    qpath=work/'artifacts/kaggle/k32-budget-tools-20261023a/validation_questions.json';rpath=root/'reports/evidence/primary_reference_consensus.json'
    require(sha(qpath)=='83ea3ea9f449619e015d98e21b1c35ec2cc6d2edd01db0c026301e718014f04e' and sha(rpath)=='deb96602c88b983ecfbf23738d2a64192057273798548ab61a7cdf7acfc7c119','Fixed question/reference projection differs')
    questions=json.loads(qpath.read_text());qh=[hashlib.sha256(q.encode()).hexdigest() for q in questions];refs=json.loads(rpath.read_text())['groups_by_question_sha256'];scored=[refs[q]['status']=='supported_consensus' for q in qh]
    require(len(qh)==len(set(qh))==1792 and sum(scored)==1529,'Fixed ordered1792/1529 mapping differs')
    report.update(plan_sha256=sha(plan),source_sha256={str(p.relative_to(root)):sha(p) for p in (source,arithmetic,augsource,qpath,rpath,Path(__file__))},reporting_rows=1792,reference_rows=1529,checks={'exact_selector_AST':True,'fixed_question_reference_mapping':True,'existing_evaluator_sources':True},input_sha256={},controls={},mixtures={},paired={},complementarity={},phase='replay');save()
    bases={'old344':(root/'artifacts/kaggle/trained-gated-sampling-v1-b','control-final','01066a6d40afcb787a8aa5667e956a2f4e32ce22eb3954b8a1f92239e82d4866','gated_sampling_result.json',1099),'uniform':(root/'artifacts/kaggle/long-c-paired-v2-c-diagnostics','run-uniform-control','3706104d23c5f5eb07ad27d32533df24386530bc892469514c18d738bacae1d2','long_c_sampling_result.json',1096),'bucket':(root/'artifacts/kaggle/long-c-paired-v2-c-diagnostics','run-bucket-control','3706104d23c5f5eb07ad27d32533df24386530bc892469514c18d738bacae1d2','long_c_sampling_result.json',1097)}
    pools={};indices={};outcomes={};rowmeta={};metrics=('legacy','augassign_fixed_output','augassign_additive1e6')
    def score(codes):
        result={k:[] for k in metrics};support={'all1792':Counter(),'scored1529':Counter()}
        for i,code in enumerate(codes):
            v=evaluate_code(code);w=aug(code)
            support['all1792']['legacy_supported' if v.supported else 'legacy_unsupported']+=1
            if scored[i]:support['scored1529']['legacy_supported' if v.supported else 'legacy_unsupported']+=1
            for tag,value in [('legacy',v),('augassign_fixed_output',w),('augassign_additive1e6',w)]:
                if not scored[i]:result[tag].append(None);continue
                target=refs[qh[i]]['consensus_value'];ok=(abs(value.value-target)<=1e-6+1e-6*abs(target) if tag=='augassign_additive1e6' else equal(value.value,target)) if value.supported else False
                result[tag].append('correct' if ok else 'wrong_numeric' if value.supported else 'unsupported')
        quality={tag:dict(Counter(v for v in values if v is not None)) for tag,values in result.items()}
        rawsha=hashlib.sha256(b''.join((json.dumps({'code':c},ensure_ascii=False)+'\n').encode() for c in codes)).hexdigest()
        return result,{'quality':quality,'selected_support':{k:dict(v) for k,v in support.items()},'ordered_code_only_projection_sha256':rawsha}
    for label,(base,role,terminal_sha,result_name,expected_correct) in bases.items():
        tp=base/'session_result.json';require(sha(tp)==terminal_sha,'Actual terminal identity differs');terminal=json.loads(tp.read_text());require(terminal['complete'] and terminal['artifact_hashes_complete'] and terminal['cleanup'] and all(c['reaped'] and c['group_exited'] for c in terminal['cleanup']),'Actual source completion/cleanup differs')
        run=base/role;files=[run/name for name in ('validation_n24_candidates.jsonl','validation_n24_predictions.jsonl','validation_n24_timing.json',result_name)]
        for p in files:require(sha(p)==terminal['artifact_sha256'][str(p.relative_to(base))],'Actual terminal file map differs:'+str(p));report['input_sha256'][str(p.relative_to(root))]=sha(p)
        report['input_sha256'][str(tp.relative_to(root))]=terminal_sha
        pool,pred=lines(files[0]),lines(files[1]);timing=json.loads(files[2].read_text());receipt=json.loads(files[3].read_text())
        require(receipt['complete'] and receipt['parity_passed_before_sampling'] and receipt['training'] is False and receipt['gold_used_for_selection'] is False,'Actual goldfree qualified pool receipt differs')
        require(all(timing[k]==v for k,v in {'rows':1792,'samples_per_question':24,'total_candidates':43008,'temperature':.8,'top_k':50,'question_batch_size':32,'seed':20261002,'length_penalty':.6,'max_new_tokens':192}.items()),'Exact fixed24 sampling configuration differs')
        require(timing['candidate_sha256']==sha(files[0]) and timing['prediction_sha256']==sha(files[1]) and len(pool)==len(pred)==1792,'Actual pool/output vector or hash differs')
        selected=[];caps=Counter();all_programs=0
        for i,(row,output,q) in enumerate(zip(pool,pred,qh)):
            require(row['row_index']==i and row['question_sha256']==q and row['samples']==24 and all(len(row[k])==24 for k in ('candidates','scores','capped','generated_lengths')),'Actual ordered vectors/count differs')
            require(all(isinstance(c,str) for c in row['candidates']) and all(math.isfinite(v) for v in row['scores']),'Raw code/score differs')
            require(all(type(c) is bool and type(n) is int and 0<=n<=192 and c==(n==192) for c,n in zip(row['capped'],row['generated_lengths'])),'Actual candidate caps/lengths differ')
            j,info=select_plurality(row['candidates'],row['scores']);require(row['selected_index']==j and all(row[k]==v for k,v in info.items()) and set(output)=={'code'} and output['code']==row['candidates'][j],'Exact original selector/rawcode replay differs')
            selected.append(j);caps['all_candidate_capped']+=sum(row['capped']);caps['selected_capped']+=row['capped'][j];all_programs+=24
        require(caps['all_candidate_capped']==timing['all_candidate_capped_rows'] and caps['selected_capped']==timing['selected_candidate_capped_rows'],'Actual cap counters differ')
        codes=[r['candidates'][j] for r,j in zip(pool,selected)];outcomes[label],summary=score(codes);require(summary['quality']['legacy']['correct']==expected_correct,'Original fixed24 control score differs')
        pools[label]=pool;indices[label]=selected;rowmeta[label]=[{'selected_index':j} for j in selected];report['controls'][label]={**summary,'caps':dict(caps),'replayed_candidates':all_programs,'retained_prediction_sha256':sha(files[1]),'sampling_receipt_sha256':sha(files[3])};report['checks']['full24_replay_'+label]=True;save()
    for left,right in [('old344','uniform'),('old344','bucket'),('uniform','bucket')]:
        pair=left+'+'+right;left_full=outcomes[left]['legacy'];right_full=outcomes[right]['legacy'];fullunion=sum(a=='correct' or b=='correct' for a,b in zip(left_full,right_full) if a is not None)
        report['complementarity'][pair]={'original_full24_selected_oracle_union_correct':fullunion,'left_full24_correct':sum(a=='correct' for a in left_full),'right_full24_correct':sum(a=='correct' for a in right_full),'original_full24_disagreement_paired':paired(left_full,right_full),'interpretation':'Offline label-using bound only; no oracle used by selector'}
        for half,start in [('first12',0),('last12',12)]:
            label=pair+'/'+half;selected_codes=[];selected_indices=[];meta=[];agreement={'all1792':Counter(),'scored1529':Counter()};support=Counter();oracle=Counter();caps=Counter();dup=Counter()
            for i,(a,b) in enumerate(zip(pools[left],pools[right])):
                lc=a['candidates'][start:start+12];rc=b['candidates'][start:start+12];ls=a['scores'][start:start+12];rs=b['scores'][start:start+12];codes=lc+rc;scores=ls+rs
                j,info=select_plurality(codes,scores);lj,li=select_plurality(lc,ls);rj,ri=select_plurality(rc,rs);lv=evaluate_code(lc[lj]);rv=evaluate_code(rc[rj]);winner=evaluate_code(codes[j]);reason='both_supported_agree' if lv.supported and rv.supported and equal(lv.value,rv.value) else 'both_supported_disagree' if lv.supported and rv.supported else 'left_only_supported' if lv.supported else 'right_only_supported' if rv.supported else 'neither_supported'
                agreement['all1792'][reason]+=1
                if scored[i]:agreement['scored1529'][reason]+=1
                left_votes=sum(v.supported and equal(v.value,winner.value) for v in map(evaluate_code,lc)) if winner.supported else 0;right_votes=sum(v.supported and equal(v.value,winner.value) for v in map(evaluate_code,rc)) if winner.supported else 0
                require(left_votes+right_votes==info['winning_votes'],'Cross-model winning group vote replay differs')
                support['winning_group_supported_by_both_models' if left_votes and right_votes else 'winning_group_left_only' if left_votes else 'winning_group_right_only' if right_votes else 'no_supported_winner']+=1
                support['winning_votes_left_total']+=left_votes;support['winning_votes_right_total']+=right_votes;support['supported_candidate_total']+=info['supported_candidates'];support['numeric_group_total']+=info['numeric_groups']
                dup['duplicate_votes_beyond_unique_programs']+=24-len(set(codes));dup['cross_model_shared_unique_programs']+=len(set(lc)&set(rc));dup['rows_with_cross_model_identical_program']+=bool(set(lc)&set(rc))
                capped=a['capped'][start:start+12]+b['capped'][start:start+12];caps['all_candidate_capped']+=sum(capped);caps['selected_capped']+=capped[j]
                if scored[i]:
                    target=refs[qh[i]]['consensus_value'];lcorrect=lv.supported and equal(lv.value,target);rcorrect=rv.supported and equal(rv.value,target)
                    oracle['left_subset12_selected_correct']+=lcorrect;oracle['right_subset12_selected_correct']+=rcorrect;oracle['either_subset12_selected_correct']+=lcorrect or rcorrect;oracle['both_subset12_selected_correct']+=lcorrect and rcorrect;oracle['any_mixed24_candidate_correct']+=any(v.supported and equal(v.value,target) for v in map(evaluate_code,codes))
                selected_codes.append(codes[j]);selected_indices.append(j);meta.append({'selected_index':j,'source_model':left if j<12 else right,'source_index':start+j%12,'left_subset_selected_index':start+lj,'right_subset_selected_index':start+rj,'left_winning_group_votes':left_votes,'right_winning_group_votes':right_votes,**info})
            outcomes[label],summary=score(selected_codes);indices[label]=selected_indices;rowmeta[label]=meta
            report['mixtures'][label]={**summary,'models':[left,right],'subset_indices_in_each_model':[start,start+11],'candidate_order':'left12 then right12','candidates_per_question':24,'total_candidates':43008,'numeric_agreement':{k:dict(v) for k,v in agreement.items()},'winner_support':dict(support),'duplicate_vote_diagnostics':dict(dup),'caps':dict(caps),'bounded_oracles':{**dict(oracle),'either_original_full24_selected_correct':fullunion},'actual_N12_generation':False,'latency_measured':False}
            report['paired'][label]={control:{tag:paired(outcomes[control][tag],outcomes[label][tag]) for tag in metrics} for control in sorted({left,right,'old344'})};save()
            print(json.dumps({'mixed_policy':label,'correct':summary['quality']['legacy']['correct'],'paired_vs_old344':report['paired'][label]['old344']['legacy']}),flush=True)
    out=args.output.with_name(args.output.stem+'_outcomes.jsonl');require(not out.exists(),'Fresh ordered outcome file required')
    with out.open('x') as f:
        for i,q in enumerate(qh):f.write(json.dumps({'row_index':i,'question_sha256':q,'scored':scored[i],'categories':{label:{m:vals[i] for m,vals in result.items()} for label,result in outcomes.items()},'selections':{label:rows[i] for label,rows in rowmeta.items()}},allow_nan=False)+'\n')
    qualifying=[]
    for left,right in [('old344','uniform'),('old344','bucket'),('uniform','bucket')]:
        pair=left+'+'+right;strong=max(report['controls'][m]['quality']['legacy']['correct'] for m in (left,right));counts=[report['mixtures'][pair+'/'+h]['quality']['legacy']['correct'] for h in ('first12','last12')]
        if all(n>strong for n in counts):qualifying.append({'pair':pair,'both_replications_correct':counts,'stronger_constituent_full24_correct':strong,'both_vs_current_old344_net':[n-1099 for n in counts]})
    report.update(complete=True,phase='finished',combinations=6,original_control_candidates=129024,mixed_candidate_slots=258048,ordered_outcomes_path=str(out.resolve().relative_to(root.resolve())),ordered_outcomes_sha256=sha(out),all_checks_pass=all(report['checks'].values()),predeclared_consistency_gate_qualifying_pairs=qualifying,evaluator_unique_programs={'legacy':evaluate_code.cache_info().currsize,'augassign':aug.cache_info().currsize},limits=['Actual retained N24 candidate subsets only; not actual N12 seed/RNG trajectories or independent seed replications.','All six combinations retained; one reused noisy1529 consensus pool and repeated validation, not official pass1 or unbiased confirmatory significance.','Legacy selector/numeric grouping/ties/raw text/duplicate votes unchanged; AugAssign/additive1e6 scoring only, additive formula not confirmed official.','Cross-model scores are original same-convention values without calibration; no oracle used by selector.','No model generation, Python code execution, training filtering/target/reference rewrite, canonical/package/provider change.','No mixed latency observed or180s proof; eventual N12 generation plus load/memory/complete singleT4 timing/reference proof required after root user-write gate.']);save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args();require(not args.output.exists(),'Fresh evidence required');started=time.monotonic();report={'schema':'mixed_model_retained_pool_diagnostic_v1','utc':datetime.now(timezone.utc).isoformat(),'complete':False,'phase':'inputs','failures':[]}
    def save():
        report['audit_CPU_elapsed_seconds']=time.monotonic()-started;args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:main(args,report,save)
    except BaseException as error:report.update(complete=False);report['failures'].append({'type':type(error).__name__,'message':str(error)[:8192]});save();raise
