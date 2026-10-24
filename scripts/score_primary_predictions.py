"""Re-score generated validation code against supplied-reference consensus.

Evaluation-only: bounded AST interpretation, no code execution, no relabeling,
no question text in this report, no official pass@1 claim.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from nlp_hw02.arithmetic import evaluate_code


def equal(left,right):
    return left==right if type(left) is int and type(right) is int else math.isclose(left,right,rel_tol=1e-9,abs_tol=1e-9)


def score(prediction_path,consensus):
    rows=[json.loads(s) for s in prediction_path.read_text().splitlines()]
    groups=consensus['groups_by_question_sha256']
    statuses=Counter(); supported=0; agree=0; strict=0; strict_agree=0
    any_conflict_match=0; pred_supported=0; missing=0; seen=set()
    single_support=single_agree=0; outcomes={}
    for row in rows:
        qhash=hashlib.sha256(row['question'].encode()).hexdigest()
        assert row.get('question_sha256',qhash)==qhash and qhash not in seen
        seen.add(qhash)
        entry=groups.get(qhash)
        if entry is None: missing+=1; continue
        status=entry['status']; statuses[status]+=1
        value=evaluate_code(row['prediction']); pred_supported+=value.supported
        original=evaluate_code(row['code'])
        single_support+=original.supported
        single_agree+=original.supported and value.supported and equal(original.value,value.value)
        if status in ('supported_consensus','supported_consensus_with_unsupported_refs'):
            supported+=1
            correct=value.supported and equal(entry['consensus_value'],value.value)
            agree+=correct
            if status=='supported_consensus':
                strict+=1; strict_agree+=correct; outcomes[qhash]=bool(correct)
        elif status=='conflict':
            any_conflict_match+=value.supported and any(equal(b['representative'],value.value) for b in entry['supported_numeric_value_buckets'])
    assert missing==0,'Predictions do not belong to the audited primary split'
    return {'rows':len(rows),'status_counts':dict(statuses),'prediction_supported':pred_supported,
        'single_reference':{'supported':single_support,'agreeing':single_agree},
        'fully_supported_consensus':{'scorable':strict,'agreeing':strict_agree,
            'agreement':strict_agree/strict if strict else None},
        'including_partial_consensus':{'scorable':supported,'agreeing':agree,
            'agreement':agree/supported if supported else None},
        'conflict_any_reference_matches_unscored':any_conflict_match,
        'prediction_sha256':hashlib.sha256(prediction_path.read_bytes()).hexdigest()},outcomes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--consensus',default='reports/evidence/primary_reference_consensus.json')
    parser.add_argument('--run',action='append',required=True,help='LABEL=/path/to/run')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    path=Path(args.consensus); consensus=json.loads(path.read_text())
    assert consensus['float_rel_tol']==1e-9 and consensus['float_abs_tol']==1e-9
    evaluator=Path(__file__).resolve().parents[1]/'src/nlp_hw02/arithmetic.py'
    assert hashlib.sha256(evaluator.read_bytes()).hexdigest()==consensus['evaluator_module_sha256']
    report={'consensus_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'note':'Bounded AST numeric agreement to synthetic supplied train references; print arguments ignored; integers exact, floats rel/abs1e-9. Unsupported predictions count as failures. Conflicts excluded, partial consensus separate. Not official pass@1.',
        'runs':{},'paired':[]}
    outcomes={}
    for spec in args.run:
        label,directory=spec.split('=',1)
        if label in outcomes: raise ValueError('Duplicate run label')
        validation=Path(directory)/'validation.json'
        assert hashlib.sha256(validation.read_bytes()).hexdigest()==consensus['primary_validation_sha256']
        report['runs'][label],outcomes[label]=score(Path(directory)/'validation_predictions.jsonl',consensus)
    labels=list(outcomes)
    for i,a in enumerate(labels):
        for b in labels[i+1:]:
            left=outcomes[a]; right=outcomes[b]
            assert left.keys()==right.keys(),'Paired predictions must use identical question groups'
            a_only=sum(left[k] and not right[k] for k in left)
            b_only=sum(right[k] and not left[k] for k in left)
            n=a_only+b_only
            p=min(1.,2*sum(math.comb(n,k) for k in range(min(a_only,b_only)+1))/2**n) if n else 1.
            report['paired'].append({'left':a,'right':b,'scorable':len(left),'left_only_correct':a_only,
                'right_only_correct':b_only,'exact_two_sided_discordance_p':p,
                'note':'Descriptive paired diagnostic, one seed; multiple experiments and noisy references limit interpretation.'})
    Path(args.output).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))


if __name__=='__main__': main()
