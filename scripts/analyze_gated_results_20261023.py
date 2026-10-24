"""Analyze fixed, recovered predictions; never execute generated programs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from nlp_hw02.arithmetic import evaluate_code
from nlp_hw02.arithmetic_augassign_audit import evaluate_code_augassign


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal(value, reference, additive=False):
    if additive:
        return abs(value-reference) <= 1e-6+1e-6*abs(reference)
    if type(value) is int and type(reference) is int:
        return value == reference
    return math.isclose(value, reference, rel_tol=1e-9, abs_tol=1e-9)


def paired(left, right):
    assert left.keys() == right.keys()
    wins = sum(left[k] and not right[k] for k in left)
    losses = sum(right[k] and not left[k] for k in left)
    n = wins+losses
    p = min(1., 2*sum(math.comb(n, k) for k in range(min(wins, losses)+1))/2**n) if n else 1.
    return {'left_only_correct': wins, 'right_only_correct': losses,
            'descriptive_exact_discordance_p': p, 'rows': len(left)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve existing evidence'
    root = args.root
    terminal_path = root/'session_result.json'
    terminal = json.loads(terminal_path.read_text())
    assert terminal['complete'] and terminal['training_complete']
    consensus_path = Path('reports/evidence/primary_reference_consensus.json')
    assert sha(consensus_path) == 'deb96602c88b983ecfbf23738d2a64192057273798548ab61a7cdf7acfc7c119'
    consensus = json.loads(consensus_path.read_text())['groups_by_question_sha256']
    runs = {}
    for role in ['control-matched330k', 'control', 'treatment']:
        for decoder in ['greedy', 'beam4-alpha06']:
            p = root/f'decoding-{role}/{decoder}/validation_predictions.jsonl'
            assert sha(p) == terminal['artifact_sha256'][str(p.relative_to(root))]
            runs[f'{role}-{decoder}'] = p
    for decoder in ['greedy', 'beam4-alpha06']:
        runs['b034-300k-'+decoder] = Path(f'artifacts/kaggle/b033-b034-v1/decoding-b034/{decoder}/validation_predictions.jsonl')
    runs['b034-300k-fixed24'] = Path('artifacts/kaggle/sample-scaling-v1/samples24/sample-plurality24_predictions.jsonl')
    metrics = {'legacy': (evaluate_code, False),
               'augassign_fixed_output': (evaluate_code_augassign, False),
               'augassign_additive_1e-6_diagnostic': (evaluate_code_augassign, True)}
    report = {'utc': datetime.now(timezone.utc).isoformat(),
              'terminal_sha256': sha(terminal_path), 'consensus_sha256': sha(consensus_path),
              'analysis_sha256': sha(__file__), 'runs': {}, 'paired': {},
              'evaluator_sha256': {p: sha(p) for p in ['src/nlp_hw02/arithmetic.py', 'src/nlp_hw02/arithmetic_augassign_audit.py']},
              'limits': ['Fixed reused noisy1529 supplied-reference consensus, not official pass1.',
                         'AugAssign extends generated-output support only; no labels or selected outputs changed.',
                         'Additive1e-6 is diagnostic; official comparison formula remains unconfirmed.',
                         'No generated-code execution, training, filtering, test targets or new selection grid.',
                         'Matched30k compares CE-best checkpoints within equal new-draw budgets; control45k is separate.',
                         'Weights-only warmstart reset moments/scaler/RNG at300k; not fullstate continuation.']}
    outcomes = {}
    ordered_questions = None
    for label, path in runs.items():
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 1792
        questions = [hashlib.sha256(row['question'].encode()).hexdigest() for row in rows]
        assert len(set(questions)) == 1792
        assert all(row.get('question_sha256', q) == q for row, q in zip(rows, questions))
        if ordered_questions is None:
            ordered_questions = questions
        assert questions == ordered_questions
        result = {'path': str(path), 'sha256': sha(path), 'metrics': {}}
        outcomes[label] = {}
        for metric, (evaluator, additive) in metrics.items():
            categories = Counter()
            failures = Counter()
            current = {}
            for row, q in zip(rows, questions):
                ref = consensus[q]
                if ref['status'] != 'supported_consensus':
                    continue
                value = evaluator(row['prediction'])
                correct = value.supported and equal(value.value, ref['consensus_value'], additive)
                category = 'correct' if correct else ('wrong_numeric' if value.supported else 'unsupported')
                categories[category] += 1
                if not value.supported:
                    failures[value.reason] += 1
                current[q] = bool(correct)
            assert len(current) == 1529
            result['metrics'][metric] = {'categories': dict(categories), 'unsupported_reasons': dict(failures)}
            outcomes[label][metric] = current
        report['runs'][label] = result
    comparisons = [('control-matched330k-greedy', 'treatment-greedy'),
                   ('control-matched330k-beam4-alpha06', 'treatment-beam4-alpha06'),
                   ('b034-300k-beam4-alpha06', 'control-beam4-alpha06'),
                   ('b034-300k-beam4-alpha06', 'control-matched330k-beam4-alpha06'),
                   ('b034-300k-beam4-alpha06', 'treatment-beam4-alpha06')]
    for left, right in comparisons:
        report['paired'][left+' vs '+right] = {
            metric: paired(outcomes[left][metric], outcomes[right][metric]) for metric in metrics}
    assert report['runs']['b034-300k-fixed24']['metrics']['legacy']['categories']['correct'] == 1087
    assert report['runs']['b034-300k-beam4-alpha06']['metrics']['legacy']['categories']['correct'] == 1002
    report['ordered_question_hashes_sha256'] = hashlib.sha256(json.dumps(ordered_questions, separators=(',', ':')).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'scores': {name: {metric: detail['categories'].get('correct', 0)
                                      for metric, detail in run['metrics'].items()}
                                 for name, run in report['runs'].items()}, 'paired': report['paired']}, indent=2))


if __name__ == '__main__':
    main()
