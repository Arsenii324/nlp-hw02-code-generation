"""Score actual E outputs against the unchanged reference consensus; no code exec."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from analyze_gated_results_20261023 import equal, paired
from nlp_hw02.arithmetic import evaluate_code
from nlp_hw02.arithmetic_augassign_audit import evaluate_code_augassign


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve previous evidence'
    terminal = json.loads((args.root/'session_result.json').read_text())
    assert terminal['complete'] and terminal['artifact_hashes_complete']
    assert all(r['reaped'] and r['group_exited'] for r in terminal['cleanup'])
    validation = Path('artifacts/kaggle/prepare-v2/cache/validation.json')
    assert sha(validation) == '73ac2a8b8ea0c16c2eec38c219b62f46857cdfe3de7fab3cab64754aa8b4c24e'
    question_path = Path('.worktrees/strong-inference-fallback-20261022/artifacts/kaggle/trained-gated-sampling-tools-20261023b/inputs/validation_questions.json')
    assert sha(question_path) == '83ea3ea9f449619e015d98e21b1c35ec2cc6d2edd01db0c026301e718014f04e'
    questions = json.loads(question_path.read_text())
    assert set(questions).issubset({r['question'] for r in json.loads(validation.read_text())})
    qhashes = [hashlib.sha256(q.encode()).hexdigest() for q in questions]
    assert len(questions) == len(set(qhashes)) == 1792
    order_sha = hashlib.sha256(json.dumps(questions, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    consensus_path = Path('reports/evidence/primary_reference_consensus.json')
    assert sha(consensus_path) == 'deb96602c88b983ecfbf23738d2a64192057273798548ab61a7cdf7acfc7c119'
    consensus = json.loads(consensus_path.read_text())['groups_by_question_sha256']
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'terminal_sha256': sha(args.root/'session_result.json'),
              'analysis_sha256': sha(Path(__file__)), 'consensus_sha256': sha(consensus_path),
              'ordered_validation_questions_sha256': order_sha, 'runs': {}, 'paired': {},
              'limits': ['Unchanged reused noisy1529 reference consensus, not official pass@1.',
                         'Matched roles have equal30k new training draws; final control has45k and is a separate comparison.',
                         'Sampling counts8/16/24 have different compute budgets and RNG trajectories.',
                         'AugAssign changes only evaluator support; additive tolerance is diagnostic, not confirmed judge code.',
                         'No generated Python execution, label changes, filtering, training or test-quality targets.']}
    metric_specs = {'legacy': (evaluate_code, False), 'augassign_fixed_output': (evaluate_code_augassign, False),
                    'augassign_additive_1e-6_diagnostic': (evaluate_code_augassign, True)}
    outcomes = {}

    def score(label, codes, provenance):
        assert len(codes) == 1792
        run = {**provenance, 'metrics': {}}
        outcomes[label] = {}
        for metric, (evaluator, additive) in metric_specs.items():
            categories, failures, current = Counter(), Counter(), {}
            for code, q in zip(codes, qhashes):
                ref = consensus[q]
                if ref['status'] != 'supported_consensus':
                    continue
                value = evaluator(code)
                correct = value.supported and equal(value.value, ref['consensus_value'], additive)
                categories['correct' if correct else ('wrong_numeric' if value.supported else 'unsupported')] += 1
                if not value.supported:
                    failures[value.reason] += 1
                current[q] = bool(correct)
            assert len(current) == 1529
            run['metrics'][metric] = {'categories': dict(categories), 'unsupported_reasons': dict(failures)}
            outcomes[label][metric] = current
        report['runs'][label] = run

    for role, counts in [('control-matched', [24]), ('control-final', [24]), ('treatment-matched', [8, 16])]:
        actual = terminal['roles'][role]
        assert actual['complete'] and actual['parity']['passed']
        for count in counts:
            label = f'{role}-n{count}'
            prefix = f'validation_n{count}'
            timing = actual['paths'][prefix]
            p = args.root/role/(prefix+'_predictions.jsonl')
            pool = args.root/role/(prefix+'_candidates.jsonl')
            for path in [p, pool]:
                assert sha(path) == terminal['artifact_sha256'][str(path.relative_to(args.root))]
            assert sha(p) == timing['prediction_sha256'] and sha(pool) == timing['candidate_sha256']
            assert timing['ordered_questions_sha256'] == order_sha
            rows = [json.loads(line) for line in p.read_text().splitlines()]
            assert len(rows) == 1792 and all(set(row) == {'code'} and isinstance(row['code'], str) for row in rows)
            pool_rows = [json.loads(line) for line in pool.read_text().splitlines()]
            assert len(pool_rows) == 1792
            for i, (row, q, prediction) in enumerate(zip(pool_rows, qhashes, rows)):
                assert row['row_index'] == i and row['question_sha256'] == q and row['samples'] == count
                assert prediction['code'] == row['candidates'][row['selected_index']]
            score(label, [r['code'] for r in rows], {'prediction_sha256': sha(p), 'candidate_sha256': sha(pool),
                  'selected_step': actual['selected_step'], 'checkpoint_sha256': actual['checkpoint_sha256'],
                  'training_budget_new_iterations': actual['training_budget_new_iterations'],
                  'test_complete_path_seconds': actual['paths'][f'test_n{count}']['complete_path_seconds']})
    old = Path('artifacts/kaggle/sample-scaling-v1/samples24/sample-plurality24_predictions.jsonl')
    rows = [json.loads(line) for line in old.read_text().splitlines()]
    assert [r['question'] for r in rows] == questions
    score('b034-300k-n24', [r['prediction'] for r in rows], {'prediction_sha256': sha(old), 'selected_step': 300000})
    assert report['runs']['b034-300k-n24']['metrics']['legacy']['categories']['correct'] == 1087
    labels = list(outcomes)
    for i, left in enumerate(labels):
        for right in labels[i+1:]:
            report['paired'][left+' vs '+right] = {m: paired(outcomes[left][m], outcomes[right][m]) for m in metric_specs}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'scores': {name: {metric: d['categories'].get('correct', 0) for metric, d in run['metrics'].items()}
                                 for name, run in report['runs'].items()}, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
