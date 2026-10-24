"""Score retained real cooldown outputs with bounded AST evaluation, never exec."""
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from nlp_hw02.arithmetic import evaluate_code
from nlp_hw02.arithmetic_augassign_audit import evaluate_code_augassign
from analyze_gated_results_20261023 import equal, paired


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = ROOT / 'artifacts/kaggle/cooldown-prefix-v1-b-diagnostics'
    terminal_path = root / 'session_result.json'
    assert sha(terminal_path) == '71e8ba12573751cef9bef71a036cf73138bfc3bdef104f7cdda12753acbb74ed'
    terminal = json.loads(terminal_path.read_text())
    assert all(terminal[k] for k in ('complete', 'paired_complete', 'training_complete', 'artifact_hashes_complete'))
    assert all(r['reaped'] and r['group_exited'] and not r['errors'] for r in terminal['cleanup'])
    reference_path = ROOT / 'reports/evidence/primary_reference_consensus.json'
    assert sha(reference_path) == 'deb96602c88b983ecfbf23738d2a64192057273798548ab61a7cdf7acfc7c119'
    references = json.loads(reference_path.read_text())['groups_by_question_sha256']
    paths = {}
    for arm in ('cooldown-full', 'cooldown-half'):
        run = next(r for r in terminal['runs'] if r['run_id'] == arm)
        assert run['actual_exit'] == 0 and run['evaluation']['complete']
        assert run['durable_state']['since_resume_iterations'] == 72384
        for decoder in ('greedy', 'beam4-alpha06'):
            path = root / f'evaluation-{arm}/{decoder}/validation_predictions.jsonl'
            assert sha(path) == terminal['artifact_sha256'][str(path.relative_to(root))]
            paths[f'{arm}-{decoder}'] = path
    old = ROOT / 'artifacts/kaggle/gated-b034-final-v1'
    old_terminal = json.loads((old / 'session_result.json').read_text())
    for decoder in ('greedy', 'beam4-alpha06'):
        path = old / f'decoding-control/{decoder}/validation_predictions.jsonl'
        assert sha(path) == old_terminal['artifact_sha256'][str(path.relative_to(old))]
        paths[f'old344-{decoder}'] = path
    evaluators = {'legacy': (lru_cache(None)(evaluate_code), False),
                  'augassign_fixed_output': (lru_cache(None)(evaluate_code_augassign), False),
                  'augassign_additive1e6': (lru_cache(None)(evaluate_code_augassign), True)}
    metrics, outcomes, ordered = {}, {}, None
    for name, path in paths.items():
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        questions = [hashlib.sha256(row['question'].encode()).hexdigest() for row in rows]
        assert len(rows) == len(set(questions)) == 1792
        assert all(row['question_sha256'] == q for row, q in zip(rows, questions))
        ordered = questions if ordered is None else ordered
        assert questions == ordered
        metrics[name] = {'prediction_sha256': sha(path), 'metrics': {}}
        outcomes[name] = {}
        for metric, (evaluate, additive) in evaluators.items():
            categories, current = Counter(), {}
            for row, q in zip(rows, questions):
                ref = references[q]
                if ref['status'] != 'supported_consensus':
                    continue
                value = evaluate(row['prediction'])
                correct = bool(value.supported and equal(value.value, ref['consensus_value'], additive))
                categories['correct' if correct else 'wrong_numeric' if value.supported else 'unsupported'] += 1
                current[q] = correct
            assert len(current) == 1529
            metrics[name]['metrics'][metric] = dict(categories)
            outcomes[name][metric] = current
    assert metrics['old344-greedy']['metrics']['legacy']['correct'] == 975
    assert metrics['old344-beam4-alpha06']['metrics']['legacy']['correct'] == 1025
    comparisons = {}
    for decoder in ('greedy', 'beam4-alpha06'):
        for left, right in [('old344', 'cooldown-full'), ('old344', 'cooldown-half'), ('cooldown-full', 'cooldown-half')]:
            a, b = f'{left}-{decoder}', f'{right}-{decoder}'
            comparisons[a + ' vs ' + b] = {m: paired(outcomes[a][m], outcomes[b][m]) for m in evaluators}
    report = {'schema': 'cooldown_actual_root_numeric_audit_v1', 'utc': datetime.now(timezone.utc).isoformat(),
              'terminal_sha256': sha(terminal_path), 'script_sha256': sha(__file__),
              'reference_sha256': sha(reference_path), 'scores': metrics, 'paired': comparisons,
              'limits': ['Reused noisy fixed1529 supplied-reference agreement, not official pass1 or unbiased selection significance.',
                         'Matched actual72384 NEW attempts for cooldownarms; old344 is a different training budget.',
                         'Actual retained greedy/beam outputs only; no fixed24 quality or coldnotebook runtime claim.',
                         'Bounded AST interpretation only; no generated Python execution, data filtering, targets or test labels.']}
    output = ROOT / 'reports/evidence/cooldown_prefix_actual_root_quality_20261023.json'
    assert not output.exists()
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'scores': {n: {m: v.get('correct', 0) for m, v in x['metrics'].items()} for n, x in metrics.items()},
                      'output': str(output.relative_to(ROOT))}))


if __name__ == '__main__':
    main()
