"""Audit fixed masked-repair outputs against existing references; no code execution."""
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
    root = Path('artifacts/kaggle/masked-repair-pilot-v1-c')
    output = Path('reports/evidence/masked_repair_quality_20261023.json')
    assert not output.exists(), 'Preserve existing evidence'
    receipt = json.loads(Path('reports/evidence/masked_repair_launch_20261023c/recovery.json').read_text())
    for item in receipt['files']:
        path = root/item['name']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
    terminal = json.loads((root/'session_result.json').read_text())
    assert terminal['training_complete'] and terminal['artifact_hashing_complete']
    assert not terminal['complete'] and not terminal['all_profiles_complete']
    qpath = Path('.worktrees/strong-inference-fallback-20261022/artifacts/kaggle/trained-gated-sampling-tools-20261023b/inputs/validation_questions.json')
    assert sha(qpath) == '83ea3ea9f449619e015d98e21b1c35ec2cc6d2edd01db0c026301e718014f04e'
    questions = json.loads(qpath.read_text())
    qhashes = [hashlib.sha256(q.encode()).hexdigest() for q in questions]
    assert len(questions) == len(set(qhashes)) == 1792
    cpath = Path('reports/evidence/primary_reference_consensus.json')
    assert sha(cpath) == 'deb96602c88b983ecfbf23738d2a64192057273798548ab61a7cdf7acfc7c119'
    consensus = json.loads(cpath.read_text())['groups_by_question_sha256']
    metrics = {'legacy': (evaluate_code, False), 'augassign_fixed_output': (evaluate_code_augassign, False),
               'augassign_additive_1e-6_diagnostic': (evaluate_code_augassign, True)}
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'analysis_sha256': sha(Path(__file__)),
              'terminal_sha256': sha(root/'session_result.json'), 'recovery_files_rehashed': len(receipt['files']),
              'consensus_sha256': sha(cpath), 'runs': {}, 'paired_vs_draft': {}, 'paired_vs_current': {},
              'limits': ['Fixed reused noisy1529 reference consensus; not official pass@1.',
                         'All eight outputs were completed and hash-bound before a final logging TypeError; overall run failed.',
                         'No generated Python execution, new labels, filtering, model inference or training.',
                         'Masked CE is not autoregressive CE or numeric quality.',
                         'Policy menu is reused validation tuning; oracle union is diagnostic, not a deployable selector.',
                         'Repair timings exclude draft generation and combined cold1919 proof has not been measured.']}
    outcomes = {}
    codes_by_label = {}

    def score(label, codes, provenance):
        assert len(codes) == 1792 and all(isinstance(c, str) for c in codes)
        run = {**provenance, 'metrics': {}}
        outcomes[label] = {}
        codes_by_label[label] = codes
        for metric, (evaluator, additive) in metrics.items():
            categories, failures, values = Counter(), Counter(), {}
            for code, q in zip(codes, qhashes):
                ref = consensus[q]
                if ref['status'] != 'supported_consensus':
                    continue
                value = evaluator(code)
                correct = value.supported and equal(value.value, ref['consensus_value'], additive)
                categories['correct' if correct else ('wrong_numeric' if value.supported else 'unsupported')] += 1
                if not value.supported:
                    failures[value.reason] += 1
                values[q] = bool(correct)
            assert len(values) == 1529
            run['metrics'][metric] = {'categories': dict(categories), 'unsupported_reasons': dict(failures)}
            outcomes[label][metric] = values
        report['runs'][label] = run

    draft = None
    for lane in (0, 1):
        profile = json.loads((root/f'masked-repair/profiles/lane-{lane}/profile_result.json').read_text())
        assert profile['checkpoint_step'] == 20000 and profile['status'] == 'complete'
        assert profile['finished_before_deadline'] and profile['elapsed_seconds'] < profile['max_seconds']
        assert profile['error']['type'] == 'TypeError'
        for policy in profile['profile']['policies']:
            assert policy['status'] == 'complete' and policy['rows_written'] == 1792
            path = root/f"masked-repair/profiles/lane-{lane}/candidates/validation_1792_{policy['policy']}.jsonl"
            assert sha(path) == policy['output_sha256'] == terminal['artifact_sha256'][str(path.relative_to(root))]
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(rows) == 1792
            assert [r['row_index'] for r in rows] == list(range(1792))
            assert [r['question'] for r in rows] == questions
            originals = [r['original_draft_text'] for r in rows]
            if draft is None:
                draft = originals
                score('original-draft', draft, {'drafts_sha256': profile['drafts_sha256']})
            assert originals == draft
            repaired = [r['repaired_text'] for r in rows]
            score(f"lane{lane}-{policy['policy']}", repaired,
                  {'output_sha256': sha(path), 'checkpoint_sha256': profile['checkpoint_sha256'],
                   'wall_seconds': policy['total_wall_seconds'],
                   'text_changed_rows': sum(a != b for a, b in zip(draft, repaired)),
                   'token_changed_rows': policy['rows_with_token_changes']})
    current = Path('artifacts/kaggle/control344-proof-v1-c/validation-parity/validation_n24_predictions.jsonl')
    proof = json.loads(Path('artifacts/kaggle/control344-proof-v1-c/session_result.json').read_text())
    assert sha(current) == proof['artifact_sha256'][str(current.relative_to(current.parents[1]))]
    current_rows = [json.loads(line) for line in current.read_text().splitlines()]
    score('current344k-fixed24', [r['code'] for r in current_rows], {'prediction_sha256': sha(current)})
    for label in outcomes:
        if label in ('original-draft', 'current344k-fixed24'):
            continue
        report['paired_vs_draft'][label] = {m: paired(outcomes[label][m], outcomes['original-draft'][m]) for m in metrics}
        report['paired_vs_current'][label] = {m: paired(outcomes[label][m], outcomes['current344k-fixed24'][m]) for m in metrics}
    repair_labels = [k for k in outcomes if k.startswith('lane')]
    report['oracle_original_or_any_repair'] = {m: sum(any(outcomes[k][m][q] for k in ['original-draft', *repair_labels])
                                                    for q in outcomes['original-draft'][m]) for m in metrics}
    report['evaluator_sha256'] = {str(p): sha(p) for p in map(Path, ['src/nlp_hw02/arithmetic.py', 'src/nlp_hw02/arithmetic_augassign_audit.py'])}
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'scores': {k: {m: v['categories'].get('correct', 0) for m, v in r['metrics'].items()} for k, r in report['runs'].items()},
                      'oracle': report['oracle_original_or_any_repair'], 'output': str(output)}))


if __name__ == '__main__':
    main()
