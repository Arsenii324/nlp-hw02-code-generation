"""Rescore preserved real predictions under a narrowly extended numeric proxy.

No model generation, training, Python execution or reference-label changes.
"""
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


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def equal(a, b):
    return a == b if isinstance(a, int) and isinstance(b, int) else math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def select(codes, scores, evaluator):
    groups = []
    for i, code in enumerate(codes):
        result = evaluator(code)
        if not result.supported:
            continue
        group = next((g for g in groups if equal(g['value'], result.value)), None)
        if group is None:
            group = {'value': result.value, 'indices': []}
            groups.append(group)
        group['indices'].append(i)
    if not groups:
        return 0
    best = max(groups, key=lambda g: (len(g['indices']), max(scores[i] for i in g['indices'])))
    return max(best['indices'], key=lambda i: scores[i])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--consensus', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh evidence output required')
    quality = args.root / 'quality'
    paths = {name: quality / name for name in (
        'beam4_predictions.jsonl', 'sample-plurality8_predictions.jsonl',
        'validation_sample_candidates.jsonl')}
    beam, selected, pools = [lines(paths[name]) for name in paths]
    assert len(beam) == len(selected) == len(pools) == 1792
    references = json.loads(args.consensus.read_text())['groups_by_question_sha256']
    outcomes = {label: [] for label in ('legacy_beam', 'legacy_selected8',
        'augassign_beam', 'augassign_fixed_selected8', 'augassign_reselected8')}
    reasons = {label: Counter() for label in outcomes}
    fixed_changes = []
    reselection_changes = []
    old_success_changed = 0
    for index, (b, s, pool) in enumerate(zip(beam, selected, pools)):
        assert b['question'] == s['question']
        question_sha = hashlib.sha256(b['question'].encode()).hexdigest()
        assert pool['question_sha256'] == question_sha
        assert len(pool['candidates']) == len(pool['scores']) == 8
        assert all(math.isfinite(score) for score in pool['scores'])
        old_index = select(pool['candidates'], pool['scores'], evaluate_code)
        assert old_index == pool['selected_index']
        assert pool['candidates'][old_index] == s['prediction']
        new_index = select(pool['candidates'], pool['scores'], evaluate_code_augassign)
        if new_index != old_index:
            reselection_changes.append({'row': index, 'question_sha256': question_sha,
                'old_index': old_index, 'new_index': new_index})
        reference = references[question_sha]
        for code in [b['prediction'], *pool['candidates']]:
            old, new = evaluate_code(code), evaluate_code_augassign(code)
            if old.supported and (not new.supported or old.value != new.value):
                old_success_changed += 1
        if reference['status'] != 'supported_consensus':
            continue
        target = reference['consensus_value']
        inputs = {
            'legacy_beam': (b['prediction'], evaluate_code),
            'legacy_selected8': (s['prediction'], evaluate_code),
            'augassign_beam': (b['prediction'], evaluate_code_augassign),
            'augassign_fixed_selected8': (s['prediction'], evaluate_code_augassign),
            'augassign_reselected8': (pool['candidates'][new_index], evaluate_code_augassign),
        }
        row = {}
        for label, (code, evaluator) in inputs.items():
            result = evaluator(code)
            category = ('correct' if equal(result.value, target) else 'wrong_numeric') if result.supported else 'unsupported'
            outcomes[label].append(category)
            row[label] = category
            if not result.supported:
                reasons[label][result.reason] += 1
        if row['legacy_beam'] != row['augassign_beam'] or row['legacy_selected8'] != row['augassign_fixed_selected8']:
            fixed_changes.append({'row': index, 'question_sha256': question_sha, 'categories': row})
    assert old_success_changed == 0
    assert all(len(values) == 1529 for values in outcomes.values())
    paired = {}
    for control, treatment in [('legacy_beam', 'legacy_selected8'),
        ('augassign_beam', 'augassign_fixed_selected8'), ('augassign_beam', 'augassign_reselected8')]:
        pairs = Counter((a == 'correct', b == 'correct') for a, b in zip(outcomes[control], outcomes[treatment]))
        wins, losses = pairs[False, True], pairs[True, False]
        n = wins + losses
        p = min(1., 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2 ** n) if n else 1.
        paired[treatment + '_vs_' + control] = {'wins': wins, 'losses': losses, 'descriptive_paired_p': p}
    report = {
        'utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Fixed legacy1529 reference-consensus rows; diagnostic generated-code AugAssign support only.',
        'file_sha256': {str(path): sha(path) for path in [*paths.values(), args.consensus,
            Path(__file__), Path('src/nlp_hw02/arithmetic.py'), Path('src/nlp_hw02/arithmetic_augassign_audit.py')]},
        'categories': {key: dict(Counter(value)) for key, value in outcomes.items()},
        'unsupported_reasons': {key: dict(value) for key, value in reasons.items()},
        'paired': paired, 'fixed_output_metric_changes': fixed_changes,
        'candidate_reselection_changes': reselection_changes,
        'legacy_supported_values_changed': old_success_changed,
        'limits': ['Not official pass@1 or a full Python evaluator.',
            'Print arguments remain ignored; imports, conditions, loops and other legacy refusals remain.',
            'No reference labels, generation policy, archived selector or selected bundle changed.',
            'Reference mistakes and question ambiguity remain.'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('categories', 'paired', 'legacy_supported_values_changed')}, indent=2))


if __name__ == '__main__':
    main()
