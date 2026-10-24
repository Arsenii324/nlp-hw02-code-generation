"""One predeclared counterfactual policy on preserved real validation pools.

No generation, model loading, test labels, dataset edits or timing prediction.
Run with PYTHONPATH=src; selector imports are the preserved audit implementation.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from audit_augassign_metric import equal, select
from nlp_hw02.arithmetic import evaluate_code
from nlp_hw02.arithmetic_augassign_audit import evaluate_code_augassign


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prefix_groups(codes):
    groups = []
    supported = 0
    for code in codes:
        result = evaluate_code(code)
        if not result.supported:
            continue
        supported += 1
        group = next((g for g in groups if equal(g['value'], result.value)), None)
        if group is None:
            group = {'value': result.value, 'count': 0}
            groups.append(group)
        group['count'] += 1
    largest = max((g['count'] for g in groups), default=0)
    unanimous = supported == 8 and len(groups) == 1
    cohort = ('unanimous8' if unanimous else 'majority6_7' if largest >= 6
              else 'no_supported' if not supported else 'disputed')
    return unanimous, cohort


def category(code, target):
    result = evaluate_code_augassign(code)
    return ('correct' if equal(result.value, target) else 'wrong_numeric') if result.supported else 'unsupported'


def analyze(path, references):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1792
    seen = set()
    cohorts = {}
    totals = {p: Counter() for p in ('prefix8', 'full24', 'adaptive')}
    sample_counts = Counter()
    changes = []
    for index, row in enumerate(rows):
        q = row['question_sha256']
        assert q not in seen
        seen.add(q)
        codes, scores = row['candidates'], row['scores']
        assert len(codes) == len(scores) == 24
        assert all(math.isfinite(x) for x in scores)
        selected24 = select(codes, scores, evaluate_code)
        assert selected24 == row['selected_index']
        selected8 = select(codes[:8], scores[:8], evaluate_code)
        stop, cohort = prefix_groups(codes[:8])
        selected = selected8 if stop else selected24
        sample_counts[8 if stop else 24] += 1
        c = cohorts.setdefault(cohort, {'rows': 0, 'scored_rows': 0,
            'prefix8': Counter(), 'full24': Counter(), 'adaptive': Counter(),
            'full24_vs_prefix8_wins': 0, 'full24_vs_prefix8_losses': 0,
            'exact_unique_programs_sum_24': 0})
        c['rows'] += 1
        c['exact_unique_programs_sum_24'] += len(set(codes))
        reference = references[q]
        if reference['status'] != 'supported_consensus':
            continue
        c['scored_rows'] += 1
        target = reference['consensus_value']
        values = {key: category(codes[j], target) for key, j in
                  (('prefix8', selected8), ('full24', selected24), ('adaptive', selected))}
        for key, value in values.items():
            totals[key][value] += 1
            c[key][value] += 1
        c['full24_vs_prefix8_wins'] += int(values['full24'] == 'correct' and values['prefix8'] != 'correct')
        c['full24_vs_prefix8_losses'] += int(values['full24'] != 'correct' and values['prefix8'] == 'correct')
        if values['full24'] != values['adaptive']:
            changes.append({'reporting_row': index, 'question_sha256': q,
                            'cohort': cohort, 'full24': values['full24'], 'adaptive': values['adaptive']})
    assert all(sum(c.values()) == 1529 for c in totals.values())
    return {'path': str(path), 'sha256': digest(path), 'rows': len(rows),
            'scored_rows': 1529, 'full24_selector_replays': len(rows),
            'categories': totals, 'cohorts': cohorts, 'counterfactual_sample_counts': sample_counts,
            'counterfactual_mean_samples': sum(n*k for n, k in sample_counts.items())/len(rows),
            'adaptive_vs_full24_changed_categories': changes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', type=Path, action='append', required=True)
    parser.add_argument('--consensus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh evidence path required')
    refs = json.loads(args.consensus.read_text())['groups_by_question_sha256']
    report = {'utc': datetime.now(timezone.utc).isoformat(),
        'policy': 'First8 all supported in one frozen-legacy numeric group:prefix8; else full24. No grid.',
        'scoring': 'Fixed1529 consensus labels; AugAssign diagnostic, original exact-int/1e-9 comparator.',
        'limits': ['Cached counterfactual only; not actual adaptive generation or measured speedup.',
                   'Real compaction/batching changes RNG shapes and generated programs.',
                   'Repeated validation diagnostic is not official pass@1.'],
        'source_sha256': {str(p): digest(p) for p in (Path(__file__),
            Path('scripts/audit_augassign_metric.py'), Path('src/nlp_hw02/arithmetic.py'),
            Path('src/nlp_hw02/arithmetic_augassign_audit.py'), args.consensus,
            Path('docs/plans/2026-10-22-adaptive-sampling-audit.md'))},
        'pools': [analyze(p, refs) for p in args.pool]}
    assert len({tuple(sorted(json.loads(line)['question_sha256'] for line in p.read_text().splitlines())) for p in args.pool}) == 1
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    for p in report['pools']:
        print(json.dumps({k: p[k] for k in ('path','categories','counterfactual_sample_counts','counterfactual_mean_samples')}))


if __name__ == '__main__':
    main()
