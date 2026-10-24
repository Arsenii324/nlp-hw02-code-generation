"""Audit recovered actual proof and exactly three predeclared CPU selectors.

No generation/model loading, program execution, data filtering or selector grid.
Reference-free choices are durably written before reference labels are loaded.
Run with PYTHONPATH=src .venv/bin/python scripts/analyze_portable_proof_crossscore.py.
"""
from collections import Counter
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from audit_augassign_metric import equal, select
from nlp_hw02.arithmetic import evaluate_code
from nlp_hw02.arithmetic_augassign_audit import evaluate_code_augassign


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/kaggle/portable-proof-crossscore-v1'
EVID = ROOT / 'reports/evidence'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def ordered_sha(values):
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def logsumexp(values):
    largest = max(values)
    return largest + math.log(math.fsum(math.exp(v - largest) for v in values))


def paired(control, treatment):
    pairs = Counter((a == 'correct', b == 'correct') for a, b in zip(control, treatment))
    wins, losses = pairs[False, True], pairs[True, False]
    n = wins + losses
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2 ** n) if n else 1.
    return {'both_correct': pairs[True, True], 'both_incorrect': pairs[False, False],
            'wins': wins, 'losses': losses, 'descriptive_exact_binomial_p': p}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=EVID / 'portable_proof_crossscore_quality_review_20261023.json')
    parser.add_argument('--choices', type=Path, default=EVID / 'portable_crossscore_reference_free_choices_20261023.jsonl')
    args = parser.parse_args()
    outpath, choicepath = args.output, args.choices
    assert not outpath.exists() and not choicepath.exists(), 'Fresh analysis outputs required'
    terminal = json.loads((ART / 'proof_terminal.json').read_text())
    manifest = json.loads((ART / 'portable_proof_manifest.json').read_text())
    result = json.loads((ART / 'crossscore/crossscore_result.json').read_text())
    timing = json.loads((ART / 'notebook-proof/b034-sampling-draft-temp08-seed20261002/candidate_timing.json').read_text())
    original = json.loads((ART / 'notebook-proof/original_notebook.ipynb.bin').read_text())
    executed = json.loads((ART / 'notebook-proof/executed_notebook.ipynb.bin').read_text())
    executor = json.loads((ART / 'notebook-proof/executed.json').read_text())
    predictions = lines(ART / 'notebook-proof/b034-sampling-draft-temp08-seed20261002/predictions.jsonl')
    frozen = ROOT / '.worktrees/strong-inference-fallback-20261022'
    stage = frozen / 'artifacts/kaggle/portable-proof-crossscore-tools-20261022b'
    checks = {'all16_terminal_artifact_hashes': len(terminal['artifact_sha256']) == 16 and all(sha(ART / name) == v for name, v in terminal['artifact_sha256'].items()),
              'manifest_frozen_stage': sha(ART / 'portable_proof_manifest.json') == sha(stage / 'portable_proof_manifest.json'),
              'driver_frozen_stage': sha(ART / 'run_driver.py') == sha(frozen / 'kaggle/portable-proof-crossscore-20261022b/run.py') == manifest['driver_sha256'],
              'executor_frozen': sha(ART / 'execute_inference_notebook.py') == manifest['inputs_sha256']['execute_inference_notebook.py'],
              'scorer_frozen': sha(ART / 'score_retained_complete_programs.py') == manifest['inputs_sha256']['score_retained_complete_programs.py'] == result['scorer_sha256'],
              'original_notebook_bytes': sha(ART / 'notebook-proof/original_notebook.ipynb.bin') == manifest['notebook_sha256'] == executor['input_sha256'],
              'notebook_all_cell_sources_unchanged': len(original['cells']) == len(executed['cells']) and all(a['cell_type'] == b['cell_type'] and ''.join(a['source']) == ''.join(b['source']) for a, b in zip(original['cells'], executed['cells'])),
              'three_actual_cells_no_error': [c['execution_count'] for c in executed['cells'] if c['cell_type'] == 'code'] == [1, 2, 3] and executor['complete'] and executor['executed_code_cells'] == 3 and all(o['output_type'] != 'error' for c in executed['cells'] for o in c.get('outputs', [])),
              'executed_notebook_hash': sha(ART / 'notebook-proof/executed_notebook.ipynb.bin') == executor['output_sha256'],
              'actual_oneT4': timing['hardware'] == {'gpu': 'Tesla T4', 'visible_gpu_count': 1} and result['gpu'] == 'Tesla T4' and result['visible_gpu_count'] == 1,
              'exact_package_model_policy': timing['package_sha256'] == manifest['candidate_package_sha256'] and timing['model_sha256'] == manifest['models']['b034']['sha256'] and timing['policy'] == manifest['default_policy'] and timing['package_member_sha256'] == manifest['runtime_member_sha256'],
              'code_only1919': len(predictions) == 1919 and all(set(r) == {'code'} and isinstance(r['code'], str) for r in predictions),
              'code_digest_original_reproduction': ordered_sha([r['code'] for r in predictions]) == manifest['original_n24_reproduction']['ordered_codes_sha256'] == timing['ordered_codes_sha256'] and sha(ART / 'notebook-proof/b034-sampling-draft-temp08-seed20261002/predictions.jsonl') == manifest['original_n24_reproduction']['code_only_jsonl_sha256'] and timing['original_n24_reproduction']['matches'],
              'core_strongest_and_whole_child_under180': timing['generation_and_selection']['seconds'] < 180 and timing['notebook_timing_seconds']['notebook_start_through_output_and_initial_sidecar'] < 180 and terminal['notebook-proof']['child_wall_seconds'] < 180,
              'complete_clean_actual_children': terminal['complete'] and terminal['proof_complete'] and terminal['optional_crossscore_complete'] and terminal['cleanup_complete'] and all(c['actual_exit'] == 0 and not c['stopped'] and c['leader_reaped'] and c['group_exited'] and c['cleanup_clean'] for c in terminal['children']),
              'no_selected_candidate_cap': timing['generation_and_selection']['selected_candidate_capped_rows'] == 0}
    import pyarrow.parquet as pq
    testpath = ROOT / '.worktrees/gated-adjacent-warmstart-20261022/artifacts/kaggle/tools-gated-b034-20261022b/test.parquet'
    questions = pq.read_table(testpath, columns=['question'])['question'].to_pylist()
    checks['pinned1919_test_order'] = sha(testpath) == timing['input_sha256'] and len(questions) == 1919 and ordered_sha(questions) == timing['ordered_questions_sha256']
    oldtestpath = ROOT / 'artifacts/kaggle/sample-scaling-v1/samples24/test_plurality24_predictions.jsonl'
    oldtest = lines(oldtestpath)
    checks['actual_original1919_bytes_by_row'] = sha(oldtestpath) == manifest['original_n24_reproduction']['source_artifact_sha256'] and all(r['row_index'] == i and r['code'] == p['code'] for i, (r, p) in enumerate(zip(oldtest, predictions))) and len(oldtest) == 1919
    poolpath = ART / 'crossscore/candidate_pool.jsonl.bin'
    pool = lines(poolpath)
    scores = {name: lines(ART / f'crossscore/{name}_scores.jsonl') for name in ('b033', 'b034')}
    checks['pool_pin_and_order'] = sha(poolpath) == result['pool_sha256'] == manifest['inputs_sha256']['candidate_pool.jsonl'] and len(pool) == 512 and [r['row_index'] for r in pool] == list(range(512))
    retainedpath = ROOT / 'artifacts/kaggle/sample-scaling-v1/samples24/validation24_sample_candidates.jsonl'
    retained = lines(retainedpath)
    checks['retained_original_inputs_pins'] = sha(retainedpath) == manifest['crossscore_input_provenance_sha256']['original_k24'] and all(sha(ROOT / f'artifacts/kaggle/b033-b034-v1/decoding-{name}/beam4-alpha06/validation_predictions.jsonl') == manifest['crossscore_input_provenance_sha256'][name] for name in ('b033', 'b034'))
    tokpath = ROOT / '.worktrees/gated-adjacent-warmstart-20261022/artifacts/kaggle/tools-gated-b034-20261022b/parent-b034/tokenizer.json'
    assert sha(tokpath) == manifest['runtime_member_sha256']['tokenizer.json']
    choices = []; exclusions = []; total = 0; fallbacks = Counter()
    for ri, row in enumerate(pool):
        assert set(row) == {'row_index', 'question', 'question_sha256', 'candidates', 'sample_to_unique', 'sample_scores', 'plurality_control_unique', 'beam_to_unique'}
        assert hashlib.sha256(row['question'].encode()).hexdigest() == row['question_sha256'] == retained[ri]['question_sha256']
        unique = row['candidates']
        assert len(unique) == len(set(unique))
        assert [unique[j] for j in row['sample_to_unique']] == retained[ri]['candidates'] and row['sample_scores'] == retained[ri]['scores']
        control = row['sample_to_unique'][select(retained[ri]['candidates'], row['sample_scores'], evaluate_code)]
        assert control == row['plurality_control_unique'] == row['sample_to_unique'][retained[ri]['selected_index']]
        values = {}; normalized = {}
        for j, code in enumerate(unique):
            a, b = (scores[name][ri] for name in ('b033', 'b034'))
            assert a['row_index'] == b['row_index'] == ri and a['question_sha256'] == b['question_sha256'] == row['question_sha256']
            assert len(a['candidates']) == len(b['candidates']) == len(unique)
            aa, bb = a['candidates'][j], b['candidates'][j]
            assert {k: v for k, v in aa.items() if k != 'logp'} == {k: v for k, v in bb.items() if k != 'logp'}
            # Both independently serialized actual model score files must carry
            # identical canonical lengths. Local shared venv lacks tokenizers;
            # no install/re-tokenization is performed or claimed here.
            length = aa['length_including_eos']
            assert type(length) is int and length >= 1
            assert aa['candidate_index'] == j and aa['code_sha256'] == hashlib.sha256(code.encode()).hexdigest() and aa['length_including_eos'] == length and aa['length_penalty'] == ((5 + length) / 6) ** .6
            assert aa['scoreable'] == (length <= 192)
            if length > 192:
                assert aa['logp'] is None and bb['logp'] is None and aa['reason'] == 'canonical_complete_target_exceeds192'
                exclusions.append({'row_index': ri, 'candidate_index': j, 'length': length}); continue
            assert aa['reason'] is None and all(math.isfinite(x) and x <= 0 for x in (aa['logp'], bb['logp']))
            values[j] = (aa['logp'] + bb['logp']) / 2
            normalized[j] = values[j] / aa['length_penalty']; total += 1
        top = max(normalized, key=normalized.get) if normalized else control
        if not normalized: fallbacks['top_no_scoreable'] += 1
        groups = []
        for j in values:
            resultj = evaluate_code(unique[j])
            if not resultj.supported: continue
            group = next((g for g in groups if equal(g['value'], resultj.value)), None)
            if group is None: group = {'value': resultj.value, 'indices': []}; groups.append(group)
            group['indices'].append(j)
        if groups:
            winning = max(groups, key=lambda g: (logsumexp([values[j] for j in g['indices']]), max(normalized[j] for j in g['indices'])))
            mass = max(winning['indices'], key=normalized.get)
        else: mass = top; fallbacks['mass_no_supported_scoreable'] += 1
        choices.append({'row_index': ri, 'question_sha256': row['question_sha256'], 'indices': {'control': control, 'top_complete': top, 'numeric_captured_mass': mass},
                        'selected_code_sha256': {name: hashlib.sha256(unique[j].encode()).hexdigest() for name, j in (('control', control), ('top_complete', top), ('numeric_captured_mass', mass))}})
    # Reference-free selections fixed BEFORE loading any reference labels.
    with choicepath.open('x') as stream:
        for row in choices: stream.write(json.dumps(row) + '\n')
        stream.flush()
        import os
        os.fsync(stream.fileno())
    checks['all_score_rows_complete_and_caps_verified'] = total == 10428 and len(exclusions) == 2
    # These old beam files also contain reference fields. Only after choices are
    # durable, read an explicit question/prediction projection for provenance.
    beams = {name: [{k: r[k] for k in ('question', 'prediction')} for r in lines(ROOT / f'artifacts/kaggle/b033-b034-v1/decoding-{name}/beam4-alpha06/validation_predictions.jsonl')] for name in ('b033', 'b034')}
    for ri, row in enumerate(pool):
        unique = list(dict.fromkeys(retained[ri]['candidates'] + [beams[n][ri]['prediction'] for n in ('b033', 'b034')]))
        assert row['candidates'] == unique and all(beams[n][ri]['question'] == row['question'] for n in beams)
        assert all(unique[row['beam_to_unique'][n]] == beams[n][ri]['prediction'] for n in beams)
    checks['original1792_first512_pool_replay'] = len(retained) == 1792 and all(len(v) == 1792 for v in beams.values())
    assert all(checks.values()), checks
    refs_path = EVID / 'primary_reference_consensus.json'
    refs = json.loads(refs_path.read_text())['groups_by_question_sha256']
    totals = {metric: {name: Counter() for name in choices[0]['indices']} for metric in ('legacy', 'augassign_fixed_output')}
    outcomes = {metric: {name: [] for name in choices[0]['indices']} for metric in totals}
    scored = []; changes = []; metricchanges = []
    for row, choice in zip(pool, choices):
        qsha = row['question_sha256']; ref = refs[qsha]
        if ref['status'] != 'supported_consensus': continue
        scored.append(qsha); target = ref['consensus_value']; categories = {}
        for metric, evaluator in (('legacy', evaluate_code), ('augassign_fixed_output', evaluate_code_augassign)):
            categories[metric] = {}
            for name, j in choice['indices'].items():
                ev = evaluator(row['candidates'][j]); category = ('correct' if equal(ev.value, target) else 'wrong_numeric') if ev.supported else 'unsupported'
                totals[metric][name][category] += 1; outcomes[metric][name].append(category); categories[metric][name] = category
        if len(set(categories['augassign_fixed_output'].values())) > 1: changes.append({'row_index': row['row_index'], 'question_sha256': qsha, 'categories': categories})
        if categories['legacy'] != categories['augassign_fixed_output']: metricchanges.append({'row_index': row['row_index'], 'question_sha256': qsha, 'categories': categories})
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'reviewer': '/root/gated_stage_review_20261022', 'scope': 'Recovered actual oneT4 notebook proof plus exactly three predeclared CPU selectors on actual fixed512 GPU score arrays; no generation/model execution/cloud/tests/fixtures, dataset edits/filtering or new selector grid.',
              'checks': checks, 'all_checks_pass': all(checks.values()), 'terminal_sha256': sha(ART / 'proof_terminal.json'),
              'pins': {'analysis_script': sha(Path(__file__)), 'manifest': sha(ART / 'portable_proof_manifest.json'), 'pool': sha(poolpath), 'consensus': sha(refs_path), 'reference_free_choices': sha(choicepath), 'legacy_evaluator': sha(ROOT / 'src/nlp_hw02/arithmetic.py'), 'augassign_evaluator': sha(ROOT / 'src/nlp_hw02/arithmetic_augassign_audit.py'), **{n + '_score_array': sha(ART / f'crossscore/{n}_scores.jsonl') for n in scores}},
              'notebook_proof': {'original_notebook_sha256': sha(ART / 'notebook-proof/original_notebook.ipynb.bin'), 'output_sha256': timing['output_sha256'], 'timing': timing['timing_seconds'], 'generation_seconds': timing['generation_and_selection']['seconds'], 'notebook_total_seconds': timing['notebook_timing_seconds']['notebook_start_through_output_and_initial_sidecar'], 'whole_child_seconds': terminal['notebook-proof']['child_wall_seconds'], 'whole_script_seconds': terminal['elapsed_seconds'], 'executor': 'Exact three plain Python notebook code cells in fresh process; not Jupyter/nbclient execution.', 'peak_allocated_bytes': timing['generation_and_selection']['peak_allocated_bytes']},
              'crossscore': {'rows': 512, 'eligible_reference_rows': len(scored), 'fixed_reference_population': 1529, 'scored_question_order_sha256': ordered_sha(scored), 'scoreable_unique': total, 'excluded': exclusions, 'fallback_counts': fallbacks, 'selectors': manifest['predeclared_cpu_selectors'], 'models': result['models'], 'whole_oneT4_child_seconds': terminal['crossscore']['child_wall_seconds'], 'totals': totals, 'paired': {metric: {name + '_vs_control': paired(v['control'], v[name]) for name in ('top_complete', 'numeric_captured_mass')} for metric, v in outcomes.items()}, 'changed_program_rows': {name: sum(c['indices'][name] != c['indices']['control'] for c in choices) for name in ('top_complete', 'numeric_captured_mass')}, 'discordant_quality_rows': changes, 'fixed_output_AugAssign_changes': metricchanges},
              'limits': ['No official pass1: bounded AST proxy and noisy existing reference consensus; AugAssign extends generated-output support only, without reselection or changed labels.', 'First512 reporting questions only; supported subset is not all1529, and crossscore full1792/1919 quality/runtime remain unestablished.', 'Captured mass is logsumexp of raw mean logP over byteexact unique programs, not repeated-sample votes or calibrated answer posterior.', 'Source caps/canonical EOS scoreability do not prove syntactic or semantic completeness; no new syntax filter or truncation was applied.', 'Canonical lengths crosschecked across both pinned actual model files and scorer source/LP/cap semantics, not independently retokenized: local shared venv lacks tokenizers; no installation.', 'Exact output/order matches pinned1919 Parquet and original generated hashes; namedbotorder/officialentrypoint/timer boundary remain unconfirmed.', 'Strongest selftimer excludes its final rewrite/print; whole executor child130.721s and complete parent133.798s independently remain below180 here.', 'Single actual run in concurrent T4x2 session with one visibleT4 per child, not a guarantee across environments. No canonical promotion/submission; B031 unchanged.']}
    outpath.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'checks': len(checks), 'all_pass': True, 'eligible_reference_rows': len(scored), 'totals': totals, 'paired': report['crossscore']['paired'], 'output': str(outpath)}, indent=2))


if __name__ == '__main__':
    main()
