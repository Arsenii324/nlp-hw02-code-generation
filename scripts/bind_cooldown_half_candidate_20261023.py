"""Bind an actual half-cooldown candidate, without building/running a model."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WT = ROOT / '.worktrees/strong-inference-fallback-20261022'
STAGE = ROOT / 'artifacts/kaggle/cooldown-paired-n24-20261023a/tools'
ACTUAL = ROOT / 'artifacts/kaggle/cooldown-paired-n24-v1-a-diagnostics'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    quality_path = ROOT / 'reports/evidence/cooldown_pair_n24_actual_quality_independent_review_20261023.json'
    assert sha(quality_path) == '5696385fea8f06e956d76d23f3a3b5dd9f3b43b187d729d751d7447fe59c8a10'
    quality = read(quality_path)
    assert quality['all_checks_pass'] is True
    terminal_path = ACTUAL / 'session_result.json'
    assert sha(terminal_path) == quality['terminal_sha256'] == '5accb52bf6b291f1b1d1db677a83bf7086aa37842dc91b205d4d6c2251c2b3b9'
    role_path = ACTUAL / 'run-cooldown-half/cooldown_sampling_result.json'
    terminal = read(terminal_path)
    assert sha(role_path) == terminal['artifact_sha256']['run-cooldown-half/cooldown_sampling_result.json']
    role = read(role_path)
    identity = role['identity']
    assert identity['run_id'] == 'cooldown-half'
    assert identity['model_sha256'] == 'edb1b328116abb82240b094093108e533a0ce026a283f7f5463602631f253089'
    assert identity['step'] == 415656
    assert identity['common_NEW_horizon'] == 72384
    assert role['complete'] is True and role['samples_per_question'] == 24
    assert quality['runtime']['cooldown-half']['identity'] == identity
    metrics = {'correct': 1117, 'wrong_numeric': 409, 'unsupported': 3}
    assert all(v == metrics for v in quality['quality']['cooldown-half']['fixed1529_categories'].values())
    reproductions = {}
    for label, rows in [('test_n24', 1919), ('validation_n24', 1792)]:
        path = ACTUAL / ('run-cooldown-half/' + label + '_predictions.jsonl')
        pool = quality['pools']['cooldown-half/' + label]
        codes = [json.loads(line)['code'] for line in path.read_text().splitlines()]
        assert len(codes) == rows and sha(path) == pool['prediction_sha256']
        ordered = hashlib.sha256(json.dumps(codes, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        assert ordered == pool['ordered_codes_sha256']
        reproductions[label] = {
            'source_artifact_sha256': sha(path), 'code_only_jsonl_sha256': sha(path),
            'ordered_codes_sha256': ordered,
            'ordered_questions_sha256': pool['timing']['ordered_questions_sha256'],
        }
    choice = read(WT / 'configs/cooldown_root_selection_b.template.json')
    choice.update(schema='cooldown_sampling_selection_b_v1', selected_after_actual_quality_audit=True,
        run_id='cooldown-half', layout='bucket', model_sha256=identity['model_sha256'],
        checkpoint_step=415656, samples_per_question=24, generation=role['sampling'],
        training_terminal_sha256=identity['training_terminal_sha256'],
        sampling_terminal_sha256=sha(terminal_path), sampling_role_sha256=sha(role_path),
        reproduction=reproductions['test_n24'], validation_reproduction=reproductions['validation_n24'],
        backend_policy=role['backend_policy'], root_quality_evidence_sha256=sha(quality_path),
        selected_quality_metrics=metrics, canonical_promotion=False,
        decision='SELECT_HALF_FIXED24_FOR_FRESH_PACKAGE_AND_EXACT_ONE_T4_PROOF_ONLY',
        utc=datetime.now(timezone.utc).isoformat(),
        limits=['Reused noisy1529 reference agreement, not official pass@1.',
                '53 wins/35 losses vs old344; descriptive p=.069346, repeated selection and one seed.',
                'Component122.168s excludes cold notebook setup/load; no inherited four-clock proof.',
                'Canonical344k/backendD and all preserved proofs remain unchanged.'])
    for key in ('cooldown_arm', 'common_NEW_horizon', 'cooldown_plan_sha256', 'decision_sha256',
                'release_sha256', 'paired_complete', 'equal_draw_comparison_supported',
                'allow_completed_lane_with_incomplete_peer', 'peer_outcomes',
                'D_recheck_artifact_sha256', 'D_recheck_status', 'evaluation_policy_skips'):
        choice[key] = identity[key]
    choice_path = ROOT / 'reports/evidence/cooldown_half_candidate_selection_20261023.json'
    write_new(choice_path, choice)
    spec = read(STAGE / 'contracts/cooldown-half.json')
    spec['schema'] = 'cooldown_selected_deployment_spec_b_v1'
    spec['training']['artifact_root'] = str(STAGE / 'training_projection')
    spec['training']['tools_root'] = str(STAGE / 'training_tools')
    spec['input_root'] = str(STAGE)
    spec['actual_sampling'] = {'artifact_root': str(ACTUAL),
        'terminal': {'path': 'session_result.json', 'sha256': sha(terminal_path)},
        'parent_sha256': terminal['parent_sha256'],
        'manifest_path': str(STAGE / 'cooldown_pair_sampling_manifest_a.json'),
        'manifest_sha256': sha(STAGE / 'cooldown_pair_sampling_manifest_a.json')}
    spec['root_selection'] = {'path': str(choice_path), 'sha256': sha(choice_path)}
    spec['root_quality_evidence'] = {'path': str(quality_path), 'sha256': sha(quality_path)}
    decision_path = STAGE / 'root_quality_decision.json'
    spec['root_quality_decision'] = {'path': str(decision_path), 'sha256': sha(decision_path)}
    spec['reproduction'] = reproductions['test_n24']
    spec['validation_reproduction'] = reproductions['validation_n24']
    spec_path = WT / 'configs/proposals/cooldown_half_selected_actual_20261023.json'
    write_new(spec_path, spec)
    print(json.dumps({'selection': str(choice_path), 'selection_sha256': sha(choice_path),
                      'spec': str(spec_path), 'spec_sha256': sha(spec_path),
                      'canonical_promotion': False, 'provider_or_model_action': False}))


if __name__ == '__main__':
    main()
