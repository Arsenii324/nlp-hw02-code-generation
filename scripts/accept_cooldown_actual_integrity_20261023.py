"""Root actual-metadata acceptance; no Torch import, model forward, or test data."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
WT = ROOT / '.worktrees/strong-inference-fallback-20261022'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    freeze = WT / 'configs/cooldown_portable_source_b_20261023.json'
    assert sha(freeze) == '02694489cef9f9c085768d83bd6d8e97a54a7377a0dad557612d3e6da62d22c5'
    assert all(sha(WT / name) == digest for name, digest in read(freeze)['source_sha256'].items())
    independent = ROOT / 'reports/evidence/cooldown_actual_whole_independent_review_20261023.json'
    assert sha(independent) == 'f43bc4b30ca3374212007ca132c047d839e91f7f3a27054cc99f99e61f0ce7a7'
    assert read(independent)['all_evidence_checks_pass']
    sys.path.insert(0, str(WT / 'scripts'))
    import cooldown_sampling_contract_b as contract
    assert 'torch' not in sys.modules
    root = ROOT / 'artifacts/kaggle/cooldown-prefix-v1-b-diagnostics'
    tools = ROOT / '.worktrees/gated-cooldown-prefix-20261023/artifacts/kaggle/cooldown-prefix-tools-20261023b'
    terminal_path = root / 'session_result.json'
    assert sha(terminal_path) == '71e8ba12573751cef9bef71a036cf73138bfc3bdef104f7cdda12753acbb74ed'
    terminal = read(terminal_path)
    manifest_path = root / 'gated_cooldown_continuation_manifest.json'
    assert sha(manifest_path) == contract.TRAINING_MANIFEST_SHA
    manifest = read(manifest_path)
    evidence = ROOT / 'reports/evidence/cooldown_prefix_launch_20261023b'
    recovery = evidence / 'training_recovery_normalized_b.json'
    assert sha(recovery) == '235ad5cd2101cd19421c08811b518590f98ec13bea848541942b956dc1a6bb31'
    assert read(recovery)['complete']

    def artifact(name, digest=None):
        path = contract.rooted(root, name)
        actual = sha(path)
        assert actual == terminal['artifact_sha256'][name]
        assert digest is None or actual == digest
        return path

    records = {}
    for run_id in ('cooldown-full', 'cooldown-half'):
        run = next(r for r in terminal['runs'] if r['run_id'] == run_id)
        definition = next(r for r in manifest['runs'] if r['run_id'] == run_id)
        worker_contract_path = tools / definition['contract_path']
        assert sha(worker_contract_path) == manifest['input_sha256'][definition['contract_path']]
        worker_contract = read(worker_contract_path)
        pointer_path = artifact(f'run-{run_id}/latest_checkpoint.json')
        pointer = read(pointer_path)
        assert pointer == run['durable_state']
        prefix = f'run-{run_id}/{pointer["slot"]}/'
        metrics = read(artifact(prefix + 'metrics.json'))
        config = read(artifact(prefix + 'run_config.json'))
        result = read(artifact(f'run-{run_id}/worker_result.json'))
        assert metrics == pointer['metrics'] == run['metrics']
        assert result == run['worker_result']
        protocol = contract.verify_cooldown_protocol(terminal, artifact, tools, manifest,
                                                      worker_contract, run, metrics, config, result, pointer)
        generic_path = evidence / f'selected_{run_id}_cpu_tensor_audit.json'
        generic = read(generic_path)
        assert generic['complete'] and generic['model_sha256'] == pointer['files_sha256']['model.pt']
        assert generic['pointer_sha256'] == sha(pointer_path)
        record = read(WT / 'configs/cooldown_root_integrity_b.template.json')
        for key in set(record) & set(generic):
            if key != 'schema':
                record[key] = generic[key]
        record.update({key: protocol[key] for key in (
            'cooldown_arm', 'common_NEW_horizon', 'cooldown_plan_sha256', 'decision_sha256',
            'release_sha256', 'paired_complete', 'equal_draw_comparison_supported',
            'peer_outcomes', 'cooldown_checkpoint', 'D_recheck_artifact_sha256',
            'D_recheck_status', 'evaluation_policy_skips')})
        record.update(complete=True, allow_completed_lane_with_incomplete_peer=False,
                      recipe_sha256=pointer['recipe_sha256'], contract_sha256=sha(worker_contract_path),
                      training_recovery_sha256=sha(recovery),
                      training_source_freeze_sha256=contract.TRAINING_SOURCE_FREEZE_SHA)
        assert all(value is not None for value in record.values())
        proposal = read(WT / f'configs/proposals/{run_id}_integrity_b_20261023a.PROPOSAL.json')
        assert all(record[key] == value for key, value in proposal.items()
                   if key not in ('complete', 'training_recovery_sha256'))
        record.update(root_owned_acceptance=True, created_utc=datetime.now(timezone.utc).isoformat(),
                      independent_actual_protocol_sha256=sha(independent),
                      actual_tensor_audit_sha256=sha(generic_path),
                      root_acceptance_script_sha256=sha(__file__),
                      scope='Actual CPU tensor receipt plus frozen B protocol verifier on actual retained metadata. '
                            'No model forward/Torch import/fullstate read/new provider observation.')
        output = evidence / f'selected_{run_id}_integrity_b.json'
        assert not output.exists()
        output.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
        records[run_id] = {'path': str(output.relative_to(ROOT)), 'sha256': sha(output)}
    assert 'torch' not in sys.modules
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
