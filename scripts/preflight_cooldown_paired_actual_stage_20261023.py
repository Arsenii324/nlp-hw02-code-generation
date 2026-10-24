"""Frozen verifier on a real staged projection: stdlib only, no tensor loading."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve prior evidence'
    assert 'torch' not in sys.modules, 'Actual metadata-only preflight requires a fresh stdlib process'
    sys.dont_write_bytecode = True  # Import the real staged adapter without adding __pycache__ payloads.
    stage = args.stage.resolve()
    tools = stage / 'tools'
    receipt = json.loads((stage / 'build_receipt.json').read_text())
    manifest_path = tools / 'cooldown_pair_sampling_manifest_a.json'
    manifest = json.loads(manifest_path.read_text())
    assert sha(manifest_path) == receipt['manifest_sha256']
    assert {str(p.relative_to(tools)) for p in tools.rglob('*') if p.is_file()} == set(receipt['tools_sha256'])
    assert all(sha(tools / name) == digest for name, digest in receipt['tools_sha256'].items())
    models = {str(p.relative_to(tools)) for p in tools.rglob('*.pt')}
    assert models == {f'training_projection/run-{role}/durable-slot1/model.pt'
                      for role in ('cooldown-full', 'cooldown-half')}
    assert not any(tools.rglob('*.npy')) and not any(p.is_symlink() for p in tools.rglob('*'))
    adapter_path = tools / 'scripts/cooldown_sampling_contract_b.py'
    assert sha(adapter_path) == '4143a53c406b4e5a01dd4e929342acb1e05d79387fed408c74ad713edf0bf9c9'
    module_spec = importlib.util.spec_from_file_location('actual_projected_cooldown_contract_b', adapter_path)
    adapter = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(adapter)
    result = {'schema': 'cooldown_paired_actual_projection_metadata_preflight_v1',
              'started_utc': datetime.now(timezone.utc).isoformat(), 'complete': False,
              'manifest_sha256': sha(manifest_path), 'build_receipt_sha256': sha(stage / 'build_receipt.json'),
              'staged_adapter_sha256': sha(adapter_path), 'script_sha256': sha(__file__),
              'roles': {}, 'no_tensor_or_generated_code_execution': True,
              'scope': 'Real staged payload and frozen stdlib verify_training on copied in-memory contracts. '
                       'Selected weights are byte-hashed, never deserialized. Fullstates/arrays absent. '
                       'No fixtures, GPU, inference, provider action, source or staged contract mutation.'}
    try:
        for role in ('cooldown-full', 'cooldown-half'):
            path = tools / manifest['roles'][role]['contract_path']
            original = json.loads(path.read_text())
            bound = deepcopy(original)
            bound['training']['artifact_root'] = str(tools / 'training_projection')
            bound['training']['tools_root'] = str(tools / 'training_tools')
            bound['input_root'] = str(tools / 'inputs')
            identity, files, inputs, source = adapter.verify_training(bound)
            assert identity['run_id'] == role and identity['step'] == 415656
            assert identity['absolute_completed_step'] == 417384 and identity['common_NEW_horizon'] == 72384
            assert identity['paired_complete'] and identity['equal_draw_comparison_supported']
            assert set(files) == {'model.pt', 'metrics.json', 'run_config.json', 'data_stats.json',
                                  'tokenizer.json', 'validation.json', 'validation_legacy.json'}
            result['roles'][role] = {'identity': identity, 'original_contract_sha256': sha(path),
                                    'verified_non_fullstate_file_count': len(files),
                                    'verified_input_sha256': {name: sha(p) for name, p in inputs.items()},
                                    'training_source_archive_sha256': sha(source)}
            assert json.loads(path.read_text()) == original, 'Staged contract mutation forbidden'
        assert 'torch' not in sys.modules
        assert all(sha(tools / name) == digest for name, digest in receipt['tools_sha256'].items())
        assert {str(p.relative_to(tools)) for p in tools.rglob('*') if p.is_file()} == set(receipt['tools_sha256'])
        result.update(complete=True, staged_payload_unchanged=True, torch_imported=False)
    except BaseException as error:
        result.update(failure_type=type(error).__name__, failure=str(error)[:2000])
        raise
    finally:
        result['finished_utc'] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': result['complete'], 'roles': list(result['roles']),
                      'torch_imported': False, 'evidence_sha256': sha(args.output)}))


if __name__ == '__main__':
    main()
