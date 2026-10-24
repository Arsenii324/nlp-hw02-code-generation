"""Bind frozen B notebook generator to actual D staging without relabelling it."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WT = ROOT / '.worktrees/strong-inference-fallback-20261022'
sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(stage):
    record = json.loads((stage / 'staging.json').read_text())
    assert record['schema'] == 'cooldown_selected_candidate_staging_d_v1'
    assert record['packager_sha256'] == '977a5d398003f52e06f2640ab132ad6c94b677ea4521d27f254be754a034b462'
    assert record['run_id'] == 'cooldown-half' and record['deployment_mode'] == 'fixed24'
    assert record['canonical_promotion'] is False
    package = stage / record['package']
    assert record['package'] == 'selected_bundle.zip' and sha(package) == record['package_sha256']
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert set(names) == set(record['member_sha256']) | {'candidate_manifest.json'}
        for name, digest in record['member_sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
        raw = archive.read('candidate_manifest.json')
        assert hashlib.sha256(raw).hexdigest() == record['manifest_sha256']
        manifest = json.loads(raw)
        assert manifest['schema'] == 'cooldown_selected_bundle_b_v1'
        assert manifest['member_sha256'] == record['member_sha256']
        assert manifest['deployment']['model_sha256'] == 'edb1b328116abb82240b094093108e533a0ce026a283f7f5463602631f253089'
        assert manifest['deployment']['root_selection_sha256'] == record['root_selection_sha256'] == '2b3c5c12067a8ff898ddf44cb25018f349a848f86f23535d46994897a6014aef'
    source = WT / 'scripts/build_cooldown_selected_notebook_b.py'
    assert sha(source) == 'e2d9e813749513dce3327b585fa1b86e192b770033b47276ae61cd6af01a6acf'
    spec = importlib.util.spec_from_file_location('frozen_cooldown_notebook_b', source)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    notebook = builder.notebook(record)
    cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    assert len(cells) == 3
    for cell in cells:
        ast.parse(''.join(cell['source']))
    output = stage / 'inference.ipynb'
    with output.open('x') as stream:
        stream.write(json.dumps(notebook, indent=1) + '\n')
    result = {'schema': 'cooldown_d_actual_notebook_binding_v1',
        'package_sha256': sha(package), 'staging_sha256': sha(stage / 'staging.json'),
        'notebook_sha256': sha(output), 'frozen_B_builder_sha256': sha(source),
        'root_binder_sha256': sha(Path(__file__)), 'code_cells': 3,
        'all_package_members_verified': True, 'B_generator_called_without_schema_relabel': True,
        'model_or_provider_action': False, 'canonical_promotion': False,
        'fresh_exact_one_T4_proof_required': True}
    with (stage / 'notebook_binding.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    main(parser.parse_args().stage)
