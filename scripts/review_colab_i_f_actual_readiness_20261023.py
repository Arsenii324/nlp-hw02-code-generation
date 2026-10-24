"""Read-only replay of three actual I receipts; no provider/model or fixtures."""
import ast
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '.worktrees/strong-inference-fallback-20261022/artifacts/colab'
E = BASE / 'continuation-i-owner-20261023e'
F = BASE / 'continuation-i-owner-20261023f'
RECOVERY = ROOT / 'artifacts/colab/continuation-i-20261023e'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
old_source = (E / 'scripts/colab_single_bucket_h_lifecycle.py').read_text()
new_source = (F / 'scripts/colab_single_bucket_h_lifecycle.py').read_text()
old_manifest = json.loads((E / 'owner_manifest.json').read_text())
manifest = json.loads((F / 'owner_manifest.json').read_text())
assert sha(F / 'owner_manifest.json') == '8f03d9305af8ea3ec2919d19941f0c7e5f53a2da8e594895d6a30a1762d0a260'
expected_source = old_source.replace('return sanitize_json(json.loads(Path(local).read_text()))', 'return json.loads(Path(local).read_text())  # Validate source-bound receipts without mutating tokenizer/token-count fields.')
assert new_source == expected_source
# Execute only the exact two stdlib-free source-bound metadata predicates.
functions = [node for node in ast.parse(new_source).body if isinstance(node, ast.FunctionDef) and node.name in ('prep_ready', 'prep_failed')]
assert len(functions) == 2
namespace = {}
exec(compile(ast.Module(body=functions, type_ignores=[]), '<F actual metadata predicates>', 'exec'), namespace)
paths = {'transport': ('prep-transport-result-json.json', '622afa3da1cebf14a6f55335162cc5eef3c0e80d2fe99fbeda1835f73dc1603c'), 'environment': ('prep-env-result-json.json', '0a3285efcad91af41745ea2c23d1e1522b79adc749df0ccf7261172ebd92af1a'), 'layout': ('prep-parent-layout-receipt-json.json', 'c81129900516e3bdd70635e4892013df6350c7cadf042b53290d626293cf4dde')}
rows = {}
for item in manifest['preparation']['progress_paths']:
    name, expected = paths[item['kind']]
    path = RECOVERY / name
    assert sha(path) == expected
    value = json.loads(path.read_text())  # Exact corrected poll_one decode.
    ready = namespace['prep_ready'](item, value, manifest['preparation'])
    failed = namespace['prep_failed'](value)
    assert ready is True and failed is False
    rows[item['kind']] = {'path': str(path.relative_to(ROOT)), 'sha256': expected, 'ready': ready, 'failed': failed}
changed_payload = [p for p in manifest['local_stage_sha256'] if sha(E / p) != sha(F / p)]
changed_manifest = [k for k in manifest if manifest[k] != old_manifest.get(k)]
checks = {'exact_one_line_parser_delta': new_source == expected_source, 'all10_payload_SHA': all(sha(F / p) == value for p, value in manifest['local_stage_sha256'].items()), 'only_controller_payload_changed': changed_payload == ['scripts/colab_single_bucket_h_lifecycle.py'], 'only_manifest_paths_and_controller_hash_changed': set(changed_manifest) == {'colab_command', 'local_controller_sha256', 'local_stage_sha256'}, 'all3_actual_receipts_ready_not_failed': all(x['ready'] and not x['failed'] for x in rows.values()), 'raw_CLI_text_logging_still_disabled': ast.dump(next(n for n in ast.parse(old_source).body if isinstance(n, ast.FunctionDef) and n.name == 'run_cli')) == ast.dump(next(n for n in ast.parse(new_source).body if isinstance(n, ast.FunctionDef) and n.name == 'run_cli'))}
assert all(checks.values())
result = {'schema': 'colab_i_owner_f_actual_readiness_independent_v1', 'utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS_SCOPED_ACTUAL_RECEIPT_REPLAY', 'manifest_sha256': sha(F / 'owner_manifest.json'), 'controller_sha256': sha(F / 'scripts/colab_single_bucket_h_lifecycle.py'), 'audit_source_sha256': sha(Path(__file__)), 'checks': checks, 'actual_receipts': rows, 'limits': ['No provider actions, model/array/PT reads, training or synthetic fixtures.', 'Actual I receipts validate readiness predicates; this is not evidence of F running or training success.', 'Only authenticated source-bound structured metadata bypasses substring redaction. Arbitrary CLI output persistence remains disabled.']}
output = ROOT / 'reports/evidence/colab_continuation_launch_20261023i/owner_f_actual_readiness_independent.json'
assert not output.exists()
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'evidence': str(output.relative_to(ROOT)), 'sha256': sha(output), 'audit_sha256': result['audit_source_sha256']}, indent=2))
