"""Stdlib checks of finite process bounds and frozen TPU staging; no cloud."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

from build_kaggle_tpu_probe import stage, MEMBERS
from tpu_probe_driver import (wait_training, cpu_import_preflight,
                              last_preflight_stage, training_bounds_with_reserve)


def main():
    checks = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        first = stage(root/'tools1', root/'kernel1', 'example/tpu-tools', 'example/tpu-probe', 'TpuV5E8')
        second = stage(root/'tools2', root/'kernel2', 'example/tpu-tools', 'example/tpu-probe', 'TpuV5E8')
        assert first == second
        with zipfile.ZipFile(root/'tools1/tpu_source.bin') as z:
            assert set(z.namelist()) == set(MEMBERS)
            for name, digest in first['source_members'].items():
                assert hashlib.sha256(z.read(name)).hexdigest() == digest
        assert json.loads((root/'kernel1/kernel-metadata.json').read_text())['enable_tpu']
        metadata = json.loads((root/'kernel1/kernel-metadata.json').read_text())
        assert metadata['title'].replace(' ', '-') == metadata['id'].split('/')[1]
        assert first['kernel_id'] == metadata['id']
        assert first['bounds']['cpu_preflight_seconds'] == 60
        try:
            stage(root/'tools1', root/'kernel1', 'example/tpu-tools', 'example/tpu-probe', 'TpuV5E8')
        except RuntimeError:
            pass
        else:
            raise AssertionError('Existing staging overwritten')
        checks.append('deterministic_allowlist_hashes_private_metadata_refuse_overwrite')
        reserve = first['bounds']
        assert training_bounds_with_reserve(reserve, 2)['child_wall_seconds'] == 440
        assert training_bounds_with_reserve(reserve, 60)['child_wall_seconds'] == 390
        assert training_bounds_with_reserve(reserve, 100)['child_wall_seconds'] == 350
        for elapsed in (2, 20, 60, 100, 400):
            effective = training_bounds_with_reserve(reserve, elapsed)
            assert elapsed+effective['child_wall_seconds']+reserve['conversion_seconds']+10 <= 540
        try:
            training_bounds_with_reserve(reserve, 435)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Training admitted without reserve')
        checks.append('preflight_consumes_training_budget_conversion_reserve_preserved')
        bounds = dict(child_wall_seconds=5, startup_seconds=5, compile_seconds=5)
        for name, phase, tighter in [('startup', None, {'startup_seconds': .1}),
                                     ('compile', 'compile_start', {'compile_seconds': .1}),
                                     ('wall', 'training_chunk', {'child_wall_seconds': .1})]:
            case = root/name; case.mkdir()
            progress = case/'progress.json'
            if phase:
                progress.write_text(json.dumps({'event': phase, 'script_seconds': 0}))
            started = time.monotonic()
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
            try:
                wait_training(child, progress, case, started, {**bounds, **tighter})
            except RuntimeError as e:
                assert 'exceeded' in str(e)
            else:
                raise AssertionError('Missing '+name+' deadline')
            assert child.poll() is not None and time.monotonic()-started < 4
            checks.append(name+'_timeout_reaped_real_child')
        case = root/'exit'; case.mkdir()
        for code in (0, 2):
            child = subprocess.Popen([sys.executable, '-c', 'raise SystemExit('+str(code)+')'], start_new_session=True)
            try:
                wait_training(child, case/'progress.json', case, time.monotonic(), bounds)
            except RuntimeError:
                assert code == 2
            else:
                assert code == 0
        checks.append('success_and_failed_child_exits')
        for name, code, expected in [('success', 'print(\'{"backend":"cpu","jax":"fixture","torch":"fixture","tokenizers":"fixture","required_jax_api":true}\')', None),
                               ('missing_dependency', 'raise ImportError("tokenizers fixture missing")', 'failed'),
                               ('wrong_backend', 'print(\'{"backend":"tpu","required_jax_api":true}\')', 'failed'),
                               ('missing_api', 'print(\'{"backend":"cpu","required_jax_api":false}\')', 'failed'),
                               ('timeout', 'import time; time.sleep(30)', 'exceeded')]:
            case = root/('preflight'+str(len(checks))); case.mkdir()
            started = time.monotonic()
            try:
                result = cpu_import_preflight(root, case, {}, .1 if expected == 'exceeded' else 5, code=code)
            except RuntimeError as e:
                assert expected is not None and expected in str(e)
            else:
                assert expected is None and result['backend'] == 'cpu'
            assert time.monotonic()-started < 4
            checks.append('preflight_'+name)
        case = root/'preflight_diagnostic_timeout'; case.mkdir()
        code = '''
import faulthandler, json, os, time
print(json.dumps({'event':'cpu_preflight_stage','stage':'slow_fixture_import','status':'start'}), flush=True)
assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert os.environ['PYTHONUNBUFFERED'] == '1'
faulthandler.dump_traceback_later(max(.01, float(os.environ['TPU_PREFLIGHT_DEADLINE_MONOTONIC'])-time.monotonic()-2))
time.sleep(30)
'''
        try:
            cpu_import_preflight(root, case, {}, .3, code=code)
        except RuntimeError as error:
            assert 'exceeded' in str(error)
        else:
            raise AssertionError('Diagnostic timeout admitted')
        log = (case/'cpu_preflight.log').read_text()
        assert 'Timeout (' in log and 'File "<string>"' in log
        marker = last_preflight_stage(case/'cpu_preflight.log')
        assert marker['stage'] == 'slow_fixture_import' and marker['status'] == 'start'
        event = json.loads((case/'parent.jsonl').read_text().splitlines()[-1])
        assert event['event'] == 'cpu_preflight_failed' and event['last_stage'] == marker
        checks.append('preflight_stack_stage_failure_record_reaped')
        case = root/'preflight_non_utf8_timeout'; case.mkdir()
        code = '''
import json, os, time
print(json.dumps({'event':'cpu_preflight_stage','stage':'native_fixture_import','status':'start'}), flush=True)
os.write(2, b'\\xffpartial-native-stderr')
time.sleep(30)
'''
        try:
            cpu_import_preflight(root, case, {}, .2, code=code)
        except RuntimeError as error:
            assert 'exceeded' in str(error)
        else:
            raise AssertionError('Native diagnostic timeout admitted')
        marker = last_preflight_stage(case/'cpu_preflight.log')
        event = json.loads((case/'parent.jsonl').read_text().splitlines()[-1])
        assert event['last_stage'] == marker and marker['stage'] == 'native_fixture_import'
        assert last_preflight_stage(root/'missing_log') is None
        checks.append('non_utf8_partial_stderr_preserves_timeout_and_stage')
    print(json.dumps({'event': 'local_tpu_staging_checks', 'checks': checks}))


if __name__ == '__main__':
    main()
