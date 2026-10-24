"""Private Kaggle TPU parent: bounded children, frozen identities, no installs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(root, event, **value):
    record = {'event': event, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **value}
    print(json.dumps(record, allow_nan=False), flush=True)
    with (root/'parent.jsonl').open('a') as f:
        f.write(json.dumps(record, allow_nan=False)+'\n')


def stop_child(child):
    """Bounded process-group cleanup; never replace the original failure."""
    prior = getattr(child, '_tpu_cleanup', None)
    if prior and prior['reaped'] and prior['group_exited']:
        return prior
    result = {'pid': child.pid, 'signals': [], 'errors': [], 'reaped': False}
    for signum, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 20)):
        try:
            # Signal the group even if its leader has already exited.
            os.killpg(child.pid, signum)
            result['signals'].append(signal.Signals(signum).name)
        except ProcessLookupError:
            pass
        except OSError as error:
            result['errors'].append(repr(error))
        try:
            child.wait(timeout=grace)
            result['reaped'] = True
        except subprocess.TimeoutExpired as error:
            result['errors'].append(repr(error))
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            break
        except OSError as error:
            result['errors'].append(repr(error))
        # A dead leader can have surviving descendants. Escalate its group.
    group_deadline = time.monotonic()+2
    while time.monotonic() < group_deadline:
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            result['group_exited'] = True
            break
        except OSError as error:
            result['errors'].append(repr(error))
            break
        time.sleep(.05)
    result.setdefault('group_exited', False)
    result['returncode'] = child.poll()
    child._tpu_cleanup = result
    return result


CPU_PREFLIGHT_CODE = '''
import json, os, sys, time
started = time.monotonic()
def record(stage, status, **values):
    print(json.dumps({'event': 'cpu_preflight_stage', 'stage': stage,
        'status': status, 'seconds': time.monotonic()-started, **values}), flush=True)
record('python_start', 'complete', executable=sys.executable,
       requested_backend=os.environ.get('JAX_PLATFORMS'))
import faulthandler, importlib, inspect
faulthandler.enable()
stack_delay = max(.01, float(os.environ['TPU_PREFLIGHT_DEADLINE_MONOTONIC'])-time.monotonic()-2)
faulthandler.dump_traceback_later(stack_delay, repeat=False)
def checked(stage, operation):
    record(stage, 'start')
    phase_started = time.monotonic()
    try:
        result = operation()
    except BaseException as error:
        record(stage, 'error', failure=repr(error))
        raise
    record(stage, 'complete', phase_seconds=time.monotonic()-phase_started)
    return result
def module(name):
    return checked('import_'+name, lambda: importlib.import_module(name))
jax = module('jax')
jnp = module('jax.numpy')
jaxlib = module('jaxlib')
torch = module('torch')
tokenizers = module('tokenizers')
sharding = module('jax.sharding')
model = module('nlp_hw02.model')
port = module('nlp_hw02.tpu_pilot')
Seq2Seq, ModelConfig = model.Seq2Seq, model.ModelConfig
convert_checkpoint, make_compiled_chunk = port.convert_checkpoint, port.make_compiled_chunk
Mesh, NamedSharding, PartitionSpec = sharding.Mesh, sharding.NamedSharding, sharding.PartitionSpec
backend = checked('cpu_backend_discovery', jax.default_backend)
assert backend == 'cpu', 'Preflight must use CPU only'
record('required_jax_api', 'start')
assert all(callable(x) for x in (jax.tree.map, jax.block_until_ready,
    jax.device_put, jax.value_and_grad, jax.lax.scan, convert_checkpoint,
    make_compiled_chunk)), 'Required JAX API missing'
assert {'in_shardings', 'out_shardings', 'donate_argnums'} <= set(inspect.signature(jax.jit).parameters)
assert callable(jax.jit(lambda x: x).lower), 'JAX lowering API missing'
record('required_jax_api', 'complete')
faulthandler.cancel_dump_traceback_later()
print(json.dumps({'event': 'cpu_import_preflight', 'backend': backend,
    'jax': jax.__version__, 'jaxlib': jaxlib.__version__, 'torch': torch.__version__,
    'tokenizers': tokenizers.__version__, 'required_jax_api': True}), flush=True)
'''


def last_preflight_stage(log_path):
    """Return a structured marker, ignoring diagnostic stacks and tracebacks."""
    try:
        lines = Path(log_path).read_text(errors='replace').splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get('event') == 'cpu_preflight_stage':
            return value
    return None


def cpu_import_preflight(source, root, base_env, seconds, code=CPU_PREFLIGHT_CODE):
    """Check deployed conversion dependencies before allocating TPU child state.

    Import/backend checks only; no model initialization, parity run or compilation.
    The separate process is reaped on every failure, timeout or interruption.
    """
    env = {**os.environ, **base_env, 'JAX_PLATFORMS': 'cpu', 'PYTHONPATH': str(Path(source)/'src'),
           'PYTHONUNBUFFERED': '1'}
    log_path = Path(root)/'cpu_preflight.log'
    started = time.monotonic()
    deadline = started+seconds
    env['TPU_PREFLIGHT_DEADLINE_MONOTONIC'] = str(deadline)
    with log_path.open('w') as log:
        child = subprocess.Popen([sys.executable, '-u', '-c', code], env=env, cwd=source,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            try:
                child.wait(timeout=max(.001, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                raise RuntimeError('CPU dependency preflight exceeded '+str(seconds)+' seconds')
            if child.returncode:
                raise RuntimeError('CPU dependency preflight failed; inspect cpu_preflight.log')
        except BaseException as error:
            stop_child(child)
            log.flush()
            emit(root, 'cpu_preflight_failed', failure=repr(error),
                 elapsed_seconds=time.monotonic()-started, bound_seconds=seconds,
                 last_stage=last_preflight_stage(log_path))
            raise
        finally:
            stop_child(child)
    try:
        result = json.loads(log_path.read_text().splitlines()[-1])
    except (OSError, IndexError, json.JSONDecodeError) as error:
        raise RuntimeError('CPU dependency preflight failed to record versions') from error
    if result.get('backend') != 'cpu' or not result.get('required_jax_api'):
        raise RuntimeError('CPU dependency preflight failed backend/API check')
    result['seconds'] = time.monotonic()-started
    emit(root, 'cpu_preflight_complete', **{k: v for k, v in result.items() if k != 'event'})
    return result


def training_bounds_with_reserve(bounds, elapsed):
    """A slower preflight consumes training time, preserving conversion reserve."""
    remaining = bounds['parent_wall_seconds']-elapsed
    child_seconds = min(bounds['child_wall_seconds'],
                        int(remaining-bounds['conversion_seconds']-40))
    if child_seconds < 20:
        raise RuntimeError('No TPU training reserve remains after CPU preflight')
    return {**bounds, 'child_wall_seconds': child_seconds}


def wait_training(child, progress, root, started, bounds, now=time.monotonic):
    """Parent budget includes startup, compile, inputs, logging and shutdown."""
    deadline = started+bounds['child_wall_seconds']
    compile_started = None
    compiled = False
    while child.poll() is None:
        elapsed = now()-started
        try:
            value = json.loads(progress.read_text())
        except (OSError, json.JSONDecodeError):
            value = {}
        phase = value.get('event')
        if phase in ('compile_start', 'lowering_complete', 'backend_compile_complete') and compile_started is None:
            compile_started = started+value['script_seconds']
        if phase in ('compile_complete', 'training_chunk', 'checkpoint_start', 'checkpoint_complete', 'finished'):
            compiled = True
        reason = None
        if now() >= deadline:
            reason = 'child_wall_budget'
        elif compile_started is None and not compiled and elapsed >= bounds['startup_seconds']:
            reason = 'startup_budget'
        elif compile_started is not None and not compiled and now()-compile_started >= bounds['compile_seconds']:
            reason = 'compile_budget'
        if reason:
            emit(root, 'child_abort', reason=reason, child_elapsed_seconds=elapsed, last_progress=value)
            emit(root, 'child_cleanup', **stop_child(child))
            raise RuntimeError('TPU child exceeded '+reason)
        time.sleep(.5)
    if child.returncode:
        raise RuntimeError('TPU training child exit '+str(child.returncode))


def main():
    allocation_start = time.monotonic()
    root = Path('/kaggle/working/tpu-pilot'); root.mkdir(parents=True, exist_ok=True)
    state = {'complete': False, 'probe_only': True, 't4_inference_verified': False}
    child = None
    try:
        inputs = list(Path('/kaggle/input').rglob('tpu_probe_manifest.json'))
        if len(inputs) != 1:
            raise RuntimeError('Require exactly one frozen TPU tool input')
        tools = inputs[0].parent
        manifest = json.loads(inputs[0].read_text())
        if digest(__file__) != manifest['driver_sha256']:
            raise RuntimeError('Driver hash mismatch')
        for name, expected in manifest['checksums'].items():
            if digest(tools/name) != expected:
                raise RuntimeError('Frozen input mismatch: '+name)
        shutil.copyfile(inputs[0], root/'tpu_probe_manifest.json')
        shutil.copyfile(__file__, root/'run_driver.py')
        state['manifest_sha256'] = digest(inputs[0])
        with tempfile.TemporaryDirectory(prefix='nlp-hw02-tpu-') as td:
            source = Path(td)
            with zipfile.ZipFile(tools/'tpu_source.bin') as archive:
                if set(archive.namelist()) != set(manifest['source_members']):
                    raise RuntimeError('Unexpected source archive members')
                for name in archive.namelist():
                    if Path(name).is_absolute() or '..' in Path(name).parts:
                        raise RuntimeError('Unsafe archive member')
                    target = source/name; target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(name))
                    if digest(target) != manifest['source_members'][name]:
                        raise RuntimeError('Extracted source member checksum mismatch')
            caches = list(Path('/kaggle/input').rglob('cache_manifest.json'))
            if len(caches) != 1 or digest(caches[0]) != manifest['cache_manifest_sha256']:
                raise RuntimeError('Expected pinned prepare-v2/v2 CPU cache')
            env = {**os.environ, 'PYTHONPATH': str(source/'src'), 'JAX_PLATFORMS': 'tpu',
                   'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false',
                   'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'python', 'PYTHONFAULTHANDLER': '1'}
            remaining = manifest['bounds']['parent_wall_seconds']-(time.monotonic()-allocation_start)
            preflight_bound = min(manifest['bounds']['cpu_preflight_seconds'], remaining-70)
            if preflight_bound < 5:
                raise RuntimeError('No CPU dependency preflight reserve remains')
            emit(root, 'cpu_preflight_start', bound_seconds=preflight_bound)
            state['cpu_preflight'] = cpu_import_preflight(source, root, env, preflight_bound)
            training_bounds = training_bounds_with_reserve(manifest['bounds'],
                                                          time.monotonic()-allocation_start)
            state['effective_training_bounds'] = training_bounds
            args = manifest['probe']
            command = [sys.executable, '-m', 'nlp_hw02.tpu_pilot', 'probe', '--cache', str(caches[0].parent),
                       '--output', str(root), '--steps', str(args['steps']), '--chunk', str(args['chunk']),
                       '--batch', str(args['batch']), '--seed', str(args['seed']),
                       '--seconds', str(training_bounds['child_wall_seconds']-40), '--optimizer', args['optimizer'],
                       '--initial', str(tools/'initial.npz')]
            with (root/'training_child.log').open('w') as log:
                child_started = time.monotonic()
                child = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                emit(root, 'child_start', stage='training', pid=child.pid, bounds=training_bounds)
                wait_training(child, root/'progress.json', root, child_started, training_bounds)
            result = json.loads((root/'probe_result.json').read_text())
            if not (result['complete'] and result['all_updates_finite'] and result['trace_count'] == 1 and
                    result['weight_change_rms'] > 0 and 20 <= result['steps'] <= args['steps'] and
                    result['optimizer_state_saved']):
                raise RuntimeError('Incomplete or numerically invalid probe evidence')
            training_cleanup = stop_child(child)
            emit(root, 'child_cleanup', **training_cleanup)
            if not (training_cleanup['reaped'] and training_cleanup['group_exited']):
                raise RuntimeError('TPU trainer group still alive; refuse CPU conversion')
            emit(root, 'child_exit', stage='training', returncode=child.returncode,
                 child_seconds=time.monotonic()-child_started)
            # New CPU-only process avoids competing TPU runtimes and separate FP32
            # conversion compilation is excluded from measured TPU training.
            env['JAX_PLATFORMS'] = 'cpu'
            remaining = manifest['bounds']['parent_wall_seconds']-(time.monotonic()-allocation_start)
            conversion_bound = min(manifest['bounds']['conversion_seconds'], remaining-40)
            if conversion_bound < 5:
                raise RuntimeError('No conversion reserve remains')
            with (root/'conversion_child.log').open('w') as log:
                child = subprocess.Popen([sys.executable, '-m', 'nlp_hw02.tpu_pilot', 'convert',
                    '--root', str(root), '--output', str(root/'model.pt')], env=env, cwd=source,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    child.wait(timeout=conversion_bound)
                except subprocess.TimeoutExpired:
                    stop_child(child)
                    raise RuntimeError('CPU checkpoint conversion exceeded reserve')
                if child.returncode:
                    raise RuntimeError('Checkpoint conversion failed; inspect conversion_child.log')
            state.update(complete=True, steps=result['steps'], conversion_verified=True,
                         converted_checkpoint_sha256=digest(root/'model.pt'))
    except BaseException as error:
        if child is not None:
            state['cleanup'] = stop_child(child)
        state['failure'] = repr(error)
        raise
    finally:
        if child is not None:
            state['cleanup'] = stop_child(child)
            if not (state['cleanup']['reaped'] and state['cleanup']['group_exited']):
                state['complete'] = False
                state.setdefault('failure', 'Child cleanup incomplete; platform cap remains final boundary')
        state['allocation_script_seconds'] = time.monotonic()-allocation_start
        state['allocation_charge_note'] = 'Script wall includes CPU preflight/startup/compile/conversion. Platform finalization and quota charge require external snapshots.'
        emit(root, 'parent_terminal', complete=state['complete'], allocation_script_seconds=state['allocation_script_seconds'])
        state['checksums'] = {str(p.relative_to(root)): digest(p) for p in root.rglob('*')
                              if p.is_file() and p.name != 'terminal.json'}
        (root/'terminal.json').write_text(json.dumps(state, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
