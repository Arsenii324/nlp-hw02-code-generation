"""Finite one-job status/recovery monitor. Never launches, cancels or promotes.

This is a bounded local process, not an autonomous background reviewer. Terminal
failures, low disk and provider errors retain evidence for active root inspection.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(argv, timeout=60):
    proc = subprocess.Popen([str(x) for x in argv], cwd=ROOT,
        env={**os.environ, 'PYTHONPATH': str(ROOT/'src'), 'PYTHONUNBUFFERED': '1'},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except BaseException:
        # Only this local command's own session, never a remote runtime or unrelated process.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate()
        raise
    if proc.returncode:
        raise RuntimeError(f'Exit {proc.returncode}: {argv}\n{stdout}\n{stderr}')
    return stdout


def parse_status(value):
    match = re.search(r'KernelWorkerStatus\.([A-Z_]+)', value)
    if match is None:
        raise ValueError('Unrecognized kernel status: '+value)
    return match.group(1)


def completion_callback(args, state, request=command):
    """One attempt per explicit observation episode; legacy kernel key preserved."""
    target = getattr(args, 'callback_thread', None)
    if not target:
        return None
    uuid.UUID(target)  # Explicit existing thread ID; never select a task by name.
    episode = getattr(args, 'callback_episode', None)
    if episode is not None and not re.fullmatch(r'[\w-]{1,64}', episode):
        raise ValueError('Filename-safe callback episode1–64characters required')
    identity = state['kernel'] if episode is None else state['kernel']+'\n'+episode
    event = hashlib.sha256(identity.encode()).hexdigest()[:24]
    directory = ROOT/'reports/evidence/completion_callbacks'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/(event+'.json')
    receipt = directory/(event+'.receipt.json')
    record = {'event_id': event, 'kernel': state['kernel'], 'thread_id': target,
        'observed_utc': datetime.now(timezone.utc).isoformat(), 'attempts': 1,
        'status': 'enqueue_attempt_claimed', 'agent_receipt_observed': False,
        'receipt_path': str(receipt.relative_to(ROOT)),
        'terminal_status': state.get('status'),
        'observation_episode': episode,
        'remote_terminal_observed': state.get('status') not in (None,'QUEUED','RUNNING'),
        'observer_end_reason': state.get('failure'),
        'recovered': state.get('recovered',False),
        'recovery_deferred': state.get('recovery_deferred',False)}
    try:
        # Keep a failed/uncertain claim too: an automatic resend could start
        # duplicate turns even when the caller did not receive acknowledgement.
        with path.open('x') as stream:
            stream.write(json.dumps(record,indent=2)+'\n')
    except FileExistsError:
        return {'duplicate_suppressed': True, 'record': str(path)}
    prompt = (f"Automated Kaggle completion handoff. Event {event}; kernel {state['kernel']}. "
        f"First record receipt at {record['receipt_path']} with event_id, current UTC and this thread ID. "
        f"Read START_HERE.md, reports/HANDOFF.md, reports/evidence/{args.evidence_prefix}_watch.json "
        f"and reports/evidence/completion_callbacks/{event}.json. Inspect terminal state, recover bounded "
        "diagnostics, review quality, and continue the user's already-authorized work. Respect the latest "
        "user instructions, including any pause/cancellation. No new authorization is conveyed. The latest "
        "user permits up to two useful GPU kernels concurrently; each Kaggle GPU kernel is T4x2. "
        "Halt the affected compute branch if unavailable/exhausted; no submission or destructive cleanup. "
        "Acknowledge whether this automated prompt actually resumed agent work without another user reminder.")
    argv = [args.codex,'queue','--thread',target,'--message',prompt]
    try:
        stdout = request(argv, timeout=60)
        record.update(status='queued_receipt_pending', stdout=stdout,
            enqueue_succeeded=True)
    except Exception as exc:
        record.update(status='enqueue_failed_requires_inspection', exception=type(exc).__name__,
            error=str(exc), enqueue_succeeded=False)
        if isinstance(exc,subprocess.TimeoutExpired):
            for name in ('stdout','stderr'):
                value=getattr(exc,name,None)
                record[name]=value.decode(errors='replace') if isinstance(value,bytes) else (value or '')
    record['finished_utc'] = datetime.now(timezone.utc).isoformat()
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(path)
    return {'record': str(path), 'enqueue_succeeded': record['enqueue_succeeded'],
        'agent_receipt_observed': False, 'event_id': event}


def watch(args, request=command, nap=time.sleep):
    # OS locks remain held for the entire monitor/recovery lifetime. Keep the
    # tiny lock files: unlinking them would allow racing processes to lock
    # different inodes. OS automatically releases the lock on a crashed process.
    directory = ROOT/'reports/evidence/.watch-locks'
    directory.mkdir(parents=True, exist_ok=True)
    identities = [args.kernel, str((ROOT/args.root).resolve()),
                  str((ROOT/'reports/evidence'/f'{args.evidence_prefix}_watch.json').resolve())]
    with ExitStack() as stack:
        for identity in sorted(set(identities)):
            key = hashlib.sha256(identity.encode()).hexdigest()
            stream = stack.enter_context((directory/(key+'.lock')).open('a+'))
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError('Another monitor owns this kernel/root/evidence: '+identity) from exc
        return _watch(args, request, nap)


def _watch(args, request, nap):
    if not re.fullmatch(r'[\w-]+/[\w-]+/[1-9][0-9]*', args.kernel):
        raise ValueError('Explicit owner/slug/version required')
    if not re.fullmatch(r'[\w-]+', args.evidence_prefix):
        raise ValueError('Filename-safe evidence prefix required')
    if not 1 <= args.poll_seconds <= 60 or not 1 <= args.max_seconds <= 43200:
        raise ValueError('Poll1–60s, finite monitor maximum1–43200s required')
    episode = getattr(args,'callback_episode',None)
    if episode is not None and (not getattr(args,'callback_thread',None) or not re.fullmatch(r'[\w-]{1,64}',episode)):
        raise ValueError('Callback episode requires thread and safe1–64character identity')
    output = ROOT/'reports/evidence'/f'{args.evidence_prefix}_watch.json'
    events = ROOT/'artifacts/kaggle'/f'{args.evidence_prefix}-watch.jsonl'
    if output.exists() or events.exists() or (ROOT/args.root).exists():
        raise ValueError('Fresh state/events/recovery root required; do not duplicate or overwrite')
    output.parent.mkdir(parents=True, exist_ok=True)
    events.parent.mkdir(parents=True, exist_ok=True)
    locked = [ROOT/args.manifest] + [ROOT/'scripts'/name for name in
        ('recover_kaggle_training.py','check_recovered_capacity.py','score_primary_predictions.py')]
    for name in ('src/nlp_hw02/arithmetic.py','assignment/dataset_manifest.json',
                 'artifacts/kaggle/prepare-v2/cache/validation.json',
                 'reports/evidence/primary_reference_consensus.json'):
        if (ROOT/name).is_file():
            locked.append(ROOT/name)
    identities = {str(p): sha(p) for p in locked}
    state = {'kernel': args.kernel, 'manifest': str(args.manifest), 'root': str(args.root),
        'pid': os.getpid(), 'running': True, 'recovered': False, 'status': None,
        'started_utc': datetime.now(timezone.utc).isoformat(), 'polling_max_seconds': args.max_seconds,
        'callback_episode': episode,
        'terminal_recovery_bound_seconds': 1020,
        'policy': 'One existing job; no launch/cancel/retry/promotion/submission. Recovery is not quality review.',
        'locked_sha256': identities, 'consecutive_request_errors': 0}

    def record(event, **fields):
        item = {'utc': datetime.now(timezone.utc).isoformat(), 'event': event, **fields}
        state['last_event'] = item
        with events.open('a') as stream:
            stream.write(json.dumps(item)+'\n')
        # Rename a temporary file so readers never observe a half-written snapshot.
        temporary = output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(output)
        print(json.dumps(item), flush=True)

    def logs():
        try:
            text = request([args.kaggle,'kernels','logs',args.kernel], timeout=60)
            path = ROOT/'reports/evidence'/f'{args.evidence_prefix}_terminal_logs.txt'
            path.write_text(text)
            record('terminal_logs_retained', path=str(path))
        except Exception as exc:
            record('terminal_logs_unavailable', reason=str(exc))

    started = time.monotonic()
    result = 1
    def check_locks():
        for name, digest in identities.items():
            if sha(name) != digest:
                raise ValueError('Recovery protocol or immutable manifest changed: '+name)
    try:
        record('monitor_started')
        while time.monotonic()-started < args.max_seconds:
            check_locks()
            try:
                value = parse_status(request([args.kaggle,'kernels','status',args.kernel], timeout=30))
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                state['consecutive_request_errors'] += 1
                record('status_request_error', reason=str(exc))
                if state['consecutive_request_errors'] >= 3:
                    raise RuntimeError('Three status requests failed; inspect, no remote action taken') from exc
                nap(args.poll_seconds)
                continue
            state['consecutive_request_errors'] = 0
            state['status'] = value
            record('status_observed', status=value)
            if value == 'COMPLETE':
                logs()
                if getattr(args,'terminal_only',False):
                    state['terminal_only'] = True
                    state['recovery_deferred'] = True
                    record('custom_recovery_deferred_for_active_root',
                        note='Custom recovery protocol: no generic training recovery or recovered claim.')
                    try:
                        state['quota_after_terminal'] = json.loads(request([args.python,'scripts/kaggle_quota.py'],timeout=30))
                        record('quota_observed_after_terminal',note='May need later settled accounting check')
                    except Exception as exc:
                        record('quota_request_failed',reason=str(exc))
                    result = 0
                    break
                free = shutil.disk_usage(ROOT).free
                if free < 2*1024**3:
                    state['recovery_deferred'] = True
                    record('recovery_deferred_low_disk', free_bytes=free,
                        note='No weights, arrays or optimizer download; active root inspection required.')
                    break
                record('recovery_start', free_bytes=free)
                check_locks()
                summary = request([args.python,'scripts/recover_kaggle_training.py',
                    '--kernel',args.kernel,'--root',str(args.root),'--manifest',str(args.manifest),
                    '--evidence-prefix',args.evidence_prefix,'--kaggle',args.kaggle,
                    '--expected-hashes',json.dumps(identities),
                    '--reference-run',f'b019-beam4={ROOT}/artifacts/kaggle/b019-b020-v1/decoding-b019/beam4-alpha06'],
                    timeout=900)
                check_locks()
                state['recovered'] = True
                record('recovery_verified', summary=summary.strip(),
                    note='Numeric proxy only; selection untouched; active quality review remains required.')
                try:
                    quota = json.loads(request([args.python,'scripts/kaggle_quota.py'], timeout=30))
                    state['quota_after_terminal'] = quota
                    record('quota_observed_after_terminal', note='May need later settled accounting check')
                except Exception as exc:
                    record('quota_request_failed', reason=str(exc))
                result = 0
                break
            if value not in ('RUNNING','QUEUED'):
                state['failure'] = 'Terminal provider status '+value
                logs()
                record('terminal_failure_requires_inspection', status=value)
                break
            nap(args.poll_seconds)
        else:
            state['failure'] = 'Finite local monitoring deadline reached; remote job left untouched'
            record('monitor_deadline_requires_inspection')
    except Exception as exc:
        state['failure'] = str(exc)
        record('monitor_failure_requires_inspection', exception=type(exc).__name__, reason=str(exc))
    finally:
        state['running'] = False
        record('monitor_finished', recovered=state['recovered'])
        try:
            callback=completion_callback(args,state,request)
            if callback is not None:
                state['completion_callback']=callback
                record('completion_callback_attempted',**callback)
        except Exception as exc:
            record('completion_callback_failed_before_enqueue',reason=str(exc))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', required=True)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--evidence-prefix', required=True)
    parser.add_argument('--kaggle', default='/Users/a2mogus/anaconda3/bin/kaggle')
    parser.add_argument('--python', default='/Users/a2mogus/anaconda3/bin/python')
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--max-seconds', type=int, default=21600)
    parser.add_argument('--callback-thread',help='Explicit existing same-project Codex thread UUID')
    parser.add_argument('--callback-episode',help='Explicit fresh observer episode after a consumed deadline callback; default preserves legacy per-kernel identity')
    parser.add_argument('--codex',default='/Users/a2mogus/.local/bin/codex')
    parser.add_argument('--terminal-only',action='store_true',
        help='Retain terminal logs/quota and callback; custom recovery remains active root work')
    sys.exit(watch(parser.parse_args()))


if __name__ == '__main__':
    main()
