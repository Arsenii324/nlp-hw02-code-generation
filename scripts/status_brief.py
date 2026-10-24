"""Read bounded local monitor snapshots; never query a provider or expose auth logs."""
from __future__ import annotations
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import argparse

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'reports/evidence'

def read(name):
    try:
        return json.loads((EVIDENCE / name).read_text())
    except (OSError, ValueError) as error:
        return {'observation_error': type(error).__name__}

def alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', action='store_true', help='Include completed older observers')
    args = parser.parse_args()
    native = read('native_offnode_monitor_20261024h_finalpair/state.json')
    latest = native.get('latest', {})
    event_path = EVIDENCE / 'native_offnode_monitor_20261024h_finalpair/events.jsonl'
    if event_path.is_file():
        with event_path.open('rb') as stream:
            stream.seek(max(0, event_path.stat().st_size - 65536))
            tail = stream.read().decode(errors='replace').splitlines()
        for line in reversed(tail):
            try:
                observation = json.loads(line).get('observation', {})
            except ValueError:
                continue
            if observation.get('utc', '') > latest.get('utc', ''):
                latest = observation
                break
    result = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'status_source': 'local snapshots; provider observations have their own timestamps',
              'kaggle': {}, 'native': {
                  'monitor_pid': native.get('pid'), 'monitor_alive': alive(native.get('pid')),
                  'observed_utc': latest.get('utc'), 'jobs': {},
                  'backup_aggregate_bytes': native.get('local_aggregate_bytes'),
                  'observation_unconfirmed': native.get('observation_unconfirmed'),
                  'errors': native.get('errors'),
              }, 'local_free_GiB': round(shutil.disk_usage(ROOT).free / 2**30, 2)}
    current = [('wide512_training', 'wide512_cooldown_20261024a'),
               ('qbatch_n16', 'qbatch_n16_20261024a'),
               ('tpu', 'tpu_split_20261024a')]
    history = [('6m', 'six_m_cooldown_pre_reset_20261023a'),
                        ('fullrange', 'fullrange_8h_pre_reset_20261023c'),
                        ('wide_budget', 'wide_t4_budget_20261024a'),
                        ('6m_n24', 'six_m_long_n24_20261024a'),
                        ('fullrange_n24', 'fullrange_long_n24_20261024a'),
                        ('graph_g_screen', 'graph_g_budget_20261024b')]
    for short, stem in current + (history if args.history else []):
        source = read(stem + '_watch.json')
        result['kaggle'][short] = {
            'kernel_ref': source.get('kernel'), 'status': source.get('status'), 'observed_utc': source.get('last_event', {}).get('utc'),
            'observer_pid': source.get('pid'), 'observer_alive': alive(source.get('pid')),
            'request_errors': source.get('consecutive_request_errors'),
            'recovered': source.get('recovered'), 'observation_error': source.get('observation_error')}
    for name, job in latest.get('jobs', {}).items():
        result['native']['jobs'][name] = {
            'attempt': job.get('progress', {}).get('step'),
            'iteration_units': 'NEW since parent' if 'committed_new_iterations' in job.get('progress', {}) else 'absolute since scratch',
            'durable': job.get('pointer', {}).get('absolute_step'),
            'best_CE': job.get('metrics', {}).get('best_validation_loss'),
            'terminal': job.get('terminal'), 'error': job.get('error')}
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
