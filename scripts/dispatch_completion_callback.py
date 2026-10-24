"""One-shot handoff for an already-finished monitor; no remote job action."""
import argparse
import json
from pathlib import Path
import time
from watch_kaggle_run import ROOT,completion_callback

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--watch-state',required=True,type=Path)
p.add_argument('--evidence-prefix',required=True)
p.add_argument('--callback-thread',required=True)
p.add_argument('--codex',default='/Users/a2mogus/.local/bin/codex')
p.add_argument('--delay-seconds',default=0,type=int)
a=p.parse_args()
if not 0<=a.delay_seconds<=60:p.error('One bounded0–60second delay only')
path=(ROOT/a.watch_state).resolve()
if not path.is_relative_to(ROOT):p.error('Monitor evidence must be inside this project')
state=json.loads(path.read_text())
if state.get('running') is not False or state.get('last_event',{}).get('event')!='monitor_finished':
 p.error('Already-finished monitor evidence required; do not duplicate a live monitor')
if a.delay_seconds:time.sleep(a.delay_seconds)
result=completion_callback(a,state)
print(json.dumps(result),flush=True)
if result.get('enqueue_succeeded') is False:raise SystemExit(1)
