"""Focused callback protocol checks; fake CLI, no real message or network."""
from argparse import Namespace
import importlib.util,json,subprocess,tempfile
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('watch',Path(__file__).with_name('watch_kaggle_run.py'))
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
for outcome in ('success','error','timeout'):
 with tempfile.TemporaryDirectory(prefix='hw02-callback-check-') as f:
  root=Path(f);state={'kernel':'owner/slug/1','status':'COMPLETE','running':False,'recovery_deferred':True,'recovered':False}
  args=Namespace(callback_thread='01a0f8a6-39b5-7192-9210-9b807ffa438f',codex='mock-codex',evidence_prefix='mock')
  calls=[]
  def request(argv,timeout=60):
   calls.append(argv)
   if outcome=='timeout':raise subprocess.TimeoutExpired(argv,timeout,output=b'partial')
   if outcome=='error':raise RuntimeError('mock denied')
   return 'Queued message.'
  with patch.object(w,'ROOT',root):
   first=w.completion_callback(args,state,request)
   second=w.completion_callback(args,state,request)
  assert len(calls)==1 and second['duplicate_suppressed']
  assert calls[0][:3]==['mock-codex','queue','--thread'] and '-m' not in calls[0]
  assert len(calls[0][-1].encode())<=1000 and 'Automated' in calls[0][-1]
  assert first['enqueue_succeeded']==(outcome=='success')
  assert not first['agent_receipt_observed']
  record=json.loads(Path(first['record']).read_text());assert record['attempts']==1
  assert record['status']==('queued_receipt_pending' if outcome=='success' else 'enqueue_failed_requires_inspection')
print('completion_callback_single_attempt_timeout_error_checks_passed')
