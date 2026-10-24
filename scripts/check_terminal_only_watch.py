"""Non-training observation must defer custom recovery yet deliver one callback."""
from argparse import Namespace
import importlib.util,json
from pathlib import Path
import tempfile
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('watch',Path(__file__).with_name('watch_kaggle_run.py'));w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
for status in ['COMPLETE','ERROR']:
 with tempfile.TemporaryDirectory(prefix='hw02-terminal-only-') as folder:
  root=Path(folder);(root/'scripts').mkdir();(root/'manifest.json').write_text('{}')
  for name in ['recover_kaggle_training.py','check_recovered_capacity.py','score_primary_predictions.py']:(root/'scripts'/name).write_text('pass\n')
  args=Namespace(kernel='owner/probe/1',root=Path('artifacts/kaggle/result'),manifest=Path('manifest.json'),evidence_prefix='probe',kaggle='fake-kaggle',python='fake-python',poll_seconds=1,max_seconds=5,terminal_only=True,callback_thread='01a0f8a6-39b5-7192-9210-9b807ffa438f',codex='fake-codex')
  calls=[]
  def request(argv,timeout=60):
   calls.append([str(x) for x in argv])
   if argv[1:3]==['kernels','status']:return 'KernelWorkerStatus.'+status
   if argv[0]=='fake-codex':return 'Queued fake message for fixture only'
   return '{}'
  with patch.object(w,'ROOT',root):result=w.watch(args,request=request,nap=lambda _:None)
  state=json.loads((root/'reports/evidence/probe_watch.json').read_text())
  assert not state['recovered'],'Non-training terminal-only observer must never call training recovery'
  assert not any(any(x.endswith('recover_kaggle_training.py') for x in c) for c in calls)
  assert not state['running'] and len([c for c in calls if c[0]=='fake-codex'])==1
  if status=='COMPLETE':assert result==0 and state['recovery_deferred'] and state['terminal_only']
  else:assert result==1 and state['failure']
print('terminal_only_watch_complete_error_callback_checks_passed')
