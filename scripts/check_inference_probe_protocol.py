"""Pinned-parent uniqueness and finite child finalization checks without GPUs."""
import json,tempfile
from pathlib import Path
from unittest.mock import patch
from inference_probe_protocol import verified_inference_parents,finish_children,sha
with tempfile.TemporaryDirectory(prefix='hw02-inference-parent-') as name:
 root=Path(name);record={};cache='same-cache'
 for run in ['b019','b024']:
  d=root/run;run_dir=d/f'run-{run}';run_dir.mkdir(parents=True)
  metadata={'config':{'learnable_xsa':run=='b024'},'torch':'2.10.0+cu128','cuda':'12.8'}
  (run_dir/'run_config.json').write_text(json.dumps(metadata));(run_dir/'model.pt').write_bytes(b'fixture-only-'+run.encode())
  (run_dir/'metrics.json').write_text(json.dumps({'completed_steps':120000,'selected_step':120000}))
  (run_dir/'tokenizer.json').write_text('{}');(run_dir/'validation.json').write_text('[]')
  files={f'run-{run}/{p.name}':sha(p) for p in run_dir.iterdir()}
  terminal={'complete':True,'training_complete':True,'training_source_sha256':run+'source','cache_manifest_sha256':cache,'artifact_sha256':files}
  (d/'session_result.json').write_text(json.dumps(terminal))
  record[run]={'run':run,'terminal_sha256':sha(d/'session_result.json'),'source_sha256':run+'source','files':files,'config':metadata['config'],'selected_step':120000}
 manifest={'parents':record,'cache_manifest_sha256':cache,'torch':'2.10.0+cu128','cuda':'12.8'}
 paths=verified_inference_parents(root,manifest);assert set(paths)=={'b019','b024'}
 duplicate=root/'duplicate';duplicate.mkdir();(duplicate/'session_result.json').write_bytes((root/'b024/session_result.json').read_bytes())
 try:verified_inference_parents(root,manifest)
 except ValueError:pass
 else:raise AssertionError('duplicate matching parent accepted')
 (duplicate/'session_result.json').unlink();(root/'b024/run-b024/model.pt').write_bytes(b'corrupt')
 try:verified_inference_parents(root,manifest)
 except ValueError:pass
 else:raise AssertionError('corrupt checkpoint accepted')
class Child:
 def __init__(self,alive):self.pid=123456;self.alive=alive;self.waits=[]
 def poll(self):return None if self.alive else 0
 def wait(self,timeout):self.waits.append(timeout);self.alive=False;return 0
live=Child(True);dead=Child(False);signals=[]
with patch('inference_probe_protocol.os.killpg',side_effect=lambda *args:signals.append(args)):
 finish_children([live,dead])
assert len(signals)==1 and live.waits==[5] and dead.waits==[]
print('inference_parent_unique_hash_software_and_child_reaping_checks_passed')
