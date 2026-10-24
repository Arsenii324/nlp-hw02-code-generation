"""Heartbeat progress must protect a healthy trainer near its final iterations."""
import importlib.util
from pathlib import Path
import json
import tempfile
s=importlib.util.spec_from_file_location('eval','scripts/kaggle_eval.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
assert hasattr(m,'remaining_seconds'),'Missing fresh-progress budget estimator'
eta=m.remaining_seconds(40000,.8,39800,39899)
assert abs(eta-126.25)<1e-9 and eta<=180 and eta+90+30<1490
assert m.remaining_seconds(40000,.8,39800,39700)==250
with tempfile.TemporaryDirectory() as d:
 root=Path(d);(root/'progress.json').write_text(json.dumps({'step':399,'phase':'training'}))
 observation=m.training_observation(root,40000)
 assert observation['steps']==399 and observation['source']=='heartbeat' and not observation['exact']
 (root/'metrics.json').write_text(json.dumps({'completed_steps':400,'optimizer_updates':399,'skipped_updates':1}))
 observation=m.training_observation(root,40000)
 assert observation['steps']==400 and observation['source']=='metrics' and observation['exact']
 (root/'metrics.json').write_text('{partial');(root/'train.jsonl').write_text(json.dumps({'event':'finished','step':401})+'\n')
 observation=m.training_observation(root,40000)
 assert observation['steps']==401 and observation['source']=='finished_event' and observation['exact']
print('PASS freshest progress protects near finish; terminal exact counts or explicit lower bounds')
