"""An exited but incomplete or terminated trainer must never be evaluated."""
import importlib.util
import json
from pathlib import Path
import tempfile
s=importlib.util.spec_from_file_location('eval',str(Path(__file__).resolve().parent/'kaggle_eval.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
assert hasattr(m,'completed_run'),'Missing completed-training evaluation gate'
with tempfile.TemporaryDirectory() as d:
    root=Path(d);(root/'model.pt').write_bytes(b'fixture')
    metrics={'completed_steps':100,'optimizer_updates':100,'skipped_updates':0,'selected_step':100}
    (root/'metrics.json').write_text(json.dumps(metrics))
    assert not m.completed_run(root,40000,exit_code=0,stopped=True)
    assert not m.completed_run(root,40000,exit_code=0,stopped=False)
    assert m.completed_run(root,100,exit_code=0,stopped=False)
    assert not m.completed_run(root,100,exit_code=1,stopped=False)
    metrics['optimizer_updates']=99;(root/'metrics.json').write_text(json.dumps(metrics))
    assert not m.completed_run(root,100,exit_code=0,stopped=False)
    (root/'metrics.json').write_text('{incomplete')
    assert not m.completed_run(root,100,exit_code=0,stopped=False)
print('PASS complete-only evaluation eligibility; abort/partial/nonzero/bad metrics rejected')
