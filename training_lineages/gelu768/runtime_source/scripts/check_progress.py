"""Actual slow child progresses despite sparse logs; unchanged state expires."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
path=Path(__file__).resolve().parents[1]/'src/nlp_hw02/progress.py';assert path.exists(),'Missing real training progress module'
spec=importlib.util.spec_from_file_location('progress',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as d:
    root=Path(d);file=root/'progress.json';observer=m.ProgressObserver(started=time.monotonic())
    code="""from pathlib import Path
import time
from nlp_hw02.progress import ProgressWriter
w=ProgressWriter(Path(__import__('sys').argv[1]),interval=.02)
for step in range(1,11):
 w.update(step,'training')
 time.sleep(.03)
print('only sparse final log')
"""
    child=subprocess.Popen([sys.executable,'-c',code,str(file)],stdout=subprocess.PIPE,text=True)
    seen=[]
    while child.poll() is None:
        now=time.monotonic();record=observer.observe(file,now)
        if record:seen.append(record['step']);assert not observer.stalled(now,timeout=.12)
        time.sleep(.02)
    stdout=child.communicate()[0];assert child.returncode==0 and 'only sparse final log' in stdout
    assert len(set(seen))>=4
    now=time.monotonic();observer.observe(file,now)
    assert observer.stalled(now+.2,timeout=.12),'Frozen progress did not expire'
    w=m.ProgressWriter(file,interval=0,clock=lambda:now+.21);w.update(10,'training',force=True)
    observer.observe(file,now+.21)
    assert observer.stalled(now+.22,timeout=.12),'Repeated same state concealed a stall'
    w.update(10,'validation',force=True);observer.observe(file,now+.23)
    assert not observer.stalled(now+.24,timeout=.12),'Real phase transition was ignored'
    file.write_text('{partial');assert observer.observe(file,now+.25) is None
    file.write_text(json.dumps({'step':-1,'phase':'training','monotonic':time.monotonic()}));assert observer.observe(file,now+.26) is None
print('PASS real sparse-log child heartbeat; frozen/repeated/invalid state; phase transition')
