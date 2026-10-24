"""Atomic actual-step progress; independent of loss logging cadence."""
import json
import math
import os
from pathlib import Path
import time


class ProgressWriter:
    def __init__(self,path,interval=5,clock=time.monotonic):
        self.path=Path(path);self.interval=interval;self.clock=clock
        self.last_write=-math.inf;self.signature=None

    def update(self,step,phase,force=False,**fields):
        now=self.clock();signature=(step,phase)
        if not force and signature==self.signature and now-self.last_write<self.interval:return False
        if not force and self.signature is not None and phase==self.signature[1] and now-self.last_write<self.interval:return False
        record={'step':step,'phase':phase,'monotonic':now,
            'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields}
        temporary=self.path.with_suffix(self.path.suffix+'.tmp')
        temporary.write_text(json.dumps(record));os.replace(temporary,self.path)
        self.last_write=now;self.signature=signature;return True


class ProgressObserver:
    def __init__(self,started):
        self.last_progress=started;self.signature=None;self.record=None

    def observe(self,path,now):
        try:
            record=json.loads(Path(path).read_text())
            step=record['step'];phase=record['phase'];stamp=record['monotonic']
            if not isinstance(step,int) or step<0 or not isinstance(phase,str) or not phase:return None
            if not isinstance(stamp,(int,float)) or not math.isfinite(stamp) or stamp>now+1:return None
            if self.record is not None and (step<self.record['step'] or stamp<self.record['monotonic']):return None
        except (OSError,ValueError,KeyError,TypeError):return None
        signature=(step,phase)
        if signature!=self.signature:
            self.last_progress=max(self.last_progress,stamp);self.signature=signature
        self.record=record;return record

    def stalled(self,now,timeout):return now-self.last_progress>timeout
