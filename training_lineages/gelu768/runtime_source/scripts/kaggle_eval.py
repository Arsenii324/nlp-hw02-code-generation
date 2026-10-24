"""Bounded child evaluation with independent VM logs and explicit completion."""
import os
from pathlib import Path
import subprocess
import time


def evaluation_count(experiment):
    lanes=len(experiment['lanes'])
    return lanes+int(experiment.get('include_b001_timing',True))+lanes*int(experiment.get('profile_decoding',False))+int(experiment.get('canonical_bundle_timing',False))


def remaining_seconds(expected_steps,rate,logged_step,heartbeat_step=None):
    if rate is None or rate<=0:return None
    latest=max(logged_step,heartbeat_step if heartbeat_step is not None else 0)
    return max(0,expected_steps-latest)/rate


def training_observation(run,expected_steps):
    import json
    run=Path(run)
    def valid(step):return isinstance(step,int) and 0<=step<=expected_steps
    try:
        metrics=json.loads((run/'metrics.json').read_text());step=metrics['completed_steps']
        if valid(step) and metrics['optimizer_updates']+metrics['skipped_updates']==step:
            return {'steps':step,'source':'metrics','exact':True}
    except (OSError,ValueError,KeyError,TypeError):pass
    try:
        for line in reversed((run/'train.jsonl').read_text().splitlines()[-30:]):
            try:record=json.loads(line)
            except ValueError:continue
            if record.get('event')=='finished' and valid(record.get('step')):
                return {'steps':record['step'],'source':'finished_event','exact':True}
    except OSError:pass
    try:
        record=json.loads((run/'progress.json').read_text());step=record['step']
        if valid(step):return {'steps':step,'source':'heartbeat','exact':record.get('phase')=='finished'}
    except (OSError,ValueError,KeyError,TypeError):pass
    return {'steps':None,'source':'no_valid_report','exact':False}


def completed_run(run,expected_steps,exit_code,stopped):
    import json
    run=Path(run)
    if exit_code!=0 or stopped or not (run/'model.pt').is_file():return False
    try:
        metrics=json.loads((run/'metrics.json').read_text())
        return (metrics['completed_steps']==expected_steps and
            metrics['optimizer_updates']+metrics['skipped_updates']==expected_steps and
            0<metrics['selected_step']<=expected_steps)
    except (OSError,ValueError,KeyError,TypeError):return False


def run_evaluation(name,command,log,env,remaining,bound,emit,required=(),sample=None,sample_interval=15):
    if remaining<bound+30:
        record={'event':'evaluation_deferred','name':name,'status':'deferred','remaining_seconds':remaining,
            'reason':'Preserve completed training; insufficient reserved evaluation time.'}
        emit(record);return record
    started=time.monotonic();log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    emit({'event':'evaluation_start','name':name,'bound_seconds':bound,'log':log.name})
    status='failed';failure=None;process=None
    with log.open('w') as stream:
        try:
            process=subprocess.Popen(command,env={**os.environ,**env},stdout=stream,stderr=subprocess.STDOUT)
            while process.poll() is None:
                elapsed=time.monotonic()-started
                if elapsed>=bound:
                    process.terminate()
                    try:process.wait(timeout=min(10,max(.1,remaining-elapsed-30)))
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    raise RuntimeError(f'{name} exceeded evaluation bound {bound}s')
                if sample is not None:sample()
                try:process.wait(timeout=min(sample_interval,max(.001,bound-elapsed)))
                except subprocess.TimeoutExpired:pass
            if process.returncode:raise RuntimeError(f'{name} child exit {process.returncode}; inspect {log}')
            missing=[str(p) for p in required if not Path(p).is_file()]
            if missing:raise RuntimeError(f'{name} missing outputs: {missing}')
            status='complete'
        except BaseException as exc:
            failure=str(exc)
            if process is not None and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
            raise
        finally:
            result={'event':'evaluation_exit','name':name,'status':status,'elapsed_seconds':time.monotonic()-started,
                'exit':process.returncode if process is not None else None,'failure':failure}
            emit(result)
    return result
