"""Unbound-source finite exact continuation parent; horizons/roles bound later."""
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import zipfile

STARTED=time.monotonic();ROOT=Path('/kaggle/working');CHILDREN=[]
PLATFORM=18000;OUTER=17940;WORK=16440;CLEANUP_END=16800;HASH_END=17895;TAIL=1500
ALARM_STAGE='unarmed';STATE={'complete':False,'phase':'startup'}


class AbsoluteDeadline(BaseException):pass


def arm_absolute(elapsed,stage):
    global ALARM_STAGE
    ALARM_STAGE=stage
    signal.setitimer(signal.ITIMER_REAL,max(.001,STARTED+elapsed-time.monotonic()))


def absolute_alarm(signum,frame):
    stage=ALARM_STAGE
    if stage=='hard_exit':os._exit(124)
    # Re-arm before raising so context-manager/exception unwinding is bounded too.
    if stage=='work':arm_absolute(CLEANUP_END,'unwind_cleanup')
    elif stage in ('cleanup','unwind_cleanup'):arm_absolute(HASH_END,'unwind_finalization')
    else:arm_absolute(OUTER,'hard_exit')
    raise AbsoluteDeadline('Absolute parent '+stage+' deadline reached')


def sha(path,deadline=None):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            if deadline is not None and time.monotonic()>=deadline:raise AbsoluteDeadline('Artifact hashing deadline reached')
            h.update(block)
    return h.hexdigest()


def emit(event,**fields):
    r={'event':event,'session_seconds':time.monotonic()-STARTED,**fields}
    with (ROOT/'session_events.jsonl').open('a') as f:f.write(json.dumps(r,allow_nan=False)+'\n')
    print(json.dumps(r,allow_nan=False),flush=True)


def start(command,env,log):
    stream=Path(log).open('w');child=subprocess.Popen(command,env={**os.environ,**env},stdout=stream,
        stderr=subprocess.STDOUT,start_new_session=True);CHILDREN.append((child,stream));return child


def stop(child,deadline=None):
    """Poll/reap leader and group independently, sharing an absolute cleanup end."""
    began=time.monotonic();end=min(deadline if deadline is not None else STARTED+WORK,began+20)
    result={'pid':child.pid,'reaped':False,'group_exited':False,'errors':[],'signals':[]}
    def observe():
        result['reaped']=child.poll() is not None
        try:os.killpg(child.pid,0)
        except ProcessLookupError:result['group_exited']=True
        except OSError as e:result['errors'].append(type(e).__name__)
        return result['reaped'] and result['group_exited']
    def send(sig):
        try:os.killpg(child.pid,sig);result['signals'].append(int(sig))
        except ProcessLookupError:pass
        except OSError as e:result['errors'].append(type(e).__name__)
    if not observe():
        send(signal.SIGTERM);term_end=min(end-5,began+15)
        while time.monotonic()<term_end and not observe():time.sleep(.05)
        if not observe():
            send(signal.SIGKILL)
            while time.monotonic()<end and not observe():time.sleep(.05)
        observe()
    result.update(seconds=time.monotonic()-began,deadline_exhausted=not(result['reaped'] and result['group_exited']))
    return result




def require(ok,message):
    if not ok:raise ValueError(message)


def snapshot_lanes(lanes):
    """JSON-safe incremental state survives alarms/errors while siblings run."""
    rows=[]
    for lane in lanes:
        metrics=lane.get('metrics');spec=lane['spec'];evaluation={k:v for k,v in lane.get('evaluation',{}).items() if k!='child'}
        rows.append({'lane':spec['lane'],'requested_new_steps':spec['new_steps'],'actual_exit':lane['child'].poll(),
            'new_step':lane['step'],'rate':lane['rate'],'eta':lane['eta'],'stopped':lane['stopped'],
            'run':str(lane['run']),'child_wall_seconds':lane.get('exited',time.monotonic())-lane['started'],
            'metrics':metrics,'cleanup':lane.get('cleanup'),'finish_failure':lane.get('finish_failure'),
            'evaluation':evaluation,'evaluation_deferred':lane.get('evaluation_deferred'),
            'complete':bool(not lane.get('finish_failure') and metrics and metrics['since_resume_iterations']==spec['new_steps'] and metrics['exact_requested_horizon_complete'] and lane['child'].poll()==0)})
    STATE['lanes']=rows


def launch_evaluation(lane,source,tools):
    # Triggered on THIS lane's real exit, without waiting for another trainer.
    spec=lane['spec'];plan=spec.get('post_training_evaluation')
    if plan is None:
        lane['evaluation_deferred']={'reason':'No root-bound evaluator/menu; no default beam/grid invented'}
        emit('post_training_evaluation_deferred',lane=spec['lane'],**lane['evaluation_deferred']);return
    bound=plan['bound_seconds'];require(type(bound) is int and 1<=bound<=300,'Explicit bounded evaluator required')
    if STARTED+WORK-time.monotonic()<bound+30:
        lane['evaluation_deferred']={'reason':'Insufficient absolute work reserve'}
        emit('post_training_evaluation_deferred',lane=spec['lane'],**lane['evaluation_deferred']);return
    command=plan['argv'];require(isinstance(command,list) and command and all(isinstance(v,str) for v in command),'Pinned evaluator argv required')
    script=Path(command[0]);require(not script.is_absolute() and '..' not in script.parts and str(script) in lane['source_members'],'Evaluator script must be pinned source member')
    required=plan['required_outputs'];require(isinstance(required,list) and required and all(isinstance(v,str) and v and not Path(v).is_absolute() and '..' not in Path(v).parts for v in required),'Explicit safe required evaluation outputs needed')
    output=ROOT/('evaluation-'+spec['lane']);output.mkdir()
    values={'{run}':str(lane['run']),'{output}':str(output),'{tools}':str(tools)}
    argv=[sys.executable,'-u',str(source/script)]+[values.get(v,v) for v in command[1:]]
    child=start(argv,{'CUDA_VISIBLE_DEVICES':str(spec['device']),'PYTHONPATH':str(source/'src'),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'},ROOT/(spec['lane']+'-evaluation.log'))
    lane['evaluation']={'child':child,'started':time.monotonic(),'bound':bound,'output':str(output),'complete':False,'required_outputs':required};emit('post_training_evaluation_started',lane=spec['lane'],pid=child.pid,bound_seconds=bound)


def main():
    candidates=list(Path('/kaggle/input').rglob('gated_continuation_manifest.json'));require(len(candidates)==1,'Exactly one future actual-bound continuation manifest required')
    path=candidates[0];tools=path.parent;m=json.loads(path.read_text())
    require(m['schema']=='gated_continuation_parent_v1' and m['verified_actual_values'] is True,'Unbound continuation source cannot launch')
    require(m['driver_sha256']==sha(__file__) and m['platform_seconds']==PLATFORM,'Source/platform mismatch')
    for name,digest in m['input_sha256'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe tools input path');require(sha(tools/name)==digest,'Bound input differs:'+name)
    cache_candidates=[p for p in Path('/kaggle/input').rglob('cache_manifest.json') if sha(p)==m['cache_manifest_sha256']]
    require(len(cache_candidates)==1,'Exactly one original6M cache required');cache=cache_candidates[0].parent
    cm=json.loads(cache_candidates[0].read_text());require(cm['complete'] is True and cm['checksums']==m['cache_checksums'],'Actual cache manifest differs')
    for name,digest in cm['checksums'].items():require(sha(cache/name)==digest,'Actual cache bytes differ:'+name)
    source=ROOT/'continuation_source';source.mkdir()
    import shutil
    with zipfile.ZipFile(tools/'continuation_source.bin') as z:
        require(len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(m['source_members_sha256']),'Exact source-member allowlist required')
        for name,digest in m['source_members_sha256'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe runtime source path');target=source/name;target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(name) as incoming,target.open('xb') as outgoing:shutil.copyfileobj(incoming,outgoing,1024*1024)
            require(sha(target)==digest,'Extracted runtime member differs:'+name)
    sys.path[:0]=[str(source/'src'),str(source/'scripts')]
    from nlp_hw02.progress import ProgressObserver
    from nlp_hw02.training_arrays import array_resident_bytes,available_memory_bytes
    specs=m['lanes'];require(1<=len(specs)<=2 and len({s['lane'] for s in specs})==len(specs) and len({s['device'] for s in specs})==len(specs),'One or two distinct root-chosen lanes/devices required')
    require(all(s['lane'] in ('control','treatment') and s['device'] in (0,1) and type(s['new_steps']) is int and s['new_steps']>=1000 and type(s['resident']) is bool for s in specs),'Explicit future lane/horizon/layout loading choices required')
    size=array_resident_bytes(cache);resident_count=sum(s['resident'] for s in specs)
    if resident_count:require(available_memory_bytes()>=(resident_count+1)*size+2*1024**3,'Actual aggregate resident RAM admission failed')
    hardware=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=15).stdout.splitlines()
    require(len(hardware)==2 and all('T4' in n for n in hardware),'T4x2 expected; each worker gets one visible GPU')
    shutil.copyfile(path,ROOT/path.name);shutil.copyfile(__file__,ROOT/'run_driver.py');shutil.copyfile(tools/'continuation_source.bin',ROOT/'continuation_source.bin')
    STATE.update(phase='training',manifest_sha256=sha(path),source_sha256=sha(tools/'continuation_source.bin'),hardware=hardware)
    lanes=[]
    for spec in specs:
        name=spec['lane'];run=ROOT/('run-'+name)
        contract=tools/spec['contract_path'];binding=json.loads(contract.read_text());require(binding['verified_actual_values'] is True and binding['lane']==name and binding['new_steps']==spec['new_steps'],'Actual worker binding differs')
        # Original kernel output remains mounted remotely; no local fullstate download/toolspayload copy.
        terminals=[p for p in Path('/kaggle/input').rglob('session_result.json') if sha(p)==binding['terminal_sha256']]
        require(len(terminals)==1,'Exactly one actual original terminal hash match required')
        terminal=terminals[0];parent=terminal.parent/('run-'+name)
        command=[sys.executable,'-u',str(source/'scripts/train_gated_continuation_b.py'),'--lane',name,'--contract',str(contract),'--terminal',str(terminal),
            '--parent',str(parent),'--data',str(cache),'--output',str(run),'--new-steps',str(spec['new_steps']),'--deadline',str(STARTED+WORK-5)]
        if spec['resident']:command.append('--resident')
        child=start(command,{'CUDA_VISIBLE_DEVICES':str(spec['device']),'PYTHONPATH':str(source/'src')+os.pathsep+str(source/'scripts'),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'},ROOT/(name+'.log'))
        lane={'spec':spec,'run':run,'child':child,'started':time.monotonic(),'observer':ProgressObserver(time.monotonic()),'step':0,'rate':None,'eta':None,'stopped':False,'source_members':m['source_members_sha256']};lanes.append(lane);snapshot_lanes(lanes);emit('continuation_started',lane=name,pid=child.pid,new_steps=spec['new_steps'],device=spec['device'])
    last_emit=0
    while any('exited' not in l or (l.get('evaluation') and 'actual_exit' not in l['evaluation']) for l in lanes):
        now=time.monotonic();remaining=STARTED+WORK-now
        for lane in lanes:
            child=lane['child'];name=lane['spec']['lane']
            if child.poll() is not None:
                if 'exited' not in lane:
                    lane['exited']=now
                    try:
                        cleanup=stop(child);lane['cleanup']=cleanup;emit('actual_trainer_exit',lane=name,actual_exit=child.returncode,cleanup=cleanup,child_seconds=now-lane['started'])
                        if child.returncode==0 and cleanup['reaped'] and cleanup['group_exited']:
                            metrics=json.loads((lane['run']/'metrics.json').read_text());binding=json.loads((tools/lane['spec']['contract_path']).read_text())
                            absolute=metrics['absolute_completed_step'];new=metrics['since_resume_iterations']
                            require(absolute==binding['parent_absolute_step']+new and metrics['parent_absolute_step']==binding['parent_absolute_step'] and metrics['counter_origin']==300000,'Absolute/since-resume origin differs')
                            require(metrics['completed_steps']==metrics['warmstart_iterations']==absolute-300000 and metrics['sampled_examples']==64*(absolute-300000),'Cumulative warmstart count differs')
                            require(metrics['optimizer_updates']+metrics['skipped_updates']==absolute-300000 and metrics['since_resume_optimizer_updates']+metrics['since_resume_skipped_updates']==new and metrics['since_resume_sampled_examples']==64*new and metrics['lr_absolute_step']==absolute,'Update/sample/LR counter binding differs')
                            lane['metrics']=metrics;snapshot_lanes(lanes)
                            # Evaluate completed saved lane immediately, including a useful bounded early stop.
                            launch_evaluation(lane,source,tools);snapshot_lanes(lanes)
                    except Exception as error:
                        lane['finish_failure']=repr(error)[:8192];snapshot_lanes(lanes);emit('finished_lane_failure',lane=name,error=lane['finish_failure'])
                        with (ROOT/(name+'.log')).open('rb') as stream:stream.seek(0,2);stream.seek(max(0,stream.tell()-8192));tail=stream.read(8192).decode(errors='replace')
                        emit('bounded_failure_diagnostic',lane=name,tail_max_bytes=8192,tail=tail)
                evaluation=lane.get('evaluation')
                if evaluation and evaluation['child'].poll() is None and now-evaluation['started']>=evaluation['bound']:
                    evaluation['cleanup']=stop(evaluation['child'],deadline=min(STARTED+WORK,now+20));evaluation['timed_out']=True
                if evaluation and evaluation['child'].poll() is not None and 'actual_exit' not in evaluation:
                    evaluation['actual_exit']=evaluation['child'].returncode;evaluation['seconds']=time.monotonic()-evaluation['started'];evaluation['cleanup']=stop(evaluation['child']);evaluation['missing_outputs']=[v for v in evaluation['required_outputs'] if not (Path(evaluation['output'])/v).is_file()];evaluation['complete']=evaluation['actual_exit']==0 and not evaluation.get('timed_out',False) and evaluation['cleanup']['group_exited'] and not evaluation['missing_outputs'];emit('actual_evaluation_exit',lane=name,**{k:v for k,v in evaluation.items() if k!='child'})
                snapshot_lanes(lanes)
                continue
            heartbeat=lane['observer'].observe(lane['run']/'progress.json',now)
            if heartbeat:lane['step']=max(lane['step'],heartbeat['step'])
            events=lane['run']/'train.jsonl';records=[]
            if events.exists():
                for line in events.read_text().splitlines()[-100:]:
                    try:records.append(json.loads(line))
                    except ValueError:pass
            complete=[r for r in records if r.get('event')=='checkpoint_completed' and r['step']>=1000]
            if complete:
                a=complete[-1];lane['rate']=a['step']/a['elapsed_seconds'];lane['eta']=max(0,lane['spec']['new_steps']-lane['step'])/lane['rate']
            phase=heartbeat['phase'] if heartbeat else 'startup';bound=360 if phase!='training' else 180
            reason='stalled_'+phase if lane['observer'].stalled(now,bound) else None
            eta=lane['eta'];near_finish=eta is not None and eta<=180 and eta+120<remaining
            if remaining<TAIL and not near_finish and (eta is None or eta+300>remaining):reason=reason or 'tail_budget'
            if reason and not lane['stopped']:
                lane['stopped']=True;lane['stop_time']=now
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                emit('graceful_training_stop',lane=name,reason=reason,new_step=lane['step'],eta=eta)
            if lane['stopped'] and now-lane['stop_time']>180:
                lane['forced_cleanup']=stop(child,deadline=min(STARTED+WORK,now+20))
        snapshot_lanes(lanes)
        if now-last_emit>=30:emit('progress',lanes=[{'lane':l['spec']['lane'],'new_step':l['step'],'eta':l['eta'],'actual_exit':l['child'].poll()} for l in lanes]);last_emit=now
        time.sleep(.5)
    results=[]
    for lane in lanes:
        metrics=lane.get('metrics');record={'lane':lane['spec']['lane'],'actual_exit':lane['child'].returncode,'metrics':metrics,'stopped':lane['stopped'],'run':str(lane['run']),
            'child_wall_seconds':lane.get('exited',time.monotonic())-lane['started'],'evaluation':{k:v for k,v in lane.get('evaluation',{}).items() if k!='child'},'evaluation_deferred':lane.get('evaluation_deferred'),'finish_failure':lane.get('finish_failure')}
        record['complete']=bool(not lane.get('finish_failure') and metrics and metrics['since_resume_iterations']==lane['spec']['new_steps'] and metrics['exact_requested_horizon_complete'] and lane['child'].returncode==0)
        results.append(record)
    STATE.update(phase='finished',lanes=results,training_complete=all(r['complete'] for r in results),complete=all(r['complete'] and (not l['spec'].get('post_training_evaluation') or r['evaluation'].get('complete',False)) for r,l in zip(results,lanes)));emit('continuation_finished',complete=STATE['complete'])
    if not STATE['training_complete']:raise RuntimeError('Continuation horizon incomplete; actual successful peer/partial states retained')


def terminate(signum,frame):
    if ALARM_STAGE=='work':raise InterruptedError('Parent termination')
    STATE['termination_during_finalization']=True


def finalize():
    """All deadlines are absolute since STARTED, not serial per-child allowances."""
    STATE['cleanup']=[]
    arm_absolute(CLEANUP_END,'cleanup')
    try:
        for child,stream in CHILDREN:
            try:record=stop(child,deadline=STARTED+CLEANUP_END)
            except Exception as e:record={'pid':child.pid,'reaped':False,'group_exited':False,'errors':[repr(e)]}
            STATE['cleanup'].append(record)
            try:stream.close()
            except Exception as e:record['errors'].append('log close: '+repr(e))
    except AbsoluteDeadline as e:
        STATE.update(complete=False,cleanup_deadline_failure=repr(e))
        arm_absolute(OUTER,'hard_exit')
        # Visit every group even if an earlier group exhausted its allowance.
        for child,stream in CHILDREN:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            except OSError:pass
        STATE['cleanup_incomplete']=True
    if not all(x['reaped'] and x['group_exited'] for x in STATE['cleanup']) or len(STATE['cleanup'])!=len(CHILDREN):
        STATE['complete']=False;STATE['cleanup_incomplete']=True
    STATE['artifact_sha256']={};STATE['artifact_hashes_complete']=False
    arm_absolute(HASH_END,'artifact_finalization')
    try:
        if STATE.get('cleanup_incomplete'):raise RuntimeError('Skip hashes while child groups may still write')
        for p in ROOT.rglob('*'):
            if p.is_file() and p.name!='session_result.json':
                STATE['artifact_sha256'][str(p.relative_to(ROOT))]=sha(p,deadline=STARTED+HASH_END)
        STATE['artifact_hashes_complete']=True
    except (Exception,AbsoluteDeadline) as e:
        STATE.update(complete=False,artifact_hash_failure=repr(e))
    # Five seconds remain for a minimal terminal sidecar after bounded hashing.
    # Python signal delivery cannot interrupt every uninterruptible native/kernel call;
    # explicit external platform18000s remains the final bound in that failure mode.
    arm_absolute(OUTER,'hard_exit')
    STATE['session_script_seconds']=time.monotonic()-STARTED
    STATE['absolute_budget_seconds']={'work_cutoff':WORK,'aggregate_cleanup_end':CLEANUP_END,
        'artifact_hash_end':HASH_END,'terminal_hard_exit':OUTER,'external_platform':PLATFORM}
    temporary=ROOT/'session_result.json.tmp'
    temporary.write_text(json.dumps(STATE,indent=2,allow_nan=False)+'\n');os.replace(temporary,ROOT/'session_result.json')
    # Keep hard alarm armed through interpreter shutdown/stream flush.


if __name__=='__main__':
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGALRM,absolute_alarm)
    arm_absolute(WORK,'work')  # Covers startup discovery, checkpoint hashing, setup and polling.
    try:main()
    except BaseException as e:STATE.update(complete=False,failure=repr(e));raise
    finally:finalize()
