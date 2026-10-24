"""Unbound uniform/bucket parent: run identity independent of architecture lane."""
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
PLATFORM=13200;OUTER=13140;WORK=11640;CLEANUP_END=12000;HASH_END=13095;TAIL=1500
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
        rows.append({'run_id':spec['run_id'],'lane':spec['lane'],'layout':spec['layout'],'requested_new_steps':spec['new_steps'],'actual_exit':lane['child'].poll(),
            'admitted_NEW_horizon':lane.get('admitted_target'),'new_step':lane['step'],'rate':lane['rate'],'eta':lane['eta'],'stopped':lane['stopped'],
            'run':str(lane['run']),'child_wall_seconds':lane.get('exited',time.monotonic())-lane['started'],
            'metrics':metrics,'worker_result':lane.get('worker_result'),'durable_state':lane.get('durable_state'),'cleanup':lane.get('cleanup'),'finish_failure':lane.get('finish_failure'),
            'evaluation':evaluation,'evaluation_deferred':lane.get('evaluation_deferred'),
            'complete':bool(not lane.get('finish_failure') and lane.get('horizon_complete',False) and lane['child'].poll()==0)})
    STATE['runs']=rows


def launch_evaluation(lane,source,tools):
    # Triggered on THIS lane's real exit, without waiting for another trainer.
    spec=lane['spec'];plan=spec.get('post_training_evaluation')
    if not lane.get('horizon_complete',False):
        lane['evaluation_deferred']={'reason':'Incomplete actual admitted horizon; preserve durable state without profiling','policy_skip':True}
        emit('post_training_evaluation_deferred',run_id=spec['run_id'],lane=spec['lane'],layout=spec['layout'],**lane['evaluation_deferred']);return
    if lane['durable_state']['files_sha256']['model.pt']=='eed1fdbb4f7e056536b322c08fad8660f5df24460f716a4d8e5382eb58ce8dc9':
        lane['evaluation_deferred']={'reason':'Exact inherited344k selected weights unchanged; retain already-proved canonical without redundant profiler','policy_skip':True}
        emit('post_training_evaluation_deferred',run_id=spec['run_id'],lane=spec['lane'],layout=spec['layout'],**lane['evaluation_deferred']);return
    if lane.get('metrics',{}).get('since_resume_iterations',0)<=0:
        lane['evaluation_deferred']={'reason':'No NEW durable updates; unchanged initial parent is not a new candidate'}
        emit('post_training_evaluation_deferred',run_id=spec['run_id'],lane=spec['lane'],layout=spec['layout'],**lane['evaluation_deferred']);return
    if plan is None:
        lane['evaluation_deferred']={'reason':'No root-bound evaluator/menu; no default beam/grid invented'}
        emit('post_training_evaluation_deferred',run_id=spec['run_id'],lane=spec['lane'],layout=spec['layout'],**lane['evaluation_deferred']);return
    bound=plan['bound_seconds'];require(type(bound) is int and 1<=bound<=300,'Explicit bounded evaluator required')
    if STARTED+WORK-time.monotonic()<bound+30:
        lane['evaluation_deferred']={'reason':'Insufficient absolute work reserve'}
        emit('post_training_evaluation_deferred',run_id=spec['run_id'],lane=spec['lane'],layout=spec['layout'],**lane['evaluation_deferred']);return
    command=plan['argv'];require(isinstance(command,list) and command and all(isinstance(v,str) for v in command),'Pinned evaluator argv required')
    script=Path(command[0]);require(not script.is_absolute() and '..' not in script.parts and str(script) in lane['source_members'],'Evaluator script must be pinned source member')
    required=plan['required_outputs'];require(isinstance(required,list) and required and all(isinstance(v,str) and v and not Path(v).is_absolute() and '..' not in Path(v).parts for v in required),'Explicit safe required evaluation outputs needed')
    output=ROOT/('evaluation-'+spec['run_id']);output.mkdir()
    values={'{run}':str(lane.get('evaluation_run',lane['run'])),'{output}':str(output),'{tools}':str(tools),'{test_parquet}':str(tools/'test.parquet')}
    argv=[sys.executable,'-u',str(source/script)]+[values.get(v,v) for v in command[1:]]
    child=start(argv,{'CUDA_VISIBLE_DEVICES':str(spec['device']),'PYTHONPATH':str(source/'src'),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'},ROOT/(spec['run_id']+'-evaluation.log'))
    lane['evaluation']={'child':child,'started':time.monotonic(),'bound':bound,'output':str(output),'complete':False,'required_outputs':required};emit('post_training_evaluation_started',run_id=spec['run_id'],lane=spec['lane'],layout=spec['layout'],pid=child.pid,bound_seconds=bound)


def fsync_directory(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.next')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path);fsync_directory(path.parent)

def main():
    candidates=list(Path('/kaggle/input').rglob('gated_cooldown_continuation_manifest.json'));require(len(candidates)==1,'Exactly one future actual-bound continuation manifest required')
    path=candidates[0];tools=path.parent;m=json.loads(path.read_text())
    require(m['schema']=='gated_cooldown_parent_a_v1' and m['verified_actual_values'] is True,'Unbound continuation source cannot launch')
    require(m['driver_sha256']==sha(__file__) and m['platform_seconds']==PLATFORM,'Source/platform mismatch')
    for name,digest in m['input_sha256'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe tools input path');require(sha(tools/name)==digest,'Bound input differs:'+name)
    cache_candidates=[p for p in Path('/kaggle/input').rglob('cache_manifest.json') if sha(p)==m['cache_manifest_sha256']]
    require(len(cache_candidates)==1,'Exactly one original6M cache required');cache=cache_candidates[0].parent
    cm=json.loads(cache_candidates[0].read_text());require(cm['complete'] is True and cm['checksums']==m['cache_checksums'],'Actual cache manifest differs')
    for name,digest in cm['checksums'].items():require(sha(cache/name)==digest,'Actual cache bytes differ:'+name)
    source=ROOT/'cooldown_continuation_source';source.mkdir()
    import shutil
    with zipfile.ZipFile(tools/'cooldown_continuation_source.bin') as z:
        require(len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(m['source_members_sha256']),'Exact source-member allowlist required')
        for name,digest in m['source_members_sha256'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe runtime source path');target=source/name;target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(name) as incoming,target.open('xb') as outgoing:shutil.copyfileobj(incoming,outgoing,1024*1024)
            require(sha(target)==digest,'Extracted runtime member differs:'+name)
    sys.path[:0]=[str(source/'src'),str(source/'scripts')]
    from nlp_hw02.progress import ProgressObserver
    from nlp_hw02.training_arrays import array_resident_bytes,available_memory_bytes
    specs=m['runs'];require(len(specs)==2 and len({s['run_id'] for s in specs})==len(specs) and len({s['device'] for s in specs})==len(specs),'One or two distinct root-chosen lanes/devices required')
    require(all(isinstance(s['run_id'],str) and 1<=len(s['run_id'])<=64 and s['run_id'][0] in 'abcdefghijklmnopqrstuvwxyz' and all(ch in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for ch in s['run_id']) and s['layout'] in ('uniform','bucket') and s['lane'] in ('control','treatment') and s['device'] in (0,1) and type(s['new_steps']) is int and s['new_steps']>=1024 and s['new_steps']%64==0 and type(s['resident']) is bool for s in specs),'Explicit future lane/horizon/layout loading choices required')
    from nlp_hw02.gated_cooldown_schedule_a import prospective_plan,digest,checkpoint_schedule,ARMS
    from gated_cooldown_coordination_b import coordinate
    require({s['cooldown_arm'] for s in specs}==set(ARMS) and all(s['lane']=='control' and s['layout']=='bucket' and s['new_steps']==81920 for s in specs),'Two standard/bucket arms with explicit81920 cap required')
    coordination=ROOT/'cooldown-coordination';coordination.mkdir();coordination_state={};STATE['cooldown']=coordination_state
    size=array_resident_bytes(cache);resident_count=sum(s['resident'] for s in specs)
    if resident_count:require(available_memory_bytes()>=(resident_count+1)*size+2*1024**3,'Actual aggregate resident RAM admission failed')
    hardware=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=15).stdout.splitlines()
    require(len(hardware)==2 and all('T4' in n for n in hardware),'T4x2 expected; each worker gets one visible GPU')
    shutil.copyfile(path,ROOT/path.name);shutil.copyfile(__file__,ROOT/'run_driver.py');shutil.copyfile(tools/'cooldown_continuation_source.bin',ROOT/'cooldown_continuation_source.bin')
    STATE.update(phase='training',manifest_sha256=sha(path),source_sha256=sha(tools/'cooldown_continuation_source.bin'),driver_sha256=sha(__file__),hardware=hardware)
    lanes=[]
    for spec in specs:
        name=spec['run_id'];run=ROOT/('run-'+name)
        require(isinstance(spec['contract_path'],str) and bool(spec['contract_path']) and not Path(spec['contract_path']).is_absolute() and '..' not in Path(spec['contract_path']).parts and spec['contract_path'] in m['input_sha256'],'Safe hash-bound worker contract required')
        contract=tools/spec['contract_path'];binding=json.loads(contract.read_text());require(binding['verified_actual_values'] is True and binding['lane']==spec['lane'] and binding['run_id']==name and binding['layout']==spec['layout'] and binding['new_steps']==spec['new_steps'] and binding['admission_mode']=='measured64_shortening' and binding['cooldown_arm']==spec['cooldown_arm'] and binding['cooldown_plan']==prospective_plan(spec['cooldown_arm']) and binding['cooldown_plan_sha256']==digest(binding['cooldown_plan']),'Actual worker binding differs')
        # Original kernel output remains mounted remotely; no local fullstate download/toolspayload copy.
        terminals=[p for p in Path('/kaggle/input').rglob('session_result.json') if sha(p)==binding['parent_terminal_sha256']]
        require(len(terminals)==1,'Exactly one actual original terminal hash match required')
        terminal=terminals[0]
        require(isinstance(binding['parent_run_path'],str) and not Path(binding['parent_run_path']).is_absolute() and '..' not in Path(binding['parent_run_path']).parts,'Safe explicit parent slot path required')
        parent=terminal.parent/binding['parent_run_path']
        require(binding['runtime_archive_sha256']==m['input_sha256']['cooldown_continuation_source.bin'] and binding['runtime_source_sha256']==m['source_members_sha256'],'Contract/current extracted archive identity differs')
        command=[sys.executable,'-u',str(source/'scripts/train_gated_cooldown_prefix_c.py'),'--run-id',name,'--lane',spec['lane'],'--contract',str(contract),'--terminal',str(terminal),
            '--parent',str(parent),'--data',str(cache),'--source-archive',str(tools/'cooldown_continuation_source.bin'),'--output',str(run),'--new-steps',str(spec['new_steps']),'--deadline',str(STARTED+WORK-5),'--coordination',str(coordination)]
        if spec['resident']:command.append('--resident')
        child=start(command,{'CUDA_VISIBLE_DEVICES':str(spec['device']),'PYTHONPATH':str(source/'src')+os.pathsep+str(source/'scripts'),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'},ROOT/(name+'.log'))
        lane={'spec':spec,'run':run,'child':child,'started':time.monotonic(),'observer':ProgressObserver(time.monotonic()),'step':0,'rate':None,'eta':None,'stopped':False,'source_members':m['source_members_sha256'],'contract_sha256':sha(contract),'plan_sha256':binding['cooldown_plan_sha256']};lanes.append(lane);snapshot_lanes(lanes);emit('continuation_started',run_id=name,lane=lane['spec']['lane'],layout=lane['spec']['layout'],pid=child.pid,new_steps=spec['new_steps'],device=spec['device'])
    def stop_pair(reason,failed_runs):
        decisions=[];now=time.monotonic();remaining=STARTED+WORK-5-now
        emit('cooldown_pair_controlled_stop',reason=reason,failed_runs=failed_runs)
        for lane in lanes:
            if lane['child'].poll() is None and not lane['stopped']:
                name=lane['spec']['run_id'];eta=lane.get('eta');rate=lane.get('rate');reserve=lane.get('checkpoint_reserve')
                if any(v is None for v in (eta,rate,reserve)) and name in coordination_state.get('requests',{}):
                    actual=coordination_state['requests'][name];rate=actual['measured_steps_per_second'];reserve=max(180.,1.5*actual['fullsave_cost_seconds'])
                    eta=max(0,coordination_state['decision']['common_NEW_horizon']-max(1024,lane['step']))/rate
                reliable=all(isinstance(v,(int,float)) and math.isfinite(v) for v in (eta,rate,reserve)) and eta>=0 and rate>0 and reserve>0
                preserve=name not in failed_runs and reliable and eta<=180 and 1.25*eta+reserve<remaining
                decision={'run_id':name,'preserved_nearfinish':preserve,'eta':eta,'checkpoint_reserve':reserve,'remaining_worker_seconds':remaining};decisions.append(decision)
                if preserve:
                    emit('cooldown_healthy_nearfinish_preserved',**decision);continue
                try:os.killpg(lane['child'].pid,signal.SIGTERM);lane['stopped']=True;lane['stop_time']=time.monotonic()
                except ProcessLookupError:pass
                except OSError as error:emit('cooldown_peer_signal_error',run_id=name,error_type=type(error).__name__)
        return decisions
    last_emit=0
    while any('exited' not in l or (l.get('evaluation') and 'actual_exit' not in l['evaluation']) for l in lanes):
        coordinate(lanes,coordination,STARTED+WORK-5,coordination_state,atomic_json,emit,stop_pair)
        now=time.monotonic();remaining=STARTED+WORK-now
        for lane in lanes:
            child=lane['child'];name=lane['spec']['run_id']
            if child.poll() is not None:
                if 'exited' not in lane:
                    lane['exited']=now
                    try:
                        cleanup=stop(child);lane['cleanup']=cleanup;emit('actual_trainer_exit',run_id=name,lane=lane['spec']['lane'],layout=lane['spec']['layout'],actual_exit=child.returncode,cleanup=cleanup,child_seconds=now-lane['started'])
                        require(cleanup['reaped'] and cleanup['group_exited'],'Owned trainer group remains; no mutable durable-state read')
                        run=lane['run'];pointer_path=run/'latest_checkpoint.json'
                        require(pointer_path.is_file(),'No committed durable checkpoint pointer')
                        pointer=json.loads(pointer_path.read_text());require(pointer['schema']=='gated_window_durable_pointer_v1' and pointer['slot'] in ('durable-slot0','durable-slot1'),'Invalid durable checkpoint slot')
                        directory=run/pointer['slot'];required={'last_training.pt','model.pt','run_config.json','metrics.json','tokenizer.json','validation.json','validation_legacy.json','data_stats.json'}
                        require(set(pointer['files_sha256'])==required,'Exact durable role allowlist differs')
                        for role_name,digest in pointer['files_sha256'].items():require(sha(directory/role_name)==digest,'Committed durable role bytes differ:'+role_name)
                        metrics=json.loads((directory/'metrics.json').read_text());binding=json.loads((tools/lane['spec']['contract_path']).read_text())
                        require(metrics['run_id']==name and metrics['lane']==lane['spec']['lane'] and metrics['layout']==lane['spec']['layout'],'Durable run/architecture/layout differs')
                        require(metrics==pointer['metrics'] and pointer['contract_sha256']==sha(tools/lane['spec']['contract_path']),'Durable pointer/metrics/current contract differs')
                        absolute=metrics['absolute_completed_step'];new=metrics['since_resume_iterations']
                        require(type(new) is int and 0<=new<=lane['spec']['new_steps'] and new%64==0 and pointer['since_resume_iterations']==new and pointer['absolute_step']==absolute,'Durable NEW window frontier differs')
                        require(absolute==binding['parent_absolute_step']+new and metrics['parent_absolute_step']==binding['parent_absolute_step'] and metrics['counter_origin']==300000,'Absolute/since-resume origin differs')
                        require(metrics['completed_steps']==metrics['warmstart_iterations']==absolute-300000 and metrics['sampled_examples']==64*(absolute-300000),'Cumulative warmstart count differs')
                        require(metrics['optimizer_updates']+metrics['skipped_updates']==absolute-300000 and metrics['since_resume_optimizer_updates']+metrics['since_resume_skipped_updates']==new and metrics['since_resume_sampled_examples']==64*new and metrics['lr_absolute_step']==absolute,'Update/sample/LR counter binding differs')
                        sampler=metrics['sampler_checkpoint'];require(sampler['absolute_step']==absolute and sampler['at_consumed_window_boundary'] is True and sampler['pending_rows']==0 and sampler['recipe_sha256']==pointer['recipe_sha256']==binding['recipe_sha256'] and sampler['runtime_archive_sha256']==binding['runtime_archive_sha256'] and sampler['runtime_source_sha256']==binding['runtime_source_sha256'],'Durable recipe/source frontier differs')
                        cooldown=metrics['cooldown_checkpoint'];require(cooldown==sampler['cooldown_checkpoint']==checkpoint_schedule(absolute,binding['cooldown_plan'],cooldown['binding']),'Durable cooldown plan/phase/LR differs')
                        if cooldown['binding'] is not None:
                            require(cooldown['binding']['decision_sha256']==sha(coordination/'common_decision.json') and cooldown['binding']['common_NEW_horizon']==coordination_state['decision']['common_NEW_horizon'],'Durable commonH/decision differs')
                            if cooldown['binding']['release_sha256'] is not None:require(cooldown['binding']['release_sha256']==sha(coordination/'common_release.json'),'Durable release binding differs')
                        lane.update(metrics=metrics,durable_state=pointer,evaluation_run=directory,horizon_complete=False)
                        result_path=run/'worker_result.json'
                        if result_path.is_file():
                            result=json.loads(result_path.read_text());require(result['schema']=='gated_window_worker_result_v1' and result['run_id']==name and result['lane']==lane['spec']['lane'] and result['layout']==lane['spec']['layout'] and result['latest_checkpoint']==pointer and result['durable_new_iterations']==new and result['durable_absolute_step']==absolute,'Actual worker/durable checkpoint differs')
                            require(new<=result['actual_model_iterations']<=lane['spec']['new_steps'] and result['requested_new_iterations']==lane['spec']['new_steps'],'Actual partial/new horizon differs')
                            lane['worker_result']=result
                            require(result['admission_mode']==metrics['admission_mode']==binding['admission_mode'] and metrics['root_bound_NEW_cap']==lane['spec']['new_steps'],'Admission mode/root cap differs')
                            target=result['admitted_NEW_horizon'];decision=result['admission']
                            require(target==metrics['admitted_NEW_horizon'] and decision==sampler['admission'],'Actual duration decision/fullstate differs')
                            if target is not None:
                                require(type(target) is int and 1024<=target<=lane['spec']['new_steps'] and target%64==0 and decision['admitted_NEW_horizon']==target and decision['root_bound_NEW_cap']==lane['spec']['new_steps'] and decision['mode']==binding['admission_mode'],'Invalid measured/fixed duration decision')
                                require(binding['admission_mode']!='fixed' or target==lane['spec']['new_steps'],'Fixed mode silently shortened')
                                lane['admitted_target']=target
                            lane['horizon_complete']=bool(child.returncode==0 and result['admitted_horizon_complete'] and decision and decision['admitted'] is True and not result['partial_nonresumable'] and result['actual_model_iterations']==new==target)
                        if child.returncode!=0:lane['finish_failure']='Trainer nonzero; verified last durable checkpoint retained'
                        snapshot_lanes(lanes)
                        # Evaluate this actual durable slot immediately, never a NONRESUMABLE partial model.
                        try:launch_evaluation(lane,source,tools)
                        except Exception as error:
                            lane['evaluation_deferred']={'reason':'launch_failure','failure_type':type(error).__name__,'failure':str(error)[:8192]}
                            emit('post_training_evaluation_launch_failure',run_id=name,**lane['evaluation_deferred'])
                        snapshot_lanes(lanes)
                    except Exception as error:
                        lane['finish_failure']=repr(error)[:8192];snapshot_lanes(lanes);emit('finished_lane_failure',run_id=name,lane=lane['spec']['lane'],layout=lane['spec']['layout'],error=lane['finish_failure'])
                        with (ROOT/(name+'.log')).open('rb') as stream:stream.seek(0,2);stream.seek(max(0,stream.tell()-8192));tail=stream.read(8192).decode(errors='replace')
                        emit('bounded_failure_diagnostic',run_id=name,lane=lane['spec']['lane'],layout=lane['spec']['layout'],tail_max_bytes=8192,tail=tail)
                evaluation=lane.get('evaluation')
                if evaluation and evaluation['child'].poll() is None and now-evaluation['started']>=evaluation['bound']:
                    evaluation['cleanup']=stop(evaluation['child'],deadline=min(STARTED+WORK,now+20));evaluation['timed_out']=True
                if evaluation and evaluation['child'].poll() is not None and 'actual_exit' not in evaluation:
                    evaluation['actual_exit']=evaluation['child'].returncode;evaluation['seconds']=time.monotonic()-evaluation['started'];evaluation['cleanup']=stop(evaluation['child']);evaluation['missing_outputs']=[v for v in evaluation['required_outputs'] if not (Path(evaluation['output'])/v).is_file()];evaluation['complete']=evaluation['actual_exit']==0 and not evaluation.get('timed_out',False) and evaluation['cleanup']['group_exited'] and not evaluation['missing_outputs'];emit('actual_evaluation_exit',run_id=name,lane=lane['spec']['lane'],layout=lane['spec']['layout'],**{k:v for k,v in evaluation.items() if k!='child'})
                snapshot_lanes(lanes)
                continue
            heartbeat=lane['observer'].observe(lane['run']/'progress.json',now)
            if heartbeat:lane['step']=max(lane['step'],heartbeat['step'])
            events=lane['run']/'train.jsonl';records=[]
            if events.exists():
                for line in events.read_text().splitlines()[-100:]:
                    try:records.append(json.loads(line))
                    except ValueError:pass
            complete=[r for r in records if r.get('event')=='checkpoint_completed' and r['step']>=1024]
            if complete:
                a=complete[-1];lane['rate']=a['step']/a['elapsed_seconds'];lane['checkpoint_reserve']=max(180.,1.5*a['checkpoint_and_CE_seconds']);target=a.get('admitted_NEW_horizon')
                if target is not None:
                    require(type(target) is int and 1024<=target<=lane['spec']['new_steps'] and target%64==0,'Invalid observed admitted target');lane['admitted_target']=target
                lane['eta']=max(0,lane.get('admitted_target',lane['spec']['new_steps'])-lane['step'])/lane['rate']
            phase=heartbeat['phase'] if heartbeat else 'startup';bound=360 if phase!='training' else 180
            reason='stalled_'+phase if lane['observer'].stalled(now,bound) else None
            eta=lane['eta'];can_finish=eta is not None and 1.25*eta+lane.get('checkpoint_reserve',300.)<remaining
            # Parent finalization already has1500s AFTER work cutoff; do not kill a
            # healthy admitted horizon merely because it enters the last work segment.
            if remaining<TAIL and not can_finish:reason=reason or 'tail_budget'
            if reason and not lane['stopped']:
                lane['stopped']=True;lane['stop_time']=now
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                emit('graceful_training_stop',run_id=name,lane=lane['spec']['lane'],layout=lane['spec']['layout'],reason=reason,new_step=lane['step'],eta=eta)
            if lane['stopped'] and now-lane['stop_time']>180:
                lane['forced_cleanup']=stop(child,deadline=min(STARTED+WORK,now+20))
        snapshot_lanes(lanes)
        if now-last_emit>=30:emit('progress',lanes=[{'run_id':l['spec']['run_id'],'lane':l['spec']['lane'],'layout':l['spec']['layout'],'new_step':l['step'],'eta':l['eta'],'actual_exit':l['child'].poll()} for l in lanes]);last_emit=now
        time.sleep(.5)
    results=[]
    for lane in lanes:
        metrics=lane.get('metrics');record={'run_id':lane['spec']['run_id'],'lane':lane['spec']['lane'],'layout':lane['spec']['layout'],'actual_exit':lane['child'].returncode,'metrics':metrics,'stopped':lane['stopped'],'run':str(lane['run']),
            'child_wall_seconds':lane.get('exited',time.monotonic())-lane['started'],'evaluation':{k:v for k,v in lane.get('evaluation',{}).items() if k!='child'},'evaluation_deferred':lane.get('evaluation_deferred'),'finish_failure':lane.get('finish_failure'),'durable_state':lane.get('durable_state'),'worker_result':lane.get('worker_result'),'root_bound_NEW_cap':lane['spec']['new_steps'],'admitted_NEW_horizon':lane.get('admitted_target')}
        record['complete']=bool(not lane.get('finish_failure') and lane.get('horizon_complete',False) and lane['child'].returncode==0)
        results.append(record)
    STATE.update(phase='final_pair_validation',runs=results,paired_complete=False)
    if all(r['complete'] for r in results):
        segments=[r['metrics']['sampler_checkpoint']['segments'][-1] for r in results]
        raw=[v['raw_segment_sha256'] for v in segments];applied=[v['applied_segment_sha256'] for v in segments];whole=[r['metrics']['sampled_indices_sha256'] for r in results]
        horizons=[r['metrics']['since_resume_iterations'] for r in results]
        pair={'common_NEW_horizons':horizons,'raw_segment_sha256':raw,'applied_segment_sha256':applied,'whole_raw_sha256':whole,
            'same_H':len(set(horizons))==1 and horizons[0]==coordination_state['decision']['common_NEW_horizon'],'same_raw':len(set(raw))==1,'same_applied':len(set(applied))==1,'same_whole_raw':len(set(whole))==1}
        STATE['pair_invariants']=pair
        require(all(pair[k] for k in ('same_H','same_raw','same_applied','same_whole_raw')),'Final paired horizon/raw/applied equality failed; both actual model artifacts retained')
        STATE['paired_complete']=not coordination_state.get('pair_incomplete',False)
    STATE.update(phase='finished',runs=results,training_complete=all(r['complete'] for r in results),complete=STATE['paired_complete'] and all(r['complete'] and (not l['spec'].get('post_training_evaluation') or r['evaluation'].get('complete',False) or (r.get('evaluation_deferred') or {}).get('policy_skip',False)) for r,l in zip(results,lanes)));emit('continuation_finished',complete=STATE['complete'])
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
    # explicit external platform13200s remains the final bound in that failure mode.
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
