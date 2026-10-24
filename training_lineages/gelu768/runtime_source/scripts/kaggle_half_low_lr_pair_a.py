"""Finite same-parent half-endpoint low-LR paired continuation; no post-training decode."""
import hashlib,json,math,os,signal,subprocess,sys,time,zipfile
from pathlib import Path
STARTED=time.monotonic();ROOT=Path('/kaggle/working');CHILDREN=[]
PLATFORM=2400;OUTER=2340;WORK=1800;WORKER=1795;CLEANUP_END=2040;HASH_END=2295
ALARM_STAGE='unarmed';STATE={'schema':'half_low_lr_parent_result_a_v1','complete':False,'phase':'startup','runs':[]}
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

def fsync_directory(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.next')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path);fsync_directory(path.parent)

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
    # Forty-five seconds remain for a minimal terminal sidecar after bounded hashing.
    # Python signal delivery cannot interrupt every uninterruptible native/kernel call;
    # explicit external platform2400s remains the final bound in that failure mode.
    arm_absolute(OUTER,'hard_exit')
    STATE['session_script_seconds']=time.monotonic()-STARTED
    STATE['absolute_budget_seconds']={'work_cutoff':WORK,'aggregate_cleanup_end':CLEANUP_END,
        'artifact_hash_end':HASH_END,'terminal_hard_exit':OUTER,'external_platform':PLATFORM}
    temporary=ROOT/'session_result.json.tmp'
    temporary.write_text(json.dumps(STATE,indent=2,allow_nan=False)+'\n');os.replace(temporary,ROOT/'session_result.json')

def locate(name,digest):
    matches=[]
    for p in Path('/kaggle/input').rglob(name):
        if p.is_file() and sha(p,STARTED+WORK)==digest:matches.append(p.resolve())
    matches=sorted(set(matches));require(len(matches)==1,'Expected one exact mounted content identity:'+name);return matches[0]

def snapshot(lanes):
    STATE['runs']=[{k:v for k,v in x.items() if k not in ('child','spec')} for x in lanes]

def main():
    ROOT.mkdir(exist_ok=True);STATE['phase']='verify_inputs'
    matches=[p for p in Path('/kaggle/input').rglob('half_low_lr_manifest_a.json') if '__TOOLS_MOUNT__' in p.parts];require(len(matches)==1,'Exact new tools mount required');mp=matches[0];tools=mp.parent
    require(sha(mp,STARTED+WORK)=='__MANIFEST_SHA256__','Root-bound stage manifest differs');m=json.loads(mp.read_text());STATE.update(manifest_sha256=sha(mp),source_sha256=m['input_sha256']['source.bin'],driver_sha256=sha(__file__),runtime_source_sha256=m['runtime_source_sha256'],parent_terminal_sha256=m['parent_terminal_sha256'],phase='hash_inputs')
    require(m['schema']=='half_low_lr_stage_a_v1' and m['verified_actual_values'] is True and m['bounds']=={'platform':2400,'parent':2340,'work':1800,'worker':1795,'cleanup':2040,'hash':2295} and m['role_order']==['low-lr-eighth','low-lr-quarter'],'Typed finite paired stage required')
    for n,h in m['input_sha256'].items():require(sha(tools/n,STARTED+WORK)==h,'Tool input changed:'+n)
    terminal=locate('session_result.json',m['parent_terminal_sha256']);parent=terminal.parent/'run-cooldown-half/durable-slot1';data=locate('cache_manifest.json',m['cache_manifest_sha256']).parent
    parent_roles=m['parent_artifact_sha256'];require(set(parent_roles)=={'last_training.pt','model.pt','metrics.json','run_config.json','tokenizer.json','validation.json','validation_legacy.json','data_stats.json'},'All8 actual endpoint roles required')
    for n,h in parent_roles.items():require(sha(parent/n,STARTED+WORK)==h,'Actual D endpoint role differs:'+n)
    for n,h in m['cache_checksums'].items():require(sha(data/n,STARTED+WORK)==h,'Exact6M cache differs:'+n)
    STATE['input_hashing_complete']=True;runtime=ROOT/'runtime';runtime.mkdir()
    with zipfile.ZipFile(tools/'source.bin') as z:
        require(len(z.namelist())==len(set(z.namelist()))==len(m['runtime_source_sha256']),'Exact unique source inventory required')
        for n,h in m['runtime_source_sha256'].items():
            rel=Path(n);require(not rel.is_absolute() and '..' not in rel.parts,'Unsafe source name');b=z.read(n);require(hashlib.sha256(b).hexdigest()==h,'Runtime member changed:'+n);p=runtime/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    hardware=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True,timeout=10,check=True).stdout.splitlines();require(len(hardware)==2 and all('T4' in v for v in hardware),'Exactly2T4 required');STATE['hardware']=hardware
    coordination=ROOT/'coordination';coordination.mkdir();lanes=[];contracts={}
    for device,run_id in enumerate(m['role_order']):
        spec=m['runs'][run_id];cp=tools/spec['contract_path'];c=json.loads(cp.read_text());fraction=.125 if device==0 else .25
        require(c['run_id']==run_id and c['lr_fraction']==fraction and c['verified_actual_values'] is True and c['new_steps']==m['root_bound_NEW_cap'],'Exact bound worker role required');contracts[run_id]=sha(cp)
        out=ROOT/('run-'+run_id);cmd=[sys.executable,'-u',str(runtime/'scripts/train_half_low_lr_a.py'),'--run-id',run_id,'--lane','control','--contract',str(cp),'--terminal',str(terminal),'--parent',str(parent),'--data',str(data),'--source-archive',str(tools/'source.bin'),'--output',str(out),'--new-steps',str(c['new_steps']),'--deadline',str(STARTED+WORKER),'--coordination',str(coordination)]
        try:
            child=start(cmd,{'CUDA_VISIBLE_DEVICES':str(device),'PYTHONPATH':str(runtime/'src')+os.pathsep+str(runtime/'scripts'),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'},ROOT/(run_id+'.log'))
            lanes.append({'run_id':run_id,'device':device,'lr_fraction':fraction,'child':child,'pid':child.pid,'done':False,'complete':False,'stopped':False,'stop_time':None,'progress':None,'spec':spec});emit('worker_started',run_id=run_id,pid=child.pid,device=device)
        except OSError as e:STATE.setdefault('launch_failures',[]).append({'run_id':run_id,'type':type(e).__name__});emit('launch_failed',run_id=run_id,error_type=type(e).__name__)
    snapshot(lanes);decision=None;first_ready=None;failed=len(lanes)!=2;last_emit=0.
    while any(not l['done'] for l in lanes):
        now=time.monotonic();require(now<STARTED+WORK,'Absolute whole-work cutoff')
        requests={}
        for l in lanes:
            run_id=l['run_id'];out=ROOT/('run-'+run_id);p=out/'progress.json'
            if p.is_file():
                try:l['progress']=json.loads(p.read_text())
                except (ValueError,OSError):pass
            rp=coordination/(run_id+'-request.json')
            if rp.is_file():requests[run_id]=json.loads(rp.read_text())
            if l['done'] or l['child'].poll() is None:continue
            l['actual_exit']=l['child'].returncode;l['done']=True
            try:
                result=json.loads((out/'worker_result.json').read_text());pointer=json.loads((out/'latest_checkpoint.json').read_text());require(result['schema']=='half_low_lr_worker_result_a_v1' and result['run_id']==run_id and result['latest_checkpoint']==pointer,'Actual child result/current pointer mismatch')
                require(pointer['contract_sha256']==contracts[run_id] and set(pointer['files_sha256'])==set(parent_roles),'Actual committed8role map required')
                for n,h in pointer['files_sha256'].items():require(sha(out/pointer['slot']/n,STARTED+WORK)==h,'Committed output role differs:'+n)
                l.update(result=result,durable_state=pointer,metrics=pointer['metrics'],complete=l['actual_exit']==0 and result['admitted_horizon_complete'] is True)
                if not l['complete']:failed=True
            except Exception as e:l['finish_failure']={'type':type(e).__name__,'message':str(e)[:2000]};failed=True
            snapshot(lanes);emit('worker_harvest',run_id=run_id,complete=l['complete'],actual_exit=l['actual_exit'])
        if requests and first_ready is None:first_ready=min(q['ready_monotonic'] for q in requests.values())
        if decision is None and len(requests)==2 and not failed:
            require(now-first_ready<120,'Common admission wait exceeded120s');rates={};fits={};details={}
            for run_id in m['role_order']:
                q=requests[run_id];fraction=.125 if run_id=='low-lr-eighth' else .25
                require(q['schema']=='half_low_lr_prefix_request_a_v1' and q['run_id']==run_id and q['lr_fraction']==fraction and q['contract_sha256']==contracts[run_id] and q['prefix_NEW_iterations']==1024 and q['absolute_step']==418408 and q['worker_deadline_monotonic']==STARTED+WORKER,'Actual1024 prefix contract/counter/deadline required')
                rate=q['measured_steps_per_second'];seconds=q['checkpoint_inclusive_seconds'];save=q['checkpoint_cost_seconds'];require(math.isfinite(q['ready_monotonic']) and q['ready_monotonic']<=now,'Actual request clock required');require(all(math.isfinite(x) and x>0 for x in (rate,seconds,save)) and math.isclose(rate,1024/seconds,rel_tol=1e-9),'Actual checkpoint-inclusive rate required')
                pp=ROOT/('run-'+run_id)/'latest_checkpoint.json';require(sha(pp,STARTED+WORK)==q['pointer_sha256'],'Stable prefix pointer required')
                reserve=max(180.,1.5*save);remaining=STARTED+WORKER-now;binding_reserve=120.;fit=1024+64*math.floor(max(0.,remaining-reserve-binding_reserve)*rate/1.25/64);fits[run_id]=min(m['root_bound_NEW_cap'],fit);rates[run_id]=rate;details[run_id]={'remaining_seconds':remaining,'checkpoint_inclusive_seconds':seconds,'rate':rate,'checkpoint_cost_seconds':save,'save_reserve_seconds':reserve,'binding_reserve_seconds':binding_reserve,'uncertainty_factor':1.25,'fit_NEW_horizon':fits[run_id]}
            raw=[requests[k]['raw_segment_sha256'] for k in m['role_order']];applied_prefix=[requests[k]['applied_segment_sha256'] for k in m['role_order']];require(len(set(raw))==len(set(applied_prefix))==1,'RAW/applied ROWORDER prefix differs; no matched-data comparison')
            H=min(fits.values());require(H>=1024 and H%64==0,'No meaningful64-aligned retained1024 horizon')
            decision={'schema':'half_low_lr_common_horizon_a_v1','root_bound_NEW_cap':m['root_bound_NEW_cap'],'common_NEW_horizon':H,'requests_sha256':{k:sha(coordination/(k+'-request.json')) for k in m['role_order']},'worker_deadline_monotonic':STARTED+WORKER,'decision_monotonic':now,'details':details,'same_RAW_prefix':True,'prefix_raw_segment_sha256':raw,'prefix_applied_segment_sha256':{k:requests[k]['applied_segment_sha256'] for k in m['role_order']},'applied_roworder_equality_required':True,'optimizer_update_skip_equality_required':False};atomic_json(coordination/'common_horizon.json',decision);STATE['common_horizon']=decision;emit('common_horizon_admitted',H=H,details=details)
        if decision is None and first_ready is not None and now-first_ready>=120:failed=True;STATE['coordination_failure']='bounded common prefix wait expired'
        for l in lanes:
            if l['done']:continue
            if failed and not l['stopped']:
                # Keep a healthy near-finish peer when conservative actual ETA/save fits.
                q=l.get('progress') or {};completed=q.get('new_step',q.get('step',0));rate=requests.get(l['run_id'],{}).get('measured_steps_per_second');save=requests.get(l['run_id'],{}).get('checkpoint_cost_seconds',0.);remaining=STARTED+WORKER-now
                eta=(decision['common_NEW_horizon']-completed)/rate if decision and isinstance(completed,(int,float)) and rate else None
                if eta is not None and 0<=eta<=180 and 1.25*eta+max(180.,1.5*save)<=remaining:l['healthy_nearfinish_retained']=True
                else:
                    try:os.killpg(l['child'].pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    l['stopped']=True;l['stop_time']=now
            if l['stopped'] and now-l['stop_time']>180:stop(l['child'],STARTED+CLEANUP_END)
        snapshot(lanes)
        if now-last_emit>=15:emit('training_progress',runs=STATE['runs']);last_emit=now
        time.sleep(.25)
    STATE['training_complete']=len(lanes)==2 and all(l['complete'] for l in lanes);STATE['paired_complete']=False
    if STATE['training_complete'] and decision:
        metrics=[l['metrics'] for l in lanes];H=decision['common_NEW_horizon'];raw=[x['sampler_checkpoint']['segments'][-1]['raw_segment_sha256'] for x in metrics];applied=[x['sampler_checkpoint']['segments'][-1]['applied_segment_sha256'] for x in metrics]
        invariants={'same_H':all(x['since_resume_iterations']==H for x in metrics),'attempt_accounting':all(x['since_resume_optimizer_updates']+x['since_resume_skipped_updates']==H for x in metrics),'same_raw':len(set(raw))==1,'same_applied_roworder':len(set(applied))==1,'raw_segment_sha256':raw,'applied_segment_sha256':applied,'updates_skips':{l['run_id']:{'updates':l['metrics']['since_resume_optimizer_updates'],'skips':l['metrics']['since_resume_skipped_updates']} for l in lanes},'applied_roworder_equality_required':True,'optimizer_update_skip_equality_required':False};STATE['pair_invariants']=invariants;STATE['paired_complete']=invariants['same_H'] and invariants['attempt_accounting'] and invariants['same_raw'] and invariants['same_applied_roworder']
    STATE.update(complete=STATE['training_complete'] and STATE['paired_complete'],phase='finished',post_training_evaluation=False);snapshot(lanes);emit('paired_training_finished',complete=STATE['complete'])

if __name__=='__main__':
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGALRM,absolute_alarm);arm_absolute(WORK,'work')
    try:main()
    except BaseException as e:STATE.update(complete=False,failure_type=type(e).__name__,failure=str(e)[:4096]);raise
    finally:finalize()
