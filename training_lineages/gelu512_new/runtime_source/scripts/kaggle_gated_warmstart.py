"""Finite pinned private T4x2 B034 warm-start comparison; no allocation calls."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

STARTED=time.monotonic();WORKING=Path('/kaggle/working');TEMP=Path('/kaggle/temp/gated-warmstart')
LANES=[];STATE={'complete':False,'phase':'startup'};OUTER=18000;TAIL=1500


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def emit(row):
    row={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'session_seconds':time.monotonic()-STARTED,**row}
    with (WORKING/'session_events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def terminate(lane,reason):
    if lane['proc'].poll() is None and not lane['stopped']:
        lane['stopped']=True;lane['stop_time']=time.monotonic()
        os.killpg(lane['proc'].pid,signal.SIGTERM)
        emit({'event':'graceful_stop','lane':lane['id'],'reason':reason,'step':lane['step'],'eta_seconds':lane.get('eta')})


def finish_children():
    for lane in LANES:
        if lane['proc'].poll() is None:
            terminate(lane,'parent_shutdown')
            try:lane['proc'].wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(lane['proc'].pid,signal.SIGKILL);lane['proc'].wait(timeout=10)
        if not lane['stream'].closed:lane['stream'].close()
        lane.setdefault('exited_at',time.monotonic()-STARTED)


def main():
    TEMP.mkdir(parents=True,exist_ok=False)
    candidates=[p for p in Path('/kaggle/input').rglob('experiment-manifest.json')
        if json.loads(p.read_text()).get('schema')=='gated_b034_warmstart_v1']
    if len(candidates)!=1:raise ValueError('Exactly one pinned gated stage required')
    tools=candidates[0].parent;experiment=json.loads(candidates[0].read_text())
    if experiment['driver_sha256']!=sha(__file__):raise ValueError('Driver SHA mismatch')
    if experiment['outer_timeout_seconds']!=OUTER or experiment['tail_reserve_seconds']!=TAIL:raise ValueError('Budget mismatch')
    for name,digest in experiment['input_sha256'].items():
        if sha(tools/name)!=digest:raise ValueError('Input SHA mismatch: '+name)
    shutil.copyfile(candidates[0],WORKING/'experiment-manifest.json');shutil.copyfile(__file__,WORKING/'run_driver.py')
    shutil.copyfile(tools/'training_source.bin',WORKING/'training_source.bin')
    source=TEMP/'source';source.mkdir()
    with zipfile.ZipFile(tools/'training_source.bin') as archive:
        for name in archive.namelist():
            if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe archive')
        archive.extractall(source)
    sys.path[:0]=[str(source/'src'),str(source/'scripts')]
    from nlp_hw02.progress import ProgressObserver
    from nlp_hw02.training_arrays import array_resident_bytes,available_memory_bytes
    from kaggle_eval import run_evaluation,training_observation
    cache_candidates=[p for p in Path('/kaggle/input').rglob('cache_manifest.json') if sha(p)==experiment['cache_manifest_sha256']]
    if len(cache_candidates)!=1:raise ValueError('Exactly one pinned6M cache required')
    cache=cache_candidates[0].parent;manifest=json.loads(cache_candidates[0].read_text())
    if not manifest['complete']:raise ValueError('Incomplete cache')
    # Stream actual immutable array bytes; no dataset code or copied token arrays.
    for name,digest in experiment['cache_checksums'].items():
        if sha(cache/name)!=digest:raise ValueError('Mounted6M cache SHA mismatch: '+name)
    import numpy as np
    for name,width,dtype in [('train_source.npy',256,np.uint16),('train_target.npy',192,np.uint16),('train_row_ids.npy',2,np.uint32)]:
        array=np.load(cache/name,mmap_mode='r')
        if array.shape!=(6000000,width) or array.dtype!=dtype:raise ValueError('Mounted array shape/dtype mismatch')
        del array
    size=array_resident_bytes(cache);available=available_memory_bytes();required=3*size+2*1024**3
    if available<required:raise RuntimeError('Insufficient RAM for two resident copies plus shared pagecache/headroom')
    emit({'event':'cache_and_ram_admitted','manifest_sha256':sha(cache/'cache_manifest.json'),
        'array_bytes_per_lane':size,'required_bytes':required,'available_bytes':available})
    env={**os.environ,'PYTHONPATH':str(source/'src'),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'}
    hardware=subprocess.run([sys.executable,'-c',"import json,torch; assert torch.cuda.device_count()==2; names=[torch.cuda.get_device_name(i) for i in range(2)]; assert all('T4' in n for n in names); print(json.dumps({'gpus':names,'torch':str(torch.__version__),'cuda':torch.version.cuda}))"],env=env,capture_output=True,text=True,check=True,timeout=90)
    emit({'event':'hardware_preflight',**json.loads(hardware.stdout)})
    if time.monotonic()-STARTED>240:raise RuntimeError('Startup exceeded240s; abort before training')
    STATE.update({'phase':'training','source_sha256':experiment['input_sha256']['training_source.bin'],
        'driver_sha256':experiment['driver_sha256'],'cache_manifest_sha256':experiment['cache_manifest_sha256']})
    for specification in experiment['lanes']:
        label=specification['id'];run=WORKING/('run-'+label);stream=(WORKING/(label+'.log')).open('w')
        command=[sys.executable,'-u',str(source/'scripts/train_gated_warmstart.py'),'--lane',label,
            '--parent',str(tools/'parent-b034'),'--data',str(cache),'--output',str(run),
            '--steps',str(specification['steps']),'--seed','42','--resident']
        proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env={**env,'CUDA_VISIBLE_DEVICES':str(specification['device'])},start_new_session=True)
        lane={'id':label,'proc':proc,'run':run,'stream':stream,'steps':specification['steps'],
            'observer':ProgressObserver(time.monotonic()),'step':0,'rate':None,'eta':None,'stopped':False,
            'admission':False,'forwarded':set()};LANES.append(lane)
        emit({'event':'lane_started','lane':label,'pid':proc.pid,'new_steps':lane['steps']})
    last_announcement=0
    while any(l['proc'].poll() is None for l in LANES):
        now=time.monotonic();remaining=OUTER-(now-STARTED)
        for lane in LANES:
            if lane['proc'].poll() is not None:
                if 'exited_at' not in lane:
                    lane['exited_at']=now-STARTED;emit({'event':'lane_training_ended','lane':lane['id'],'exit':lane['proc'].returncode})
                continue
            heartbeat=lane['observer'].observe(lane['run']/'progress.json',now)
            if heartbeat:lane['step']=max(lane['step'],heartbeat['step'])
            path=lane['run']/'train.jsonl';records=[]
            if path.exists():
                for line in path.read_text().splitlines()[-80:]:
                    try:records.append(json.loads(line))
                    except ValueError:pass
            train=[r for r in records if r.get('event')=='train']
            if len(train)>=2:
                a,b=train[-min(len(train),6)],train[-1]
                lane['rate']=(b['step']-a['step'])/(b['elapsed_seconds']-a['elapsed_seconds'])
            for row in records:
                event=row.get('event');step=row.get('step',0);key=(event,step)
                if key not in lane['forwarded'] and (event in ('initial_validation','validation','gate_diagnostics','initial_copied_ffn_gradients','matched_snapshot','finished') or (event=='train' and (step<=300 or step%1000==0))):
                    lane['forwarded'].add(key);emit({'event':'lane_record','lane':lane['id'],'record':row})
            lane['eta']=None if not lane['rate'] else max(0,lane['steps']-lane['step'])/lane['rate']
            admission_records=[r for r in records if r.get('event')=='checkpoint_completed' and r.get('step',0)>=1000]
            if admission_records and not lane['admission']:
                admission_record=admission_records[0]
                admission_rate=admission_record['step']/admission_record['elapsed_seconds']
                lane['eta']=max(0,lane['steps']-lane['step'])/admission_rate
                admitted=lane['eta']+TAIL<=remaining;lane['admission']=True
                emit({'event':'budget_admission','lane':lane['id'],'new_iterations_observed':lane['step'],
                    'estimated_remaining_seconds':lane['eta'],'tail_reserve_seconds':TAIL,'admitted':admitted})
                if not admitted:terminate(lane,'actual1000NEW_eta_does_not_fit')
            phase=heartbeat['phase'] if heartbeat else 'startup'
            bound=240 if phase in ('startup','initializing','load_source','load_target','model_initialization','initial_validation','validation','final_validation','checkpoint') else 120
            if lane['observer'].stalled(now,bound):terminate(lane,'stalled_'+phase)
            eta=lane['eta'];near_finish=eta is not None and eta<=180 and eta+120<remaining
            if remaining<TAIL and not near_finish and (eta is None or eta+1200>remaining):terminate(lane,'tail_budget')
            if lane['stopped'] and now-lane['stop_time']>90:
                os.killpg(lane['proc'].pid,signal.SIGKILL);emit({'event':'stuck_child_killed','lane':lane['id']})
        if now-last_announcement>=60:
            emit({'event':'progress','lanes':[{'id':l['id'],'step':l['step'],'eta':l['eta'],'exit':l['proc'].poll()} for l in LANES]});last_announcement=now
        time.sleep(5)
    finish_children();completed=[]
    for lane in LANES:
        observation=training_observation(lane['run'],lane['steps']);lane['observation']=observation
        if lane['proc'].returncode==0 and not lane['stopped'] and observation['steps']==lane['steps'] and observation['exact']:completed.append(lane)
    STATE.update({'training_complete':len(completed)==2,'phase':'evaluation'})
    # Same first30k sampled row IDs are independently digested by both actual trainers.
    if len(completed)==2:
        snapshots=[json.loads((l['run']/'matched_330k/metrics.json').read_text()) for l in completed]
        if snapshots[0]['sampled_indices_sha256']!=snapshots[1]['sampled_indices_sha256']:raise ValueError('Matched-horizon sample streams differ')
        emit({'event':'matched_sample_stream_verified','sha256':snapshots[0]['sampled_indices_sha256'],'draws':1920000})
    evaluations=[]
    for lane in completed:
        entries=[(lane['id'],lane['run'])]
        if lane['id']=='control':entries.insert(0,('control-matched330k',lane['run']/'matched_330k'))
        for label,run in entries:
            out=WORKING/('decoding-'+label)
            result=run_evaluation(label,[sys.executable,'-u',str(source/'scripts/profile_decoding_gpu.py'),
                '--run',str(run),'--output',str(out),'--test-parquet',str(tools/'test.parquet'),'--skip-synthetic-check'],
                WORKING/(label+'-evaluation.log'),{**env,'CUDA_VISIBLE_DEVICES':'0'},
                OUTER-(time.monotonic()-STARTED),300,emit,
                required=[out/'decoding_profile.json',out/'beam_test_timing.json',out/'beam_test_predictions.jsonl'])
            evaluations.append({'label':label,**result})
    STATE.update({'complete':len(completed)==2 and len(evaluations)==3 and all(r['status']=='complete' for r in evaluations),
        'phase':'finished','evaluations':evaluations})
    emit({'event':'finished',**STATE})
    if not STATE['training_complete']:raise RuntimeError('Training incomplete; retained actual evidence')


if __name__=='__main__':
    def stop_parent(signum,frame):raise SystemExit('Parent termination requested')
    signal.signal(signal.SIGTERM,stop_parent)
    try:main()
    except BaseException as error:
        STATE.update({'complete':False,'failure_type':type(error).__name__,'failure':str(error)})
        raise
    finally:
        finish_children()
        STATE['session_script_seconds']=time.monotonic()-STARTED
        STATE['lanes']=[{'id':l['id'],'exit':l['proc'].returncode,'stopped':l['stopped'],
            'observation':l.get('observation'),'idle_tail_seconds':time.monotonic()-STARTED-l['exited_at']} for l in LANES]
        paths=[p for p in WORKING.rglob('*') if p.is_file() and p.name!='session_result.json' and p.suffix in ('.json','.jsonl','.pt','.log','.bin','.py')]
        STATE['artifact_sha256']={str(p.relative_to(WORKING)):sha(p) for p in paths}
        (WORKING/'session_result.json').write_text(json.dumps(STATE,indent=2))
