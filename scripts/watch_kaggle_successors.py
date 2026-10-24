"""Finite authorized pipeline: keep two useful GPU kernels, recover, then stop.

No cancellation, submissions, blind retries or automatic model promotion.
Stops new launches at the authorized reset, preserves healthy running kernels.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PYTHON='/Users/a2mogus/anaconda3/bin/python'
KAGGLE='/Users/a2mogus/anaconda3/bin/kaggle'
DEADLINE=datetime(2026,10,3,0,0,tzinfo=timezone.utc)
WATCH=ROOT/'reports/evidence/successor_watch.json'
EVENTS=ROOT/'artifacts/kaggle/successor-watch.jsonl'
CURRENT=[
    {'name':'b013-b014','kernel':'arsen4ikvar/nlp-hw02-xsa-retry-20261002/1',
     'manifest':'artifacts/kaggle/xsa-retry-tools/experiment-manifest.json','root':'artifacts/kaggle/b013-b014-v1'},
    {'name':'b015-b016','kernel':'arsen4ikvar/nlp-hw02-model-followup-20261002/1',
     'manifest':'artifacts/kaggle/model-followup-tools-20261002/experiment-manifest.json','root':'artifacts/kaggle/b015-b016-v1'}]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(argv,timeout=300):
    result=subprocess.run([str(x) for x in argv],cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src'),
        'PYTHONUNBUFFERED':'1'},text=True,capture_output=True,timeout=timeout)
    if result.returncode:raise RuntimeError(f'Command exit{result.returncode}: {argv}\n{result.stdout}\n{result.stderr}')
    return result.stdout


def status(job):
    value=command([KAGGLE,'kernels','status',job['kernel']],timeout=30)
    match=re.search(r'KernelWorkerStatus\.([A-Z_]+)',value)
    if match is None:raise ValueError('Unrecognized kernel status: '+value)
    return match.group(1)


def snapshot(state,event,**fields):
    record={'utc':datetime.now(timezone.utc).isoformat(),'event':event,**fields}
    EVENTS.parent.mkdir(parents=True,exist_ok=True)
    with EVENTS.open('a') as stream:stream.write(json.dumps(record)+'\n')
    state['last_event']=record
    WATCH.write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(record),flush=True)


def refresh_handoff(state):
    path=ROOT/'reports/HANDOFF.md';body=path.read_text();marker='## Selected result\n'
    if body.count(marker)!=1:raise ValueError('Unexpected handoff structure')
    rows=['| Pair | Kernel/version | Last observed status | Recovered |',
        '| --- | --- | --- | --- |']
    for job in state['jobs']:
        rows.append(f'| {job["name"]} | `{job["kernel"]}` | {job.get("status","just pushed; pending first status")} | {bool(job.get("recovered"))} |')
    failure=state.get('failure')
    head='# Agent handoff — B012 selected; finite GPU successor pipeline\n\n'
    head+='Updated '+datetime.now(timezone.utc).isoformat()+'.\n'
    head+='B012 selection/canonical proof unchanged; no submission. Current experiment\nstate is preserved in [watch evidence](evidence/successor_watch.json).\nTwo useful GPU kernels are authorized until October3 00:00UTC; no third\nallocation. Preserve healthy admitted jobs through completion after reset.\nThe watcher launches only the two declared successors, recovers bounded\noutputs and stops on a failure or changed source; no retries or promotion.\nTPU probe is separate: last root observation19:04UTC QUEUED, cause unknown,\ncompleted identity verification attested by user. This GPU watcher does not\nrefresh TPU status.\n\n'+'\n'.join(rows)+'\n\n'
    if failure:head+='**Watcher stopped; inspect before continuing:** '+failure.replace('\n',' ')+'\n\n'
    elif state['pipeline_complete']:head+='All four declared GPU pairs are recovered. Quality reports need review and\na new candidate needs exact canonical one-T4 proof before promotion.\n\n'
    else:head+='Work remains active. Read the current head and evidence before treating the\ndated sections below as present status.\n\n'
    path.write_text(head+marker+body.split(marker,1)[1])


def recover(job,state):
    snapshot(state,'recovery_start',kernel=job['kernel'])
    output=command([PYTHON,'scripts/recover_kaggle_training.py','--kernel',job['kernel'],
        '--root',job['root'],'--manifest',job['manifest'],'--evidence-prefix',job['name'].replace('-','_')],timeout=600)
    job['recovered']=True
    snapshot(state,'recovery_passed',kernel=job['kernel'],summary=output.strip())
    refresh_handoff(state)
    stamp=datetime.now(timezone.utc).isoformat()
    with (ROOT/'experiments/LOG.md').open('a') as f:
        f.write(f'\n{stamp}: {job["kernel"]} COMPLETE, bounded models/metadata/source/predictions recovered at{job["root"]}; terminal hashes/counts/config/split and oneT4 profiles verified; paired numeric proxy saved. Optimizer states stay remote. SelectedB012 unchanged.\n')


def checked_loop():
    stage=ROOT/'kaggle/loop-depth-20261002';tools=ROOT/'artifacts/kaggle/tools-loop-depth-20261002'
    manifest=json.loads((stage/'experiment-manifest.json').read_text())
    if manifest['source_sha256']!='1df6e945be2f18b959f1d81c36e68204e16caad098080be42ac59a19bff36b74':
        raise ValueError('Unreviewed loop source')
    if sha(stage/'run.py')!=manifest['driver_sha256'] or sha(tools/'training_source.bin')!=manifest['source_sha256']:
        raise ValueError('Loop stage changed')
    proof=json.loads((ROOT/'reports/evidence/loop_depth_upload_checks.json').read_text())
    if proof['version']!=1 or not all(proof['readback_equal'].values()):raise ValueError('Loop input readback missing')
    return {'name':'b017-b018','stage':str(stage.relative_to(ROOT)),
        'manifest':str((tools/'experiment-manifest.json').relative_to(ROOT)),
        'root':'artifacts/kaggle/b017-b018-v1','tools':str(tools.relative_to(ROOT)),
        'kernel':'arsen4ikvar/nlp-hw02-loop-depth-20261002/1'}


def prepare_resume(state):
    output=command([PYTHON,'scripts/build_kaggle_successor.py','resume',
        '--parent-terminal','artifacts/kaggle/b013-b014-v1/session_result.json'],timeout=120)
    snapshot(state,'resume_stage_frozen',summary=output.strip())
    tools=ROOT/'artifacts/kaggle/tools-polar-resume-20261002'
    dataset=json.loads((tools/'dataset-metadata.json').read_text())['id']
    command([KAGGLE,'datasets','create','--path',tools,'--keep-tabular','--quiet'],timeout=120)
    started=time.monotonic()
    while True:
        readiness=json.loads(command([KAGGLE,'datasets','status',dataset,'--format','json'],timeout=30))
        if readiness['status']=='ready':break
        if readiness['status'] in ('error','failed') or time.monotonic()-started>180:
            raise RuntimeError('Resume input did not become READY; no GPU push')
        time.sleep(10)
    readback=ROOT/'artifacts/kaggle/polar-resume-upload-check'
    if readback.exists():raise ValueError('Fresh resume readback directory required')
    command([KAGGLE,'datasets','download',dataset,'--path',readback,'--unzip','--quiet'],timeout=120)
    equal={name:(tools/name).read_bytes()==(readback/name).read_bytes() for name in
        ('training_source.bin','experiment-manifest.json','training_checksums.json','test.parquet')}
    if not all(equal.values()):raise ValueError('Resume private input readback differs')
    proof={'observed_utc':datetime.now(timezone.utc).isoformat(),'dataset':dataset,
        'version':readiness['current_version_number'],'status':'READY','readback_equal':equal,'GPU_launched':False}
    (ROOT/'reports/evidence/polar_resume_upload_checks.json').write_text(json.dumps(proof,indent=2)+'\n')
    return {'name':'b019-b020','stage':'kaggle/polar-resume-20261002',
        'manifest':'artifacts/kaggle/tools-polar-resume-20261002/experiment-manifest.json',
        'tools':str(tools.relative_to(ROOT)),'root':'artifacts/kaggle/b019-b020-v1',
        'kernel':'arsen4ikvar/nlp-hw02-polar-resume-20261002/1'}


def prepared_resume():
    """Reuse a completed private upload after inspection; never upload twice."""
    tools=ROOT/'artifacts/kaggle/tools-polar-resume-20261002'
    stage=ROOT/'kaggle/polar-resume-20261002'
    manifest=json.loads((tools/'experiment-manifest.json').read_text())
    proof=json.loads((ROOT/'reports/evidence/polar_resume_upload_checks.json').read_text())
    if proof['status']!='READY' or proof['version']!=1 or not all(proof['readback_equal'].values()):
        raise ValueError('Prepared resume private input proof missing')
    if sha(tools/'training_source.bin')!=manifest['source_sha256'] or sha(stage/'run.py')!=manifest['driver_sha256']:
        raise ValueError('Prepared resume source/driver changed')
    pin=json.loads((ROOT/'reports/evidence/polar_resume_image_pin.json').read_text())
    metadata=json.loads((stage/'kernel-metadata.json').read_text())
    if (metadata['docker_image']!=pin['docker_image'] or metadata['docker_image_pinning_type']!='original' or
        sha(stage/'kernel-metadata.json')!=pin['kernel_metadata_sha256_after']):
        raise ValueError('Prepared resume parent-image pin mismatch')
    if manifest['resume_parent']['terminal_sha256']!=sha(ROOT/'artifacts/kaggle/b013-b014-v1/session_result.json'):
        raise ValueError('Prepared resume parent changed')
    return {'name':'b019-b020','stage':str(stage.relative_to(ROOT)),
        'manifest':str((tools/'experiment-manifest.json').relative_to(ROOT)),
        'tools':str(tools.relative_to(ROOT)),'root':'artifacts/kaggle/b019-b020-v1',
        'kernel':'arsen4ikvar/nlp-hw02-polar-resume-20261002/1','requested_parent_image':pin['docker_image']}


def launch(job,state):
    # Only the main loop calls this after current known statuses show a free slot.
    if datetime.now(timezone.utc)>=DEADLINE:raise RuntimeError('Launch window ended; do not allocate')
    quota=json.loads(command([PYTHON,'scripts/kaggle_quota.py'],timeout=30))
    def seconds(value):return float(value.rstrip('s'))
    gpu=quota['gpuQuota']
    available=seconds(gpu['totalTimeAllowed'])-seconds(gpu['timeUsed'])-seconds(gpu['timeReserved'])
    if available<18000:raise RuntimeError('Insufficient unreserved quota for finite18000s successor; stop for user')
    manifest=json.loads((ROOT/job['manifest']).read_text())
    if sha(ROOT/job['stage']/'run.py')!=manifest['driver_sha256']:raise ValueError('Driver changed before push')
    if sha(ROOT/job['tools']/'training_source.bin')!=manifest['source_sha256']:raise ValueError('Source changed before push')
    snapshot(state,'launch_attempt_once',kernel=job['kernel'],quota=quota,source_sha256=manifest['source_sha256'])
    output=command([KAGGLE,'kernels','push','--path',job['stage'],'--timeout','18000','--accelerator','NvidiaTeslaT4'],timeout=120)
    slug=job['kernel'].rsplit('/',1)[0]
    if f'https://www.kaggle.com/code/{slug}' not in output or 'Kernel version 1 successfully pushed.' not in output:
        raise ValueError('Push returned unexpected slug/version; inspect, never retry')
    job['launched']=True;job['launch_utc']=datetime.now(timezone.utc).isoformat()
    (ROOT/'reports/evidence'/f'{job["name"].replace("-","_")}_launch.json').write_text(json.dumps(
        {'observed_utc':job['launch_utc'],'kernel':job['kernel'],'stdout':output,'quota_before':quota},indent=2)+'\n')
    # Do not interfere with the preserved followers; each new finite stream gets
    # its own unbuffered file and contains no fabricated heartbeat.
    follower=ROOT/'artifacts/kaggle'/f'{job["name"]}-follow.log'
    with follower.open('w') as stream:
        proc=subprocess.Popen([KAGGLE,'kernels','logs',slug,'--follow'],cwd=ROOT,
            env={**os.environ,'PYTHONUNBUFFERED':'1'},stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    job['follower_pid']=proc.pid
    snapshot(state,'successor_launched',kernel=job['kernel'],stdout=output.strip())
    with (ROOT/'experiments/LOG.md').open('a') as f:
        f.write(f'\n{job["launch_utc"]}: {job["kernel"]} pushed once by finite successor watcher after a GPU slot freed; private readback verified, platform18000s, no third kernel/retry. Admission and final quality pending; selectedB012 unchanged.\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true')
    parser.add_argument('--resume-state',type=Path);parser.add_argument('--use-prepared-resume',action='store_true')
    args=parser.parse_args()
    loop=checked_loop()
    if not args.run:
        print(json.dumps({'current':CURRENT,'next_loop':loop,'next_resume':'B01360k→120k NS5/PE5',
            'launch_deadline_UTC':DEADLINE.isoformat(),'launches_allowed':False}));return
    if WATCH.exists() and args.resume_state is None:
        raise ValueError('Watcher state already exists; inspect rather than duplicate scheduling')
    locked_files=['scripts/build_kaggle_successor.py','scripts/successor_driver_patch.py','scripts/kaggle_resume_inputs.py',
        'scripts/recover_kaggle_training.py','scripts/check_recovered_capacity.py',
        'src/nlp_hw02/model.py','src/nlp_hw02/train.py','src/nlp_hw02/resuming.py','src/nlp_hw02/optim.py',
        'src/nlp_hw02/optim_batched.py','src/nlp_hw02/optim_polar.py',
        'scripts/check_looped_model.py','scripts/check_training_resumption.py','scripts/check_polar_muon.py']
    state={'running':True,'pipeline_complete':False,'launch_deadline_UTC':DEADLINE.isoformat(),
        'source_snapshot':{name:sha(ROOT/name) for name in locked_files},'jobs':CURRENT.copy(),'next_loop':loop,
        'next_resume':None,'selected_bundle_changed':False,'submissions':False}
    loop_launched=resume_launched=False
    if args.resume_state is not None:
        prior=json.loads(args.resume_state.read_text())
        if prior['running'] or prior['pipeline_complete']:raise ValueError('Only an inspected stopped watcher may resume')
        if prior['source_snapshot']!=state['source_snapshot']:raise ValueError('Locked training/protocol source changed')
        archive=ROOT/'reports/evidence'/('successor_watch_before_resume_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'.json')
        archive.write_bytes(args.resume_state.read_bytes())
        state=prior;state['running']=True;state['previous_stop']={'archive':str(archive.relative_to(ROOT)),
            'failure':state.pop('failure',None),'failure_type':state.pop('failure_type',None)}
        loop_launched=any(j['name']=='b017-b018' for j in state['jobs'])
        resume_launched=any(j['name']=='b019-b020' for j in state['jobs'])
        if args.use_prepared_resume:
            if resume_launched:raise ValueError('Resume runtime already launched; do not launch again')
            state['next_resume']=prepared_resume()
    try:
        snapshot(state,'watch_started',policy='At mosttwo useful kernels, explicit finite successors only; stop on failure, never cancel current work.')
        refresh_handoff(state)
        while True:
            for name,digest in state['source_snapshot'].items():
                if sha(ROOT/name)!=digest:raise ValueError('Prepared code changed during watcher: '+name)
            jobs=state['jobs']
            with ThreadPoolExecutor(max_workers=4) as pool:statuses=list(pool.map(status,jobs))
            for job,value in zip(jobs,statuses):job['status']=value
            snapshot(state,'statuses',statuses={j['kernel']:j['status'] for j in jobs})
            if any(value not in ('RUNNING','QUEUED','COMPLETE') for value in statuses):
                raise RuntimeError('A kernel failed/cancelled; stop further launches and inspect retained diagnostics')
            active=sum(value in ('RUNNING','QUEUED') for value in statuses)
            if active>2:raise RuntimeError('More than two known active GPU kernels; inspect, do not cancel')
            # Start the already-ready loop before lengthy recovery, using the
            # newly released slot while the other healthy session progresses.
            if not loop_launched and active<2 and any(value=='COMPLETE' for value in statuses) and datetime.now(timezone.utc)<DEADLINE:
                launch(loop,state);jobs.append(loop);loop_launched=True;active+=1
                refresh_handoff(state)
            for job in list(jobs):
                if job.get('status')=='COMPLETE' and not job.get('recovered'):recover(job,state)
            parent=next(j for j in jobs if j['name']=='b013-b014')
            if parent.get('recovered') and state['next_resume'] is None and datetime.now(timezone.utc)<DEADLINE:
                state['next_resume']=prepare_resume(state)
                snapshot(state,'resume_ready',job=state['next_resume'])
            if not resume_launched and state['next_resume'] is not None and active<2 and datetime.now(timezone.utc)<DEADLINE:
                launch(state['next_resume'],state);jobs.append(state['next_resume']);resume_launched=True
                refresh_handoff(state)
            command([str(ROOT/'.venv/bin/python'),'scripts/docs_inventory.py','--write','--check'],timeout=30)
            if loop_launched and resume_launched and all(j.get('recovered') for j in jobs):
                state['pipeline_complete']=True;break
            if datetime.now(timezone.utc)>=DEADLINE and all(j.get('recovered') for j in jobs):break
            time.sleep(60)
    except BaseException as exc:
        state['failure_type']=type(exc).__name__;state['failure']=str(exc)
        snapshot(state,'watch_stopped_requires_inspection',reason=str(exc),
            policy='No retry/cancellation; healthy existing kernels remain running. Report blocker to user at next interaction.')
        raise
    finally:
        state['running']=False
        snapshot(state,'watch_finished',pipeline_complete=state['pipeline_complete'])
        refresh_handoff(state)
        command([str(ROOT/'.venv/bin/python'),'scripts/docs_inventory.py','--write','--check'],timeout=30)


if __name__=='__main__':main()
