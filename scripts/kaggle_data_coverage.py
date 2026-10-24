"""Finite T4×2 experiment batch; executed as a private Kaggle script."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

STARTED=time.monotonic()
WORKING=Path('/kaggle/working')
TEMP=Path('/kaggle/temp/nlp-hw02'); TEMP.mkdir(parents=True,exist_ok=True)
LANES=[]
RUN_STATE={'complete':False,'phase':'startup'}
TAIL_RESERVE_SECONDS=1200
OUTER_SECONDS=9000
INFERENCE_BOUND_SECONDS=300
CHECKPOINT_GRACE_SECONDS=90
SHUTDOWN_SECONDS=30
NEAR_FINISH_SECONDS=180


def emit(record):
    record={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'session_seconds':time.monotonic()-STARTED,**record}
    with (WORKING/'session_events.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)


def unpack(path,dest):
    dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
        z.extractall(dest)


def gpu_sample():
    try:
        usage=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    except (subprocess.TimeoutExpired,OSError) as error:
        emit({'event':'gpu_telemetry_unavailable','error':type(error).__name__})
        return
    if usage.returncode:
        emit({'event':'gpu_telemetry_unavailable','returncode':usage.returncode})
        return
    emit({'event':'gpu_sample','sample':usage.stdout.strip()})


def evaluate(name,command,log,env,required):
    from kaggle_eval import run_evaluation
    return run_evaluation(name,command,log,env,OUTER_SECONDS-(time.monotonic()-STARTED),
        INFERENCE_BOUND_SECONDS,emit,required=required,sample=gpu_sample)


def phase_test(source,cache,tools,runs):
    # Both trainers have exited: no training contention during one-T4 timing.
    code='''import json,pathlib,sys,torch,pyarrow.parquet as pq
from tokenizers import Tokenizer
from nlp_hw02.model import load_model
from nlp_hw02.inference import generate_questions,syntax_valid
torch.set_num_threads(2)
tools=pathlib.Path(sys.argv[1]); run=pathlib.Path(sys.argv[2]); out=pathlib.Path(sys.argv[3]); out.mkdir(exist_ok=True)
questions=pq.read_table(tools/'test.parquet',columns=['question'])['question'].to_pylist()
model,checkpoint=load_model(run/'model.pt'); tok=Tokenizer.from_file(str(run/'tokenizer.json'))
warm=json.loads((run/'validation.json').read_text())[0]['question']; generate_questions(model,tok,[warm],batch_size=256)
preds,timing=generate_questions(model,tok,questions,batch_size=256)
assert len(preds)==1919 and torch.cuda.device_count()==1 and 'T4' in torch.cuda.get_device_name(0)
timing.update({'gpu_count':torch.cuda.device_count(),'gpu':torch.cuda.get_device_name(0),'selected_step':checkpoint['step'],
              'syntax_valid_fraction':sum(map(syntax_valid,preds))/len(preds),'meets_180_seconds_on_one_T4':timing['seconds']<=180})
(out/'test_timing.json').write_text(json.dumps(timing,indent=2))
with (out/'test_predictions.jsonl').open('w') as f:
 for i,p in enumerate(preds): f.write(json.dumps({'row_index':i,'code':p},ensure_ascii=False)+'\\n')
print(json.dumps(timing)); print('ONE_T4_INFERENCE_OK')
'''
    script=TEMP/'test.py'; script.write_text(code)
    baseline=TEMP/'baseline'
    entries=list(runs)
    if RUN_STATE.get('include_b001_timing',True):
        unpack(tools/'baseline_bundle.bin',baseline)
        entries=[('b001-256',baseline),*entries]
    completed=[]; deferred=[]
    for name,run in entries:
        out=WORKING/f'inference-{name}'
        result=evaluate(name,[sys.executable,'-u',str(script),str(tools),str(run),str(out)],
            WORKING/f'inference-{name}.log',{'PYTHONPATH':str(source/'src'),'CUDA_VISIBLE_DEVICES':'0'},
            [out/'test_timing.json',out/'test_predictions.jsonl'])
        if result['status']=='deferred': deferred.append(name); continue
        emit({'event':'inference','run':name,**json.loads((out/'test_timing.json').read_text())})
        completed.append(name)
        if name=='b001-256':
            old=[json.loads(line)['code'] for line in (baseline/'test_predictions.jsonl').read_text().splitlines()]
            new=[json.loads(line)['code'] for line in (out/'test_predictions.jsonl').read_text().splitlines()]
            emit({'event':'batch_numerical_comparison','changed_rows_vs_old_colab128':sum(a!=b for a,b in zip(old,new)),
                  'note':'Batch and runtime software differ; no semantic or isolated batch-effect claim.'})
    return completed,deferred


def phase_extra(source,tools,runs,experiment):
    completed=[];deferred=[]
    env={'PYTHONPATH':str(source/'src'),'CUDA_VISIBLE_DEVICES':'0',
         'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'}
    if experiment.get('profile_decoding'):
        for name,run in runs:
            label=f'decoding-{name}';out=WORKING/label
            result=evaluate(label,[sys.executable,'-u',str(source/'scripts/profile_decoding_gpu.py'),
                '--run',str(run),'--output',str(out),'--test-parquet',str(tools/'test.parquet'),'--skip-synthetic-check'],
                WORKING/f'{label}.log',env,[out/'decoding_profile.json',out/'beam_test_timing.json',out/'beam_test_predictions.jsonl'])
            (deferred if result['status']=='deferred' else completed).append(label)
    if experiment.get('canonical_bundle_timing'):
        label='canonical-b008';out=WORKING/label;notebook=tools/'canonical_inference.ipynb'
        assert hashlib.sha256(notebook.read_bytes()).hexdigest()==experiment['canonical_notebook_sha256']
        assert hashlib.sha256((tools/'selected_bundle.bin').read_bytes()).hexdigest()==experiment['canonical_bundle_sha256']
        result=evaluate(label,[sys.executable,'-u',str(source/'scripts/execute_inference_notebook.py'),
            '--notebook',str(notebook),'--output',str(out/'executed.ipynb')],
            WORKING/f'{label}.log',env,[out/'executed.ipynb',out/'executed.json',WORKING/'canonical-inference/test_timing.json',
                                      WORKING/'canonical-inference/test_predictions.jsonl'])
        if result['status']=='deferred':deferred.append(label)
        else:
            evidence=json.loads((out/'executed.json').read_text())
            assert evidence['complete'] and evidence['executed_code_cells']==3
            assert evidence['input_sha256']==experiment['canonical_notebook_sha256']
            timing=json.loads((WORKING/'canonical-inference/test_timing.json').read_text())
            assert timing['bundle_sha256']==experiment['canonical_bundle_sha256']
            assert timing['rows']==1919 and timing['gpu_count']==1 and timing['decoder']=='beam'
            assert timing['meets_180_seconds_on_one_T4'] and timing['seconds']<=180
            completed.append(label);emit({'event':'canonical_notebook_verified',**timing})
    return completed,deferred


def main():
    global OUTER_SECONDS,TAIL_RESERVE_SECONDS
    # Parent does not own a CUDA context: only fixed-device subprocesses do.
    tools_files=[p for p in Path('/kaggle/input').rglob('training_checksums.json') if all((p.parent/n).is_file() for n in ('training_source.bin','data-transition.json','experiment-manifest.json'))]
    assert len(tools_files)==1
    tools=tools_files[0].parent
    experiment=json.loads((tools/'experiment-manifest.json').read_text())
    outer_seconds=experiment.get('outer_timeout_seconds',9000)
    assert 1800<=outer_seconds<=18000
    RUN_STATE['include_b001_timing']=experiment.get('include_b001_timing',True)
    specifications=experiment['lanes']
    assert len(specifications)==2 and {l['device'] for l in specifications}=={0,1}
    import re
    assert len({l['id'].lower() for l in specifications})==2
    assert all(re.fullmatch(r'b[0-9]+',l['id'].lower()) and 1000<=l['iterations']<=500000 for l in specifications)
    OUTER_SECONDS=outer_seconds
    hashes=json.loads(tools_files[0].read_text())
    assert experiment['source_sha256']==hashes['training_source.bin']
    assert experiment['driver_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    import shutil
    shutil.copyfile(tools/'experiment-manifest.json',WORKING/'experiment-manifest.json')
    shutil.copyfile(Path(__file__),WORKING/'run_driver.py')
    RUN_STATE.update({'driver_sha256':experiment['driver_sha256'],
        'experiment_manifest_sha256':hashlib.sha256((tools/'experiment-manifest.json').read_bytes()).hexdigest(),
        'expected_iterations':{l['id'].lower():l['iterations'] for l in specifications}})
    for name,digest in hashes.items(): assert hashlib.sha256((tools/name).read_bytes()).hexdigest()==digest,name
    source=TEMP/'source'; unpack(tools/'training_source.bin',source)
    sys.path.insert(0,str(source/'scripts'))
    sys.path.insert(0,str(source/'src'))
    from kaggle_eval import evaluation_count,completed_run,remaining_seconds,training_observation
    from nlp_hw02.progress import ProgressObserver
    inference_count=evaluation_count(experiment)
    required_tail=inference_count*INFERENCE_BOUND_SECONDS+CHECKPOINT_GRACE_SECONDS+SHUTDOWN_SECONDS
    TAIL_RESERVE_SECONDS=required_tail+NEAR_FINISH_SECONDS
    emit({'event':'evaluation_reserve','planned_evaluations':inference_count,'tail_reserve_seconds':TAIL_RESERVE_SECONDS})
    import importlib.util
    assert all(importlib.util.find_spec(m) is not None for m in ('nbformat','tokenizers','pyarrow')), 'Missing preinstalled dependencies'

    all_manifests=list(Path('/kaggle/input').rglob('cache_manifest.json'))
    caches={}
    for name,expected in experiment['cache_manifests_sha256'].items():
        candidates=[p for p in all_manifests if hashlib.sha256(p.read_bytes()).hexdigest()==expected]
        assert len(candidates)==1,'Exactly one content-pinned cache for '+name
        caches[name]=candidates[0].parent
    cache=caches['b033'];new_cache=caches['b034'];manifests=[cache/'cache_manifest.json']
    cache_digest=hashlib.sha256(manifests[0].read_bytes()).hexdigest()
    assert cache_digest==experiment['cache_manifest_sha256']
    RUN_STATE.update({'training_source_sha256':hashes['training_source.bin'],
        'cache_manifest_sha256':cache_digest,'cache_manifests_sha256':experiment['cache_manifests_sha256'],
        'cpu_kernel_version':experiment['cpu_kernel_version'],'cpu_kernel_versions':experiment['cpu_kernel_versions'],
        'controlled_data_transition':experiment['data_transition']})
    import numpy as np
    for name,root in caches.items():
        cm=json.loads((root/'cache_manifest.json').read_text());assert cm['complete']
        rows=2000000 if name=='b033' else 6000000
        for filename,expected in cm['checksums'].items():
            if filename.endswith('.json'):assert hashlib.sha256((root/filename).read_bytes()).hexdigest()==expected,filename
        for filename,width,dtype in [('train_source.npy',256,np.uint16),('train_target.npy',192,np.uint16),('train_row_ids.npy',2,np.uint32)]:
            a=np.load(root/filename,mmap_mode='r');assert a.shape==(rows,width) and a.dtype==dtype;del a
    verified=subprocess.run([sys.executable,str(source/'scripts/verify_training_caches.py'),
        '--old',str(cache),'--new',str(new_cache),'--manifest',str(tools/'experiment-manifest.json')],
        capture_output=True,text=True,check=True,timeout=150)
    emit(json.loads(verified.stdout))
    from nlp_hw02.training_arrays import available_memory_bytes,array_resident_bytes
    sizes={n:array_resident_bytes(p) for n,p in caches.items()};available=available_memory_bytes()
    required=2*sum(sizes.values())+2*1024**3
    assert available>=required,'Insufficient RAM for both resident caches/pagecache/headroom'
    emit({'event':'resident_arrays_admission','array_bytes_by_lane':sizes,'available_bytes':available,
          'required_bytes':required,'reserved_copies':2,'note':'Two different immutable caches; arrays stay remote.'})
    env={**os.environ,'PYTHONPATH':str(source/'src'),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
         'TOKENIZERS_PARALLELISM':'false'}
    hardware=subprocess.run([sys.executable,'-c',
        "import json,torch; assert torch.cuda.device_count()==2; names=[torch.cuda.get_device_name(i) for i in range(2)]; assert all('T4' in n for n in names); [torch.ones(64,64,device=f'cuda:{i}').sum().item() for i in range(2)]; print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpus':names}))"],
        capture_output=True,text=True,env=env,check=True,timeout=90)
    emit({'event':'hardware_preflight','hardware':json.loads(hardware.stdout)})
    selected_muon=None
    parent_directory=None
    from kaggle_resume_inputs import verified_parent
    parent_directory=verified_parent('/kaggle/input',experiment['resume_parent'],cache_digest,json.loads(hardware.stdout))
    emit({'event':'resume_parent_verified','parent':experiment['resume_parent'],'seconds':time.monotonic()-STARTED})
    shutil.copyfile(tools/'data-transition.json',WORKING/'data-transition.json')
    from nlp_hw02.data_transition import validate_transition
    _,transition_record=validate_transition(tools/'data-transition.json',parent_directory,new_cache)
    emit({'event':'controlled_data_transition_verified',**transition_record})
    if time.monotonic()-STARTED>240: raise RuntimeError('Successor startup exceeded240s; abort before training')
    lanes=LANES
    RUN_STATE['phase']='training'
    for spec in specifications:
        device=spec['device']; name=spec['id'].lower(); iterations=spec['iterations']
        config=spec.get('model_config',{'exclusive_self_attention':spec.get('variant')=='xsa'})
        optimizer=spec.get('optimizer','adamw')
        if optimizer=='muon_profile_selected':
            assert selected_muon is not None,'Predeclared Muon timing selector requires profile'
            optimizer=selected_muon
        lane_cache=caches[name]
        assert spec['optimizer']=='muon_grouped' and not spec['allow_orthogonalizer_change']
        run=WORKING/f'run-{name}'
        code="from nlp_hw02.train import train; from nlp_hw02.model import ModelConfig; "
        code+=f"train({str(run)!r},cfg=ModelConfig(**{config!r}),optimizer_kind={optimizer!r},seed={spec['seed']!r},data_dir={str(lane_cache)!r},max_steps={iterations!r},seconds=1_000_000,resident_data={experiment.get('resident_training_arrays',False)!r}); print('TRAINING_OK')"
        resumed_step=spec.get('resume_from_step',0)
        assert parent_directory is not None and resumed_step==experiment['resume_parent']['step']
        assert iterations-resumed_step>=1000
        assert resumed_step==240000 and iterations==300000
        transition_path=str(tools/'data-transition.json') if name=='b034' else None
        suffix=f",resume_dir={str(parent_directory)!r},allow_orthogonalizer_change=False,data_transition_manifest={transition_path!r},expected_initial_validation={experiment['parent_last_validation_loss']!r}"
        code=code.replace("resident_data=True); print('TRAINING_OK')","resident_data=True"+suffix+"); print('TRAINING_OK')")
        assert 'resume_dir=' in code
        log_path=WORKING/f'{name}.log'; stream=log_path.open('w')
        process=subprocess.Popen([sys.executable,'-u','-c',code],env={**env,'CUDA_VISIBLE_DEVICES':str(device)},
                                 stdout=stream,stderr=subprocess.STDOUT)
        lanes.append({'name':name,'device':device,'run':run,'log':log_path,'stream':stream,'proc':process,
                      'last_progress':time.monotonic(),'observer':ProgressObserver(time.monotonic()),'step':0,'eta_seconds':None,'terminated':False,
                      'failure_logged':False,'admission_checked':False,'iterations':iterations,'measured_rate':None,
                      'last_validation_forwarded':-1,'last_timing_forwarded':0,'last_train_timing_forwarded':0,
                      'resumed_step':resumed_step,'admission_step':resumed_step+1000,'diagnostics_forwarded':{},
                      'array_loading_forwarded':False,'exit_observed_seconds':None})
        emit({'event':'lane_started','run':name,'device':device,'pid':process.pid,'iterations':iterations})
    try:
        last_sample=0; last_announcement=0
        while any(l['proc'].poll() is None for l in lanes):
            now=time.monotonic()
            for lane in lanes:
                if lane['proc'].poll() is not None:
                    if lane['exit_observed_seconds'] is None:
                        lane['exit_observed_seconds']=now-STARTED
                        emit({'event':'lane_training_ended','run':lane['name'],'exit':lane['proc'].returncode})
                    if lane['proc'].returncode and not lane['failure_logged']:
                        emit({'event':'lane_failure','run':lane['name'],'exit':lane['proc'].returncode,
                              'policy':'Keep the already admitted useful sibling; no failed-lane retry or new training jobs.'})
                        lane['failure_logged']=True
                    continue
                heartbeat=lane['observer'].observe(lane['run']/'progress.json',now)
                if heartbeat is not None:
                    lane['last_progress']=lane['observer'].last_progress
                    lane['heartbeat_step']=heartbeat['step'];lane['heartbeat_phase']=heartbeat['phase']
                events=lane['run']/'train.jsonl'
                if events.exists():
                    records=[]
                    for line in events.read_text().splitlines()[-20:]:
                        try: records.append(json.loads(line))
                        except json.JSONDecodeError: pass
                    for record in records:
                        step=record.get('step',0)
                        if step>lane['step']:
                            lane['step']=step
                        if record.get('event')=='train':
                            rate=record['steps_per_second']
                            lane['measured_rate']=rate
                        if record.get('event')=='array_loading' and not lane['array_loading_forwarded']:
                            emit({'event':'lane_array_loading','run':lane['name'],**{k:v for k,v in record.items() if k not in ('event','utc')}})
                            lane['array_loading_forwarded']=True
                        if record.get('event')=='step_timing' and step>lane['last_timing_forwarded']:
                            emit({'event':'lane_step_timing','run':lane['name'],**{k:v for k,v in record.items() if k not in ('event','utc')}})
                            lane['last_timing_forwarded']=step
                        if record.get('event')=='train' and step>lane['last_train_timing_forwarded'] and (step<=lane['resumed_step']+300 or step%1000==0):
                            emit({'event':'lane_training_timing','run':lane['name'],**{k:v for k,v in record.items() if k not in ('event','utc')}})
                            lane['last_train_timing_forwarded']=step
                        diagnostic=record.get('event')
                        if diagnostic in ('xsa_gates','loop_norms','update_rms','parent_initial_validation_verified') and step>lane['diagnostics_forwarded'].get(diagnostic,-1):
                            emit({'event':'lane_'+diagnostic,'run':lane['name'],**{k:v for k,v in record.items() if k not in ('event','utc')}})
                            lane['diagnostics_forwarded'][diagnostic]=step
                        if record.get('event') in ('initial_validation','validation') and step>lane['last_validation_forwarded']:
                            emit({'event':'lane_validation','run':lane['name'],'step':step,'loss':record['loss'],
                                  'legacy_loss':record.get('legacy_validation_loss'),
                                  'optimizer_updates':record.get('optimizer_updates'),'skipped_updates':record.get('skipped_updates')})
                            lane['last_validation_forwarded']=step
                    recent=[r for r in records if r.get('event')=='train'][-6:]
                    if len(recent)>=2:
                        rate=(recent[-1]['step']-recent[0]['step'])/(recent[-1]['elapsed_seconds']-recent[0]['elapsed_seconds'])
                        lane['measured_rate']=rate
                lane['eta_seconds']=remaining_seconds(lane['iterations'],lane['measured_rate'],lane['step'],lane.get('heartbeat_step'))
                phase=lane.get('heartbeat_phase','startup')
                stall_bound=180 if phase in ('startup','initializing','load_source','load_target','model_initialization','initial_validation','validation','final_validation') else 120
                stalled=lane['observer'].stalled(now,timeout=stall_bound)
                near_platform_cap=now-STARTED>outer_seconds-TAIL_RESERVE_SECONDS
                eta=lane['eta_seconds']
                early_admission_failure=False
                if lane['step']>=lane['admission_step'] and eta is not None and not lane['admission_checked']:
                    early_admission_failure=eta+TAIL_RESERVE_SECONDS>outer_seconds-(now-STARTED)
                    lane['admission_checked']=True
                    emit({'event':'budget_admission','run':lane['name'],'new_iterations_observed':lane['step']-lane['resumed_step'],'estimated_remaining_seconds':eta,
                          'tail_reserve_seconds':TAIL_RESERVE_SECONDS,'admitted':not early_admission_failure})
                # Derive reservation from the actual finite evaluation queue.
                # Prioritize a healthy near-finished trainer over optional
                # inference: retain its final checkpoint and defer evaluation.
                remaining=outer_seconds-(now-STARTED)
                near_finish_can_checkpoint=(eta is not None and eta<=NEAR_FINISH_SECONDS and
                    eta+CHECKPOINT_GRACE_SECONDS+SHUTDOWN_SECONDS<remaining)
                cannot_finish=near_platform_cap and not near_finish_can_checkpoint and (
                    eta is None or eta+required_tail>remaining)
                if (stalled or cannot_finish or early_admission_failure) and not lane['terminated']:
                    emit({'event':'graceful_stop','run':lane['name'],'stalled':stalled,
                          'early_admission_failure':early_admission_failure,'estimated_remaining_seconds':eta,
                          'heartbeat_phase':phase,'heartbeat_step':lane.get('heartbeat_step'),'stall_bound_seconds':stall_bound})
                    lane['proc'].send_signal(signal.SIGTERM); lane['terminated']=True; lane['stop_time']=now
                if lane['terminated'] and now-lane['stop_time']>90:
                    lane['proc'].kill(); emit({'event':'stuck_child_killed','run':lane['name']})
            if now-last_sample>=15:
                gpu_sample()
                last_sample=now
            if now-last_announcement>=60:
                emit({'event':'progress','lanes':[{'run':l['name'],'step':l['step'],
                     'eta_seconds':l['eta_seconds'],'exit':l['proc'].poll(),
                     'heartbeat_step':l.get('heartbeat_step'),'phase':l.get('heartbeat_phase')} for l in lanes]})
                last_announcement=now
            time.sleep(5)
    finally:
        for lane in lanes:
            if lane['proc'].poll() is None: lane['proc'].terminate()
            try: lane['proc'].wait(timeout=90)
            except subprocess.TimeoutExpired: lane['proc'].kill(); lane['proc'].wait()
            if lane['exit_observed_seconds'] is None: lane['exit_observed_seconds']=time.monotonic()-STARTED
            lane['stream'].close()
            emit({'event':'lane_exit','run':lane['name'],'exit':lane['proc'].returncode})
    successful=[(l['name'],l['run']) for l in lanes if completed_run(l['run'],l['iterations'],l['proc'].returncode,l['terminated'])]
    incomplete=[l['name'] for l in lanes if l['name'] not in {n for n,_ in successful}]
    if incomplete:emit({'event':'incomplete_training_evaluation_skipped','runs':incomplete,'reason':'Only complete nonterminated admitted training may enter optional model evaluations.'})
    observations={l['name']:training_observation(l['run'],l['iterations']) for l in lanes}
    for lane in lanes:
        metrics_path=lane['run']/'metrics.json'
        if metrics_path.exists():
            metrics=json.loads(metrics_path.read_text())
            assert metrics['resumed_from_step']==lane['resumed_step']
            assert metrics['new_iterations']==metrics['completed_steps']-lane['resumed_step']
    iterations={name:record['steps'] for name,record in observations.items()}
    expected_iterations={l['name']:l['iterations'] for l in lanes}
    result={'complete':len(successful)==2 and iterations==expected_iterations,'iterations':iterations,
            'training_source_sha256':hashes['training_source.bin'],
            'cache_manifest_sha256':cache_digest,'cache_manifests_sha256':experiment['cache_manifests_sha256'],
            'controlled_data_transition':experiment['data_transition'],
            'session_script_seconds':time.monotonic()-STARTED,'iteration_count_evidence':observations,
            'training_lane_idle_tail_seconds':{l['name']:max(0,time.monotonic()-STARTED-l['exit_observed_seconds']) for l in lanes},
            'lanes':[{'run':l['name'],'exit':l['proc'].returncode,'gracefully_stopped':l['terminated']} for l in lanes]}
    RUN_STATE.update({**result,'complete':False,'training_complete':result['complete'],'phase':'inference'})
    import shutil
    shutil.copyfile(tools/'training_source.bin',WORKING/'training_source.bin')
    (WORKING/'session_result.json').write_text(json.dumps(RUN_STATE,indent=2))
    evaluated,deferred=phase_test(source,cache,tools,successful)
    extras,extra_deferred=phase_extra(source,tools,successful,experiment)
    deferred.extend(extra_deferred);evaluated.extend(extras)
    result['complete']=result['complete'] and not deferred
    RUN_STATE.update({**result,'phase':'finished','inference_complete':not deferred and not incomplete,
                      'incomplete_training_evaluation_skipped':incomplete,
                      'evaluated_runs':evaluated,'deferred_inference':deferred,
                      'training_lane_idle_tail_seconds':{l['name']:max(0,time.monotonic()-STARTED-l['exit_observed_seconds']) for l in lanes}})
    (WORKING/'session_result.json').write_text(json.dumps(RUN_STATE,indent=2))
    emit({'event':'finished',**result})
    if not RUN_STATE['training_complete']: raise RuntimeError('Some lanes failed; inspect retained diagnostics')
    print('KAGGLE_DUAL_RUN_OK' if not deferred else 'KAGGLE_TRAINING_OK_INFERENCE_DEFERRED',flush=True)


if __name__=='__main__':
    try: main()
    except BaseException as exc:
        RUN_STATE.update({'complete':False,'failure_type':type(exc).__name__,'failure':str(exc)})
        raise
    finally:
        # Cover failures during lane launch as well as inside the monitor.
        for lane in LANES:
            if lane['proc'].poll() is None:
                lane['terminated']=True
                lane['proc'].terminate()
                try: lane['proc'].wait(timeout=90)
                except subprocess.TimeoutExpired: lane['proc'].kill(); lane['proc'].wait()
            if lane['exit_observed_seconds'] is None:lane['exit_observed_seconds']=time.monotonic()-STARTED
            if not lane['stream'].closed: lane['stream'].close()
        RUN_STATE['session_script_seconds']=time.monotonic()-STARTED
        if LANES:
            from kaggle_eval import training_observation
        observations={l['name']:training_observation(l['run'],l['iterations']) for l in LANES}
        RUN_STATE['iterations']={name:r['steps'] for name,r in observations.items()}
        RUN_STATE['iteration_count_evidence']=observations
        RUN_STATE['training_lane_idle_tail_seconds']={l['name']:(max(0,time.monotonic()-STARTED-l['exit_observed_seconds']) if l['exit_observed_seconds'] is not None else None) for l in LANES}
        RUN_STATE['lanes']=[{'run':l['name'],'exit':l['proc'].returncode,'gracefully_stopped':l['terminated']} for l in LANES]
        paths=[WORKING/'training_source.bin',WORKING/'session_events.jsonl',WORKING/'muon_profile.json',
               WORKING/'experiment-manifest.json',WORKING/'run_driver.py',WORKING/'data-transition.json']
        for lane in LANES:
            paths.append(lane['log'])
            paths.extend(p for p in lane['run'].glob('*') if p.suffix in ['.pt','.json','.jsonl'])
        paths.extend(p for p in WORKING.glob('inference-*/*') if p.suffix in ['.json','.jsonl'])
        for pattern in ('decoding-*/*','decoding-*/*/*','canonical-*/*'):
            paths.extend(p for p in WORKING.glob(pattern) if p.suffix in ['.json','.jsonl','.ipynb'])
        paths.extend(p for p in WORKING.glob('*.log') if p.is_file())
        digests={}
        for path in paths:
            if not path.is_file(): continue
            h=hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
            digests[str(path.relative_to(WORKING))]=h.hexdigest()
        RUN_STATE['artifact_sha256']=digests
        (WORKING/'session_result.json').write_text(json.dumps(RUN_STATE,indent=2))
