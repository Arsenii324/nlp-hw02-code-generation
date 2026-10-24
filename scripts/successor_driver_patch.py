"""Small explicit changes to the preserved, measured finite two-lane driver."""


def once(body,old,new):
    if body.count(old)!=1:raise ValueError('Unexpected preserved driver marker: '+old[:100])
    return body.replace(old,new)


def patched_driver(body,mode):
    if mode=='xsa-polar-resume':
        # Optional utilization telemetry must not destroy healthy training peers.
        old="""def gpu_sample():
    usage=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    emit({'event':'gpu_sample','sample':usage.stdout.strip()})"""
        new="""def gpu_sample():
    try:
        usage=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    except (subprocess.TimeoutExpired,OSError) as error:
        emit({'event':'gpu_telemetry_unavailable','error':type(error).__name__})
        return
    if usage.returncode:
        emit({'event':'gpu_telemetry_unavailable','returncode':usage.returncode})
        return
    emit({'event':'gpu_sample','sample':usage.stdout.strip()})"""
        return once(patched_driver(body,'resume'),old,new)
    if mode=='long-resume':
        # Preserve the reviewed independent-parent path; change only its step fence.
        return once(patched_driver(body,'xsa-resume'),
            "assert resumed_step==experiment['resume_parents'][name]['step']==60000",
            "assert resumed_step==experiment['resume_parents'][name]['step']==120000")
    if mode not in ('loop','resume','adjacent','xsa-resume'):raise ValueError(mode)
    check={'loop':'check_looped_model.py','resume':'check_training_resumption.py',
           'adjacent':'check_adjacent_sharing.py','xsa-resume':'check_xsa_resumption.py'}[mode]
    marker="    subprocess.run([sys.executable,str(source/'scripts/check_baseline.py')],env=env,check=True,timeout=90)"
    extra=f"\n    subprocess.run([sys.executable,str(source/'scripts/{check}'),'--cuda'],env={{**env,'CUDA_VISIBLE_DEVICES':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8'}},check=True,timeout=90)"
    if mode=='resume':
        extra+="\n    subprocess.run([sys.executable,str(source/'scripts/check_polar_muon.py'),'--cuda'],env={**env,'CUDA_VISIBLE_DEVICES':'0'},check=True,timeout=60)"
    body=once(body,marker,marker+extra)
    marker="    if time.monotonic()-STARTED>180: raise RuntimeError('Startup exceeded180s; abort before training')"
    # No additional allocation: finite preflight inside this session. This bound
    # includes CPU/CUDA restart+polar probes; ordinary startup stays well below it.
    replacement="    parent_directory=None\n"
    if mode=='resume':
        replacement+="    from kaggle_resume_inputs import verified_parent\n    parent_directory=verified_parent('/kaggle/input',experiment['resume_parent'],cache_digest,json.loads(hardware.stdout))\n    emit({'event':'resume_parent_verified','parent':experiment['resume_parent'],'seconds':time.monotonic()-STARTED})\n"
    elif mode=='xsa-resume':
        replacement+="    from kaggle_resume_inputs import verified_parent\n    parent_directories={}\n    for lane_name,pins in experiment['resume_parents'].items():\n        parent_directories[lane_name]=verified_parent('/kaggle/input',pins,cache_digest,json.loads(hardware.stdout))\n        emit({'event':'resume_parent_verified','run':lane_name,'parent':pins,'seconds':time.monotonic()-STARTED})\n"
    replacement+="    if time.monotonic()-STARTED>240: raise RuntimeError('Successor startup exceeded240s; abort before training')"
    body=once(body,marker,replacement)
    marker="        log_path=WORKING/f'{name}.log'; stream=log_path.open('w')"
    replacement="        resumed_step=spec.get('resume_from_step',0)\n"
    if mode=='resume':
        replacement+="        assert parent_directory is not None and resumed_step==experiment['resume_parent']['step']\n        assert iterations-resumed_step>=1000\n        suffix=f\",resume_dir={str(parent_directory)!r},allow_orthogonalizer_change={spec.get('allow_orthogonalizer_change',False)!r}\"\n        code=code.replace(\"resident_data=True); print('TRAINING_OK')\",\"resident_data=True\"+suffix+\"); print('TRAINING_OK')\")\n        assert 'resume_dir=' in code\n"
    elif mode=='xsa-resume':
        replacement+="        parent_directory=parent_directories[name]\n        assert resumed_step==experiment['resume_parents'][name]['step']==60000\n        assert iterations-resumed_step>=1000 and not spec.get('allow_orthogonalizer_change',False)\n        suffix=f\",resume_dir={str(parent_directory)!r},allow_orthogonalizer_change=False\"\n        code=code.replace(\"resident_data=True); print('TRAINING_OK')\",\"resident_data=True\"+suffix+\"); print('TRAINING_OK')\")\n        assert 'resume_dir=' in code\n"
    else:replacement+="        assert resumed_step==0\n"
    body=once(body,marker,replacement+marker)
    marker="                      'array_loading_forwarded':False,'exit_observed_seconds':None})"
    body=once(body,marker,"                      'resumed_step':resumed_step,'admission_step':resumed_step+1000,'diagnostics_forwarded':{},\n"+marker)
    marker="                        if record.get('event') in ('initial_validation','validation') and step>lane['last_validation_forwarded']:"
    replacement="                        diagnostic=record.get('event')\n                        if diagnostic in ('xsa_gates','loop_norms','update_rms') and step>lane['diagnostics_forwarded'].get(diagnostic,-1):\n                            emit({'event':'lane_'+diagnostic,'run':lane['name'],**{k:v for k,v in record.items() if k not in ('event','utc')}})\n                            lane['diagnostics_forwarded'][diagnostic]=step\n"
    body=once(body,marker,replacement+marker)
    body=once(body,"(step<=300 or step%1000==0)","(step<=lane['resumed_step']+300 or step%1000==0)")
    body=once(body,"if lane['step']>=1000 and eta is not None and not lane['admission_checked']:",
        "if lane['step']>=lane['admission_step'] and eta is not None and not lane['admission_checked']:")
    marker="emit({'event':'budget_admission','run':lane['name'],'estimated_remaining_seconds':eta,"
    body=once(body,marker,"emit({'event':'budget_admission','run':lane['name'],'new_iterations_observed':lane['step']-lane['resumed_step'],'estimated_remaining_seconds':eta,")
    marker="    observations={l['name']:training_observation(l['run'],l['iterations']) for l in lanes}"
    extra="\n    for lane in lanes:\n        metrics_path=lane['run']/'metrics.json'\n        if metrics_path.exists():\n            metrics=json.loads(metrics_path.read_text())\n            assert metrics['resumed_from_step']==lane['resumed_step']\n            assert metrics['new_iterations']==metrics['completed_steps']-lane['resumed_step']"
    body=once(body,marker,marker+extra)
    compile(body,'successor-driver','exec')
    return body
