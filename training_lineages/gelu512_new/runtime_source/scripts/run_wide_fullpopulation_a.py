"""Finite owned-node strict wide full-population continuation; no provider/allocation calls."""
import argparse,datetime,hashlib,json,math,os,pathlib,signal,subprocess,time

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def stop(child):
    result={'pid':child.pid,'reaped':False,'group_exited':False};end=time.monotonic()+20
    def observe():
        result['reaped']=child.poll() is not None
        try:os.killpg(child.pid,0)
        except ProcessLookupError:result['group_exited']=True
        return result['reaped'] and result['group_exited']
    for sig,wait in [(signal.SIGTERM,15),(signal.SIGKILL,5)]:
        if observe():break
        try:os.killpg(child.pid,sig)
        except ProcessLookupError:pass
        cutoff=min(end,time.monotonic()+wait)
        while time.monotonic()<cutoff and not observe():time.sleep(.1)
    observe();return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,choices=(0,1),required=True);p.add_argument('--job-root',type=pathlib.Path,required=True);p.add_argument('--data',type=pathlib.Path,required=True);p.add_argument('--runtime',type=pathlib.Path,required=True);p.add_argument('--source-archive',type=pathlib.Path,required=True);p.add_argument('--contract',type=pathlib.Path,required=True);p.add_argument('--contract-sha256',required=True);p.add_argument('--parent-contract',type=pathlib.Path,required=True);p.add_argument('--end-utc',required=True);p.add_argument('--reserve-seconds',type=int,default=3600);a=p.parse_args()
    end_utc=datetime.datetime.fromisoformat(a.end_utc.replace('Z','+00:00'));assert end_utc.tzinfo is not None and end_utc<=datetime.datetime(2026,10,24,20,59,tzinfo=datetime.timezone.utc)
    seconds=end_utc.timestamp()-time.time();assert math.isfinite(seconds) and seconds>a.reserve_seconds+900 and 600<=a.reserve_seconds<=7200
    started=time.monotonic();worker_deadline=started+seconds-a.reserve_seconds;hard_deadline=started+seconds;child=None
    def hard_exit(*_):
        if child is not None:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        os._exit(124) # Last bound covers startup/IO/unwind; committed states remain on disk.
    signal.signal(signal.SIGALRM,hard_exit);signal.setitimer(signal.ITIMER_REAL,max(.001,hard_deadline-time.monotonic()))
    assert all(x.is_absolute() for x in (a.job_root,a.data,a.runtime,a.source_archive,a.contract,a.parent_contract)) and not a.job_root.exists()
    assert sha(a.contract)==a.contract_sha256;c=json.loads(a.contract.read_text());assert c['verified_actual_values'] is True and c['schema']=='wide_fullpopulation_training_contract_a_v1' and sha(a.parent_contract)==c['parent_binding_sha256'] and sha(a.source_archive)==c['runtime_archive_sha256']
    a.job_root.mkdir(mode=0o700);out=a.job_root/'training';report={'schema':'wide_fullpopulation_owned_parent_a_v1','physical_device':a.device,'complete':False,'worker_deadline_monotonic':worker_deadline,'hard_deadline_monotonic':hard_deadline,'end_UTC':a.end_utc,'reserve_seconds':a.reserve_seconds,'contract_sha256':a.contract_sha256,'parent_binding_sha256':c['parent_binding_sha256'],'source_sha256':sha(pathlib.Path(__file__)),'provider_action':False,'hard_alarm_armed':True,'fixed_NEW_horizon':c['new_steps']}
    def save():
        report['elapsed_seconds']=time.monotonic()-started;q=a.job_root/'owner-result.json.partial';q.write_text(json.dumps(report,indent=2)+'\n');os.replace(q,a.job_root/'owner-result.json')
    requested=False
    def stopping(*_):
        nonlocal requested;requested=True
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    command=[str(a.runtime/'scripts/train_wide_fullpopulation_a.py'),'--contract',str(a.contract),'--parent-contract',str(a.parent_contract),'--data',str(a.data),'--source-archive',str(a.source_archive),'--output',str(out),'--new-steps',str(c['new_steps']),'--deadline',str(worker_deadline)]
    save()
    try:
        with (a.job_root/'train.log').open('x') as log:
            worker_seconds=max(.001,worker_deadline-time.monotonic()+180);report['worker_external_timeout_seconds']=worker_seconds;report['worker_external_kill_grace_seconds']=30
            child=subprocess.Popen(['timeout','--signal=TERM','--kill-after=30s',str(worker_seconds)+'s',os.sys.executable,'-u',*command],env={**os.environ,'CUDA_VISIBLE_DEVICES':str(a.device),'PYTHONPATH':str(a.runtime/'src')+os.pathsep+str(a.runtime/'scripts'),'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','TOKENIZERS_PARALLELISM':'false'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True);report['worker_pid']=child.pid;save()
            while child.poll() is None:
                now=time.monotonic()
                if requested or now>=worker_deadline+180:break
                progress=out/'progress.json'
                if progress.is_file():
                    try:report['progress']=json.loads(progress.read_text())
                    except (ValueError,OSError):pass
                save();time.sleep(min(5,max(.1,worker_deadline+180-now)))
            report['cleanup']=stop(child);report['actual_exit']=child.poll()
        result=out/'worker_result.json'
        if result.is_file():
            report['worker_result']=json.loads(result.read_text());report['complete']=report['actual_exit']==0 and report['worker_result']['admitted_horizon_complete'] is True and report['cleanup']['reaped'] and report['cleanup']['group_exited']
    except BaseException as error:
        report['complete']=False;report['error_type']=type(error).__name__
        if child is not None:report['cleanup']=stop(child)
    finally:save() # Hardalarm remains armed throughout finalization.
    return 0 if report['complete'] else 1
if __name__=='__main__':raise SystemExit(main())
