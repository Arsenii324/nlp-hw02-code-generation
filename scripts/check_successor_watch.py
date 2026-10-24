"""Offline schedule fixtures: no network, launches, state downloads or GPUs."""
from copy import deepcopy
from datetime import datetime,timezone,timedelta
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


def fixture(mode):
    spec=importlib.util.spec_from_file_location('watch_fixture','scripts/watch_kaggle_successors.py')
    watch=importlib.util.module_from_spec(spec);spec.loader.exec_module(watch)
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);watch.ROOT=root;watch.WATCH=root/'watch.json';watch.EVENTS=root/'events.jsonl'
        # Snapshot locks are ordinary local files; the fake builder never reads
        # these fixture contents and no production source is edited.
        for name in ['scripts/build_kaggle_successor.py','scripts/successor_driver_patch.py','scripts/kaggle_resume_inputs.py',
            'scripts/recover_kaggle_training.py','scripts/check_recovered_capacity.py','src/nlp_hw02/model.py',
            'src/nlp_hw02/train.py','src/nlp_hw02/resuming.py','src/nlp_hw02/optim.py','src/nlp_hw02/optim_batched.py',
            'src/nlp_hw02/optim_polar.py','scripts/check_looped_model.py','scripts/check_training_resumption.py','scripts/check_polar_muon.py']:
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
        loop={'name':'b017-b018','kernel':'fixture/loop/1'};resume={'name':'b019-b020','kernel':'fixture/resume/1'}
        watch.checked_loop=lambda:deepcopy(loop)
        watch.prepare_resume=lambda state:deepcopy(resume)
        watch.prepared_resume=lambda:deepcopy(resume)
        watch.command=lambda *a,**kw:''
        watch.refresh_handoff=lambda state:None
        watch.time=SimpleNamespace(sleep=lambda seconds:None)
        watch.DEADLINE=datetime.now(timezone.utc)+timedelta(days=1)
        launches=[];events=[];cycle=[0]
        def snapshot(state,event,**fields):
            events.append(event)
            if event=='statuses':cycle[0]+=1
        watch.snapshot=snapshot
        def launch(job,state):
            active=[j for j in state['jobs'] if j.get('status') in ('RUNNING','QUEUED')]
            assert len(active)<2,'attempted third allocation'
            assert job['kernel'] not in launches,'attempted retry'
            launches.append(job['kernel']);job['launched']=True
        watch.launch=launch
        def recover(job,state):
            assert job.get('status')=='COMPLETE';job['recovered']=True
        watch.recover=recover
        def status(job):
            if mode=='resume':
                if job['name']=='b013-b014':return 'COMPLETE'
                if job['name']=='b015-b016':return 'RUNNING' if cycle[0]==0 else 'COMPLETE'
                return 'RUNNING' if cycle[0]<2 else 'COMPLETE'
            if mode=='error':return 'ERROR' if job['name']=='b013-b014' else 'RUNNING'
            if mode=='expired':return 'COMPLETE'
            if cycle[0]==0:return 'RUNNING'
            if cycle[0]==1:return 'COMPLETE' if job['name']=='b013-b014' else 'RUNNING'
            if cycle[0]==2:return 'COMPLETE' if job['name'] in ('b013-b014','b015-b016') else 'RUNNING'
            return 'COMPLETE'
        watch.status=status
        if mode=='expired':watch.DEADLINE=datetime(2000,1,1,tzinfo=timezone.utc)
        saved=sys.argv;sys.argv=['watch_fixture','--run']
        if mode=='resume':
            (root/'reports/evidence').mkdir(parents=True)
            jobs=deepcopy(watch.CURRENT);jobs[0].update({'status':'COMPLETE','recovered':True})
            jobs[1]['status']='RUNNING';jobs.append({**loop,'launched':True,'status':'RUNNING'})
            locked=[p for p in root.rglob('*.py')]
            prior={'running':False,'pipeline_complete':False,'source_snapshot':{str(p.relative_to(root)):watch.sha(p) for p in locked},
                'jobs':jobs,'next_loop':deepcopy(loop),'next_resume':None,'failure':'fixture403','failure_type':'RuntimeError',
                'selected_bundle_changed':False,'submissions':False}
            watch.WATCH.write_text(json.dumps(prior))
            sys.argv+=['--resume-state',str(watch.WATCH),'--use-prepared-resume']
        try:
            try:watch.main()
            except RuntimeError:
                assert mode=='error'
            else:assert mode!='error'
        finally:sys.argv=saved
        if mode=='normal':assert launches==['fixture/loop/1','fixture/resume/1'] and cycle[0]==4
        elif mode=='resume':assert launches==['fixture/resume/1'] and cycle[0]==3
        else:assert launches==[]
        assert events[-1]=='watch_finished'
        if mode=='error':assert 'watch_stopped_requires_inspection' in events
        return {'case':mode,'launches':launches,'terminal_cleanup_recorded':True}


def main():
    print(json.dumps({'event':'offline_successor_watch_checks_passed','cases':[fixture(mode) for mode in ('normal','expired','error','resume')],
        'network_used':False,'actual_GPU_launches':0}))


if __name__=='__main__':main()
