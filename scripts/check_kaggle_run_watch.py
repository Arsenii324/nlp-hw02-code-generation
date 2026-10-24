"""Focused local protocol checks; mocked provider, no credentials or remote requests."""
from argparse import Namespace
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('watch',Path(__file__).with_name('watch_kaggle_run.py'))
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
assert w.parse_status('Kernel status: KernelWorkerStatus.RUNNING')=='RUNNING'
try:w.parse_status('uncertain')
except ValueError:pass
else:raise AssertionError('Unrecognized status accepted')

for terminal,low_disk in [('COMPLETE',False),('ERROR',False),('COMPLETE',True)]:
    with tempfile.TemporaryDirectory(prefix='hw02-watch-check-') as folder:
        root=Path(folder);(root/'reports/evidence').mkdir(parents=True);(root/'artifacts/kaggle').mkdir(parents=True)
        (root/'manifest.json').write_text('{}');(root/'scripts').mkdir()
        for name in ['recover_kaggle_training.py','check_recovered_capacity.py','score_primary_predictions.py']:
            (root/'scripts'/name).write_text('pass\n')
        args=Namespace(kernel='owner/slug/1',root=Path('artifacts/kaggle/result'),manifest=Path('manifest.json'),evidence_prefix='mock',
            kaggle='mock-kaggle',python='mock-python',poll_seconds=1,max_seconds=10)
        statuses=iter(['RUNNING',terminal]);calls=[]
        def request(argv,timeout=60):
            calls.append([str(x) for x in argv])
            if argv[1:3]==['kernels','status']:return 'KernelWorkerStatus.'+next(statuses)
            return '{}'
        space=shutil._ntuple_diskusage(10*1024**3,0,(1 if low_disk else 5)*1024**3)
        with patch.object(w,'ROOT',root),patch.object(w.shutil,'disk_usage',return_value=space):
            result=w.watch(args,request=request,nap=lambda _:None)
        state=json.loads((root/'reports/evidence/mock_watch.json').read_text())
        assert not state['running'] and state['status']==terminal
        assert state['recovered']==(terminal=='COMPLETE' and not low_disk)
        assert result==(0 if state['recovered'] else 1)
        assert all('push' not in c and 'cancel' not in c and 'stop' not in c for c in calls)
        if terminal=='COMPLETE' and not low_disk:
            recovery=[c for c in calls if any(x.endswith('recover_kaggle_training.py') for x in c)]
            assert len(recovery)==1 and '--reference-run' in recovery[0]
        else:
            assert any(c[1:3]==['kernels','logs'] for c in calls)
            assert not any(any(x.endswith('recover_kaggle_training.py') for x in c) for c in calls)
print('finite_one_job_watch_mock_checks_passed')

# A concurrent invocation cannot share kernel, recovery directory, or evidence.
with tempfile.TemporaryDirectory(prefix='hw02-watch-lock-check-') as folder:
    root=Path(folder);(root/'scripts').mkdir();(root/'manifest.json').write_text('{}')
    for name in ['recover_kaggle_training.py','check_recovered_capacity.py','score_primary_predictions.py']:
        (root/'scripts'/name).write_text('pass\n')
    args=Namespace(kernel='owner/slug/1',root=Path('result'),manifest=Path('manifest.json'),evidence_prefix='mock',
        kaggle='mock-kaggle',python='mock-python',poll_seconds=1,max_seconds=10)
    def request(argv,timeout=60):
        if argv[1:3]==['kernels','status']:
            from copy import copy
            for variant in ('kernel','root','evidence'):
                other=copy(args)
                if variant=='kernel':other.root=Path('different');other.evidence_prefix='different'
                if variant=='root':other.kernel='other/slug/1';other.evidence_prefix='different'
                if variant=='evidence':other.kernel='other/slug/1';other.root=Path('different')
                try:w.watch(other,request=lambda *_: 'KernelWorkerStatus.ERROR',nap=lambda _:None)
                except ValueError as exc:assert 'Another monitor owns' in str(exc)
                else:raise AssertionError('Concurrent monitor was accepted')
            # Change locked code during the status request: must reject recovery.
            (root/'scripts/recover_kaggle_training.py').write_text('changed\n')
            return 'KernelWorkerStatus.COMPLETE'
        assert argv[1:3]==['kernels','logs'],'Recovery happened after code change'
        return '{}'
    with patch.object(w,'ROOT',root),patch.object(w.shutil,'disk_usage',return_value=shutil._ntuple_diskusage(10*1024**3,0,5*1024**3)):
        assert w.watch(args,request=request,nap=lambda _:None)==1
    state=json.loads((root/'reports/evidence/mock_watch.json').read_text())
    assert 'changed' in state['failure'] and not state['recovered']

# A timed-out local recovery command must not leave its download descendants live.
import subprocess
import sys
import time
with tempfile.TemporaryDirectory(prefix='hw02-watch-timeout-check-') as folder:
    root=Path(folder);marker=root/'descendant-finished.txt'
    child="import time;from pathlib import Path;time.sleep(1);Path("+repr(str(marker))+").write_text('unexpected')"
    parent="import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',"+repr(child)+"]);time.sleep(30)"
    with patch.object(w,'ROOT',root):
        try:w.command([sys.executable,'-c',parent],timeout=.15)
        except subprocess.TimeoutExpired:pass
        else:raise AssertionError('Expected finite timeout')
    time.sleep(1.2);assert not marker.exists(),'Local download descendant survived timeout'
print('finite_one_job_watch_lock_mutation_timeout_checks_passed')

with tempfile.TemporaryDirectory(prefix='hw02-watch-recovery-mutation-') as folder:
    root=Path(folder);(root/'scripts').mkdir();(root/'manifest.json').write_text('{}')
    for name in ['recover_kaggle_training.py','check_recovered_capacity.py','score_primary_predictions.py']:
        (root/'scripts'/name).write_text('pass\n')
    args=Namespace(kernel='owner/slug/1',root=Path('result'),manifest=Path('manifest.json'),evidence_prefix='mock',
        kaggle='mock-kaggle',python='mock-python',poll_seconds=1,max_seconds=10)
    def request(argv,timeout=60):
        if argv[1:3]==['kernels','status']:return 'KernelWorkerStatus.COMPLETE'
        if argv[1:3]==['kernels','logs']:return '{}'
        assert '--expected-hashes' in argv
        (root/'scripts/score_primary_predictions.py').write_text('changed-during-recovery\n')
        return '{}'
    with patch.object(w,'ROOT',root),patch.object(w.shutil,'disk_usage',return_value=shutil._ntuple_diskusage(10*1024**3,0,5*1024**3)):
        assert w.watch(args,request=request,nap=lambda _:None)==1
    state=json.loads((root/'reports/evidence/mock_watch.json').read_text())
    assert not state['recovered'] and 'changed' in state['failure']

spec=importlib.util.spec_from_file_location('recover',Path(__file__).with_name('recover_kaggle_training.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
with tempfile.TemporaryDirectory(prefix='hw02-recovery-timeout-evidence-') as folder:
    root=Path(folder);(root/'reports/evidence').mkdir(parents=True);(root/'manifest.json').write_text('{}')
    args=['recover','--kernel','owner/slug/1','--root',str(root/'result'),'--manifest',str(root/'manifest.json'),'--evidence-prefix','mock']
    with patch.object(r,'ROOT',root),patch.object(sys,'argv',args),patch.object(r.shutil,'disk_usage',return_value=shutil._ntuple_diskusage(10*1024**3,0,5*1024**3)),patch.object(r.subprocess,'run',side_effect=subprocess.TimeoutExpired(['mock'],300,output=b'partial-out',stderr=b'partial-err')):
        try:r.main()
        except subprocess.TimeoutExpired:pass
        else:raise AssertionError('Timeout not propagated')
    state=json.loads((root/'reports/evidence/mock_recovery_commands.json').read_text())
    entry=state['commands'][0]
    assert entry['failure']=='TimeoutExpired' and entry['timeout_seconds']==300
    assert entry['stdout']=='partial-out' and entry['stderr']=='partial-err'
print('finite_one_job_watch_recovery_integrity_timeout_evidence_checks_passed')
