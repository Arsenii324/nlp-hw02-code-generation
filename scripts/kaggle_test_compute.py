"""Finite two-T4 B012 validation probe; no training, install or submission."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started=time.monotonic();working=Path('/kaggle/working');temp=Path('/kaggle/temp/b012-probe');temp.mkdir(parents=True,exist_ok=True)
    terminal={'complete':False};children=[]
    def emit(record):
        record={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'seconds':time.monotonic()-started,**record}
        with (working/'probe_events.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    try:
        files=list(Path('/kaggle/input').rglob('probe_manifest.json'));assert len(files)==1
        tools=files[0].parent;manifest=json.loads(files[0].read_text())
        for name,digest in manifest['tools_sha256'].items():assert Path(name).name==name and sha(tools/name)==digest,name
        assert sha(__file__)==manifest['driver_sha256']
        bundles=[p for p in Path('/kaggle/input').rglob('selected_bundle.bin') if sha(p)==manifest['bundle_sha256']];assert len(bundles)==1
        run=temp/'run';run.mkdir()
        with zipfile.ZipFile(bundles[0]) as archive:
            assert archive.testzip() is None;meta=json.loads(archive.read('bundle_manifest.json'))
            for name,digest in meta['file_sha256'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==digest,name
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in archive.namelist());archive.extractall(run)
        assert sha(run/'model.pt')==manifest['checkpoint_sha256']
        source=temp/'source';source.mkdir()
        with zipfile.ZipFile(run/'baseline_source.zip') as archive:
            assert archive.testzip() is None and all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in archive.namelist());archive.extractall(source)
        hardware=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=15).stdout.splitlines()
        assert len(hardware)==2 and all('T4' in name for name in hardware)
        shutil.copyfile(__file__,working/'run_driver.py');shutil.copyfile(files[0],working/'probe_manifest.json')
        shutil.copyfile(tools/'test_compute_probe.py',working/'test_compute_probe.py')
        emit({'event':'startup_complete','gpus':hardware})
        assert time.monotonic()-started<60
        for device,lane in enumerate(('speed','quality')):
            log=(working/f'{lane}.log').open('w');out=working/lane
            proc=subprocess.Popen([sys.executable,'-u',str(tools/'test_compute_probe.py'),'--run',str(run),'--output',str(out),
                '--consensus',str(tools/'primary_reference_consensus.json'),'--lane',lane],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'PYTHONPATH':str(source/'src'),'CUDA_VISIBLE_DEVICES':str(device),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'})
            children.append((lane,proc,log));emit({'event':'probe_started','lane':lane,'device':device})
        announced=set();last=0;child_started=time.monotonic()
        while any(p.poll() is None for _,p,_ in children):
            if time.monotonic()-child_started>500:raise TimeoutError('Finite probe bound exceeded')
            for lane,proc,_ in children:
                if proc.poll() is not None and lane not in announced:
                    emit({'event':'probe_exit','lane':lane,'exit':proc.returncode});announced.add(lane)
                    assert proc.returncode==0,lane
                result=working/lane/'probe_result.json'
                if result.exists() and time.monotonic()-started-last>15:
                    try:record=json.loads(result.read_text())
                    except json.JSONDecodeError:continue
                    emit({'event':'probe_progress','lane':lane,'completed_paths':list(record['paths'])});last=time.monotonic()-started
            time.sleep(1)
        for lane,proc,_ in children:
            assert proc.returncode==0,lane
            result=json.loads((working/lane/'probe_result.json').read_text());assert result['complete'] and result['gpu_count']==1
        terminal['complete']=True;emit({'event':'all_probes_complete'})
    except BaseException as exc:terminal.update(failure_type=type(exc).__name__,failure=str(exc));raise
    finally:
        for lane,proc,log in children:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
            log.close()
        terminal['elapsed_seconds']=time.monotonic()-started
        terminal['artifact_sha256']={str(p.relative_to(working)):sha(p) for p in working.rglob('*') if p.is_file() and p.name!='probe_terminal.json'}
        (working/'probe_terminal.json').write_text(json.dumps(terminal,indent=2)+'\n')


if __name__=='__main__':main()
