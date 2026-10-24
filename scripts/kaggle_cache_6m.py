"""One bounded CPU-only cache job; no GPU allocation, training or submission."""
from datetime import datetime,timezone
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


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    started=time.monotonic();working=Path('/kaggle/working');temp=Path('/kaggle/temp/cache-6m')
    result={'complete':False,'cache_ready':False,'GPU_requested':False,'phase':'startup'}
    worker=None
    def termination(signum,frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        raise RuntimeError('Parent SIGTERM received; worker shutdown required')
    signal.signal(signal.SIGTERM,termination)
    def emit(event,**fields):
        record={'utc':datetime.now(timezone.utc).isoformat(),'event':event,'elapsed_seconds':time.monotonic()-started,**fields}
        with (working/'prepare_events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    try:
        matches=list(Path('/kaggle/input').rglob('cache_6m_manifest.json'))
        if len(matches)!=1:raise ValueError('Exactly one frozen tools input required')
        tools=matches[0].parent;manifest=json.loads(matches[0].read_text())
        if sha(__file__)!=manifest['driver_sha256'] or sha(tools/'source.bin')!=manifest['source_sha256']:
            raise ValueError('Source/driver hash mismatch')
        if manifest['outer_seconds']!=5400 or manifest['worker_seconds']!=4800:raise ValueError('Unexpected bounds')
        parents=[p.parent for p in Path('/kaggle/input').rglob('cache_manifest.json')
                 if sha(p)==manifest['parent_cache_sha256']['cache_manifest.json']]
        if len(parents)!=1:raise ValueError('Exactly one pinned2M parent cache required')
        parent=parents[0]
        for name,expected in manifest['parent_cache_sha256'].items():
            if sha(parent/name)!=expected:raise ValueError('Parent identity mismatch: '+name)
        required=manifest['train_rows']*(2*(256+192)+8)+manifest['raw_bytes']+3*1024**3
        free=shutil.disk_usage(working).free
        if free<required:raise ValueError('Insufficient remote disk before raw fetch')
        info=Path('/proc/meminfo').read_text();available=int(next(x for x in info.splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
        for limit,used in [('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current')]:
            if Path(limit).exists() and Path(limit).read_text().strip()!='max':available=min(available,int(Path(limit).read_text())-int(Path(used).read_text()))
        if available<8*1024**3:raise ValueError('Insufficient remote RAM for CPU preparation')
        if time.monotonic()-started>120:raise TimeoutError('CPU startup admission exceeded')
        temp.mkdir(parents=True,exist_ok=False);source=temp/'source';source.mkdir()
        with zipfile.ZipFile(tools/'source.bin') as archive:
            names=archive.namelist()
            if len(names)!=len(set(names)) or any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):raise ValueError('Unsafe source archive')
            if archive.testzip() is not None:raise ValueError('Source CRC failure')
            for name,expected in manifest['source_members_sha256'].items():
                if hashlib.sha256(archive.read(name)).hexdigest()!=expected:raise ValueError('Source member mismatch')
            if set(names)!=set(manifest['source_members_sha256']):raise ValueError('Unexpected source members')
            archive.extractall(source)
        for name in ('cache_6m_manifest.json','source.bin'):shutil.copyfile(tools/name,working/name)
        shutil.copyfile(__file__,working/'run_driver.py')
        emit('CPU_prepare_admitted',free_bytes=free,required_bytes=required,available_RAM_bytes=available,
             train_rows=manifest['train_rows'],GPU_requested=False)
        result['phase']='prepare_and_audit'
        command=[sys.executable,'-u',str(source/'scripts/prepare_cache_6m.py'),'--raw',str(temp/'raw'),
                 '--output',str(working/'cache'),'--parent-cache',str(parent),'--manifest',str(tools/'cache_6m_manifest.json')]
        remaining=manifest['outer_seconds']-(time.monotonic()-started)-120
        if remaining<manifest['worker_seconds']:raise TimeoutError('Insufficient worker/tail reserve')
        with (working/'prepare.log').open('w') as log:
            worker=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,'PYTHONPATH':str(source/'src'),'TOKENIZERS_PARALLELISM':'false',
                     'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
            worker.wait(timeout=manifest['worker_seconds'])
        if worker.returncode!=0:raise RuntimeError('CPU worker failed: '+str(worker.returncode))
        coverage=json.loads((working/'cache/coverage_audit.json').read_text())
        if not coverage['complete'] or not coverage['fixed_validation_tokenizer_legacy_bytes']:raise ValueError('Coverage audit incomplete')
        result.update(complete=True,cache_ready=True,phase='finished',coverage=coverage,
                      source_sha256=manifest['source_sha256'],manifest_sha256=sha(matches[0]))
        emit('CPU_cache_ready',coverage=coverage)
    except BaseException as exc:
        result.update(complete=False,cache_ready=False,failure_type=type(exc).__name__,failure=str(exc));raise
    finally:
        if worker is not None and worker.poll() is None:
            try:os.killpg(worker.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:worker.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:os.killpg(worker.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                worker.wait(timeout=5)
            result['worker_shutdown']='terminated process group before artifact hashing'
        if worker is not None:result['worker_returncode']=worker.returncode
        result['elapsed_seconds']=time.monotonic()-started
        # Arrays stay remote; their hashes are in cache_manifest. Recover tiny diagnostics only.
        result['artifact_sha256']={str(p.relative_to(working)):sha(p) for p in working.rglob('*')
             if p.is_file() and not p.is_symlink() and p.suffix in ('.json','.jsonl','.log','.py','.bin')
             and p.name!='prepare_result.json'}
        (working/'prepare_result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'event':'terminal',**result}),flush=True)


if __name__=='__main__':main()
