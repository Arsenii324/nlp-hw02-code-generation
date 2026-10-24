"""Finite private exact-notebook verification, parent without a CUDA context."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_execution(executed,output,manifest):
    executed=Path(executed);output=Path(output)
    evidence=json.loads((executed/'executed.json').read_text())
    assert evidence['complete'] and evidence['executed_code_cells']==3
    assert evidence['input_sha256']==manifest['canonical_notebook_sha256']
    assert evidence['output_sha256']==digest(executed/'executed.ipynb')
    notebook=json.loads((executed/'executed.ipynb').read_text())
    cells=[c for c in notebook['cells'] if c['cell_type']=='code']
    assert len(cells)==3 and [c['execution_count'] for c in cells]==[1,2,3]
    assert all(not any(o['output_type']=='error' for o in c['outputs']) for c in cells)
    timing=json.loads((output/'test_timing.json').read_text())
    for key in ('bundle_sha256','checkpoint_sha256','training_source_sha256','inference_source_sha256',
        'selected_step','test_parquet_sha256'):assert timing[key]==manifest[key],key
    for key,value in manifest['decoding'].items():assert timing[key]==value,key
    assert timing['rows']==1919 and timing['gpu_count']==1 and 'T4' in timing['gpu']
    assert timing['meets_180_seconds_on_one_T4'] and 0<timing['seconds']<=180
    rows=[json.loads(line) for line in (output/'test_predictions.jsonl').read_text().splitlines()]
    assert [r['row_index'] for r in rows]==list(range(1919)) and all(isinstance(r['code'],str) for r in rows)
    return timing


def main():
    started=time.monotonic();working=Path('/kaggle/working')
    terminal={'complete':False,'phase':'startup'}
    def emit(record):
        record={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'session_seconds':time.monotonic()-started,**record}
        with (working/'session_events.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    try:
        matches=list(Path('/kaggle/input').rglob('verification_checksums.json'));assert len(matches)==1
        tools=matches[0].parent;manifest=json.loads((tools/'verification_manifest.json').read_text())
        for name,expected in json.loads(matches[0].read_text()).items():
            assert name==Path(name).name and digest(tools/name)==expected,name
        assert digest(Path(__file__))==manifest['driver_sha256']
        for name in ('verification_manifest.json','verification_checksums.json','canonical_inference.ipynb',
            'execute_inference_notebook.py','kaggle_eval.py'):shutil.copyfile(tools/name,working/name)
        shutil.copyfile(Path(__file__),working/'run_driver.py')
        hardware=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],
            check=True,capture_output=True,text=True,timeout=15).stdout.splitlines()
        assert len(hardware)==2 and all('T4' in name for name in hardware)
        emit({'event':'hardware','gpus':hardware,'note':'Only GPU0 visible to notebook child; full session is charged.'})
        assert time.monotonic()-started<180,'Startup admission exceeded'
        sys.path.insert(0,str(tools));from kaggle_eval import run_evaluation
        out=working/'executed';inference=working/'canonical-inference'
        terminal['phase']='canonical_execution'
        result=run_evaluation('exact-canonical',[sys.executable,'-u',str(tools/'execute_inference_notebook.py'),
            '--notebook',str(tools/'canonical_inference.ipynb'),'--output',str(out/'executed.ipynb')],
            working/'canonical.log',{'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
            'TOKENIZERS_PARALLELISM':'false'},manifest['outer_timeout_seconds']-(time.monotonic()-started),
            manifest['evaluation_bound_seconds'],emit,required=[out/'executed.ipynb',out/'executed.json',
                inference/'test_timing.json',inference/'test_predictions.jsonl'])
        assert result['status']=='complete'
        timing=validate_execution(out,inference,manifest)
        terminal.update(complete=True,phase='finished',timing=timing,execution=result,
            manifest_sha256=digest(tools/'verification_manifest.json'))
        emit({'event':'canonical_notebook_verified',**timing})
    except BaseException as exc:
        terminal.update(failure_type=type(exc).__name__,failure=str(exc));raise
    finally:
        terminal['script_elapsed_seconds']=time.monotonic()-started
        terminal['artifact_sha256']={str(p.relative_to(working)):digest(p) for p in working.rglob('*')
            if p.is_file() and not p.is_symlink() and p.name!='session_result.json'}
        (working/'session_result.json').write_text(json.dumps(terminal,indent=2))
        print(json.dumps({'event':'session_terminal',**terminal}),flush=True)


if __name__=='__main__':main()
