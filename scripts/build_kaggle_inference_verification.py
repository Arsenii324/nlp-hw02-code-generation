"""Stage an exact selected-bundle notebook verification; no upload or launch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile
from package_inference import validate_bundle

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage(bundle,tools,kernel):
    bundle=Path(bundle);tools=Path(tools);kernel=Path(kernel)
    assert not tools.exists() or not any(tools.iterdir()), 'Preserve existing staged inputs'
    assert not kernel.exists() or not any(kernel.iterdir()), 'Preserve existing staged kernel'
    selected=validate_bundle(bundle)
    raw=Path(os.environ.get('NLP_HW02_DATA_ROOT',ROOT.parent/'data/tinysms_qa_12m'))/'data/test/test.parquet'
    expected=next(r['sha256'] for r in json.loads((ROOT/'assignment/dataset_manifest.json').read_text())['files']
        if r['path']=='data/test/test.parquet')
    assert digest(raw)==expected
    tools.mkdir(parents=True,exist_ok=True);kernel.mkdir(parents=True,exist_ok=True)
    sources={'selected_bundle.bin':bundle,'canonical_inference.ipynb':ROOT/'submit/inference.ipynb',
        'execute_inference_notebook.py':ROOT/'scripts/execute_inference_notebook.py',
        'kaggle_eval.py':ROOT/'scripts/kaggle_eval.py','test.parquet':raw}
    for name,path in sources.items():
        assert path.is_file() and not path.is_symlink();shutil.copyfile(path,tools/name)
    shutil.copyfile(ROOT/'scripts/kaggle_inference_verification.py',kernel/'run.py')
    with zipfile.ZipFile(bundle) as archive:metrics=json.loads(archive.read('metrics.json'))
    manifest={'bundle_sha256':digest(bundle),'checkpoint_sha256':selected['file_sha256']['model.pt'],
        'training_source_sha256':selected['training_source_sha256'],
        'inference_source_sha256':selected.get('inference_source_sha256',{}),
        'selected_run':selected['selected_run'],'selected_step':metrics['selected_step'],
        'decoding':selected.get('inference',{'decoder':'greedy','batch_size':256}),
        'test_parquet_sha256':expected,'canonical_notebook_sha256':digest(tools/'canonical_inference.ipynb'),
        'driver_sha256':digest(kernel/'run.py'),'outer_timeout_seconds':600,'evaluation_bound_seconds':420,
        'GPU_sessions_concurrent_limit':1,
        'note':'Inference only; exactly one visible T4 in a fresh child, no training/filler/install/submission.'}
    (tools/'verification_checksums.json').write_text(json.dumps({n:digest(tools/n) for n in sources},indent=2))
    for path in (tools,kernel):(path/'verification_manifest.json').write_text(json.dumps(manifest,indent=2))
    dataset='arsen4ikvar/nlp-hw02-selected-verification-tools-20261002'
    identity='arsen4ikvar/nlp-hw02-selected-verification-20261002'
    (tools/'dataset-metadata.json').write_text(json.dumps({'id':dataset,'title':'NLP HW02 private inference check',
        'licenses':[{'name':'other'}]},indent=2))
    (kernel/'kernel-metadata.json').write_text(json.dumps({'id':identity,'title':'NLP HW02 selected verification 20261002',
        'code_file':'run.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,
        'enable_tpu':False,'enable_internet':False,'machine_shape':'NvidiaTeslaT4','dataset_sources':[dataset],
        'competition_sources':[],'kernel_sources':[]},indent=2))
    assert all(p.is_file() and not p.is_symlink() for p in tools.iterdir())
    return manifest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',required=True)
    parser.add_argument('--tools',default=str(ROOT/'artifacts/kaggle/selected-verification-tools'))
    parser.add_argument('--kernel',default=str(ROOT/'kaggle/selected-verification'))
    args=parser.parse_args();record=stage(args.bundle,args.tools,args.kernel)
    print(json.dumps({'tools':args.tools,'kernel':args.kernel,**record},indent=2))


if __name__=='__main__':main()
