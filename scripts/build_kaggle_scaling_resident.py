"""Stage one corrected resident-array width/depth pair; no upload/launch here."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile
from package_baseline import package_source

ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    tools=ROOT/'artifacts/kaggle/scaling-resident-tools';kernel=ROOT/'kaggle/scaling-resident'
    tools.mkdir(parents=True,exist_ok=True);kernel.mkdir(parents=True,exist_ok=True)
    cache=ROOT/'artifacts/kaggle/prepare-v2/cache/cache_manifest.json'
    assert json.loads(cache.read_text())['complete']
    source=package_source(tools/'training_source.bin')
    with zipfile.ZipFile(tools/'training_source.bin') as z:
        assert z.testzip() is None
        assert {'scripts/profile_decoding_gpu.py','scripts/execute_inference_notebook.py','scripts/kaggle_eval.py'}<=set(z.namelist())
    raw=Path(os.environ.get('NLP_HW02_DATA_ROOT',ROOT.parent/'data/tinysms_qa_12m'))
    expected=next(r['sha256'] for r in json.loads((ROOT/'assignment/dataset_manifest.json').read_text())['files'] if r['path']=='data/test/test.parquet')
    assert sha(raw/'data/test/test.parquet')==expected
    shutil.copyfile(raw/'data/test/test.parquet',tools/'test.parquet')
    shutil.copyfile(ROOT/'scripts/kaggle_dual_run.py',kernel/'run.py')
    checks={name:sha(tools/name) for name in ('training_source.bin','test.parquet')}
    (tools/'training_checksums.json').write_text(json.dumps(checks,indent=2))
    dataset='arsen4ikvar/nlp-hw02-scaling-resident-tools-20261002';identity='arsen4ikvar/nlp-hw02-width-depth-resident-20261002'
    (tools/'dataset-metadata.json').write_text(json.dumps({'id':dataset,'title':'NLP HW02 private scaling tools','licenses':[{'name':'other'}]},indent=2))
    (kernel/'kernel-metadata.json').write_text(json.dumps({'id':identity,'title':'NLP HW02 width depth resident 20261002','code_file':'run.py','language':'python','kernel_type':'script',
        'is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'machine_shape':'NvidiaTeslaT4',
        'dataset_sources':[dataset],'competition_sources':[],'kernel_sources':['arsen4ikvar/nlp-hw02-prepare-v2']},indent=2))
    record={'source_sha256':source['sha256'],'driver_sha256':sha(kernel/'run.py'),'cpu_kernel_version':2,'cache_manifest_sha256':sha(cache),
        'lanes':[{'id':'B011','device':0,'seed':42,'model_config':{'d_model':512,'ff':2048,'heads':4,'encoder_layers':2,'decoder_layers':2},'optimizer':'muon_grouped','iterations':40000},
                 {'id':'B012','device':1,'seed':42,'model_config':{'d_model':384,'ff':1536,'heads':4,'encoder_layers':4,'decoder_layers':4},'optimizer':'muon_grouped','iterations':40000}],
        'batch_size':64,'outer_timeout_seconds':14400,'GPU_sessions_concurrent_limit':1,'include_b001_timing':False,'profile_decoding':True,
        'canonical_bundle_timing':False,'resident_training_arrays':True,'predecessor_failed_kernel':'arsen4ikvar/nlp-hw02-width-depth-20261002/1',
        'evaluation_bound_seconds':300,'planned_evaluation_count':4,'tail_reserve_seconds':1500,
        'parameter_counts':{'b011':15992832,'b012':17524224},
        'note':'Near-sized width/depth capacity-allocation comparison; same data/seed/Muon recipe/40k iterations, not exactly matched parameters. No filler/retry. Fixed beam4/alpha.6 validation and test timing, resident immutable arrays and actual progress/timings; no incomplete-model evaluation. B008 canonical already verified, no repeat.'}
    for directory in (tools,kernel):(directory/'experiment-manifest.json').write_text(json.dumps(record,indent=2))
    assert {p.name for p in tools.iterdir()}=={'training_source.bin','test.parquet','training_checksums.json','dataset-metadata.json','experiment-manifest.json'}
    assert all(p.is_file() and not p.is_symlink() for p in tools.iterdir())
    print(json.dumps({'tools':str(tools),'kernel':str(kernel),'tools_bytes':sum(p.stat().st_size for p in tools.iterdir()),**record},indent=2))

if __name__=='__main__':main()
