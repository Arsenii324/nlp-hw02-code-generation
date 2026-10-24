"""Stage the next private capacity/optimizer batch, without uploading/launching.

Separate tool dataset/kernel IDs keep the current running batch immutable.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


ROOT=Path(__file__).resolve().parents[1]


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--xsa',action='store_true',help='Use XSA in BOTH lanes, after clean-split results support it')
    parser.add_argument('--iterations',type=int,default=40000)
    args=parser.parse_args()
    assert 1000<=args.iterations<=60000
    prepared=ROOT/'artifacts/kaggle/prepare-v2/cache/cache_manifest.json'
    cache=json.loads(prepared.read_text()); assert cache['complete']
    tools=ROOT/'artifacts/kaggle/capacity-tools'; tools.mkdir(parents=True,exist_ok=True)
    from package_baseline import package_source
    package_source(tools/'training_source.bin')
    raw=Path(os.environ.get('NLP_HW02_DATA_ROOT',ROOT.parent/'data/tinysms_qa_12m'))
    manifest=json.loads((ROOT/'assignment/dataset_manifest.json').read_text())
    test=next(x for x in manifest['files'] if x['path']=='data/test/test.parquet')
    assert digest(raw/'data/test/test.parquet')==test['sha256']
    shutil.copyfile(raw/'data/test/test.parquet',tools/'test.parquet')
    hashes={name:digest(tools/name) for name in ('training_source.bin','test.parquet')}
    (tools/'training_checksums.json').write_text(json.dumps(hashes,indent=2))
    (tools/'dataset-metadata.json').write_text(json.dumps({'id':'arsen4ikvar/nlp-hw02-capacity-tools-20261001',
        'title':'NLP HW02 private capacity tools','licenses':[{'name':'other'}]},indent=2))
    kernel=ROOT/'kaggle/capacity'; kernel.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/'scripts/kaggle_dual_run.py',kernel/'run.py')
    (kernel/'kernel-metadata.json').write_text(json.dumps({'id':'arsen4ikvar/nlp-hw02-capacity-muon-20261001',
        'title':'NLP HW02 capacity Muon 20261001','code_file':'run.py','language':'python','kernel_type':'script',
        'is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,
        'machine_shape':'NvidiaTeslaT4','dataset_sources':['arsen4ikvar/nlp-hw02-capacity-tools-20261001'],
        'competition_sources':[],'kernel_sources':['arsen4ikvar/nlp-hw02-prepare-v2']},indent=2))
    config={'d_model':384,'ff':1536,'heads':4,'exclusive_self_attention':args.xsa}
    record={'source_sha256':hashes['training_source.bin'],'driver_sha256':digest(kernel/'run.py'),
        'cpu_kernel_version':2,'cache_manifest_sha256':digest(prepared),
        'lanes':[{'id':'B007','device':0,'seed':42,'model_config':config,'optimizer':'adamw','iterations':args.iterations},
                 {'id':'B008','device':1,'seed':42,'model_config':config,'optimizer':'muon_profile_selected','iterations':args.iterations}],
        'batch_size':64,'outer_timeout_seconds':9000,'GPU_sessions_concurrent_limit':1,
        'include_b001_timing':False,'optimizer_probe_width':384,
        'muon_selection_rule':'Faster finite stock-BF16/grouped-FP16 optimizer path by predeclared median timing gate; record actual selected implementation/precision. No quality tuning.',
        'note':'Matched architecture/data/seed/iterations; larger capacity vs B005/B006 is a recipe comparison. Muon changes optimizer and possibly NS precision, not an isolated batching ablation. Integrated timing gate; no filler runs.'}
    for directory in (kernel,tools): (directory/'experiment-manifest.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__=='__main__': main()
