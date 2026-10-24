"""Freeze the finite GPU batch; archives use .bin to prevent Kaggle expansion."""
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
prepared=ROOT/'artifacts/kaggle/prepare-v2'
cache_manifests=list(prepared.rglob('cache_manifest.json'))
assert len(cache_manifests)==1,'Recover completed CPU v2 cache manifest before GPU admission'
cache=json.loads(cache_manifests[0].read_text()); assert cache['complete']
cache_digest=hashlib.sha256(cache_manifests[0].read_bytes()).hexdigest()
tools=ROOT/'artifacts/kaggle/train-tools'; tools.mkdir(parents=True,exist_ok=True)
raw=Path(os.environ.get('NLP_HW02_DATA_ROOT',ROOT.parent/'data/tinysms_qa_12m'))
files={'training_source.bin':ROOT/'artifacts/baseline_source.zip',
       'baseline_bundle.bin':ROOT/'artifacts/baseline_bundle.zip','test.parquet':raw/'data/test/test.parquet'}
manifest=json.loads((ROOT/'assignment/dataset_manifest.json').read_text())
test=next(x for x in manifest['files'] if x['path']=='data/test/test.parquet')
assert hashlib.sha256(files['test.parquet'].read_bytes()).hexdigest()==test['sha256']
for name,path in files.items(): shutil.copyfile(path,tools/name)
hashes={name:hashlib.sha256((tools/name).read_bytes()).hexdigest() for name in files}
(tools/'training_checksums.json').write_text(json.dumps(hashes,indent=2))
(tools/'dataset-metadata.json').write_text(json.dumps({'id':'arsen4ikvar/nlp-hw02-train-tools-20261001',
    'title':'NLP HW02 private training tools','licenses':[{'name':'other'}]},indent=2))
kernel=ROOT/'kaggle/train'; kernel.mkdir(parents=True,exist_ok=True)
shutil.copyfile(ROOT/'scripts/kaggle_dual_run.py',kernel/'run.py')
(kernel/'kernel-metadata.json').write_text(json.dumps({'id':'arsen4ikvar/nlp-hw02-data-xsa-20261001',
    'title':'NLP HW02 data XSA 20261001','code_file':'run.py','language':'python','kernel_type':'script',
    'is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,
    'machine_shape':'NvidiaTeslaT4','dataset_sources':['arsen4ikvar/nlp-hw02-train-tools-20261001'],
    'competition_sources':[],'kernel_sources':['arsen4ikvar/nlp-hw02-prepare-v2']},indent=2))
record={'source_sha256':hashes['training_source.bin'],'driver_sha256':hashlib.sha256((kernel/'run.py').read_bytes()).hexdigest(),
        'cpu_kernel_version':2,'cache_manifest_sha256':cache_digest,
        'lanes':[{'id':'B005','device':0,'variant':'baseline','seed':42,'iterations':60000},
                                     {'id':'B006','device':1,'variant':'xsa','seed':42,'iterations':60000}],
        'batch_size':64,'outer_timeout_seconds':9000,'GPU_sessions_concurrent_limit':1,
        'note':'Same clean split, data, tokenizer and initial tensors; only XSA projection differs. Slow final lane may finish with other lane idle; no filler runs.'}
(kernel/'experiment-manifest.json').write_text(json.dumps(record,indent=2))
(tools/'experiment-manifest.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
