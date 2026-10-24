"""Stage a private allowlisted tools dataset and a CPU-only cache job."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
tools=ROOT/'artifacts/kaggle/tools'; tools.mkdir(parents=True,exist_ok=True)
mapping={'source.zip':ROOT/'artifacts/baseline_source.zip',
         'tokenizer.json':ROOT/'artifacts/baseline/tokenizer.json',
         'legacy_validation.json':ROOT/'artifacts/baseline/validation.json',
         'baseline_bundle.zip':ROOT/'artifacts/baseline_bundle.zip'}
for name,path in mapping.items(): shutil.copyfile(path,tools/name)
hashes={name:hashlib.sha256((tools/name).read_bytes()).hexdigest() for name in mapping}
with zipfile.ZipFile(tools/'source.zip') as z:
    source_hashes={name:hashlib.sha256(z.read(name)).hexdigest() for name in z.namelist()}
(tools/'checksums.json').write_text(json.dumps(hashes,indent=2))
(tools/'dataset-metadata.json').write_text(json.dumps({'id':'arsen4ikvar/nlp-hw02-tools-20261001',
    'title':'NLP HW02 private tools 20261001','licenses':[{'name':'other'}]},indent=2))
kernel=ROOT/'kaggle/prepare'; kernel.mkdir(parents=True,exist_ok=True)
(kernel/'kernel-metadata.json').write_text(json.dumps({'id':'arsen4ikvar/nlp-hw02-prepare-v2',
    'title':'NLP HW02 prepare v2','code_file':'prepare.py','language':'python','kernel_type':'script',
    'is_private':True,'enable_gpu':False,'enable_tpu':False,'enable_internet':True,
    'dataset_sources':['arsen4ikvar/nlp-hw02-tools-20261001'],'competition_sources':[],
    'kernel_sources':[]},indent=2))
(kernel/'prepare.py').write_text(('''"""CPU-only pinned-data preparation; no GPU quota allocation."""
import hashlib,json,os,pathlib,subprocess,sys,time,zipfile
started=time.monotonic()
working=pathlib.Path('/kaggle/working'); temp=pathlib.Path('/kaggle/temp/nlp-hw02'); temp.mkdir(parents=True,exist_ok=True)
matches=list(pathlib.Path('/kaggle/input').rglob('checksums.json'))
matches=[p for p in matches if (p.parent/'source.zip').exists() or (p.parent/'source').is_dir()]
assert len(matches)==1, f'Expected exactly one tools bundle: {matches}'
tools=matches[0].parent
hashes=json.loads((tools/'checksums.json').read_text())
for name,digest in hashes.items():
    if not name.endswith('.zip'): assert hashlib.sha256((tools/name).read_bytes()).hexdigest()==digest, name
source=tools/'source'
if not source.is_dir():
    source=temp/'source'; source.mkdir(exist_ok=True)
    with zipfile.ZipFile(tools/'source.zip') as z:
        for name in z.namelist():
            assert not pathlib.PurePosixPath(name).is_absolute() and '..' not in pathlib.PurePosixPath(name).parts
        z.extractall(source)
for name,digest in __SOURCE_HASHES__.items():
    assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest, name
env={**os.environ,'PYTHONPATH':str(source/'src'),'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'2'}
def run(args):
    subprocess.run([sys.executable,'-u',*map(str,args)],env=env,check=True)
run([source/'scripts/check_data_v2.py'])
if (source/'scripts/check_arithmetic.py').exists(): run([source/'scripts/check_arithmetic.py'])
run([source/'scripts/prepare_representative.py','--raw',temp/'raw','--output',working/'cache',
     '--tokenizer',tools/'tokenizer.json','--legacy-validation',tools/'legacy_validation.json'])
record={'source_sha256':hashes['source.zip'],'elapsed_seconds':time.monotonic()-started,
        'cache_ready':True,'gpu_requested':False,'dataset_revision':'b2724689ff079e9c22efd548fe134bfb29036dbb'}
(working/'prepare_result.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
print('KAGGLE_PREPARE_OK',flush=True)
''').replace('__SOURCE_HASHES__',repr(source_hashes)))
print(json.dumps({'dataset_bytes':sum((tools/name).stat().st_size for name in mapping),
                  'source_sha256':hashes['source.zip'],'kernel':str(kernel)},indent=2))
