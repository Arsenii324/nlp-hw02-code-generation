"""Execute after the canonical notebook; scoped temporary Colab artifacts only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

base=Path('/content/selected_model');source=base/'source'
addon=Path('/content/decoding_addon.zip')
assert hashlib.sha256(addon.read_bytes()).hexdigest()=='54027edbe29b2f57a8cba661e1e4e7f48481ddb9a001820b8b4ca780c9377e8c'
with zipfile.ZipFile(addon) as archive:
    assert archive.testzip() is None
    assert set(archive.namelist())=={'src/nlp_hw02/beam.py','scripts/profile_decoding_gpu.py'}
    archive.extractall(source)
reference=Path('/content/reference_b005');reference.mkdir(exist_ok=True)
bundle=Path('/content/baseline_bundle_b005.zip')
assert hashlib.sha256(bundle.read_bytes()).hexdigest()=='99605d0d520118fddd249d770468961060ea6cd7ce38e9499a4995ebd699963d'
with zipfile.ZipFile(bundle) as archive:
    assert archive.testzip() is None;archive.extractall(reference)
# Both checkpoints use byte-identical model/inference/tokenizer definitions.
with zipfile.ZipFile(reference/'baseline_source.zip') as archive:
    for name in ('src/nlp_hw02/model.py','src/nlp_hw02/inference.py','src/nlp_hw02/tokenizer.py'):
        assert archive.read(name)==(source/name).read_bytes()
import torch
assert torch.cuda.device_count()==1 and 'T4' in torch.cuda.get_device_name(0)
if 'model' in globals(): del model
torch.cuda.empty_cache()
environment=os.environ.copy();environment['PYTHONPATH']=str(source/'src');environment['CUDA_VISIBLE_DEVICES']='0'
outputs=[]
for label,run in [('b005',reference),('b008',base)]:
    output=Path(f'/content/decoding-{label}');outputs.append(output)
    command=[sys.executable,'-u',str(source/'scripts/profile_decoding_gpu.py'),'--run',str(run),'--output',str(output)]
    if label=='b008': command.extend(['--test-parquet',str(base/'data/data/test/test.parquet')])
    subprocess.run(command,env=environment,check=True,timeout=180)
archive_path=Path('/content/decoding_checks.zip');digests={}
with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as archive:
    for output in outputs:
        names=['decoding_profile.json','greedy/validation.json','greedy/validation_predictions.jsonl',
               'beam4-alpha06/validation.json','beam4-alpha06/validation_predictions.jsonl']
        if output.name=='decoding-b008': names.extend(['beam_test_timing.json','beam_test_predictions.jsonl'])
        for name in names:
            path=output/name;relative=f'{output.name}/{name}';archive.write(path,relative)
            digests[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    archive.writestr('checks_manifest.json',json.dumps({'file_sha256':digests,
        'addon_sha256':hashlib.sha256(addon.read_bytes()).hexdigest(),
        'selected_bundle_sha256':hashlib.sha256(Path('/content/selected_bundle.zip').read_bytes()).hexdigest(),
        'note':'Same one-T4/runtime comparison, remaining1792 validation rows, fixed beam4alpha.6; no test tuning.'},indent=2))
with zipfile.ZipFile(archive_path) as archive: assert archive.testzip() is None
print(json.dumps({'archive':str(archive_path),'bytes':archive_path.stat().st_size,
    'sha256':hashlib.sha256(archive_path.read_bytes()).hexdigest()}))
print('COLAB_DECODING_CHECKS_FINISHED_OK')
