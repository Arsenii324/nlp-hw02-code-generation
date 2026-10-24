"""Run with Colab exec --env NLP_HW02_VARIANT=xsa; archive only bounded artifacts."""
import hashlib
import json
import os
from pathlib import Path
import zipfile

variant=os.environ.get('NLP_HW02_VARIANT','baseline')
if variant not in ['baseline','xsa','muon','relu2']: raise ValueError(variant)
root=Path('/content/nlp-hw02')
run=root/('run' if variant=='baseline' else f'run-{variant}')
exit_path=root/('test.exit' if variant=='baseline' else f'{variant}.exit')
assert exit_path.read_text()=='0', 'Only bundle completed runs'
names=['model.pt','tokenizer.json','run_config.json','data_stats.json','metrics.json',
       'train.jsonl','validation.json','validation_predictions.jsonl','test_predictions.jsonl','test_timing.json']
if (run/'batch_profile.json').exists(): names.append('batch_profile.json')
source=Path('/content/baseline_source.zip' if variant=='baseline' else '/content/followup_source.zip')
archive=Path(f'/content/{variant}_bundle.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for name in names: z.write(run/name,name)
    z.write(source,'baseline_source.zip')
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,
    'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
    'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'variant':variant}))
print('BUNDLE_FINISHED_OK')
