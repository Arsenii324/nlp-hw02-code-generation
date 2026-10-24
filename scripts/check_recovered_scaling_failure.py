"""Verify failed training separately from successful exact B008 reproduction."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
root=ROOT/'artifacts/kaggle/b009-b010-v1'
expected_path=ROOT/'reports/evidence/b009_b010_prelaunch_manifest.json'
expected=json.loads(expected_path.read_text());terminal=json.loads((root/'session_result.json').read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert not terminal['complete'] and not terminal['training_complete'] and terminal['inference_complete']
assert terminal['phase']=='finished' and terminal['failure_type']=='RuntimeError'
assert terminal['iterations']=={'b009':100,'b010':100}
assert all(l['exit']==0 and l['gracefully_stopped'] for l in terminal['lanes'])
assert (root/'experiment-manifest.json').read_bytes()==expected_path.read_bytes()
assert sha(root/'run_driver.py')==expected['driver_sha256']==terminal['driver_sha256']
assert sha(root/'training_source.bin')==expected['source_sha256']==terminal['training_source_sha256']
assert terminal['cache_manifest_sha256']==expected['cache_manifest_sha256']
checked=[];not_downloaded=[]
for name,digest in terminal['artifact_sha256'].items():
    assert not Path(name).is_absolute() and '..' not in Path(name).parts
    path=root/name
    if not path.exists():not_downloaded.append(name);continue
    assert not path.is_symlink() and sha(path)==digest,name
    checked.append(name)
runs={}
for lane in expected['lanes']:
    name=lane['id'].lower();run=root/f'run-{name}'
    for filename in ('model.pt','metrics.json','run_config.json','train.jsonl','validation.json'):
        assert str((run/filename).relative_to(root)) in checked
    m=json.loads((run/'metrics.json').read_text());cfg=json.loads((run/'run_config.json').read_text())
    assert m['completed_steps']==m['optimizer_updates']==100 and m['skipped_updates']==0
    assert m['selected_step']==100 and cfg['max_steps']==40000 and cfg['seed']==42
    for key,value in lane['model_config'].items():assert cfg['config'][key]==value
    records=[json.loads(line) for line in (run/'train.jsonl').read_text().splitlines()]
    train=next(r for r in records if r['event']=='train');assert train['step']==100
    assert records[-1]['event']=='finished' and records[-1]['stop_requested']
    runs[name]={'completed_iterations':100,'training_seconds':m['training_elapsed_seconds'],
        'first100_seconds':train['elapsed_seconds'],'first100_steps_per_second':train['steps_per_second'],
        'primary_ce':m['best_validation_loss'],'stop_requested':True}
executed=root/'canonical-b008/executed.ipynb';result=json.loads((root/'canonical-b008/executed.json').read_text())
assert result['complete'] and result['executed_code_cells']==3
assert result['input_sha256']==expected['canonical_notebook_sha256']==sha(ROOT/'reports/evidence/selected_beam_inference_input.ipynb')
assert result['output_sha256']==sha(executed)
input_nb=json.loads((ROOT/'reports/evidence/selected_beam_inference_input.ipynb').read_text());output_nb=json.loads(executed.read_text())
assert [(c['cell_type'],c['source'],c['id']) for c in input_nb['cells']]==[(c['cell_type'],c['source'],c['id']) for c in output_nb['cells']]
cells=[c for c in output_nb['cells'] if c['cell_type']=='code']
assert [c['execution_count'] for c in cells]==[1,2,3] and all(c['outputs'] for c in cells)
assert all(not any(o['output_type']=='error' for o in c['outputs']) for c in cells)
for n in ['canonical-b008/executed.ipynb','canonical-b008/executed.json','canonical-inference/test_timing.json','canonical-inference/test_predictions.jsonl']:
    assert n in checked
bundle=ROOT/'artifacts/beam_candidate_b008.zip';assert sha(bundle)==expected['canonical_bundle_sha256']
with zipfile.ZipFile(bundle) as z:
    manifest=json.loads(z.read('bundle_manifest.json'));assert z.testzip() is None
    assert manifest['inference']=={'decoder':'beam','batch_size':256,'beam_width':4,'length_penalty':.6}
    for n,digest in manifest['file_sha256'].items():assert hashlib.sha256(z.read(n)).hexdigest()==digest
    model_sha=hashlib.sha256(z.read('model.pt')).hexdigest()
timing=json.loads((root/'canonical-inference/test_timing.json').read_text())
assert timing['bundle_sha256']==sha(bundle) and timing['checkpoint_sha256']==model_sha
assert timing['inference_source_sha256']==manifest['inference_source_sha256']
assert timing['training_source_sha256']==manifest['training_source_sha256']
assert timing['rows']==1919 and timing['gpu_count']==1 and 'T4' in timing['gpu']
assert timing['seconds']<=180 and timing['meets_180_seconds_on_one_T4']
assert timing['decoder']=='beam' and timing['beam_width']==4 and timing['length_penalty']==.6 and timing['batch_size']==256
new=[json.loads(line) for line in (root/'canonical-inference/test_predictions.jsonl').read_text().splitlines()]
old=[json.loads(line) for line in (ROOT/'artifacts/colab-decoding-checks/decoding-b008/beam_test_predictions.jsonl').read_text().splitlines()]
assert [r['row_index'] for r in new]==list(range(1919)) and all(isinstance(r['code'],str) for r in new)
assert [r['row_index'] for r in old]==list(range(1919))
quota=json.loads((ROOT/'reports/evidence/scaling_failure_quota.json').read_text())
assert quota['gpuQuota']['timeReserved']=='0s'
charged=float(quota['gpuQuota']['timeUsed'][:-1])-7033.658
report={'training_complete':False,'runs':runs,'verified_artifacts':checked,'not_downloaded':not_downloaded,
    'canonical_b008':{'verified':True,'input_sha256':result['input_sha256'],'executed_sha256':sha(executed),
        'bundle_sha256':sha(bundle),'seconds':timing['seconds'],'rows':1919,'capped_rows':timing['capped_rows'],
        'changed_predictions_vs_prior_colab_beam':sum(a['code']!=b['code'] for a,b in zip(new,old))},
    'charged_session_seconds':charged,'remaining_quota_hours':(108000-float(quota['gpuQuota']['timeUsed'][:-1]))/3600,
    'note':'Failed100step training comparison; successful exact B008 beam reproduction verified separately. No architecture gain claim.'}
out=ROOT/'reports/evidence/scaling_failure_recovery_checks.json';out.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ('verified_artifacts','not_downloaded')},indent=2))
print('Verified artifacts:',len(checked),'explicitly not downloaded:',not_downloaded)
