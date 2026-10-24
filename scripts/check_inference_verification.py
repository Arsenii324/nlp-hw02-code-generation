"""Verify private staging and exact execution gates using preserved real evidence."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

ROOT=Path(__file__).resolve().parents[1]
assert importlib.util.find_spec('build_kaggle_inference_verification'), 'Missing private inference staging builder'
assert importlib.util.find_spec('kaggle_inference_verification'), 'Missing exact execution verification driver'
from build_kaggle_inference_verification import stage
from kaggle_inference_verification import validate_execution

with tempfile.TemporaryDirectory() as directory:
    base=Path(directory)
    manifest=stage(ROOT/'artifacts/beam_candidate_b008.zip',base/'tools',base/'kernel')
    expected={'selected_bundle.bin','canonical_inference.ipynb','execute_inference_notebook.py',
        'kaggle_eval.py','test.parquet','verification_checksums.json','verification_manifest.json','dataset-metadata.json'}
    assert {p.name for p in (base/'tools').iterdir()}==expected
    metadata=json.loads((base/'kernel/kernel-metadata.json').read_text())
    assert metadata['is_private'] and not metadata['enable_internet'] and metadata['machine_shape']=='NvidiaTeslaT4'
    assert metadata['kernel_sources']==[] and len(metadata['dataset_sources'])==1
    assert manifest['bundle_sha256']=='97d65b75b0b6141a0a0119b50de55269e0095187bd89695323012064d9d55b62'
    assert (base/'tools/canonical_inference.ipynb').read_bytes()==(ROOT/'submit/inference.ipynb').read_bytes()
    try:stage(ROOT/'artifacts/beam_candidate_b008.zip',base/'tools',base/'kernel')
    except AssertionError:pass
    else:raise AssertionError('Existing staging inputs must not be overwritten')
    executed=base/'executed';executed.mkdir();output=base/'output';output.mkdir()
    shutil.copyfile(ROOT/'reports/evidence/selected_beam_inference_output.ipynb',executed/'executed.ipynb')
    shutil.copyfile(ROOT/'reports/evidence/selected_beam_inference_execution.json',executed/'executed.json')
    shutil.copyfile(ROOT/'reports/evidence/selected_kaggle_beam_test_timing.json',output/'test_timing.json')
    shutil.copyfile(ROOT/'artifacts/selected_kaggle_beam_test_predictions.jsonl',output/'test_predictions.jsonl')
    result=validate_execution(executed,output,manifest)
    assert result['rows']==1919 and result['seconds']<180
    path=output/'test_timing.json';original=path.read_bytes()
    for key,value in [('checkpoint_sha256','wrong'),('bundle_sha256','wrong'),('seconds',181),
        ('gpu_count',2),('beam_width',8),('decoder','greedy'),('rows',1918)]:
        record=json.loads(original);record[key]=value;path.write_text(json.dumps(record))
        try:validate_execution(executed,output,manifest)
        except AssertionError:pass
        else:raise AssertionError(f'Invalid timing field accepted: {key}')
    path.write_bytes(original)
    notebook=executed/'executed.ipynb';original=notebook.read_bytes()
    record=json.loads(original);next(c for c in record['cells'] if c['cell_type']=='code')['execution_count']=None
    notebook.write_text(json.dumps(record))
    try:validate_execution(executed,output,manifest)
    except AssertionError:pass
    else:raise AssertionError('Tampered notebook accepted')
    notebook.write_bytes(original)
print('INFERENCE_VERIFICATION_CHECKS_OK')
