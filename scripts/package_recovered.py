"""Create a portable selected-model bundle from a completed, hashed Kaggle run."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',default='artifacts/selected_bundle.zip')
    args=parser.parse_args()
    root=Path(args.root); run=root/f'run-{args.run}'
    terminal=json.loads((root/'session_result.json').read_text())
    assert terminal['complete'] and terminal['inference_complete'],'Only package measured completed runs'
    timing=root/f'inference-{args.run}/test_timing.json'
    t=json.loads(timing.read_text()); assert t['gpu_count']==1 and t['meets_180_seconds_on_one_T4']
    files={name:run/name for name in ['model.pt','tokenizer.json','run_config.json','data_stats.json',
        'metrics.json','train.jsonl','validation.json','validation_predictions.jsonl']}
    files.update({'test_timing.json':timing,'test_predictions.jsonl':timing.with_name('test_predictions.jsonl'),
                  'baseline_source.zip':root/'training_source.bin','session_result.json':root/'session_result.json'})
    for name,path in files.items():
        assert path.is_file() and not path.is_symlink()
        if name!='session_result.json':
            assert hashlib.sha256(path.read_bytes()).hexdigest()==terminal['artifact_sha256'][str(path.relative_to(root))]
    predictions=[json.loads(s) for s in files['test_predictions.jsonl'].read_text().splitlines()]
    assert [r['row_index'] for r in predictions]==list(range(1919))
    record={'selected_run':args.run,'kernel_outputs':str(root),'training_source_sha256':terminal['training_source_sha256'],
        'file_sha256':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()},
        'note':'Portable from-scratch model and internal predictions; no submission schema or official pass@1 claim.'}
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,path in files.items(): archive.write(path,name)
        archive.writestr('bundle_manifest.json',json.dumps(record,indent=2))
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None and set(archive.namelist())==set(files)|{'bundle_manifest.json'}
        for name,digest in record['file_sha256'].items(): assert hashlib.sha256(archive.read(name)).hexdigest()==digest
    print(json.dumps({'output':str(output),'bytes':output.stat().st_size,
        'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),**record},indent=2))


if __name__=='__main__': main()
