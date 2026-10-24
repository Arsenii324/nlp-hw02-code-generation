"""Add a hashed inference addon to a verified bundle without altering training files."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
TRAINING_MEMBERS={'model.pt','tokenizer.json','run_config.json','data_stats.json','metrics.json',
    'train.jsonl','validation.json','validation_predictions.jsonl','test_timing.json','test_predictions.jsonl',
    'baseline_source.zip','session_result.json'}


def sha(data):return hashlib.sha256(data).hexdigest()


def validate_bundle(path,require_beam=False):
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist();assert archive.testzip() is None and len(names)==len(set(names))
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
        manifest=json.loads(archive.read('bundle_manifest.json'))
        assert set(names)==set(manifest['file_sha256'])|{'bundle_manifest.json'}
        assert TRAINING_MEMBERS<=set(names)
        for name,digest in manifest['file_sha256'].items():assert sha(archive.read(name))==digest,name
        assert sha(archive.read('baseline_source.zip'))==manifest['training_source_sha256']
        with zipfile.ZipFile(io.BytesIO(archive.read('baseline_source.zip'))) as source:
            assert source.testzip() is None and len(source.namelist())==len(set(source.namelist()))
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in source.namelist())
        decoding=manifest.get('inference',{'decoder':'greedy','batch_size':256})
        if require_beam:assert decoding=={'decoder':'beam','batch_size':256,'beam_width':4,'length_penalty':.6}
        if decoding['decoder']=='beam':
            assert {'inference_addon.zip','inference_validation.json','inference_test_timing.json','inference_test_predictions.jsonl'}<=set(names)
            with zipfile.ZipFile(io.BytesIO(archive.read('inference_addon.zip'))) as addon:
                expected=manifest['inference_source_sha256']
                assert addon.testzip() is None and len(addon.namelist())==2
                assert set(addon.namelist())==set(expected)=={'src/nlp_hw02/beam.py','src/nlp_hw02/beam_inference.py'}
                for name,digest in expected.items():assert sha(addon.read(name))==digest,name
    return manifest


def build_inference_bundle(original,output,profile_root,validation_report,validation_label):
    original=Path(original);output=Path(output);profile_root=Path(profile_root);validation_report=Path(validation_report)
    with zipfile.ZipFile(original) as archive:
        assert archive.testzip() is None and set(archive.namelist())==TRAINING_MEMBERS|{'bundle_manifest.json'}
        assert len(archive.namelist())==len(TRAINING_MEMBERS)+1
        files={name:archive.read(name) for name in TRAINING_MEMBERS}
        manifest=json.loads(archive.read('bundle_manifest.json'))
    assert set(manifest['file_sha256'])==TRAINING_MEMBERS
    for name,digest in manifest['file_sha256'].items():assert sha(files[name])==digest,name
    assert sha(files['baseline_source.zip'])==manifest['training_source_sha256']
    profile=json.loads((profile_root/'decoding_profile.json').read_text())
    assert profile['checkpoint_sha256']==sha(files['model.pt']) and profile['gpu_count']==1 and 'T4' in profile['gpu']
    beam=ROOT/'src/nlp_hw02/beam.py';wrapper=ROOT/'src/nlp_hw02/beam_inference.py'
    assert not beam.is_symlink() and not wrapper.is_symlink()
    assert sha(beam.read_bytes())==profile['beam_source_sha256'],'Beam implementation differs from measured source'
    assert (profile_root/'beam4-alpha06/validation.json').read_bytes()==files['validation.json']
    validation=json.loads(validation_report.read_text())
    score=validation['runs'][validation_label]
    prediction_path=profile_root/'beam4-alpha06/validation_predictions.jsonl'
    assert score['prediction_sha256']==sha(prediction_path.read_bytes())
    assert score['rows']==1792 and profile['paths']['beam4-alpha06']['rows']==1792
    timing_path=profile_root/'beam_test_timing.json';prediction_path=profile_root/'beam_test_predictions.jsonl'
    timing=json.loads(timing_path.read_text())
    expected_test=next(r['sha256'] for r in json.loads((ROOT/'assignment/dataset_manifest.json').read_text())['files']
                       if r['path']=='data/test/test.parquet')
    assert timing['gpu_count']==1 and 'T4' in timing['gpu'] and timing['test_parquet_sha256']==expected_test
    assert timing['rows']==1919 and timing['meets_180_seconds_on_one_T4'] and timing['seconds']<=180
    assert timing['beam_width']==4 and timing['length_penalty']==.6 and timing['batch_size']==256
    rows=[json.loads(s) for s in prediction_path.read_text().splitlines()]
    assert [r['row_index'] for r in rows]==list(range(1919))
    addon=io.BytesIO()
    with zipfile.ZipFile(addon,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('src/nlp_hw02/beam.py',beam.read_bytes())
        archive.writestr('src/nlp_hw02/beam_inference.py',wrapper.read_bytes())
    files.update({'inference_addon.zip':addon.getvalue(),'inference_validation.json':validation_report.read_bytes(),
        'inference_test_timing.json':timing_path.read_bytes(),'inference_test_predictions.jsonl':prediction_path.read_bytes()})
    manifest.update({'file_sha256':{name:sha(data) for name,data in files.items()},
        'inference':{'decoder':'beam','batch_size':256,'beam_width':4,'length_penalty':.6},
        'inference_source_sha256':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in (beam,wrapper)},
        'inference_base_bundle_sha256':sha(original.read_bytes()),'inference_validation_label':validation_label,
        'note':'Training files/source are unchanged. Separate inference addon uses validation-selected fixed beam; exact new notebook GPU execution remains required.'})
    output.parent.mkdir(parents=True,exist_ok=True);temporary=output.with_suffix(output.suffix+'.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,data in files.items():archive.writestr(name,data)
        archive.writestr('bundle_manifest.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None and set(archive.namelist())==set(files)|{'bundle_manifest.json'}
        for name,digest in manifest['file_sha256'].items():assert sha(archive.read(name))==digest
    os.replace(temporary,output)
    strict=score['fully_supported_consensus']
    return {'output':str(output),'bytes':output.stat().st_size,'sha256':sha(output.read_bytes()),
        'validation_agreeing':strict['agreeing'],'validation_scorable':strict['scorable'],
        'test_seconds':timing['seconds'],'test_rows':timing['rows'],**manifest}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--profile-root',required=True);parser.add_argument('--validation-report',required=True)
    parser.add_argument('--validation-label',required=True);args=parser.parse_args()
    print(json.dumps(build_inference_bundle(args.input,args.output,args.profile_root,args.validation_report,args.validation_label),indent=2))


if __name__=='__main__':main()
