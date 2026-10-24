"""Packaging contract: preserve immutable training data and reject tampering."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import zipfile


def main():
    path=Path('scripts/package_inference.py');assert path.exists(),'Missing inference bundle packager'
    spec=importlib.util.spec_from_file_location('package_inference',path)
    package=importlib.util.module_from_spec(spec);spec.loader.exec_module(package)
    original=Path('artifacts/selected_greedy_b008.zip')
    assert hashlib.sha256(original.read_bytes()).hexdigest()=='f93f25d5196d192bef9017ccf250217a684e9606644d16afd7d3b7ed832c6e52'
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);out=root/'beam.zip'
        record=package.build_inference_bundle(original,out,
            Path('artifacts/colab-decoding-checks/decoding-b008'),
            Path('reports/evidence/colab_remaining_numeric.json'),'B008-beam4')
        with zipfile.ZipFile(original) as old,zipfile.ZipFile(out) as new:
            assert new.testzip() is None
            for name in old.namelist():
                if name!='bundle_manifest.json':assert old.read(name)==new.read(name),name
            manifest=json.loads(new.read('bundle_manifest.json'))
            assert manifest['inference']=={'decoder':'beam','batch_size':256,'beam_width':4,'length_penalty':.6}
            for name,sha in manifest['file_sha256'].items():assert hashlib.sha256(new.read(name)).hexdigest()==sha
            assert manifest['training_source_sha256']=='6248912874660473eb1247b7481af4455eb69dc6e10fcc79895206d3e320a832'
            assert record['validation_agreeing']==645 and record['validation_scorable']==1529
            assert record['test_seconds']<180 and record['test_rows']==1919
            assert not any(n.endswith('.npy') or 'last_training' in n or 'key' in n for n in new.namelist())
        bad=root/'tampered.zip'
        with zipfile.ZipFile(original) as old,zipfile.ZipFile(bad,'w') as new:
            for name in old.namelist():new.writestr(name,b'changed' if name=='model.pt' else old.read(name))
        try:package.build_inference_bundle(bad,root/'bad-output.zip',
            Path('artifacts/colab-decoding-checks/decoding-b008'),
            Path('reports/evidence/colab_remaining_numeric.json'),'B008-beam4')
        except AssertionError:pass
        else:raise AssertionError('Tampered model accepted')
        assert not (root/'bad-output.zip').exists()
    print('PASS immutable training members, hashed separate addon/evidence, full-row timing, tamper rejection')


if __name__=='__main__':main()
