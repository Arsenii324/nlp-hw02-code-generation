"""Reject corruption even if an outer staging checksum was regenerated."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import zipfile
spec=importlib.util.spec_from_file_location('pack','scripts/package_inference.py');pack=importlib.util.module_from_spec(spec);spec.loader.exec_module(pack)
assert hasattr(pack,'validate_bundle'),'Missing preallocation bundle validator'
path=Path('artifacts/beam_candidate_b008.zip');pack.validate_bundle(path,require_beam=True)
with tempfile.TemporaryDirectory() as d:
    bad=Path(d)/'bad.zip'
    with zipfile.ZipFile(path) as old:
        files={n:old.read(n) for n in old.namelist()};manifest=json.loads(files['bundle_manifest.json'])
    addon=io.BytesIO()
    with zipfile.ZipFile(addon,'w') as z:z.writestr('src/nlp_hw02/beam.py',b'wrong, wrapper missing')
    files['inference_addon.zip']=addon.getvalue()
    manifest['file_sha256']['inference_addon.zip']=hashlib.sha256(addon.getvalue()).hexdigest()
    files['bundle_manifest.json']=json.dumps(manifest).encode()
    with zipfile.ZipFile(bad,'w') as z:
        for name,data in files.items():z.writestr(name,data)
    try:pack.validate_bundle(bad,require_beam=True)
    except AssertionError:pass
    else:raise AssertionError('Rehashed damaged addon accepted')
print('PASS complete staged bundle validation; rehashed missing/tampered addon rejected before launch')
