"""Exercise canonical manifest loading on real current and legacy archives."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile
import nbformat
cell=next(c.source for c in nbformat.read('submit/inference.ipynb',as_version=4).cells if c.cell_type=='code')
nodes=[n for n in ast.parse(cell).body if isinstance(n,ast.FunctionDef) and n.name=='read_bundle_manifest']
assert len(nodes)==1,'Missing legacy-aware bundle manifest loader'
namespace={'json':json,'hashlib':hashlib}
exec(compile(ast.Module(body=nodes,type_ignores=[]),'canonical-manifest','exec'),namespace)
read=namespace['read_bundle_manifest']
for path,decoder in [(Path('artifacts/baseline_bundle.zip'),'greedy'),(Path('artifacts/beam_candidate_b008.zip'),'beam')]:
    with zipfile.ZipFile(path) as z:
        m=read(z,hashlib.sha256(path.read_bytes()).hexdigest())
        assert set(z.namelist())==set(m['file_sha256'])|({'bundle_manifest.json'} if 'bundle_manifest.json' in z.namelist() else set())
        assert m.get('inference',{}).get('decoder','greedy')==decoder
        assert m['training_source_sha256']==hashlib.sha256(z.read('baseline_source.zip')).hexdigest()
with zipfile.ZipFile('artifacts/baseline_bundle.zip') as z:
    try:read(z,'0'*64)
    except AssertionError:pass
    else:raise AssertionError('Untrusted manifestless legacy archive accepted')
print('PASS canonical real legacy/current bundle manifests; unknown legacy digest rejected')
