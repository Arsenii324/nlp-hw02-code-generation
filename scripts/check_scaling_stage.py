"""Check actual staged archives, notebook cells, driver and CPU model shapes."""
import hashlib
import json
from pathlib import Path
import zipfile
import nbformat
import torch
from nlp_hw02.model import Seq2Seq,ModelConfig

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(2)
root=Path('artifacts/kaggle/scaling-tools');manifest=json.loads((root/'experiment-manifest.json').read_text())
assert manifest['driver_sha256']==sha(Path('kaggle/scaling/run.py'))
# This is the preserved B009/B010 stage, not the now-repaired working driver.
assert manifest==json.loads(Path('reports/evidence/b009_b010_prelaunch_manifest.json').read_text())
assert (root/'experiment-manifest.json').read_bytes()==Path('kaggle/scaling/experiment-manifest.json').read_bytes()
checks=json.loads((root/'training_checksums.json').read_text())
for name,digest in checks.items():assert sha(root/name)==digest
assert manifest['source_sha256']==checks['training_source.bin']
assert manifest['canonical_notebook_sha256']==checks['canonical_inference.ipynb']==sha(Path('submit/inference.ipynb'))
assert manifest['canonical_bundle_sha256']==checks['selected_bundle.bin']
nb=nbformat.read(root/'canonical_inference.ipynb',as_version=4);nbformat.validate(nb)
cells=[c for c in nb.cells if c.cell_type=='code'];assert len(cells)==3
for i,c in enumerate(cells):compile(c.source,f'canonical-cell{i}','exec');assert not c.outputs
assert "+'\\n')" in cells[-1].source
assert "CUDA_VISIBLE_DEVICES" in cells[0].source and "inference_source_sha256" in cells[0].source
with zipfile.ZipFile(root/'selected_bundle.bin') as z:
    assert z.testzip() is None;m=json.loads(z.read('bundle_manifest.json'));assert m['inference']['decoder']=='beam'
    for n,digest in m['file_sha256'].items():assert hashlib.sha256(z.read(n)).hexdigest()==digest
    with zipfile.ZipFile(root/'training_source.bin') as source:
        assert source.testzip() is None
        assert source.read('src/nlp_hw02/beam.py')==Path('src/nlp_hw02/beam.py').read_bytes()
for lane in manifest['lanes']:
    model=Seq2Seq(ModelConfig(**lane['model_config']))
    count=sum(p.numel() for p in model.parameters());assert count==manifest['parameter_counts'][lane['id'].lower()]
    src=torch.tensor([[4,5,2],[6,2,0]]);target=torch.tensor([[1,5,2],[1,6,2]])
    logits=model(src,target);assert logits.shape==(2,3,2048) and torch.isfinite(logits).all()
    logits.sum().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print(lane['id'],count,'forward/backward finite')
    del model
print('PASS archive checksums/immutable identities, canonical cells, selected addon, actual candidate configs')
