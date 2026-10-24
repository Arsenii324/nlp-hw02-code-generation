"""Catch stale B019 tokenizer anchor or ordered-row errors in new recovery."""
import json,tempfile,shutil,hashlib
from pathlib import Path
from check_recovered_inference_mixture import verify
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(prefix='hw02-new-inference-recovery-') as td:
 root=Path(td);old=ROOT/'artifacts/kaggle/b025-b026-v1'
 for name in ['run_driver.py','inference_source.bin']:shutil.copyfile(old/name,root/name)
 for before,after in [('b025','b029'),('b026','b030')]:shutil.copytree(old/before,root/after)
 m=json.loads((old/'inference_probe_manifest.json').read_text());m['validation_parent']='b027';m['parents']={new:{**m['parents'][prev],'run':new,'files':{n.replace('run-'+prev,'run-'+new):h for n,h in m['parents'][prev]['files'].items()}} for prev,new in [('b019','b027'),('b024','b028')]};m['lanes']={new:{**m['lanes'][prev],'parents':['b027','b028'] if new=='b029' else ['b028'],'mechanism':'equal_token_probability_ensemble' if new=='b029' else 'learned_wider_beam'} for prev,new in [('b025','b029'),('b026','b030')]}
 terminal={'complete':True,'lanes':{}}
 for lane in ['b029','b030']:
  p=root/lane/'probe_result.json';r=json.loads(p.read_text());r['lane']=lane;r['decoder']=m['lanes'][lane];r['checkpoint_sha256']={name:m['parents'][name]['files'][f'run-{name}/model.pt'] for name in m['lanes'][lane]['parents']};p.write_text(json.dumps(r));terminal['lanes'][lane]={'exit':0,'report':r}
 mp=root/'inference_probe_manifest.json';mp.write_text(json.dumps(m));terminal['manifest_sha256']=sha(mp)
 def seal():
  terminal['artifact_sha256']={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='session_result.json'}
  (root/'session_result.json').write_text(json.dumps(terminal))
 seal();assert set(verify(root,mp)['lanes'])=={'b029','b030'}
 path=root/'b030/validation_predictions.jsonl';original=path.read_bytes();rows=path.read_text().splitlines();path.write_text('\n'.join(reversed(rows))+'\n');seal()
 try:verify(root,mp)
 except AssertionError:pass
 else:raise AssertionError('Reorderedrows accepted despite resealedhashes')
 path.write_bytes(original);seal();path=root/'b030/probe_result.json';r=json.loads(path.read_text());r['checkpoint_sha256']={'b027':m['parents']['b027']['files']['run-b027/model.pt']};path.write_text(json.dumps(r));terminal['lanes']['b030']['report']=r;seal()
 try:verify(root,mp)
 except AssertionError:pass
 else:raise AssertionError('Wrong learnedmodel role accepted')
print('PASS new anchor/lane recovery / wrongmodel and resealedroworder rejected')
