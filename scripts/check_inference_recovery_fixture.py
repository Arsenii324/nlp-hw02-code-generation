"""Require exact pilot exclusion/order and corruption rejection on fake artifacts."""
import hashlib,json,tempfile
from pathlib import Path
from check_recovered_inference_mixture import verify

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(prefix='hw02-inference-recovery-') as folder:
 root=Path(folder);validation=[{'question':str(i),'code':'def solve():\n return 1'} for i in range(2048)];raw=json.dumps(validation)
 manifest={'driver_sha256':hashlib.sha256(b'driver').hexdigest(),'tools_sha256':{'inference_source.bin':hashlib.sha256(b'source').hexdigest(),'test.parquet':'test'},'torch':'test-torch','cuda':'test-cuda','parents':{'b019':{'files':{'run-b019/model.pt':'ckpt','run-b019/tokenizer.json':'tokenizer','run-b019/validation.json':hashlib.sha256(raw.encode()).hexdigest()}}},'validation_generation_indices':list(range(256)), 'lanes':{}}
 (root/'run_driver.py').write_bytes(b'driver');(root/'inference_source.bin').write_bytes(b'source')
 terminal={'complete':True,'lanes':{}}
 for lane in ['b025','b026']:
  spec={'parents':['b019'],'batch_size':256,'beam_width':4,'length_penalty':.6};manifest['lanes'][lane]=spec;out=root/lane;out.mkdir();(out/'validation.json').write_text(raw)
  timing={'seconds':40.,'rows':1919,'gpu_count':1,'gpu':'Tesla T4','test_parquet_sha256':'test','meets_180_seconds_on_one_T4':True,'peak_allocated_bytes':1,'batch_size':256,'beam_width':4,'length_penalty':.6,'max_new_tokens':192,'capped_rows':0}
  valtime={**timing,'rows':1792};report={'complete':True,'lane':lane,'decoder':spec,'gpu_count':1,'gpu':'Tesla T4','torch':'test-torch','cuda':'test-cuda','selection_uses_reference_or_numeric_interpreter':False,'checkpoint_sha256':{'b019':'ckpt'},'source_sha256':manifest['tools_sha256']['inference_source.bin'],'tokenizer_sha256':'tokenizer','validation_sha256':sha(out/'validation.json'),'validation_timing':valtime,'test_timing':timing}
  (out/'probe_result.json').write_text(json.dumps(report));(out/'test_timing.json').write_text(json.dumps(timing));terminal['lanes'][lane]={'exit':0,'report':report}
  (out/'validation_predictions.jsonl').write_text(''.join(json.dumps({**r,'prediction':'def solve():\n return 1'})+'\n' for r in validation[256:]))
  (out/'test_predictions.jsonl').write_text(''.join(json.dumps({'row_index':i,'code':'def solve():\n return 1'})+'\n' for i in range(1919)))
 mp=root/'inference_probe_manifest.json';mp.write_text(json.dumps(manifest));terminal['manifest_sha256']=sha(mp)
 def seal():
  terminal['artifact_sha256']={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='session_result.json'}
  (root/'session_result.json').write_text(json.dumps(terminal))
 seal();assert set(verify(root,mp)['lanes'])=={'b025','b026'}
 predictions=root/'b025/validation_predictions.jsonl';original=predictions.read_text();lines=original.splitlines()
 for kind,bad in [('order',list(reversed(lines))),('pilot',[json.dumps({**validation[0],'prediction':'def solve():\n return 1'})]+lines[1:])]:
  predictions.write_text('\n'.join(bad)+'\n');seal()
  try:verify(root,mp)
  except AssertionError:pass
  else:raise AssertionError(kind+' mismatch accepted despite resealed artifact hash')
 predictions.write_text(original);seal();predictions.write_text(original+'corrupt\n')
 try:verify(root,mp)
 except AssertionError:pass
 else:raise AssertionError('corrupt downloaded prediction accepted')
 print('inference_recovery_order_pilot_corruption_checks_passed')
