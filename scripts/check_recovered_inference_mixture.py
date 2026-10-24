"""Hash/row/provenance verification for bounded inference-only probe recovery."""
import argparse,hashlib,json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root,manifest_path):
 root=Path(root);manifest_path=Path(manifest_path);manifest=json.loads(manifest_path.read_text());terminal=json.loads((root/'session_result.json').read_text())
 assert terminal['complete'] is True and terminal['manifest_sha256']==sha(manifest_path)
 assert sha(root/'inference_probe_manifest.json')==sha(manifest_path)
 assert sha(root/'run_driver.py')==manifest['driver_sha256']
 assert sha(root/'inference_source.bin')==manifest['tools_sha256']['inference_source.bin']
 for filename,digest in terminal['artifact_sha256'].items():
  path=Path(filename);assert not path.is_absolute() and '..' not in path.parts
  assert (root/path).is_file() and sha(root/path)==digest,filename
 assert not list(root.rglob('*.pt')) and not list(root.rglob('*.npy')) and not list(root.rglob('*.npz'))
 result={};anchor=manifest.get('validation_parent','b019')
 for lane,spec in manifest['lanes'].items():
  directory=root/lane;report=json.loads((directory/'probe_result.json').read_text())
  assert terminal['lanes'][lane]['exit']==0 and terminal['lanes'][lane]['report']==report
  assert report['complete'] is True and report['lane']==lane and report['decoder']==spec
  assert report['gpu_count']==1 and 'T4' in report['gpu'] and report['torch']==manifest['torch'] and report['cuda']==manifest['cuda']
  assert report['selection_uses_reference_or_numeric_interpreter'] is False
  expected={name:manifest['parents'][name]['files'][f'run-{name}/model.pt'] for name in spec['parents']};assert report['checkpoint_sha256']==expected
  assert report['source_sha256']==manifest['tools_sha256']['inference_source.bin']
  assert report['tokenizer_sha256']==manifest['parents'][anchor]['files'][f'run-{anchor}/tokenizer.json']
  assert report['validation_sha256']==sha(directory/'validation.json')==manifest['parents'][anchor]['files'][f'run-{anchor}/validation.json']
  validation=json.loads((directory/'validation.json').read_text());rows=[json.loads(s) for s in (directory/'validation_predictions.jsonl').read_text().splitlines()]
  assert len(rows)==1792 and report['validation_timing']['rows']==1792
  excluded=set(manifest['validation_generation_indices'])
  expected=[(r['question'],r['code']) for i,r in enumerate(validation) if i not in excluded]
  assert [(r['question'],r['code']) for r in rows]==expected and all(isinstance(r['prediction'],str) for r in rows)
  assert len({r['question'] for r in rows})==1792
  test=[json.loads(s) for s in (directory/'test_predictions.jsonl').read_text().splitlines()];timing=json.loads((directory/'test_timing.json').read_text());assert timing==report['test_timing']
  assert [x['row_index'] for x in test]==list(range(1919)) and all(isinstance(x['code'],str) for x in test)
  assert timing['gpu_count']==1 and 'T4' in timing['gpu'] and timing['rows']==1919 and timing['test_parquet_sha256']==manifest['tools_sha256']['test.parquet']
  assert 0<timing['seconds']<540 and timing['meets_180_seconds_on_one_T4']==(timing['seconds']<=180)
  for t in (report['validation_timing'],timing):
   assert all(t[k]==spec[v] for k,v in [('batch_size','batch_size'),('beam_width','beam_width'),('length_penalty','length_penalty')])
   assert t['max_new_tokens']==192 and 0<=t['capped_rows']<=t['rows']
  result[lane]={'validation_seconds':report['validation_timing']['seconds'],'test_seconds':timing['seconds'],'test_capped_rows':timing['capped_rows'],'meets_180':timing['meets_180_seconds_on_one_T4'],'peak_allocated_bytes':timing['peak_allocated_bytes']}
 return {'terminal_sha256':sha(root/'session_result.json'),'manifest_sha256':sha(manifest_path),'verified_artifacts':len(terminal['artifact_sha256']),'lanes':result,'official_pass_at_1':False,'optimizer_or_arrays_recovered':False,'canonical_notebook_proof':False,'selected_bundle_changed':False}


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args();record=verify(a.root,a.manifest);Path(a.output).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
