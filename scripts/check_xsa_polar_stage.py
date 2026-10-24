"""Execute the real learned180k stage, validate parent guards; no allocation."""
import contextlib,hashlib,io,json,sys,tempfile,zipfile
from pathlib import Path
from unittest.mock import patch
import build_kaggle_successor as builder

origin=Path.cwd()
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);(root/'artifacts/kaggle').mkdir(parents=True);(root/'kaggle').mkdir();(root/'reports/evidence').mkdir(parents=True)
 for name in ('b011-b012-v1','xsa-retry-tools','tools-training-180k-20261003'):
  (root/'artifacts/kaggle'/name).symlink_to((origin/'artifacts/kaggle'/name).resolve(),target_is_directory=True)
 for name in ('scripts','src'):(root/name).symlink_to(origin/name,target_is_directory=True)
 parent=origin/'artifacts/kaggle/b027-b028-v1/session_result.json'
 argv=['builder','xsa-polar-resume','--parent-terminal',str(parent),'--tag','fixture']
 with patch.object(builder,'ROOT',root),patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):builder.main()
 tools=root/'artifacts/kaggle/tools-xsa-polar-240k-fixture';kernel=root/'kaggle/xsa-polar-240k-fixture'
 manifest=json.loads((tools/'experiment-manifest.json').read_text());metadata=json.loads((kernel/'kernel-metadata.json').read_text())
 assert (kernel/'experiment-manifest.json').read_bytes()==(tools/'experiment-manifest.json').read_bytes()
 assert [(s['id'],s['resume_from_step'],s['iterations'],s['optimizer'],s['allow_orthogonalizer_change']) for s in manifest['lanes']]==[('B031',180000,240000,'muon_grouped',False),('B032',180000,240000,'muon_grouped_pe',True)]
 assert all(s['model_config']['learnable_xsa'] and not s['model_config']['exclusive_self_attention'] for s in manifest['lanes'])
 assert manifest['resume_parent']['run']=='b028' and manifest['resume_parent']['step']==180000
 assert manifest['new_iterations_per_lane']==60000 and manifest['GPU_sessions_concurrent_limit']==1
 assert manifest['parameter_counts']=={'b031':17524256,'b032':17524256}
 assert metadata['kernel_sources']==['arsen4ikvar/nlp-hw02-prepare-v2','arsen4ikvar/nlp-hw02-training-180k-20261003']
 assert metadata['is_private'] and metadata['enable_gpu'] and not metadata['enable_tpu'] and not metadata['enable_internet']
 assert metadata['docker_image_pinning_type']=='original' and metadata['machine_shape']=='NvidiaTeslaT4'
 assert metadata['title'].lower().replace(' ','-')==metadata['id'].split('/')[1]
 assert builder.sha(kernel/'run.py')==manifest['driver_sha256'];compile((kernel/'run.py').read_text(),'driver','exec')
 assert builder.sha(tools/'training_source.bin')==manifest['source_sha256']
 with zipfile.ZipFile(tools/'training_source.bin') as archive:
  assert archive.testzip() is None and len(archive.namelist())==len(set(archive.namelist()))
  assert set(archive.namelist())==set(manifest['source_member_sha256'])
  for name,digest in manifest['source_member_sha256'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==digest
 assert not list(tools.rglob('*.pt')) and not list(tools.rglob('*.npy')) and not list(tools.rglob('*.npz'))
 before={p.name:builder.sha(p) for p in tools.iterdir()}
 with patch.object(builder,'ROOT',root),patch.object(sys,'argv',argv):
  try:builder.main()
  except ValueError:pass
  else:raise AssertionError('Overwrote immutable stage')
 assert before=={p.name:builder.sha(p) for p in tools.iterdir()}
 # Reseal tiny metadata copies, retaining the actual terminal's remote last-state pins.
 fake=root/'parent';directory=fake/'run-b028';directory.mkdir(parents=True)
 terminal=json.loads(parent.read_text());original=json.loads((parent.parent/'run-b028/run_config.json').read_text())
 metrics=(parent.parent/'run-b028/metrics.json').read_bytes();(directory/'metrics.json').write_bytes(metrics)
 rejections=0
 with patch.object(builder,'ROOT',root):
  for bad in [None,{**original,'seed':43},{**original,'torch':'wrong'},
              {**original,'schedule':'restart'},
              {**original,'config':{**original['config'],'learnable_xsa':False}},
              {**original,'optimizer':{**original['optimizer'],'orthogonalizer':'polar_express_v5_5'}}]:
   (directory/'run_config.json').write_text(json.dumps(original if bad is None else bad))
   altered={**terminal,'artifact_sha256':{**terminal['artifact_sha256'],'run-b028/run_config.json':builder.sha(directory/'run_config.json')}}
   if bad is None:altered['iterations']={**terminal['iterations'],'b028':179000}
   (fake/'session_result.json').write_text(json.dumps(altered))
   try:builder.learned_180k_parent_record(fake/'session_result.json')
   except ValueError:rejections+=1
   else:raise AssertionError('Accepted bad parent recipe/count')
 assert rejections==6
print(json.dumps({'event':'xsa_polar_stage_checks_passed','actual_builder':True,'same_learned180k_parent240k_lanes':True,'private_pinned_single_parent_mount':True,'resealed_parent_rejections':rejections,'source_crc_members_sha':True,'immutable_overwrite_rejected':True,'CUDA_verified':False}))
