"""Run the real180k builder in tiny isolated staging; no upload or allocation."""
import contextlib,hashlib,io,json,sys,tempfile,zipfile
from pathlib import Path
from unittest.mock import patch
import build_kaggle_successor as builder

origin=Path.cwd()
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);(root/'artifacts/kaggle').mkdir(parents=True);(root/'kaggle').mkdir();(root/'reports/evidence').mkdir(parents=True)
    for name in ['b011-b012-v1','xsa-retry-tools','tools-polar-resume-20261002','tools-xsa-continuation-20261003']:
        (root/'artifacts/kaggle'/name).symlink_to((origin/'artifacts/kaggle'/name).resolve(),target_is_directory=True)
    for name in ('scripts','src'):(root/name).symlink_to(origin/name,target_is_directory=True)
    argv=['builder','long-resume','--ordinary-parent-terminal',str(origin/'artifacts/kaggle/b019-b020-v1/session_result.json'),
          '--learned-parent-terminal',str(origin/'artifacts/kaggle/b023-b024-v1/session_result.json'),'--tag','fixture']
    with patch.object(builder,'ROOT',root),patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):builder.main()
    tools=root/'artifacts/kaggle/tools-training-180k-fixture';kernel=root/'kaggle/training-180k-fixture'
    manifest=json.loads((tools/'experiment-manifest.json').read_text());metadata=json.loads((kernel/'kernel-metadata.json').read_text())
    assert (kernel/'experiment-manifest.json').read_bytes()==(tools/'experiment-manifest.json').read_bytes()
    assert [(s['id'],s['resume_from_step'],s['iterations'],s['optimizer'],s['allow_orthogonalizer_change']) for s in manifest['lanes']]==[('B027',120000,180000,'muon_grouped',False),('B028',120000,180000,'muon_grouped',False)]
    assert [(s['model_config']['exclusive_self_attention'],s['model_config']['learnable_xsa']) for s in manifest['lanes']]==[(False,False),(False,True)]
    assert manifest['resume_parents']['b027']['run']=='b019' and manifest['resume_parents']['b028']['run']=='b024'
    assert manifest['new_iterations_per_lane']==60000 and manifest['GPU_sessions_concurrent_limit']==1
    assert metadata['is_private'] and not metadata['enable_internet'] and metadata['enable_gpu'] and not metadata['enable_tpu']
    assert len(set(metadata['kernel_sources']))==3 and 'polar-resume-20261002' in metadata['kernel_sources'][1] and 'xsa-continuation-20261003' in metadata['kernel_sources'][2]
    assert metadata['docker_image_pinning_type']=='original' and metadata['machine_shape']=='NvidiaTeslaT4'
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
print(json.dumps({'event':'long_stage_checks_passed','actual_builder_executed':True,'independent_parents120k_to180k':True,'source_crc_members_sha':True,'private_one_kernel_no_state_upload':True,'immutable_overwrite_rejected':True}))
