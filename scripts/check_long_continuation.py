"""Reject wrong parent steps/recipes and bind an actual 120k continuation."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import build_kaggle_successor as builder
from nlp_hw02.model import ModelConfig
from successor_driver_patch import patched_driver
from kaggle_resume_inputs import verified_parent,sha


def main():
    assert hasattr(builder,'continuation_parent_record'),'Missing explicit 120k parent validation'
    specifications=[('b019','tools-polar-resume-20261002',False),('b024','tools-xsa-continuation-20261003',True)]
    rejections=0
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        for run,tools,learned in specifications:
            directory=root/run/f'run-{run}';directory.mkdir(parents=True)
            config=asdict(ModelConfig(d_model=384,ff=1536,heads=4,encoder_layers=4,decoder_layers=4,learnable_xsa=learned))
            metadata={'config':config,'torch':'2.10.0+cu128','cuda':'12.8','seed':42,'batch_size':64,
                      'optimizer':{'kind':'hybrid grouped FP16 Muon+AdamW','orthogonalizer':'newton_schulz','muon_ns_steps':5},'schedule':'warmup=200, inverse_sqrt decay'}
            metrics={'completed_steps':120000,'optimizer_updates':119950,'skipped_updates':50,'sampled_examples':7680000}
            for name,value in [('run_config.json',metadata),('metrics.json',metrics)]:
                (directory/name).write_text(json.dumps(value))
            (directory/'model.pt').write_bytes(b'fixture-best');(directory/'last_training.pt').write_bytes(b'fixture-state')
            expected={'source_sha256':run+'source','driver_sha256':run+'driver'}
            toolsdir=root/'artifacts/kaggle'/tools;toolsdir.mkdir(parents=True)
            (toolsdir/'experiment-manifest.json').write_text(json.dumps(expected))
            terminal={'complete':True,'training_complete':True,'iterations':{run:120000},'training_source_sha256':expected['source_sha256'],'driver_sha256':expected['driver_sha256'],'cache_manifest_sha256':'cache','artifact_sha256':{str(p.relative_to(directory.parent)):sha(p) for p in directory.iterdir()}}
            path=directory.parent/'session_result.json';path.write_text(json.dumps(terminal))
            with patch.object(builder,'ROOT',root):
                record,cfg=builder.continuation_parent_record(path,run)
                assert record['step']==120000 and record['run']==run and cfg.learnable_xsa==learned
                assert set(record['files'])=={f'run-{run}/'+n for n in ('last_training.pt','model.pt','run_config.json','metrics.json')}
                assert verified_parent(path.parent,record,'cache',metadata)==directory
                # Changed count/source/completion must fail despite otherwise matching pins.
                for field,value in [('iterations',{run:60000}),('training_complete',False),('training_source_sha256','wrong')]:
                    path.write_text(json.dumps({**terminal,field:value}))
                    try:builder.continuation_parent_record(path,run)
                    except ValueError:rejections+=1
                    else:raise AssertionError('Accepted wrong '+field)
                path.write_text(json.dumps(terminal))
                # Reseal metadata to ensure recipe guards are independent of file hashes.
                for bad in [{**metadata,'seed':43},{**metadata,'schedule':'restart-warmup'},
                            {**metadata,'optimizer':{**metadata['optimizer'],'orthogonalizer':'polar_express'}},
                            {**metadata,'config':{**config,'learnable_xsa':not learned}}]:
                    (directory/'run_config.json').write_text(json.dumps(bad))
                    path.write_text(json.dumps({**terminal,'artifact_sha256':{**terminal['artifact_sha256'],f'run-{run}/run_config.json':sha(directory/'run_config.json')}}))
                    try:builder.continuation_parent_record(path,run)
                    except ValueError:rejections+=1
                    else:raise AssertionError('Accepted resealed wrong recipe')
                (directory/'run_config.json').write_text(json.dumps(metadata));path.write_text(json.dumps(terminal))
                (directory/'last_training.pt').write_bytes(b'corrupt')
                try:verified_parent(path.parent,record,'cache',metadata)
                except ValueError:rejections+=1
                else:raise AssertionError('Accepted corrupted full state')
        assert rejections==16
    body=patched_driver(Path('artifacts/kaggle/b011-b012-v1/run_driver.py').read_text(),'long-resume')
    compile(body,'long-resume','exec')
    # Execute the actual generated step guard, rather than only assert source text.
    import ast
    tree=ast.parse(body)
    guards=[n for n in ast.walk(tree) if isinstance(n,ast.Assert) and isinstance(n.test,ast.Compare) and ast.unparse(n.test).startswith('resumed_step == experiment')]
    assert len(guards)==1
    guard=compile(ast.fix_missing_locations(ast.Module(body=guards,type_ignores=[])),'resume-step-guard','exec')
    for step,parent,passes in [(120000,120000,True),(60000,60000,False),(120000,60000,False),(60000,120000,False)]:
        env={'resumed_step':step,'experiment':{'resume_parents':{'b027':{'step':parent}}},'name':'b027'}
        try:exec(guard,env)
        except AssertionError:assert not passes
        else:assert passes
    print(json.dumps({'event':'long_continuation_checks_passed','parent_recipe_hash_rejections':rejections,'120k_driver_step_guard_executed':True,'CUDA_verified':False}))


if __name__=='__main__':main()
