"""Focused finite-driver/new-step admission and pinned-parent fixtures."""
import hashlib
import json
from pathlib import Path
import tempfile
from kaggle_resume_inputs import verified_parent,sha
from successor_driver_patch import patched_driver


def main():
    preserved=Path('artifacts/kaggle/b011-b012-v1/run_driver.py').read_text()
    for mode in ('loop','resume'):
        body=patched_driver(preserved,mode)
        assert "lane['step']>=lane['admission_step']" in body
        assert "'admission_step':resumed_step+1000" in body
        assert "if lane['step']>=1000" not in body
        assert 'near_finish_can_checkpoint=' in body
        assert "Only complete nonterminated admitted training" in body
        assert "time.sleep(5)" in body
        assert "metrics['new_iterations']==metrics['completed_steps']-lane['resumed_step']" in body
        if mode=='resume':assert "verified_parent('/kaggle/input'" in body and 'resume_dir=' in body
        compile(body,'successor-'+mode,'exec')
    # Inherited60000 cannot satisfy admission; 61000 has1000 new iterations.
    assert not 60000>=60000+1000 and not 60999>=60000+1000 and 61000>=60000+1000
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);run=root/'run-b013';run.mkdir()
        hardware={'torch':'fixtureTorch','cuda':'fixtureCUDA'}
        metadata={**hardware,'seed':42,'batch_size':64}
        metrics={'completed_steps':60000,'optimizer_updates':59990,'skipped_updates':10,'sampled_examples':3840000}
        (run/'model.pt').write_bytes(b'fixture-best');(run/'last_training.pt').write_bytes(b'fixture-state')
        (run/'run_config.json').write_text(json.dumps(metadata));(run/'metrics.json').write_text(json.dumps(metrics))
        files={str(p.relative_to(root)):sha(p) for p in run.iterdir()}
        terminal={'complete':True,'training_complete':True,'iterations':{'b013':60000},
            'training_source_sha256':'source','driver_sha256':'driver','cache_manifest_sha256':'cache','artifact_sha256':files}
        path=root/'session_result.json';path.write_text(json.dumps(terminal))
        record={'terminal_sha256':sha(path),'run':'b013','step':60000,'seed':42,'source_sha256':'source',
            'driver_sha256':'driver','files':files}
        assert verified_parent(root,record,'cache',hardware)==run
        rejected=0
        for field,value in [('terminal_sha256','wrong'),('source_sha256','wrong'),('driver_sha256','wrong'),('step',59999),('seed',43)]:
            try:verified_parent(root,{**record,field:value},'cache',hardware)
            except ValueError:rejected+=1
            else:raise AssertionError('Accepted bad '+field)
        try:verified_parent(root,record,'cache',{**hardware,'torch':'different'})
        except ValueError:rejected+=1
        else:raise AssertionError('Accepted software mismatch')
        (run/'last_training.pt').write_bytes(b'corrupted')
        try:verified_parent(root,record,'cache',hardware)
        except ValueError:rejected+=1
        else:raise AssertionError('Accepted corrupted parent state')
        assert rejected==7
    print(json.dumps({'event':'successor_driver_checks_passed','admission_requires_1000_new_steps':True,
        'pinned_parent_hash_and_metadata_rejections':rejected,'preserved_tail_heartbeat_near_finish':True,'CUDA_run_verified':False}))


if __name__=='__main__':main()
