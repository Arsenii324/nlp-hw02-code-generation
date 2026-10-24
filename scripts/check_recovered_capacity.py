"""Verify bounded recovered artifacts against a prelaunch capacity manifest."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_training_counts(metrics,lane,batch_size):
    assert metrics['completed_steps']==lane['iterations']
    assert metrics['optimizer_updates']+metrics['skipped_updates']==lane['iterations']
    assert metrics['sampled_examples']==lane['iterations']*batch_size


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True); parser.add_argument('--manifest',required=True)
    parser.add_argument('--validation',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--metadata-only',action='store_true',help='Verify diagnostics; do not claim checkpoint bytes were checked')
    args=parser.parse_args(); root=Path(args.root)
    expected=json.loads(Path(args.manifest).read_text()); terminal=json.loads((root/'session_result.json').read_text())
    assert terminal['complete'] and terminal['training_complete'] and terminal['inference_complete']
    assert terminal['phase']=='finished' and terminal['training_source_sha256']==expected['source_sha256']
    assert terminal['cache_manifest_sha256']==expected['cache_manifest_sha256']
    assert digest(root/'training_source.bin')==expected['source_sha256']
    assert digest(root/'run_driver.py')==expected['driver_sha256']
    assert (root/'experiment-manifest.json').read_bytes()==Path(args.manifest).read_bytes()
    iterations={lane['id'].lower():lane['iterations'] for lane in expected['lanes']}
    assert terminal['iterations']==iterations and len(terminal['lanes'])==len(iterations)
    assert {lane['run'] for lane in terminal['lanes']}==set(iterations)
    assert all(l['exit']==0 and not l['gracefully_stopped'] for l in terminal['lanes'])
    with zipfile.ZipFile(root/'training_source.bin') as z:
        assert z.testzip() is None
        assert len(set(z.namelist()))==len(z.namelist())
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
        if 'source_member_sha256' in expected:
            assert set(z.namelist())==set(expected['source_member_sha256'])
            for name,sha in expected['source_member_sha256'].items():
                assert hashlib.sha256(z.read(name)).hexdigest()==sha,name
    checked=[]; skipped=[]
    for name,sha in terminal['artifact_sha256'].items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        path=root/name
        if not path.exists(): skipped.append(name); continue
        assert path.is_file() and not path.is_symlink() and digest(path)==sha,name
        checked.append(name)
    results={}
    for lane in expected['lanes']:
        name=lane['id'].lower(); run=root/f'run-{name}'; inference=root/f'inference-{name}'
        required=[run/n for n in ('tokenizer.json','run_config.json','data_stats.json','metrics.json',
            'train.jsonl','validation.json','validation_predictions.jsonl')]
        if not args.metadata_only:required.append(run/'model.pt')
        required.extend(inference/n for n in ('test_timing.json','test_predictions.jsonl'))
        assert all(str(p.relative_to(root)) in checked for p in required),'Missing required hashed artifact'
        assert digest(run/'validation.json')==digest(Path(args.validation))
        metrics=json.loads((run/'metrics.json').read_text()); config=json.loads((run/'run_config.json').read_text())
        verify_training_counts(metrics,lane,expected['batch_size'])
        assert 0<metrics['selected_step']<=lane['iterations']
        assert config['seed']==lane['seed'] and config['batch_size']==expected['batch_size']
        assert config['max_steps']==lane['iterations'] and math.isfinite(metrics['best_validation_loss'])
        if 'parameter_counts' in expected:assert config['parameters']==expected['parameter_counts'][name]
        if 'resume_from_step' in lane:
            start=lane['resume_from_step']
            assert metrics['resumed_from_step']==start and metrics['new_iterations']==lane['iterations']-start
            if start:
                assert config['resumption']['step']==start
                change=config['resumption']['orthogonalizer_change']
                if lane.get('allow_orthogonalizer_change'):
                    assert change['from']=='newton_schulz' and change['to']=='polar_express_v5_5'
                    assert change['momentum_and_other_state_restored']
                else:assert change is None
            else:assert config['resumption'] is None
        if lane['optimizer']=='adamw': assert config['optimizer']['kind']=='AdamW'
        else: assert 'Muon' in config['optimizer']['kind']
        if lane['optimizer']=='muon_grouped_pe':assert config['optimizer']['orthogonalizer']=='polar_express_v5_5'
        for key,value in lane['model_config'].items(): assert config['config'][key]==value
        rows=[json.loads(s) for s in (inference/'test_predictions.jsonl').read_text().splitlines()]
        assert [r['row_index'] for r in rows]==list(range(1919)) and all(isinstance(r['code'],str) for r in rows)
        timing=json.loads((inference/'test_timing.json').read_text())
        assert timing['rows']==1919 and timing['gpu_count']==1 and 'T4' in timing['gpu']
        assert timing['meets_180_seconds_on_one_T4'] and timing['seconds']<=180
        results[name]={'completed_iterations':metrics['completed_steps'],'selected_step':metrics['selected_step'],
            'optimizer_updates':metrics['optimizer_updates'],'skipped_updates':metrics['skipped_updates'],
            'primary_ce':metrics['best_validation_loss'],'training_seconds':metrics['training_elapsed_seconds'],
            'test_seconds':timing['seconds'],'test_capped_rows':timing['capped_rows']}
    extra_results={}
    if expected.get('profile_decoding'):
        for lane in expected['lanes']:
            name=lane['id'].lower();directory=root/f'decoding-{name}'
            required=['decoding_profile.json','beam_test_timing.json','beam_test_predictions.jsonl',
                'greedy/validation.json','greedy/validation_predictions.jsonl',
                'beam4-alpha06/validation.json','beam4-alpha06/validation_predictions.jsonl']
            assert all(str((directory/n).relative_to(root)) in checked for n in required),'Missing hashed beam evidence'
            profile=json.loads((directory/'decoding_profile.json').read_text())
            checkpoint_sha=(terminal['artifact_sha256'][f'run-{name}/model.pt'] if args.metadata_only
                else digest(root/f'run-{name}/model.pt'))
            assert profile['checkpoint_sha256']==checkpoint_sha
            assert profile['gpu_count']==1 and 'T4' in profile['gpu']
            with zipfile.ZipFile(root/'training_source.bin') as archive:
                assert profile['beam_source_sha256']==hashlib.sha256(archive.read('src/nlp_hw02/beam.py')).hexdigest()
                assert profile['profiler_sha256']==hashlib.sha256(archive.read('scripts/profile_decoding_gpu.py')).hexdigest()
            for decoder in ('greedy','beam4-alpha06'):
                assert digest(directory/decoder/'validation.json')==digest(Path(args.validation))
                rows=[json.loads(line) for line in (directory/decoder/'validation_predictions.jsonl').read_text().splitlines()]
                assert len(rows)==1792 and all(isinstance(r['prediction'],str) for r in rows)
                assert profile['paths'][decoder]['rows']==1792
            rows=[json.loads(line) for line in (directory/'beam_test_predictions.jsonl').read_text().splitlines()]
            assert [r['row_index'] for r in rows]==list(range(1919)) and all(isinstance(r['code'],str) for r in rows)
            timing=json.loads((directory/'beam_test_timing.json').read_text())
            assert timing['rows']==1919 and timing['gpu_count']==1 and 'T4' in timing['gpu']
            assert timing['beam_width']==4 and timing['length_penalty']==.6 and timing['batch_size']==256
            assert timing['meets_180_seconds_on_one_T4'] and timing['seconds']<=180
            test_manifest=json.loads((Path(__file__).resolve().parents[1]/'assignment/dataset_manifest.json').read_text())
            test_sha=next(r['sha256'] for r in test_manifest['files'] if r['path']=='data/test/test.parquet')
            assert timing['test_parquet_sha256']==test_sha
            extra_results[name]={'beam_test_seconds':timing['seconds'],'beam_test_capped_rows':timing['capped_rows']}
    if expected.get('canonical_bundle_timing'):
        directory=root/'canonical-b008';required=[directory/'executed.ipynb',directory/'executed.json',
            root/'canonical-inference/test_timing.json',root/'canonical-inference/test_predictions.jsonl']
        assert all(str(p.relative_to(root)) in checked for p in required),'Missing canonical hashed evidence'
        execution=json.loads((directory/'executed.json').read_text())
        assert execution['complete'] and execution['executed_code_cells']==3
        assert execution['input_sha256']==expected['canonical_notebook_sha256']
        assert execution['output_sha256']==digest(directory/'executed.ipynb')
        notebook=json.loads((directory/'executed.ipynb').read_text());cells=[c for c in notebook['cells'] if c['cell_type']=='code']
        assert len(cells)==3 and [c['execution_count'] for c in cells]==[1,2,3]
        assert all(not any(o['output_type']=='error' for o in c['outputs']) for c in cells)
        timing=json.loads((root/'canonical-inference/test_timing.json').read_text())
        assert timing['bundle_sha256']==expected['canonical_bundle_sha256'] and timing['decoder']=='beam'
        assert timing['rows']==1919 and timing['gpu_count']==1 and 'T4' in timing['gpu']
        assert timing['beam_width']==4 and timing['length_penalty']==.6 and timing['batch_size']==256
        assert timing['meets_180_seconds_on_one_T4'] and timing['seconds']<=180
        rows=[json.loads(line) for line in (root/'canonical-inference/test_predictions.jsonl').read_text().splitlines()]
        assert [r['row_index'] for r in rows]==list(range(1919)) and all(isinstance(r['code'],str) for r in rows)
        extra_results['canonical_b008']={'test_seconds':timing['seconds'],'execution_seconds':execution['elapsed_seconds'],
                                        'test_capped_rows':timing['capped_rows']}
    report={'root':str(root),'prelaunch_manifest_sha256':digest(Path(args.manifest)),
        'terminal_sha256':digest(root/'session_result.json'),'verified_artifacts':checked,
        'not_downloaded':skipped,'runs':results,'extra_evaluations':extra_results,
        'checkpoint_bytes_verified':not args.metadata_only,
        'note':('Downloaded diagnostic hashes/config/counts/split/source and recorded one-T4 timings verified; checkpoint identity is terminal/profile metadata only, actual weights not retrieved or verified.' if args.metadata_only else 'Complete matched-iteration batch and fixed one-T4 timing verified; no official pass@1 claim.')}
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__': main()
