"""Actual local-stage source/JSON/hash readback only; no model/tests/cloud."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for b in iter(lambda:stream.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 evidence_path=ROOT/'reports/evidence/gated_cooldown_bound_stage_a_20261023.json';e=json.loads(evidence_path.read_text());tools=ROOT/e['tools'];kernel=ROOT/e['kernel'];m=json.loads((tools/'gated_cooldown_continuation_manifest.json').read_text());km=json.loads((kernel/'kernel-metadata.json').read_text());dm=json.loads((tools/'dataset-metadata.json').read_text());checks={}
 def check(k,v):
  checks[k]=bool(v)
  if not v:raise ValueError(k)
 check('actual_bound_local_only_no_launch',e['status']=='LOCAL_ONLY_ROOT_AND_INDEPENDENT_WHOLE_STAGE_REVIEW_REQUIRED' and e['tests_cloud_privateupload'] is False and e['future_launch_authority'] is False and e['no_model_optimizer_raw_read_copy_download'] is True)
 check('all6_tool_file_hashes',len(e['tools_sha256'])==6 and set(e['tools_sha256'])=={p.name for p in tools.iterdir()} and all(sha(tools/n)==d for n,d in e['tools_sha256'].items()))
 check('all4_runtime_input_pins',len(m['input_sha256'])==4 and all(sha(tools/n)==d for n,d in m['input_sha256'].items()))
 check('exact_manifest_driver_metadata',sha(tools/'gated_cooldown_continuation_manifest.json')==e['manifest_sha256'] and sha(kernel/'run.py')==m['driver_sha256']==e['driver_sha256']=='1836b30f498cf46f7e693ea5a327811c18e9f1124b415f3244b6b27bd0a7aad9' and sha(kernel/'kernel-metadata.json')==e['kernel_metadata_sha256'])
 archive=tools/'cooldown_continuation_source.bin';check('accepted_C_archive_unchanged',sha(archive)==m['input_sha256'][archive.name]==e['source_sha256']=='dd0c3e62c2d87128270ce2e1281015e3b4bd6133e3ccc2467d9f1a7e0175f911')
 with zipfile.ZipFile(archive) as z:
  check('all58_unique_safe_member_pins',len(z.namelist())==58==len(set(z.namelist())) and set(z.namelist())==set(m['source_members_sha256']) and all(not Path(n).is_absolute() and '..' not in Path(n).parts and hashlib.sha256(z.read(n)).hexdigest()==m['source_members_sha256'][n] for n in z.namelist()))
  check('no_weights_arrays_data_in_source_archive',not any(n.endswith(('.pt','.npy','.npz','.parquet')) for n in z.namelist()))
  for n in z.namelist():
   if n.endswith('.py'):ast.parse(z.read(n))
  driver=ast.parse(z.read('scripts/kaggle_gated_cooldown_prefix_b.py'));worker=ast.parse(z.read('scripts/train_gated_cooldown_prefix_b.py'))
  check('driver_whole_archivebytes_equal_kernel',z.read('scripts/kaggle_gated_cooldown_prefix_b.py')==(kernel/'run.py').read_bytes())
  launcher=next(n for n in driver.body if isinstance(n,ast.FunctionDef) and n.name=='launch_evaluation');maps=[n for n in ast.walk(launcher) if isinstance(n,ast.Dict) and any(isinstance(k,ast.Constant) and k.value=='{test_parquet}' for k in n.keys)];check('exact_placeholder_mapping_single_dictionary',len(maps)==1)
  mapping={k.value:v for k,v in zip(maps[0].keys,maps[0].values)};check('all4_placeholderkeys_and_exact_testpath',set(mapping)=={'{run}','{output}','{tools}','{test_parquet}'} and ast.dump(mapping['{test_parquet}'],include_attributes=False)==ast.dump(ast.parse("str(tools/'test.parquet')",mode='eval').body,include_attributes=False))
  check('source_aggregate_RAM_guard_unchanged',"available_memory_bytes()>=(resident_count+1)*size+2*1024**3" in z.read('scripts/kaggle_gated_cooldown_prefix_b.py').decode())
  check('unbound_guards_before_runtime',"m['verified_actual_values'] is True" in z.read('scripts/kaggle_gated_cooldown_prefix_b.py').decode() and "c['verified_actual_values'] is True" in z.read('scripts/train_gated_cooldown_prefix_b.py').decode())
 terminal_path=ROOT/'artifacts/kaggle/gated-b034-final-v1/session_result.json';terminal=json.loads(terminal_path.read_text());check('actual_original_terminal',sha(terminal_path)==m['parent_terminal_sha256']==e['parent_terminal_sha256']=='991448f5aab75e7a7e8dad030f05f2a8af9a6327c75133831fade72e0c13e609' and terminal['training_complete'])
 check('both_2bucket_arms_cap81920_and_distinct_devices',len(m['runs'])==2 and {r['run_id'] for r in m['runs']}=={'cooldown-full','cooldown-half'} and {r['device'] for r in m['runs']}=={0,1} and {r['cooldown_arm'] for r in m['runs']}=={'full-remaining','half-remaining'} and all(r['lane']=='control' and r['layout']=='bucket' and r['new_steps']==81920 and r['resident'] is True for r in m['runs']))
 for r in m['runs']:
  c=json.loads((tools/r['contract_path']).read_text());prefix=r['run_id']+':'
  check(prefix+'contract_exact_rolesource_hash',c['verified_actual_values'] is True and c['run_id']==r['run_id'] and c['lane']=='control' and c['layout']=='bucket' and c['new_steps']==81920 and c['cooldown_arm']==r['cooldown_arm'] and c['runtime_archive_sha256']==e['source_sha256'] and c['runtime_source_sha256']==m['source_members_sha256'])
  check(prefix+'all8_original_roles_terminalbound',len(c['parent_artifact_sha256'])==8 and c['parent_artifact_sha256']==e['parent_artifact_sha256'] and all(terminal['artifact_sha256']['run-control/'+n]==d for n,d in c['parent_artifact_sha256'].items()))
  check(prefix+'endpoint_not_best_no_reset',c['parent_absolute_step']==345000 and c['parent_run_path']=='run-control' and c['parent_kind']=='original_uniform_gated' and c['allow_incomplete_window_parent'] is False and c['parent_artifact_sha256']['last_training.pt']=='f9d4f20ca116b4d71089cbafeecab9c6cfb210752bd572e8da5bcca9627cfb0d' and c['parent_artifact_sha256']['model.pt']=='eed1fdbb4f7e056536b322c08fad8660f5df24460f716a4d8e5382eb58ce8dc9')
  check(prefix+'actual_source_driver_manifest_original',c['parent_source_sha256']==terminal['source_sha256'] and c['parent_driver_sha256']==terminal['driver_sha256'] and c['parent_manifest_path']=='experiment-manifest.json' and c['parent_manifest_sha256']==terminal['artifact_sha256']['experiment-manifest.json'])
  check(prefix+'exact6Mcache_and4metadata_roles',c['cache_manifest_sha256']==m['cache_manifest_sha256']=='8ce0939534bdb5f2b791d19d0ab13f6b4c79402b8627fecad4e9902cd08bb13c' and c['cache_checksums']==m['cache_checksums'] and all(c['parent_artifact_sha256'][n]==c['cache_checksums'][n] for n in ('tokenizer.json','validation.json','validation_legacy.json','data_stats.json')))
  check(prefix+'H_decision_release_unfabricated',c['actual_H'] is None and c['actual_decision_sha256'] is None and c['actual_release_sha256'] is None and c['admission_mode']=='measured64_shortening' and c['cooldown_plan']['prefix_NEW_iterations']==1024)
  ev=r['post_training_evaluation'];argv=ev['argv'];placeholders={v for v in argv if '{' in v or '}' in v}
  check(prefix+'all_eval_placeholders_resolve_statically',placeholders=={'{run}','{output}','{test_parquet}'} and placeholders<=set(mapping) and argv==['scripts/profile_decoding_gpu.py','--run','{run}','--output','{output}','--test-parquet','{test_parquet}','--skip-synthetic-check'] and argv[0] in m['source_members_sha256'] and ev['bound_seconds']==300)
  check(prefix+'required_eval_outputs_complete',ev['required_outputs']==['decoding_profile.json','greedy/validation_predictions.jsonl','beam4-alpha06/validation_predictions.jsonl','beam_test_timing.json','beam_test_predictions.jsonl'])
 check('exact13500_all_absolute_bounds',m['platform_seconds']==13500 and e['external_platform_cap_seconds']==13500 and m['bounds_seconds']=={'platform':13500,'parent':13440,'work':11940,'worker':11935,'aggregate_cleanup':12300,'hash':13395,'tail':1500})
 check('both_H_roles_measured_at_runtime',m['actual_common_H'] is None and m['actual_terminal_sha256'] is None and m['resident_policy'].startswith('Requestedbothresident'))
 check('short_matching_title_slug',km['title']==km['id'].split('/')[1] and dm['title']==dm['id'].split('/')[1] and len(km['title'])<=50 and len(dm['title'])<=50 and km['id']=='arsen4ikvar/nlp-hw02-cooldown-prefix-20261023a')
 check('private_offline_T4_no_TPU_image',km['is_private'] is True and km['enable_internet'] is False and km['enable_gpu'] is True and km['enable_tpu'] is False and km['machine_shape']=='NvidiaTeslaT4' and km['docker_image']=='gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461' and km['docker_image_pinning_type']=='original')
 check('exact_tools_originalparent_cache_mounts',km['dataset_sources']==[dm['id']] and km['kernel_sources']==['arsen4ikvar/nlp-hw02-gated-b034-20261022b','arsen4ikvar/nlp-hw02-cache-6m-20261004c'] and km['competition_sources']==[])
 check('pinned_questiononly_test',sha(tools/'test.parquet')=='8bde59c8d8bb7edc4e8a503be22fc132c4a47773025d333432e78c5489e6522a')
 check('all_C_source_actual_readback_unchanged',m['source_manifest_sha256']=='36bd37a3e481f9c5835884047fa13ceb68b7daf1dcaf44524607782e182208fe' and m['source_review_sha256']=='963157f652e9501b55f0c0c9ef0e4be6cd61da8d69a548d79224008e7e1a021b' and sha(ROOT/'reports/evidence/gated_cooldown_source_c_root_review_20261023.json')==m['source_root_review_sha256'])
 check('canonical_pair_unchanged',sha(ROOT/'artifacts/selected_bundle.zip')=='7c9c7885912e867cbf2bd8eba9ee583cc6424265a2f8f3032299c2a1411c6083' and sha(ROOT/'submit/inference.ipynb')=='2b1f2d877cb89d861f8a360a67fa108aa3dce332b2c593064f1940fd3307f977')
 out=ROOT/'reports/evidence/gated_cooldown_bound_stage_a_static_20261023.json';out.write_text(json.dumps({'schema':'gated_cooldown_bound_stage_a_static_v1','utc':datetime.now(timezone.utc).isoformat(),'all_checks_pass':all(checks.values()),'checks':checks,'stage_evidence_sha256':sha(evidence_path),'driver_sha256':e['driver_sha256'],'manifest_sha256':e['manifest_sha256'],'source_sha256':e['source_sha256'],'audit_script_sha256':sha(__file__),'all_placeholder_readback':'Static AST key/value inspection only; no evaluator/model executed','local_RAM_claim':False,'tests_models_optimizer_or_raw_downloads_cloud':False,'independent_whole_stage_review_pending':True},indent=2)+'\n');print(json.dumps({'checks':len(checks),'all_pass':all(checks.values()),'evidence':str(out)}))
if __name__=='__main__':main()
