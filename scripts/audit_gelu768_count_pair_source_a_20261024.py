"""Actual source/AST audit only; no candidate imports, synthetic data or models."""
from pathlib import Path
import ast,hashlib,json
R=Path(__file__).resolve().parents[1];W=R/'.worktrees/strong-inference-fallback-20261022'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fs(s):return {n.name:n for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def dump(n):return ast.dump(n,include_attributes=False)
def checks():
 pa=(W/'scripts/probe_own_gelu768_validation_a.py').read_text();pc=(W/'scripts/probe_own_gelu768_validation_c.py').read_text();fa,fc=map(fs,[pa,pc]);parent=(W/'scripts/kaggle_gelu768_count_pair_a.py').read_text();old=(W/'scripts/kaggle_wide_t4_budget_a.py').read_text();fp,fo=map(fs,[parent,old]);builder=(W/'scripts/build_kaggle_gelu768_count_pair_a.py').read_text();ob=(W/'scripts/build_kaggle_wide_t4_budget_a.py').read_text();fb,fob=map(fs,[builder,ob]);worker=(W/'scripts/probe_gelu768_count_pair_a.py').read_text();out={}
 files=['scripts/own_gelu768_validation_contract_c.py','scripts/probe_own_gelu768_validation_c.py','scripts/probe_gelu768_count_pair_a.py','scripts/kaggle_gelu768_count_pair_a.py','scripts/build_kaggle_gelu768_count_pair_a.py'];out['all5_new_Python_AST_parse']=all(ast.parse((W/n).read_text()) for n in files)
 same=['atomic_json','verify_topology','actual_prefix_parity','sample_fixed','observe_backend'];out['5_sampler_helpers_AST_exact']=all(dump(fa[n])==dump(fc[n]) for n in same)
 ca=ast.get_source_segment(pa,fa['generate_candidates']);cc=ast.get_source_segment(pc,fc['generate_candidates']);out['generate_candidates_exact_except_countguard']=cc.replace('count in (8,12)','count in (4,6,8)').replace('prescribed8/12','prescribed4/6/8')==ca
 out['adapter_and_internal_countguard_both8_12']='x in (8,12)' in (W/files[0]).read_text() and 'count in (8,12)' in pc and 'count in (4,6,8)' not in pc
 out['exactF_only_additional_guard']=all(x in (W/files[0]).read_text() for x in ['wide_gelu768_scratch_f_v1','5b7713f7e6a8a66c403853b9c0db4eab758f23f7dd3f9547051e7ddea81ae833','f2223a305d46488662f3baf2877f03c5c82291a6322a0e4e344388671a4ab51b'])
 ph=['Deadline','require','arm','alarm','sha','atomic_json','emit','stop','safe_relative','locate_stage','ordered_sha','expected_question_orders','validate_parity','validate_pool'];out['14_parent_helpers_classes_AST_exact']=all(dump(fp[n])==dump(fo[n]) for n in ph)
 out['parent_finalizer_exact_namespace_only']=dump(fp['finalize'])==dump(fs(old.replace('wide_t4_budget','gelu768_count_pair'))['finalize'])
 out['7_builder_helpers_AST_exact']=all(dump(fb[n])==dump(fob[n]) for n in ['require','sha','safe','ref_path','read','copy_small','write'])
 out['900_tuple_exact']='PLATFORM, PARENT, WORK, WORKER = 900, 840, 740, 735' in parent and 'CLEANUP, HASH_END = 785, 835' in parent
 out['parent_worker_result_route_agree']=all(x in parent and x in worker for x in ['gelu768_count_worker_result_a_v1','gelu768_count_worker_result_a.json'])
 out['one_count_per_concurrent_role']='counts==([8]' in worker and "else [12])" in worker and "expected_counts={'gelu768-n8':([8],[8]),'gelu768-n12':([12],[12])}" in parent
 out['shared_single_projection_path']="tools_root/'roles'/'gelu768'" in parent and "tools/'roles/gelu768'" in builder and "'model_APFS_clones':1" in builder
 out['parent_binding_tokens_once']=all(parent.count(x)==1 for x in ['__TOOLS_MOUNT__','__SOURCE_FREEZE_SHA256__','__STAGE_MANIFEST_SHA256__'])
 sf=W/'configs/own_gelu768_validation_source_c_20261024.json';sampler=json.loads(sf.read_text());out['all5_sampler_C_pins']=all(sha(W/n)==h for n,h in sampler['source_sha256'].items())
 out['old_sampler_A_freeze_preserved']=sha(W/'configs/own_gelu768_validation_source_a_20261024.json')=='68ae905f0aea1ca3120e5ea8e61a6e5ca71136f7dc48f711ce15ad90d76faeb6'
 out['preliminary_B_preserved']=sha(W/'configs/own_gelu768_validation_source_b_20261024.json')=='ccdfccadf04ac811d4da82cca15e85ffc2a7fa30f49d33829608cba9880f58ab'
 out['pair_templates_NULL']=json.loads((W/'configs/gelu768_count_worker_a.template.json').read_text())['verified_actual_values'] is False and json.loads((W/'configs/gelu768_count_pair_input_a.template.json').read_text())['verified_actual_values'] is False
 assert all(out.values());return out
if __name__=='__main__':print(json.dumps(checks(),indent=2))
