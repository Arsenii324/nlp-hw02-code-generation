"""Actual source-file AST/hash inspection only; no models/tests/synthetic runs."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'.worktrees/gated-cooldown-prefix-20261023'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def norm(n):return ast.dump(n,include_attributes=False)
def fn(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);a=p.parse_args();source=a.source.resolve();freeze=json.loads((source/'source_manifest.json').read_text());checks={}
 worker_name=next(n for n in freeze['source_members_sha256'] if n.startswith('scripts/train_gated_cooldown_prefix_'))
 parent_name=next(n for n in freeze['source_members_sha256'] if n.startswith('scripts/kaggle_gated_cooldown_prefix_'))
 recipe_name=next(n for n in freeze['source_members_sha256'] if n.startswith('src/nlp_hw02/gated_cooldown_recipe_'))
 coordination_name=next(n for n in freeze['source_members_sha256'] if n.startswith('scripts/gated_cooldown_coordination_'))
 def check(k,v):
  checks[k]=bool(v)
  if not v:raise ValueError(k)
 check('source_archive_actual_pin',sha(source/'cooldown_continuation_source.bin')==freeze['source_sha256'])
 with zipfile.ZipFile(source/'cooldown_continuation_source.bin') as z:
  check('all58_safe_unique_member_pins',len(z.namelist())==58==len(set(z.namelist())) and set(z.namelist())==set(freeze['source_members_sha256']) and all(not Path(n).is_absolute() and '..' not in Path(n).parts and hashlib.sha256(z.read(n)).hexdigest()==freeze['source_members_sha256'][n]==sha(W/n) for n in z.namelist()))
 foundation=ROOT/'.worktrees/bucketed-gated-continuation-20261023/artifacts/kaggle/gated-window-tools-20261023c/window_continuation_source.bin'
 with zipfile.ZipFile(foundation) as old:check('all53_foundation_source_bytes_unchanged',len(old.namelist())==53 and all((W/n).read_bytes()==old.read(n) for n in old.namelist()))
 wt=ast.parse((W/worker_name).read_text());wo=ast.parse((W/'scripts/train_gated_window_continuation_b.py').read_text());pt=ast.parse((W/parent_name).read_text());po=ast.parse((W/'scripts/kaggle_gated_window_continuation_b.py').read_text());rt=ast.parse((W/recipe_name).read_text());ro=ast.parse((W/'src/nlp_hw02/gated_window_recipe_b.py').read_text())
 for n in ('require','sha','fsync_directory','atomic_json','atomic_torch_save','safe_relative','safe_run_id'):check('worker_exact_whole_helper_AST:'+n,norm(fn(wt,n))==norm(fn(wo,n)))
 for n in ('arm_absolute','absolute_alarm','sha','emit','start','stop','require','snapshot_lanes','launch_evaluation','terminate','finalize'):check('parent_exact_whole_lifecycle_AST:'+n,norm(fn(pt,n))==norm(fn(po,n)))
 old_guard=next(n for n in po.body if isinstance(n,ast.If) and '__name__' in ast.unparse(n.test));new_guard=next(n for n in pt.body if isinstance(n,ast.If) and '__name__' in ast.unparse(n.test));check('parent_exact_guard_AST',norm(old_guard)==norm(new_guard))
 for n in ('fingerprint','recipe_for','valid_sha','validate_window_parent','draw_window','segment_record'):check('recipe_exact_whole_helper_AST:'+n,norm(fn(rt,n))==norm(fn(ro,n)))
 old=(W/'src/nlp_hw02/gated_window_recipe_b.py').read_text();expected=old.replace("require(old['lr']==absolute_lr(step)*old['lr_multiplier'],'Saved group LR does not equal absolute checkpoint step')","require(old['lr']==saved_lr_oracle(state,metadata)*old['lr_multiplier'],'Saved group LR differs from strict saved schedule/phase oracle')").replace("metrics=json.loads((parent/'metrics.json').read_text())","metrics=json.loads((parent/'metrics.json').read_text())\n    if 'cooldown_checkpoint' in state:require(metrics['cooldown_checkpoint']==state['cooldown_checkpoint'],'Saved metrics/fullstate cooldown binding differs')")
 check('strict_restore_only_scheduleLR_and_metadata_delta',norm(fn(rt,'restore_window_training'))==norm(fn(ast.parse(expected),'restore_window_training')))
 oldworker=(W/'scripts/train_gated_window_continuation_b.py').read_text().replace('absolute=start+completed+1;lr=absolute_lr(absolute)',"absolute=start+completed+1;lr=scheduled_lr(absolute,c['cooldown_plan'],cooldown_binding)")
 def batch(t):return next(n for n in ast.walk(fn(t,'run')) if isinstance(n,ast.For) and ast.unparse(n.target)=='batch_index')
 check('entire64batch_training_loop_only_LR_oracle_change',norm(batch(wt))==norm(batch(ast.parse(oldworker))))
 check('all5_new_sources_parse',all(ast.parse((W/n).read_text()) for n in freeze['source_members_sha256'] if 'cooldown' in n))
 parent=(W/parent_name).read_text();worker=(W/worker_name).read_text();sched=(W/'src/nlp_hw02/gated_cooldown_schedule_a.py').read_text();coord=(W/coordination_name).read_text()
 check('exact_absolute_13500_budget_constants','PLATFORM=13500;OUTER=13440;WORK=11940;CLEANUP_END=12300;HASH_END=13395;TAIL=1500' in parent and 'STARTED+WORK-5' in parent)
 check('worker_initial_only_original345k_not_descendant_coercion',"c['parent_kind']=='original_uniform_gated'" in worker and "args.lane=='control'" in worker and 'args.new_steps==81920' in worker)
 check('worker_template_guard_before_Torch',worker.index("c['verified_actual_values'] is True")<worker.index('    import torch',worker.index('def run(')))
 check('prospective_recipe_H_separate','schedule_binding_separate' in sched and "cooldown_plan=deepcopy(c['cooldown_plan'])" in worker and "cooldown_binding=None" in worker)
 check('never_step1025_unbound_or_unreleased',"require(j<=PREFIX,'Never advance past1024" in sched and 'allow_pending_release=j<=PREFIX' in sched and "valid_sha(binding['release_sha256'])" in sched)
 check('exact_remaining_envelopes_static','(remaining-k)/(remaining-1)' in sched and "(1. if k<=remaining//2 else (remaining-k)/(remaining-remaining//2))" in sched)
 check('common_decision_min64_actualrate_and_fullcost',all(s in coord for s in ['horizon=min(fits.values())','remaining=worker_deadline-now','r[\'measured_steps_per_second\']/1.25','binding_reserve=max(60.,2.5*','reserve=max(180.,1.5*']))
 check('common_prefix_request_checks_pointer_frontier_raw_applied',all(s in coord for s in ['committed[\'since_resume_iterations\']','schedule[\'binding\'] is None','segment[\'raw_segment_sha256\']','segment[\'applied_segment_sha256\']']))
 check('decision_ack_release_before_postprefix_return',coord.index("ack=checkpoint_bound(provisional,False)")<coord.index("release=wait(release_path)")<coord.index('checkpoint_bound(binding,True);guard();return binding,decision'))
 check('bounded_barrier_wait_and_peer_failure_stop','time.monotonic()+120' in coord and "now>=state['barrier_cutoff']" in coord and 'postprefix_reject.json' in coord and 'stop_peer(' in coord)
 check('postprefix_reject_never_changes_H','elif completed==PREFIX+1024 and completed<target' in worker and 'never re-admit changedH after decay' in worker)
 for n,d in freeze['templates_sha256'].items():
  t=json.loads((source/n).read_text());check('NULL_unbound_template:'+n,sha(source/n)==d and t['verified_actual_values'] is False and (t.get('actual_H') is None) and (t.get('actual_common_H') is None))
 check('current_canonical_untouched',sha(ROOT/'artifacts/selected_bundle.zip')=='7c9c7885912e867cbf2bd8eba9ee583cc6424265a2f8f3032299c2a1411c6083' and sha(ROOT/'submit/inference.ipynb')=='2b1f2d877cb89d861f8a360a67fa108aa3dce332b2c593064f1940fd3307f977')
 prior=W/'artifacts/gated-cooldown-source-20261023b/source_manifest.json';prior_map=json.loads(prior.read_text())['source_members_sha256']
 check('all58_priorB_archive_members_still_match_worktree_files',all(sha(W/n)==d for n,d in prior_map.items()))
 check('worker_truthful_current_schedule_label',"'schedule':'retained1024 absoluteISQRT then immutableH remaining-envelope cooldown; no warmup/reset'" in worker)
 stop_pair=next(n for n in ast.walk(fn(pt,'main')) if isinstance(n,ast.FunctionDef) and n.name=='stop_pair');stop_source=ast.unparse(stop_pair)
 check('peer_graceful_stop_records_stop_time',"lane['stop_time'] = time.monotonic()" in stop_source)
 check('healthy_nearfinish_preserved_from_actual_rate_and_save',all(v in stop_source for v in ['name not in failed_runs','eta <= 180','1.25 * eta + reserve < remaining','preserved_nearfinish']))
 check('coordination_reacts_only_to_lost_training',"(l.get('finish_failure') and not l.get('horizon_complete',False))" in coord)
 launch_try=next(n for n in ast.walk(fn(pt,'main')) if isinstance(n,ast.Try) and len(n.body)==1 and isinstance(n.body[0],ast.Expr) and isinstance(n.body[0].value,ast.Call) and ast.unparse(n.body[0].value.func)=='launch_evaluation')
 check('eval_launch_failure_separate_from_training',"evaluation_deferred" in ast.unparse(launch_try.handlers[0]) and "finish_failure" not in ast.unparse(launch_try.handlers[0]) and "['child']" not in ast.unparse(launch_try.handlers[0]))
 check('actual_shared_barrier_cutoff_in_decision_and_worker',all(v in coord for v in ["'barrier_cutoff_monotonic':barrier_cutoff", "record['ready_monotonic']", "state['barrier_started']+120", 'end=min(end,cutoff);guard()', 'guard();ack=checkpoint_bound(provisional,False);guard()', 'guard();checkpoint_bound(binding,True);guard();return']))
 check('existing_decision_release_files_also_guarded','guard();value=read_json(path);guard();return value' in coord)
 check('firstpostprefix_draw_barrier_cutoff_guard',"time.monotonic()<cooldown_decision['barrier_cutoff_monotonic']" in worker and worker.index("'Common barrier elapsed before firstpostprefix draw")<worker.index("raw,applied,guard=draw_window"))
 check('final_pair_SHA_gate_preserves_both_results',all(v in parent for v in ["STATE.update(phase='final_pair_validation',runs=results,paired_complete=False)","'same_raw':len(set(raw))==1","'same_applied':len(set(applied))==1","'same_whole_raw':len(set(whole))==1"]))
 evidence={'schema':'gated_cooldown_source_c_static_20261023_v2','utc':datetime.now(timezone.utc).isoformat(),'source':str(source.relative_to(ROOT)),'source_manifest_sha256':sha(source/'source_manifest.json'),'source_sha256':freeze['source_sha256'],'C_archive_member_count':58,'C_unchanged_foundation_members':53,'C_unchanged_schedule_member':1,'C_fresh_runtime_route_members':4,'priorB_preservation_scope':'All58priorB archive member files remain byteexact on disk; C contains54shared+4newroutes, not all58Bmembers.','checks':checks,'all_checks_pass':all(checks.values()),'audit_script_sha256':sha(__file__),'tests_models_arrays_optimizer_downloads_cloud':False,'future_actual_bindings':None,'independent_review_pending':True}
 path=ROOT/'reports/evidence/gated_cooldown_source_c_static_20261023.json';path.write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps({'checks':len(checks),'all_pass':all(checks.values()),'evidence':str(path)}))
if __name__=='__main__':main()
