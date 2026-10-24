"""Actual copied file/hash/source audit; no tensors/models/arrays imported."""
from pathlib import Path
import ast,datetime,hashlib,json,shutil
R=Path(__file__).resolve().parents[1];E=R/'reports/evidence/native_offnode_monitor_20261023a';D=R/'deployments/native-offnode-monitor-20261023a'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
f=read(D/'source_freeze_c.json');checks={'all6_source_freeze_pins':len(f['source_files_sha256'])==6 and all(sha(R/p)==h for p,h in f['source_files_sha256'].items()),'source_AST_parse':all(ast.parse((R/p).read_text()) for p in f['source_files_sha256'] if p.endswith('.py'))};jobs={};roles={'model.pt','last_training.pt','metrics.json','run_config.json','tokenizer.json','validation.json','validation_legacy.json','data_stats.json'}
for name in ('gelu','swiglu'):
 entries=[(p,read(p)) for p in (E/'attempts'/name).rglob('receipt.json')];good=[(p,x) for p,x in entries if x['complete']];assert good
 receipt,x=sorted(good,key=lambda t:t[1]['verified_utc'])[-1];d=R/x['local_path'];pointer=read(d/'latest_checkpoint.json');snap=read(d/'snapshot_receipt.json');metric=read(d/'metrics.json');cfg=read(d/'run_config.json')
 checks[name+'_10_exact_local_files']=set(p.name for p in d.iterdir())==roles|{'latest_checkpoint.json','snapshot_receipt.json'} and all(p.is_file() and not p.is_symlink() for p in d.iterdir())
 checks[name+'_all_file_sizes_SHAs']=all((d/n).stat().st_size==v['bytes'] and sha(d/n)==v['sha256'] for n,v in x['files'].items())
 checks[name+'_all8_pointer_and_receipt_SHAs']=set(pointer['files_sha256'])==roles and all(sha(d/n)==pointer['files_sha256'][n]==snap['files'][n]['sha256'] for n in roles)
 checks[name+'_stable_committed_snapshot']=snap['complete'] and pointer['schema']=='wide_scratch_durable_pointer_a_v1' and sha(d/'latest_checkpoint.json')==snap['pointer_sha256']==x['pointer_sha256'] and pointer['absolute_step']==snap['absolute_step']==x['absolute_step']>0
 expected=sha(R/('artifacts/paid-wide-gelu-launch-20261023b/contract.json' if name=='gelu' else 'artifacts/paid-wide-swiglu-launch-20261023c/wide-swiglu-contract-c.json'))
 checks[name+'_root_actual_contract']=pointer['contract_sha256']==expected
 c=x['remote_duplicate_cleanup'];checks[name+'_only_redundant_remote_binaries_removed']=set(c['removed_only_ephemeral_binaries'])=={'model.pt','last_training.pt'} and c['verified_local_copy'] and c['live_durable_touched'] is False
 jobs[name]={'absolute_step':pointer['absolute_step'],'selected_step':metric.get('selected_step'),'best_validation_loss':metric.get('best_validation_loss'),'bytes':x['bytes'],'local_path':x['local_path'],'pointer_sha256':sha(d/'latest_checkpoint.json'),'complete8roles_sha256':pointer['files_sha256'],'verified_utc':x['verified_utc'],'attempt_receipt':str(receipt.relative_to(R)),'attempt_receipt_sha256':sha(receipt),'failed_attempts_retained':sum(not y['complete'] for _,y in entries),'no_tensor_deserialization':True}
checks['first_pair_below_16GB']=sum(x['bytes'] for x in jobs.values())<16000000000;checks['actual_free_above_6GB']=shutil.disk_usage(R).free>6000000000
# The C successor changes only subprocess deadline clamping.
b=ast.parse((D/'monitor_b.py').read_text());c=ast.parse((D/'monitor_c.py').read_text());bf={x.name:ast.dump(x,include_attributes=False) for x in b.body if isinstance(x,ast.FunctionDef)};cf={x.name:ast.dump(x,include_attributes=False) for x in c.body if isinstance(x,ast.FunctionDef)}
checks['C_only_invoke_changed']=set(bf)==set(cf) and [n for n in bf if bf[n]!=cf[n]]==['invoke']
assert all(checks.values()),checks
out={'schema':'paid_native_first_complete_offnode_backups_v1','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS_ACTUAL_FIRST_BACKUPS_SOURCE_C_READY_FOR_ROOT_REVIEW','audit_source_sha256':sha(Path(__file__)),'source_freeze_c_sha256':sha(D/'source_freeze_c.json'),'daemon_source_c_sha256':sha(D/'monitor_c.py'),'checks':checks,'jobs':jobs,'actual_local_free_bytes':shutil.disk_usage(R).free,'long_daemon_started':False,'limits':['No tensor deserialization/forward/full-cache copies; fullstate and model bytes were streamed and SHA-checked only.','Remote first failed SwiGLU snapshot is retained; all8 role hashes completed but final stability assertion failed, consistent with concurrent pointer advance.','Local monitor is not resilient to local laptop/network loss. Remote billing guard remains independently armed for12:00UTC.','SourceC uses11:45UTC subprocess cutoff plus bounded native subprocess cleanup; no sourceA/B long daemon was launched.','No local checkpoint deletion is automatic; space guard alerts root and stops copies while preserving verified states.']}
p=E/'first_complete_backup_audit.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'checks':len(checks),'audit_sha256':sha(p),'total_verified_bytes':sum(x['bytes'] for x in jobs.values()),'source_sha256':out['audit_source_sha256']}))
