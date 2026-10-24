"""Static author audit only; no candidate imports or candidate function execution."""
import ast,hashlib,json,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];W=R/'.worktrees/strong-inference-fallback-20261022'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def functions(s):return {n.name:n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
def dump(n):return ast.dump(n,include_attributes=False)
def verify():
 b=(W/'scripts/bind_wide_fullpopulation_validation_a.py').read_text();p=(W/'scripts/wide_fullpopulation_inference_protocol_a.py').read_text();donor=(W/'scripts/bind_own_wide_validation_b.py').read_text();origin=(W/'src/nlp_hw02/wide_fullpopulation_a.py').read_text();schedule=(W/'src/nlp_hw02/wide_continuation_schedule_a.py').read_text();bf,pf,df,of,sf=map(functions,[b,p,donor,origin,schedule]);checks={}
 checks['both_new_modules_AST_parse']=bool(ast.parse(b)) and bool(ast.parse(p))
 checks['protocol_stdlib_only']=all(n.module in ('pathlib',) for n in ast.parse(p).body if isinstance(n,ast.ImportFrom)) and all(a.name in {'datetime','hashlib','json','math','zipfile'} for n in ast.parse(p).body if isinstance(n,ast.Import) for a in n.names)
 checks['six_binder_helpers_AST_exact']=all(dump(bf[n])==dump(df[n]) for n in ['require','sha','read','write','clone','ref'])
 begin=' # Explicit authorized CPU model.pt only; never last_training.pt and never forward.';end=' out=a.output.resolve();';checks['entire_selected_CPU_block_byte_exact']=b[b.index(begin):b.index(end)]==donor[donor.index(begin):donor.index(end)]
 checks['bind_function_AST_exact']=dump(bf['bind'])==dump(df['bind'])
 checks['recipe_and_parent_data_comparison_AST_exact']=all(dump(pf[n])==dump(of[n]) for n in ['recipe_for','compare_parent_data'])
 checks['schedule_functions_AST_exact']=all(dump(pf[n])==dump(sf[n]) for n in ['validate_schedule','scheduled_lr'])
 helpers=(W/'configs/own_wide_validation_source_a_20261024.json');f=json.loads(helpers.read_text());checks['unchanged_generic_producer_freeze']=sha(helpers)=='c19dc11506fe3cbf1f9b90f12554c37ded8690706f1e16e9ab14c3171f93c770' and all(sha(W/n)==h for n,h in f['source_sha256'].items())
 checks['original512_binder_A_preserved']=sha(W/'scripts/bind_own_wide_validation_a.py')=='a574d964c2770c22dd26c9a708d9ac92fd6badd122f293d79b186a87e3853a00'
 checks['original512_binder_B_preserved']=sha(W/'scripts/bind_own_wide_validation_b.py')=='ba703eae70fec7128c033b6f07d5a64a223597a7ed92543ce64cd7aedd9afb25'
 checks['training_source_freeze_exact']=sha(W/'configs/wide_fullpopulation_source_a_20261024.json')=='ac069fca9e920ba538854175b1aa6add12a1a2e1460512bf287093ae866333e4'
 receipt=json.loads((R/'reports/evidence/native_fullpopulation_prepare_20261024a/source_archive_prepared.json').read_text());archive=R/receipt['archive']
 with zipfile.ZipFile(archive) as z:checks['actual_prepared73_source_members_exact']=sha(archive)=='27a4c82d2fdfb86eac03b7b82c91ac215ea048f6bfbe455555797ff69a8430bb' and len(z.namelist())==len(set(z.namelist()))==73 and all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in receipt['member_sha256'].items())
 checks['NULL_input_template']=json.loads((W/'configs/wide_fullpopulation_inference_input_a.template.json').read_text())['verified_actual_values'] is False
 checks['NULL_owner_status_template']=all(json.loads((W/'configs/wide_fullpopulation_actual_owner_status_a.template.json').read_text())[k] is None for k in ['observed_UTC','contract_sha256','snapshot_pointer_sha256','observed_durable_step','worker_group_alive'])
 checks['source_guard_before_CLI_dispatch']=b.index('verify_binder_source(a.binder_source_freeze_sha256);')<b.index('prepare(a) if a.action')
 assert all(checks.values());return checks
if __name__=='__main__':print(json.dumps(verify(),indent=2))
