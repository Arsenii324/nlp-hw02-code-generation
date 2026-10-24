import ast,hashlib,json
from pathlib import Path
r=Path('/Users/a2mogus/build-projs/dl2-hse/nlp-hw02-code-generation');w=r/'.worktrees/strong-inference-fallback-20261022'
files=['src/nlp_hw02/wide_kaggle_cooldown_a.py','scripts/wide_kaggle_cooldown_coordination_a.py','scripts/train_wide_kaggle_cooldown_a.py','scripts/kaggle_wide_cooldown_pair_a.py','scripts/build_kaggle_wide_cooldown_pair_a.py']
for n in files:ast.parse((w/n).read_text())
def tree(n):return ast.parse((w/n).read_text())
def funcs(n):return {a.name:ast.dump(a,include_attributes=False) for a in tree(n).body if isinstance(a,(ast.FunctionDef,ast.ClassDef))}
checks={n+':parses':True for n in files}
a=funcs('scripts/train_wide_continuation_a.py');b=funcs(files[2]);checks['worker_utility_AST_exact']=all(a[n]==b[n] for n in ['require','sha','fsync_directory','atomic_json','atomic_torch_save'])
a=funcs('scripts/kaggle_half_low_lr_pair_a.py');b=funcs(files[3]);same=['AbsoluteDeadline','arm_absolute','absolute_alarm','sha','emit','start','stop','require','fsync_directory','atomic_json','terminate','finalize','locate'];checks['parent13_lifecycle_AST_exact']=all(a[n]==b[n] for n in same)
# Whole 64-update loop is byte-identical; LR call uses a new schedule object outside loop.
s=(w/'scripts/train_wide_continuation_a.py').read_text();t=(w/files[2]).read_text();cut=lambda x:x[x.index('        for batch_index in range(64):'):x.index('        torch.cuda.synchronize();window_seconds=',x.index('        for batch_index in range(64):'))];checks['entire64_update_loop_byte_exact']=cut(s)==cut(t)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checks['strictloader_unchanged']=sha(w/'src/nlp_hw02/wide_origin0_resume_a.py')=='b0b3f497ae981df396226395cedbe764b1c302cd99b25128f41387a686446ed6'
import zipfile
p=r/'artifacts/kaggle/wide-t4-budget-20261024a/tools/roles/gelu512/evidence/source.bin'
with zipfile.ZipFile(p) as z:
 members={n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()};checks['actual_native64_foundation']=sha(p)=='97494fb88adab12d244716a782b079b74f3072ae11f5f17166c5f22159c857bd' and len(members)==64
 additions=[*files[:4],'src/nlp_hw02/wide_origin0_resume_a.py'];checks['five_additions_no_foundation_replacement']=not(set(additions)&set(members)) and len(set(members)|set(additions))==69
checks['all_templates_unbound']=all(json.loads((w/'configs'/n).read_text()).get('verified_actual_values',False) is False for n in ['wide_kaggle_cooldown_training_a.template.json','wide_kaggle_cooldown_parent_a.template.json','wide_kaggle_cooldown_stage_input_a.template.json'])
print(json.dumps(checks,indent=2));assert all(checks.values())
