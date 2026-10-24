"""Execute new generated parent guards and sampler, preserve a real peer."""
import ast,json,subprocess,sys
from unittest.mock import patch
from pathlib import Path
from successor_driver_patch import patched_driver

original=Path('artifacts/kaggle/b011-b012-v1/run_driver.py').read_text()
body=patched_driver(original,'xsa-polar-resume');tree=ast.parse(body)
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gpu_sample')
events=[];env={'subprocess':subprocess,'emit':events.append}
exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'actual_gpu_sample','exec'),env)
peer=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'])
try:
 for error in (subprocess.TimeoutExpired('nvidia-smi',10),OSError('missing sampler')):
  with patch.object(subprocess,'run',side_effect=error):env['gpu_sample']()
  assert peer.poll() is None and events[-1]['event']=='gpu_telemetry_unavailable'
 for code in (1,0):
  with patch.object(subprocess,'run',return_value=subprocess.CompletedProcess(['nvidia-smi'],code,'sample','query failure')):env['gpu_sample']()
  assert peer.poll() is None and events[-1]['event']==('gpu_sample' if code==0 else 'gpu_telemetry_unavailable')
finally:
 peer.terminate();peer.wait(timeout=5)
guards=[n for n in ast.walk(tree) if isinstance(n,ast.Assert) and 'resumed_step==experiment' in ast.unparse(n).replace(' ','')]
assert len(guards)==1
guard=compile(ast.fix_missing_locations(ast.Module(body=guards,type_ignores=[])),'actual_parent_step_guard','exec')
for step,pin,passes in [(180000,180000,True),(60000,180000,False),(180000,60000,False)]:
 try:exec(guard,{'resumed_step':step,'experiment':{'resume_parent':{'step':pin}},'parent_directory':'parent'})
 except AssertionError:assert not passes
 else:assert passes
assert 'allow_orthogonalizer_change={spec.get' in body
assert 'check_training_resumption.py' in body and 'check_polar_muon.py' in body
assert body.count("raise RuntimeError('Successor startup exceeded240s; abort before training')")==1
print(json.dumps({'event':'xsa_polar_driver_checks_passed','actual_generated_sampler_executed':True,'timeout_oserror_nonzero_success':True,'real_peer_preserved':True,'actual_parent_step_guard_executed':True,'CUDA_oracles_declared':True,'CUDA_verified':False}))
