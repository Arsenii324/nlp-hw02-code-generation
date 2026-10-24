"""One local-state failure alert; no provider calls, allocation or stop."""
from pathlib import Path
import os,json,time,hashlib,datetime,subprocess
R=Path(__file__).resolve().parents[1];E=R/'reports/evidence/colab_continuation_launch_20261023i2';S=R/'artifacts/colab/continuation-i2-20261023g/owner-result.json';thread='01a0f8a6-39b5-7192-9210-9b807ffa438f';pid=json.loads((E/'owner_launch.json').read_text())['pid'];end=datetime.datetime(2026,10,23,22,30,tzinfo=datetime.timezone.utc).timestamp();out=E/'early_failure_callback.json';assert not out.exists()
(E/'early_failure_monitor.json').write_text(json.dumps({'pid':os.getpid(),'controller_pid':pid,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'provider_calls':False,'maximum_callbacks':1},indent=2)+'\n')
while time.time()<end:
 if(E/'completion.json').exists():break
 try:os.kill(pid,0)
 except ProcessLookupError:break
 try:r=json.loads(S.read_text())
 except (FileNotFoundError,json.JSONDecodeError):time.sleep(15);continue
 terminal=(r.get('latest_kernel_heartbeat') or {}).get('terminal')
 bad=bool(r.get('failure') or r.get('unexpected_owner_error_type') or r.get('requires_urgent_root_attention') or (isinstance(terminal,dict) and terminal.get('complete') is False))
 if bad:
  event=hashlib.sha256(('colab-I2-urgent|'+r.get('owner_manifest_sha256','')).encode()).hexdigest()[:24]
  message=f'Automated urgent Colab owner-state handoff. Event {event}. First record actual receipt in reports/evidence/completion_callbacks/{event}.receipt.json with UTC/thread ID. Read START_HERE.md, reports/HANDOFF.md and artifacts/colab/continuation-i2-20261023g/owner-result.json. A local-state failure flag or incomplete terminal was observed while the owner may still be active. Inspect exact current owned endpoint/process/recovery before action. No duplicate allocation/rerun, submission, destructive cleanup or new authorization. Respect latest user instructions; halt affected compute on actual refusal/exhaustion. If already resolved, reconcile without repeating actions.'
  p=subprocess.run(['/Users/a2mogus/.local/bin/codex','queue','--thread',thread,'--message',message],cwd=R,capture_output=True,text=True,timeout=120);out.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event_id':event,'enqueue_succeeded':p.returncode==0,'returncode':p.returncode,'agent_receipt_observed':False,'stdout':p.stdout,'stderr':p.stderr},indent=2)+'\n');break
 time.sleep(15)
