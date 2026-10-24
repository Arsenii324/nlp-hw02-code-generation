"""One finite local wake; never performs provider actions or allocates compute."""
import datetime,json,time,subprocess,hashlib,fcntl,os
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'reports/evidence/quota_reset_wake_20261024a';D.mkdir(exist_ok=True)
thread='01a0f8a6-39b5-7192-9210-9b807ffa438f';target=datetime.datetime(2026,10,24,tzinfo=datetime.timezone.utc);event=hashlib.sha256((thread+'|quota-reset|2026-10-24').encode()).hexdigest()[:24]
with (D/'owner.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (D/'enqueue_result.json').exists(),'Already attempted; no duplicate wake'
 state={'pid':os.getpid(),'thread_id':thread,'event_id':event,'target_utc':target.isoformat(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'provider_actions':False,'agent_receipt_observed':False}
 (D/'owner.json').write_text(json.dumps(state,indent=2)+'\n')
 while True:
  left=(target-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
  if left<=0:break
  time.sleep(min(60,left))
 message=f'Automated quota-reset readiness handoff. Event {event}. First record actual receipt in reports/evidence/completion_callbacks/{event}.receipt.json with current UTC/thread ID. Read START_HERE.md and reports/HANDOFF.md plus docs/plans/2026-10-23-post-reset-measured-budget.md. Reconcile current callbacks and latest user instructions; check actual quota reset/remaining deadline/current provider identities/private input readback/free slots before any already-authorized action. Two useful GPU kernels max, eachT4x2; halt affected compute on actual unavailable/exhausted; no submission/destructive cleanup. This wake provides no new authorization. Do not duplicate existing allocations/observers. Enqueue acknowledgment is not delivery receipt.'
 try:
  p=subprocess.run(['/Users/a2mogus/.local/bin/codex','queue','--thread',thread,'--message',message],cwd=R,capture_output=True,text=True,timeout=120);result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event_id':event,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'enqueue_succeeded':p.returncode==0,'agent_receipt_observed':False}
 except Exception as e:result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event_id':event,'enqueue_succeeded':False,'error_type':type(e).__name__,'agent_receipt_observed':False}
 (D/'enqueue_result.json').write_text(json.dumps(result,indent=2)+'\n')
