"""One owned local controller, active terminal notification and one idle queue wake.

Never allocates independently, retries a controller, stops compute, or scores output.
The explicit command in the root-owned config supplies the sole allocation authority.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    uuid.UUID(config["thread_id"])
    root = Path(config["root"]).resolve()
    launch = Path(config["launch"]).resolve()
    evidence = Path(config["owner_evidence"]).resolve()
    assert launch.is_relative_to(root) and evidence.is_relative_to(root)
    assert config["maximum_allocation_requests"] == 1
    started = now()
    event = hashlib.sha256((config["thread_id"] + started +
                            config["stage_manifest_sha256"]).encode()).hexdigest()[:24]
    write_new(launch / "launch_claim.json", {
        "utc": started, "one_owner_invocation": True, "never_retry_uncertain": True,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
    })
    with (launch / "owner.log").open("xb") as log:
        child = subprocess.Popen(config["command"], cwd=root, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        write_new(launch / "launch.json", {
            **config, "started_utc": started, "wrapper_pid": os.getpid(),
            "owner_pid": child.pid, "completion_event_id": event,
            "maximum_allocation_requests": 1,
            "allocation_request_actually_started": "inspect owner_result.json",
        })
        print(json.dumps({"owner_pid": child.pid, "wrapper_pid": os.getpid()}), flush=True)
        try:
            returncode = child.wait(timeout=1950)
            observer_expired = False
        except subprocess.TimeoutExpired:
            returncode = None
            observer_expired = True
    result_path = evidence / "owner_result.json"
    result_read_error = None
    try:
        result = json.loads(result_path.read_text()) if result_path.is_file() else {}
        if not isinstance(result, dict):
            raise ValueError("Owner result is not an object")
    except Exception as error:
        result = {}
        result_read_error = type(error).__name__
    record = {
        "event_id": event, "utc": now(), "thread_id": config["thread_id"],
        "session": config["session"], "owner_returncode": returncode,
        "observer_expired": observer_expired, "owner_pid": child.pid,
        "owner_still_running": child.poll() is None,
        "owner_result_read_error_type": result_read_error,
        "allocation_request_actually_started": "allocation_request_started_utc" in result,
        "owner_result_path": str(result_path.relative_to(root)),
        "owner_result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest()
        if result_path.is_file() else None,
        "actual_agent_receipt_observed": False,
    }
    write_new(launch / "completion.json", record)
    callback = root / "reports/evidence/completion_callbacks" / (event + ".json")
    write_new(callback, record)
    message = (
        f"Automated Colab sequential-reference handoff. Event {event}; session {config['session']}. "
        f"First record actual receipt at reports/evidence/completion_callbacks/{event}.receipt.json "
        "with event_id/current UTC/thread ID. Read START_HERE.md and reports/HANDOFF.md plus "
        f"{(launch / 'completion.json').relative_to(root)} and {result_path.relative_to(root)}. "
        "Inspect exact owner/owned endpoint/terminal/archive/recovery/stop state. Score only completed "
        "SHA-bound actual reference outputs; a reference path is not a cold notebook proof. "
        "Continue authorized benchmark work, no submission/new allocation authority/duplicate rerun; "
        "halt affected new compute branch on actual refusal/exhaustion. Acknowledge actual automatic "
        "resumption; if already reviewed while active, reconcile without repeating actions."
    )
    try:
        queued = subprocess.run(["/Users/a2mogus/.local/bin/codex", "queue", "--thread",
                                 config["thread_id"], "--message", message],
                                capture_output=True, text=True, timeout=30)
        enqueue = {"utc": now(), "returncode": queued.returncode,
                   "stdout": queued.stdout[-2000:], "stderr": queued.stderr[-2000:],
                   "message": message, "queue_ack_is_not_delivery": True}
    except Exception as error:
        enqueue = {"utc": now(), "error_type": type(error).__name__,
                   "message": message, "no_retry": True}
    write_new(launch / "callback_enqueue.json", enqueue)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
