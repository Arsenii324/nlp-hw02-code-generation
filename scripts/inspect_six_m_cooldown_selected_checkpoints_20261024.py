"""Weights-only CPU integrity audit for the completed 6M cooldown run."""
import hashlib
import json
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "artifacts/kaggle/6m-cooldown-20261024a/tools"
DIAGNOSTICS = ROOT / "artifacts/kaggle/six-m-cooldown-8h-v1-a-diagnostics-retry1"
MODELS = ROOT / "artifacts/kaggle/six-m-cooldown-8h-v1-a-selected-models"
EVIDENCE = ROOT / "reports/evidence/six_m_cooldown_launch_20261024a"
EXPECTED = {
    "ref": "arsen4ikvar/nlp-hw02-6m-cooldown-20261024a",
    "numeric_id": 140435753,
    "version": 1,
    "driver_sha256": "dc9c00a1f5a312f315193e36eae439aebfd56ed2104fa96843136d9fc80d879a",
    "manifest_sha256": "095642208b9853fb6ad71457f638ffd3d593d14bcde09bbe45686399e7c8473d",
    "source_sha256": "03a646beb045780da3534f2bb9f2efa97bda7bcaac5be6d8da596de2e23d3e57",
    "freeze_sha256": "23cf6dec5cf62a134c3bc247504905be07108852f485c5928caff936a2d2e04b",
    "terminal_sha256": "bb549eb45cbacf0eda66edd338c179c759dc4323c19fc9821c4e63da3eeafe5d",
    "model_source_sha256": "07e58176c39eed8ed03134074282d74589be97120e6d6569a15f27eee1538b26",
}


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    torch.set_num_threads(2)
    manifest_path = TOOLS / "gated_cooldown_continuation_manifest.json"
    archive = TOOLS / "cooldown_continuation_source.bin"
    terminal_path = DIAGNOSTICS / "session_result.json"
    normalized_path = EVIDENCE / "training_recovery_normalized_retry2.json"
    weights_receipt_path = EVIDENCE / "selected_model_recovery_retry2.json"
    manifest = json.loads(manifest_path.read_text())
    terminal = json.loads(terminal_path.read_text())
    normalized = json.loads(normalized_path.read_text())
    weights_receipt = json.loads(weights_receipt_path.read_text())
    require(sha_file(manifest_path) == EXPECTED["manifest_sha256"], "manifest SHA mismatch")
    require(sha_file(archive) == EXPECTED["source_sha256"], "source archive SHA mismatch")
    require(sha_file(terminal_path) == EXPECTED["terminal_sha256"], "terminal SHA mismatch")
    require(normalized["complete"] and normalized["version"] == 1, "normalized recovery incomplete")
    require(weights_receipt["complete"] and weights_receipt["version"] == 1, "weight recovery incomplete")
    require(weights_receipt["numeric_id"] == EXPECTED["numeric_id"] and weights_receipt["ref"] == EXPECTED["ref"], "provider identity mismatch")
    require(weights_receipt["driver_sha256"] == EXPECTED["driver_sha256"], "driver SHA mismatch")
    require(weights_receipt["manifest_sha256"] == EXPECTED["manifest_sha256"], "receipt manifest mismatch")
    require(weights_receipt["training_source_sha256"] == EXPECTED["source_sha256"], "receipt source mismatch")
    require(weights_receipt["source_freeze_sha256"] == EXPECTED["freeze_sha256"], "freeze SHA mismatch")
    require(normalized["training_terminal_sha256"] == EXPECTED["terminal_sha256"], "normalized terminal mismatch")
    require(normalized["training_source_sha256"] == EXPECTED["source_sha256"], "normalized source mismatch")
    require(normalized["training_driver_sha256"] == EXPECTED["driver_sha256"], "normalized driver mismatch")
    require(terminal["complete"] and terminal["training_complete"] and terminal["paired_complete"] and terminal["artifact_hashes_complete"], "actual terminal incomplete")
    require(terminal["driver_sha256"] == EXPECTED["driver_sha256"] and terminal["manifest_sha256"] == EXPECTED["manifest_sha256"] and terminal["source_sha256"] == EXPECTED["source_sha256"], "actual terminal identity mismatch")
    require(normalized["common_NEW_horizon"] == 180160 and normalized["origin_absolute_step"] == 345000 and normalized["prefix_NEW_iterations"] == 1024, "actual common horizon mismatch")
    require(all(x["reaped"] and x["group_exited"] for x in terminal["cleanup"]), "owned child cleanup incomplete")

    member_map = manifest["source_members_sha256"]
    results = []
    with zipfile.ZipFile(archive) as source:
        require(set(source.namelist()) == set(member_map), "archive member set differs from frozen map")
        for name, expected_sha in member_map.items():
            require(sha_bytes(source.read(name)) == expected_sha, f"archive member SHA mismatch: {name}")
        with tempfile.TemporaryDirectory(prefix="six-m-cooldown-model-audit-") as temp_name:
            temp = Path(temp_name)
            for name in ("src/nlp_hw02/__init__.py", "src/nlp_hw02/model.py", "src/nlp_hw02/tokenizer.py"):
                target = temp / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read(name))
            sys.path.insert(0, str(temp / "src"))
            from nlp_hw02.model import ModelConfig, Seq2Seq

            for run_id in ("cooldown-full", "cooldown-half"):
                run = normalized["runs"][run_id]
                require(run["complete"] and run["actual_exit"] == 0 and run["admitted_NEW_horizon"] == 180160, f"run incomplete: {run_id}")
                pointer_path = DIAGNOSTICS / f"run-{run_id}/latest_checkpoint.json"
                pointer = json.loads(pointer_path.read_text())
                slot = pointer["slot"]
                relpath = f"run-{run_id}/{slot}/model.pt"
                model_path = MODELS / relpath
                digest = sha_file(model_path)
                model_record = weights_receipt["files"][relpath]
                require(digest == model_record["sha256"] == pointer["files_sha256"]["model.pt"] == terminal["artifact_sha256"][relpath], f"model SHA mismatch: {run_id}")
                require(model_path.stat().st_size == model_record["bytes"] == 70160913, f"model size mismatch: {run_id}")
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
                require(set(checkpoint) == {"config", "model", "step", "validation_loss", "seed", "dataset_revision"}, f"checkpoint fields mismatch: {run_id}")
                run_config = json.loads((DIAGNOSTICS / f"run-{run_id}/run_config.json").read_text())
                config = run_config["config"]
                require(checkpoint["config"] == config and checkpoint["seed"] == run_config["seed"] == 42, f"config/seed mismatch: {run_id}")
                require(checkpoint["dataset_revision"] == run_config["data"]["dataset_revision"] == "b2724689ff079e9c22efd548fe134bfb29036dbb", f"dataset revision mismatch: {run_id}")
                require(checkpoint["step"] == pointer["metrics"]["selected_step"] == run["selected_step"], f"selected step mismatch: {run_id}")
                require(checkpoint["validation_loss"] == pointer["metrics"]["best_validation_loss"] == run["best_validation_loss"], f"selected loss mismatch: {run_id}")

                model = Seq2Seq(ModelConfig(**config))
                state = checkpoint["model"]
                expected_state = model.state_dict()
                require(set(state) == set(expected_state), f"state keys mismatch: {run_id}")
                require(all(state[k].shape == expected_state[k].shape and state[k].dtype == expected_state[k].dtype and bool(torch.isfinite(state[k]).all()) for k in expected_state), f"shape/dtype/finite mismatch: {run_id}")
                aliases = {}
                for name, parameter in model.named_parameters(remove_duplicate=False):
                    aliases.setdefault(id(parameter), []).append(name)
                alias_groups = [names for names in aliases.values() if len(names) > 1]
                require(all(all(torch.equal(state[names[0]], state[name]) for name in names[1:]) for names in alias_groups), f"tied alias mismatch: {run_id}")
                model.load_state_dict(state, strict=True)
                require(all(torch.equal(value, state[name]) for name, value in model.state_dict().items()), f"strict-load mismatch: {run_id}")
                parameter_count = sum(parameter.numel() for parameter in model.parameters())
                xsa_names = [name for name, _ in model.named_parameters() if name.endswith(".xsa_alpha")]
                require(parameter_count == 17524256 and sum(model.state_dict()[name].numel() for name in xsa_names) == 32, f"parameter/XSA mismatch: {run_id}")
                require(len(model.encoder) == len(model.decoder) == 4 and model.encoder_repeat_gates is None and model.decoder_repeat_gates is None, f"topology mismatch: {run_id}")

                result = {
                    "schema": "six_m_cooldown_selected_weights_cpu_integrity_v1",
                    "complete": True,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "training_terminal_sha256": EXPECTED["terminal_sha256"],
                    "training_source_sha256": EXPECTED["source_sha256"],
                    "training_manifest_sha256": EXPECTED["manifest_sha256"],
                    "training_driver_sha256": EXPECTED["driver_sha256"],
                    "source_freeze_sha256": EXPECTED["freeze_sha256"],
                    "run_id": run_id,
                    "slot": slot,
                    "model_sha256": digest,
                    "model_bytes": model_path.stat().st_size,
                    "model_source_sha256": EXPECTED["model_source_sha256"],
                    "config": config,
                    "step": checkpoint["step"],
                    "validation_loss": checkpoint["validation_loss"],
                    "seed": checkpoint["seed"],
                    "dataset_revision": checkpoint["dataset_revision"],
                    "parameter_count": parameter_count,
                    "state_tensor_count": len(state),
                    "checkpoint_fields": sorted(checkpoint),
                    "finite_state": True,
                    "strict_load": True,
                    "alias_groups": alias_groups,
                    "xsa_alpha_tensors": len(xsa_names),
                    "xsa_alpha_scalars": 32,
                    "protocol_scope": {
                        "common_NEW_horizon": 180160,
                        "origin_absolute_step": 345000,
                        "prefix_NEW_iterations": 1024,
                        "requested_NEW_cap": 196608,
                        "actual_NEW_steps": run["actual_model_iterations"],
                        "durable_absolute_step": run["durable_absolute_step"],
                        "final_scheduled_lr": run["last_scheduled_lr"],
                        "raw_prefix_sha256": run["prefix_raw_segment_sha256"],
                        "applied_prefix_sha256": run["prefix_applied_segment_sha256"],
                        "postprefix_decision": run["postprefix_recheck"]["decision"],
                        "postprefix_recheck_step": run["postprefix_recheck"]["NEW_step"],
                        "cleanup_groups_reaped": all(x["reaped"] and x["group_exited"] for x in terminal["cleanup"]),
                    },
                    "scope": "Weights-only CPU tensor integrity and strict-load audit. No forward, inference, optimizer/fullstate load, arrays, or GPU use. Training/protocol facts are bound to actual terminal and normalized recovery receipts.",
                }
                output = EVIDENCE / f"selected_{run_id}_cpu_integrity.json"
                require(not output.exists(), f"refusing to overwrite {output.name}")
                output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
                results.append({"run_id": run_id, "receipt": str(output.relative_to(ROOT)), "receipt_sha256": sha_file(output), "model_sha256": digest, "step": checkpoint["step"], "validation_loss": checkpoint["validation_loss"], "parameters": parameter_count})
                del model, checkpoint, state, expected_state
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
