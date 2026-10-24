"""Strict CPU tensor integrity for the two selected full-range weights; no forward pass."""
from __future__ import annotations
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "artifacts/kaggle/fullrange-cooldown-v1-c-diagnostics-retry1"
WEIGHTS = ROOT / "artifacts/kaggle/fullrange-cooldown-v1-c-selected-models"
RUNTIME = DIAG / "fullrange_bucket_cooldown_runtime"
MODEL_SOURCE_SHA = "07e58176c39eed8ed03134074282d74589be97120e6d6569a15f27eee1538b26"
TERMINAL_SHA = "18647c3cb0bb0fe263cdb8b3e741573c0b2d758babda00d2807999f3bf830afc"
DRIVER_SHA = "fb7b52c0ed3d823b2710db8dcef918cf807ef169770ed7f56b3be4183de83f08"
MANIFEST_SHA = "a596fb5228854491c1781ca668c87d9b95cd2c235a35b753cec6f606eb81529f"
REVISION = "b2724689ff079e9c22efd548fe134bfb29036dbb"
ARMS = {'full-remaining': {'model_sha256': '0b1366c37e05f7720e6edcd60a6ae56949c64e183fc40769b8844aef0c813407', 'step': 373480, 'validation_loss': 0.24646930431611383}, 'half-remaining': {'model_sha256': 'ee0fa73eb3aad900c0a78066a3cad6748663a73a7a6c1b0e66dda3dca5c7043f', 'step': 373480, 'validation_loss': 0.24691140272223983}}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def inspect() -> dict:
    torch.set_num_threads(2)
    terminal_path = DIAG / "session_result.json"
    require(sha(terminal_path) == TERMINAL_SHA, "actual terminal SHA mismatch")
    terminal = json.loads(terminal_path.read_text())
    require(terminal.get("complete") and terminal.get("training_complete") and terminal.get("artifact_hashes_complete"), "terminal incomplete")
    require(terminal.get("driver_sha256") == DRIVER_SHA and terminal.get("manifest_sha256") == MANIFEST_SHA, "terminal driver/manifest mismatch")
    model_source = RUNTIME / "src/nlp_hw02/model.py"
    require(sha(model_source) == MODEL_SOURCE_SHA, "actual model source SHA mismatch")

    sys.path.insert(0, str(RUNTIME / "src"))
    from nlp_hw02.model import ModelConfig, Seq2Seq

    receipts = {}
    for arm, expected in ARMS.items():
        runroot = DIAG / f"run-{arm}"
        pointer_path = runroot / "latest_checkpoint.json"
        config_path = runroot / "run_config.json"
        pointer_rel = f"run-{arm}/latest_checkpoint.json"
        config_rel = f"run-{arm}/run_config.json"
        require(sha(pointer_path) == terminal["artifact_sha256"].get(pointer_rel), f"{arm}: pointer not terminal-bound")
        require(sha(config_path) == terminal["artifact_sha256"].get(config_rel), f"{arm}: run config not terminal-bound")
        pointer = json.loads(pointer_path.read_text())
        run_config = json.loads(config_path.read_text())
        require(pointer.get("slot") == "durable-slot0", f"{arm}: selected slot mismatch")
        model_rel = f"run-{arm}/durable-slot0/model.pt"
        model_path = WEIGHTS / model_rel
        digest = sha(model_path)
        require(digest == expected["model_sha256"] == terminal["artifact_sha256"].get(model_rel), f"{arm}: model SHA mismatch")
        require(pointer.get("files_sha256", {}).get("model.pt") == digest, f"{arm}: pointer model SHA mismatch")

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        required_fields = {"config", "model", "step", "validation_loss", "seed", "dataset_revision"}
        require(set(checkpoint) == required_fields, f"{arm}: checkpoint field set mismatch")
        require(checkpoint["config"] == run_config.get("config"), f"{arm}: config differs from terminal-bound run config")
        require(checkpoint["step"] == expected["step"] == pointer["metrics"].get("selected_step"), f"{arm}: selected step mismatch")
        require(checkpoint["validation_loss"] == expected["validation_loss"] == pointer["metrics"].get("best_validation_loss"), f"{arm}: CE-best loss mismatch")
        require(checkpoint["seed"] == run_config.get("seed") == 42, f"{arm}: seed mismatch")
        require(checkpoint["dataset_revision"] == run_config.get("data", {}).get("dataset_revision") == REVISION, f"{arm}: dataset revision mismatch")
        require(run_config.get("parameters") == 17_524_256, f"{arm}: terminal-bound parameter count mismatch")

        model = Seq2Seq(ModelConfig(**checkpoint["config"]))
        expected_state = model.state_dict()
        state = checkpoint["model"]
        require(set(expected_state) == set(state), f"{arm}: state key set mismatch")
        require(all(state[k].shape == expected_state[k].shape and state[k].dtype == expected_state[k].dtype for k in expected_state), f"{arm}: state shape/dtype mismatch")
        require(all(bool(torch.isfinite(t).all()) for t in state.values()), f"{arm}: nonfinite state tensor")
        aliases: dict[int, list[str]] = {}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            aliases.setdefault(id(parameter), []).append(name)
        alias_groups = [names for names in aliases.values() if len(names) > 1]
        require(all(all(torch.equal(state[names[0]], state[name]) for name in names[1:]) for names in alias_groups), f"{arm}: aliased checkpoint weights differ")
        model.load_state_dict(state, strict=True)
        require(all(torch.equal(value, state[name]) for name, value in model.state_dict().items()), f"{arm}: strict state reload differs")
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        xsa = {name: parameter.detach().tolist() for name, parameter in model.named_parameters() if name.endswith(".xsa_alpha")}
        require(parameter_count == 17_524_256 and sum(len(values) for values in xsa.values()) == 32, f"{arm}: model parameter/XSA count mismatch")
        require(len(model.encoder) == len(model.decoder) == 4 and model.encoder_repeat_gates is None and model.decoder_repeat_gates is None, f"{arm}: encoder-decoder topology mismatch")

        record = {
            "schema": "fullrange_selected_weights_cpu_tensor_integrity_v1",
            "complete": True,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "terminal_sha256": TERMINAL_SHA,
            "driver_sha256": DRIVER_SHA,
            "manifest_sha256": MANIFEST_SHA,
            "model_source_sha256": MODEL_SOURCE_SHA,
            "model_sha256": digest,
            "pointer_sha256": sha(pointer_path),
            "run_config_sha256": sha(config_path),
            "run_id": arm,
            "slot": pointer["slot"],
            "step": checkpoint["step"],
            "validation_loss": checkpoint["validation_loss"],
            "seed": checkpoint["seed"],
            "dataset_revision": checkpoint["dataset_revision"],
            "config": checkpoint["config"],
            "parameters": parameter_count,
            "checkpoint_fields": sorted(checkpoint),
            "finite_state_tensors": True,
            "strict_alias_check": True,
            "unique_xsa_scalars": 32,
            "state_tensor_count": len(state),
            "alias_groups": alias_groups,
            "verifier_python": sys.version.split()[0],
            "verifier_torch": torch.__version__,
            "scope": "Weights-only CPU tensor inspection and strict state loading. No forward pass, CUDA, training, inference, optimizer/fullstate read, arrays or generated code execution. This is tensor integrity, not the producer's complete protocol receipt.",
        }
        out = ROOT / "reports/evidence/fullrange_bucket_cooldown_launch_20261023c" / f"selected_{arm}_cpu_tensor_integrity.json"
        require(not out.exists(), f"{arm}: preserve existing integrity receipt")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
        receipts[arm] = {"path": str(out.relative_to(ROOT)), "receipt_sha256": sha(out), "model_sha256": digest, "step": checkpoint["step"], "validation_loss": checkpoint["validation_loss"], "parameters": parameter_count, "state_tensor_count": len(state), "alias_group_count": len(alias_groups)}
        del model, checkpoint, state, expected_state
    return receipts


if __name__ == "__main__":
    print(json.dumps(inspect(), indent=2, sort_keys=True))
