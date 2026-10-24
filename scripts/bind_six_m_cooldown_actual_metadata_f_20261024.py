"""Bind actual 6M cooldown facts through the frozen F-family metadata adapter."""
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
WORKTREE = ROOT / ".worktrees/strong-inference-fallback-20261022"
TOOLS = ROOT / "artifacts/kaggle/6m-cooldown-20261024a/tools"
DIAGNOSTICS = ROOT / "artifacts/kaggle/six-m-cooldown-8h-v1-a-diagnostics-retry1"
MODELS = ROOT / "artifacts/kaggle/six-m-cooldown-8h-v1-a-selected-models"
INPUTS = ROOT / "artifacts/kaggle/cooldown-paired-n24-20261023a/tools/inputs"
EVIDENCE = ROOT / "reports/evidence/six_m_cooldown_launch_20261024a"
ADAPTER_PATH = WORKTREE / "scripts/cooldown_sampling_contract_f.py"
SOURCE_FREEZE_PATH = WORKTREE / "configs/cooldown_pair_sampling_source_f_20261024.json"
EXPECTED = {
    "training_source": "03a646beb045780da3534f2bb9f2efa97bda7bcaac5be6d8da596de2e23d3e57",
    "training_driver": "dc9c00a1f5a312f315193e36eae439aebfd56ed2104fa96843136d9fc80d879a",
    "training_manifest": "095642208b9853fb6ad71457f638ffd3d593d14bcde09bbe45686399e7c8473d",
    "source_freeze": "23cf6dec5cf62a134c3bc247504905be07108852f485c5928caff936a2d2e04b",
    "paired_source_freeze": "f61f71aa329860e839a7b8820c221dbb5c15c5f1eafb65135a142cda78185a0e",
    "terminal": "bb549eb45cbacf0eda66edd338c179c759dc4323c19fc9821c4e63da3eeafe5d",
    "normalized_recovery": "39c9c6388754b472ae8912ec11c3d61c9ea07aa2249eafdfb8581fa7f8e20230",
    "source_model": "07e58176c39eed8ed03134074282d74589be97120e6d6569a15f27eee1538b26",
}
INPUT_NAMES = {
    "validation_questions": "validation_questions.json",
    "parity_pairs": "actual_first4_parity_pairs.json",
    "test_parquet": "test.parquet",
    "legacy_selector_source": "sampling_candidate.py",
}
RUNS = ("cooldown-full", "cooldown-half")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise FileExistsError(path)
        return
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    assert sha(ADAPTER_PATH) == "1a7514857e295471223483ad6f2533fc16401928f70e3e47861c51fbede5647a"
    assert sha(SOURCE_FREEZE_PATH) == EXPECTED["paired_source_freeze"]
    freeze = read(SOURCE_FREEZE_PATH)
    assert all(sha(WORKTREE / name) == digest for name, digest in freeze["source_sha256"].items())
    spec_ref = importlib.util.spec_from_file_location("cooldown_sampling_contract_f_actual_6m", ADAPTER_PATH)
    adapter = importlib.util.module_from_spec(spec_ref)
    spec_ref.loader.exec_module(adapter)
    assert "torch" not in sys.modules

    terminal_path = DIAGNOSTICS / "session_result.json"
    recovery_path = DIAGNOSTICS / "training_recovery_normalized_f.json"
    source_recovery_path = EVIDENCE / "training_recovery_normalized_f.json"
    if not recovery_path.exists():
        shutil.copyfile(source_recovery_path, recovery_path)
    terminal = read(terminal_path)
    recovery = read(recovery_path)
    manifest = read(TOOLS / "gated_cooldown_continuation_manifest.json")
    assert sha(terminal_path) == EXPECTED["terminal"]
    assert sha(recovery_path) == EXPECTED["normalized_recovery"]
    assert terminal["complete"] and terminal["training_complete"] and terminal["paired_complete"]
    assert terminal["source_sha256"] == EXPECTED["training_source"]
    assert terminal["driver_sha256"] == EXPECTED["training_driver"]
    assert terminal["manifest_sha256"] == EXPECTED["training_manifest"]
    assert all(item["reaped"] and item["group_exited"] and not item["errors"] for item in terminal["cleanup"])
    mapped_manifest = DIAGNOSTICS / "gated_cooldown_continuation_manifest.json"
    if not mapped_manifest.exists():
        mapped_manifest.write_bytes((TOOLS / "gated_cooldown_continuation_manifest.json").read_bytes())
    assert sha(mapped_manifest) == EXPECTED["training_manifest"] == terminal["artifact_sha256"]["gated_cooldown_continuation_manifest.json"]

    # Source-policy lineages bind the observed family facts but remain proposals.
    lineages = {}
    cpu_records = {}
    fixed_role_projection = {}
    cached_metadata = ROOT / "artifacts/kaggle/fullrange-paired-20261023a/tools/parent_metadata"
    for run_id in RUNS:
        cpu_path = EVIDENCE / f"selected_{run_id}_cpu_tensor_audit_f.json"
        cpu = read(cpu_path)
        run = next(row for row in terminal["runs"] if row["run_id"] == run_id)
        pointer_path = DIAGNOSTICS / f"run-{run_id}/latest_checkpoint.json"
        pointer = read(pointer_path)
        model_rel = f"run-{run_id}/{pointer['slot']}/model.pt"
        assert cpu["model_sha256"] == pointer["files_sha256"]["model.pt"] == terminal["artifact_sha256"][model_rel]
        assert cpu["step"] == run["metrics"]["selected_step"] == pointer["metrics"]["selected_step"]
        assert cpu["validation_loss"] == run["metrics"]["best_validation_loss"] == pointer["metrics"]["best_validation_loss"]
        slot_prefix = f"run-{run_id}/{pointer['slot']}/"
        fixed_role_projection[run_id] = {}
        for filename in ("tokenizer.json", "validation.json", "validation_legacy.json", "data_stats.json"):
            source_path = cached_metadata / filename
            expected_sha = pointer["files_sha256"][filename]
            assert sha(source_path) == expected_sha == terminal["artifact_sha256"][slot_prefix + filename]
            destination = DIAGNOSTICS / (slot_prefix + filename)
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source_path.read_bytes())
            assert sha(destination) == expected_sha
            fixed_role_projection[run_id][filename] = {
                "source": str(source_path.relative_to(ROOT)),
                "source_sha256": sha(source_path),
                "destination": str(destination.relative_to(ROOT)),
                "destination_sha256": sha(destination),
            }
        model_source = MODELS / model_rel
        model_destination = DIAGNOSTICS / model_rel
        if not model_destination.exists():
            model_destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(model_source, model_destination)
        assert sha(model_destination) == cpu["model_sha256"]
        fixed_role_projection[run_id]["model.pt"] = {
            "source": str(model_source.relative_to(ROOT)),
            "destination": str(model_destination.relative_to(ROOT)),
            "sha256": sha(model_destination),
            "materialized_as": "hardlink; no second model byte copy",
        }
        lineage = {
            "schema": "cooldown_actual_training_lineage_proposal_f_v1",
            "complete": True,
            "root_accepted": False,
            "status": "FACTS_BOUND_PROPOSAL_PENDING_ROOT_ACCEPTANCE",
            "training_terminal_sha256": EXPECTED["terminal"],
            "training_recovery_sha256": EXPECTED["normalized_recovery"],
            "training_source_sha256": EXPECTED["training_source"],
            "training_manifest_sha256": EXPECTED["training_manifest"],
            "training_driver_sha256": EXPECTED["training_driver"],
            "run_id": run_id,
            "slot": pointer["slot"],
            "pointer_sha256": sha(pointer_path),
            "selected_model_sha256": cpu["model_sha256"],
            "selected_step": cpu["step"],
            "selected_validation_loss": cpu["validation_loss"],
            "model_source_sha256": EXPECTED["source_model"],
            "cpu_integrity_receipt_sha256": sha(cpu_path),
            "parameters": cpu["parameters"],
            "unique_xsa_scalars": cpu["unique_xsa_scalars"],
        }
        lineage_path = EVIDENCE / f"lineage_{run_id}_proposal_f.json"
        save(lineage_path, lineage)
        lineages[run_id] = {"path": str(lineage_path.relative_to(ROOT)), "sha256": sha(lineage_path)}
        cpu_records[run_id] = {"path": str(cpu_path.relative_to(ROOT)), "sha256": sha(cpu_path)}

    expected_inputs = {
        "validation_questions.json": adapter.INPUT_PINS["validation_questions"],
        "actual_first4_parity_pairs.json": adapter.INPUT_PINS["parity_pairs"],
        "test.parquet": adapter.INPUT_PINS["test_parquet"],
        "sampling_candidate.py": adapter.INPUT_PINS["legacy_selector_source"],
    }
    assert set(INPUT_NAMES.values()) == set(expected_inputs)
    for name, digest in expected_inputs.items():
        assert sha(INPUTS / name) == digest

    all_identities = {}
    source_projection = {}
    for run_id in RUNS:
        cpu = read(EVIDENCE / f"selected_{run_id}_cpu_tensor_audit_f.json")
        pointer_path = DIAGNOSTICS / f"run-{run_id}/latest_checkpoint.json"
        pointer = read(pointer_path)
        contract_path = TOOLS / next(row for row in manifest["runs"] if row["run_id"] == run_id)["contract_path"]
        spec = read(WORKTREE / "configs/cooldown_sampling_producer_f.template.json")
        spec.update(verified_actual_values=True, samples_per_question=24, input_root=str(ROOT))
        spec["reviewed_source"]["sha256"] = EXPECTED["paired_source_freeze"]
        spec["training"].update(
            artifact_root=str(DIAGNOSTICS),
            tools_root=str(TOOLS),
            terminal={"path": "session_result.json", "sha256": EXPECTED["terminal"]},
            run_id=run_id,
            pointer_sha256=sha(pointer_path),
            recovery={"path": "training_recovery_normalized_f.json", "sha256": EXPECTED["normalized_recovery"]},
            root_lineage=lineages[run_id],
        )
        spec["selected"].update(model_sha256=cpu["model_sha256"], config=cpu["config"], step=cpu["step"], validation_loss=cpu["validation_loss"])
        spec["root_cpu_integrity"] = {"path": cpu_records[run_id]["path"], "sha256": cpu_records[run_id]["sha256"]}
        spec["inputs"] = {key: {"path": str((INPUTS / filename).relative_to(ROOT)), "sha256": expected_inputs[filename]} for key, filename in INPUT_NAMES.items()}
        spec_path = EVIDENCE / f"{run_id}_producer_f_actual_proposal.json"
        save(spec_path, spec)
        assert spec["backend_policy"] == adapter.BACKEND_POLICY
        identity, files, inputs, source_archive = adapter.verify_training(spec)
        assert "torch" not in sys.modules
        assert identity["model_sha256"] == cpu["model_sha256"] and identity["step"] == cpu["step"]
        assert identity["validation_loss"] == cpu["validation_loss"] and identity["root_lineage_sha256"] == lineages[run_id]["sha256"]
        assert identity["common_NEW_horizon"] == 180160 and identity["cooldown_arm"] == {"cooldown-full": "full-remaining", "cooldown-half": "half-remaining"}[run_id]
        identity_path = EVIDENCE / f"{run_id}_adapter_identity_proposal_f.json"
        save(identity_path, identity)
        source_projection[run_id] = {
            "spec": {"path": str(spec_path.relative_to(ROOT)), "sha256": sha(spec_path)},
            "identity": {"path": str(identity_path.relative_to(ROOT)), "sha256": sha(identity_path)},
            "verified_file_count": len(files),
            "verified_input_roles": sorted(inputs),
            "training_source_archive_sha256": sha(source_archive),
            "root_accepted": False,
        }
        all_identities[run_id] = identity

    assert "torch" not in sys.modules
    projection = {
        "schema": "six_m_cooldown_actual_f_metadata_projection_v1",
        "complete": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "adapter_path": str(ADAPTER_PATH.relative_to(ROOT)),
        "adapter_sha256": sha(ADAPTER_PATH),
        "source_freeze_sha256": EXPECTED["source_freeze"],
        "paired_source_freeze_sha256": EXPECTED["paired_source_freeze"],
        "terminal_sha256": EXPECTED["terminal"],
        "normalized_recovery_sha256": EXPECTED["normalized_recovery"],
        "normalized_recovery_field_provenance": recovery["normalization_provenance"],
        "protocol_verification_sha256": sha(EVIDENCE / "actual_f_protocol_verification.json"),
        "root_acceptance": False,
        "candidate_or_sampler_execution": False,
        "roles": source_projection,
        "fixed_role_projection": fixed_role_projection,
        "limits": [
            "F-family verify_training metadata/source/protocol adapter passed for both actual roles; no candidate producer, model, sampler, or provider code executed.",
            "Both lineage records are factual proposals; root_accepted remains false. No screen decision, package, provider asset, or launch is bound.",
            "Training recovery normalization explicitly maps GetKernel identity_before/after driver SHA fields and observed current-version fields; raw retry2 receipt remains preserved.",
        ],
    }
    out = EVIDENCE / "actual_metadata_projection_f.json"
    save(out, projection)
    print(json.dumps({"projection": str(out.relative_to(ROOT)), "sha256": sha(out), "roles": {r: {"identity_sha256": source_projection[r]["identity"]["sha256"], "model_sha256": all_identities[r]["model_sha256"], "step": all_identities[r]["step"], "loss": all_identities[r]["validation_loss"], "H": all_identities[r]["common_NEW_horizon"]} for r in all_identities}, "torch_imported": "torch" in sys.modules}, sort_keys=True))


if __name__ == "__main__":
    main()
