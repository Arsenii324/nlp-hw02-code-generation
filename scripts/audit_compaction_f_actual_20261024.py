#!/usr/bin/env python3
"""CPU-only audit of the completed F compaction N8 run and its matched control."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "wide_audit", ROOT / "scripts/audit_wide_n8_validation_actual_b.py")
wide = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(wide)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def read_lines(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> None:
    stage = ROOT / "artifacts/paid-wide-compaction-f-20261024a"
    run = ROOT / "artifacts/paid-wide-compaction-f-result-20261024a"
    ev = ROOT / "reports/evidence/finished_sequence_compaction_launch_20261024f"
    expected = {
        "caller_sha256": "a2c47f62ed0d9a18baf03b63e8d5e63fb68496d0a8c59b343ff05ab59ceca127",
        "contract_sha256": "48e391a160576a6385ec8587336dda0f79a56f9c959608c7bd4c64a7cbcd375e",
        "stage_manifest_sha256": "8dd704dc0162de87e91522bea07c2cb599c4195436df38d565282a717c76217c",
        "source_freeze_sha256": "7a72f2f9c5197c7afc1505d5d625bf070e941f3b904aa6223bb83ba32faae6f4",
        "source_archive_sha256": "97494fb88adab12d244716a782b079b74f3072ae11f5f17166c5f22159c857bd",
        "producer_sha256": "1ceb912e5b1aa12fd2d72c12b1abe36fc889c72a28b1c27d263a89a50cd7827b",
        "compaction_helper_sha256": "fc0b120b8d56283195c79546c24f711df983e4bf92cb45908970c4293c244c78",
        "generic_sampler_sha256": "6fe63312e6828481308a5067cfd36a00add6d814b772d03105f80dd7578ef304",
        "model_sha256": "78e5b9660ed115c5ecc4d30e0b29d419beee54896ea25597f3e3a1068d877875",
        "selector_sha256": wide.SELECTOR_SHA,
        "questions_sha256": wide.QUESTIONS_SHA,
        "parity_pairs_sha256": "1fbf9051c99ed84e816123b561f691cd942f8c742cf19b4a2850c58f4e7190ed",
        "reference_sha256": wide.REFERENCE_SHA,
        "baseline_predictions_sha256": wide.BASELINE_SHA,
        "baseline_report_sha256": wide.BASELINE_REPORT_SHA,
    }

    # Verify the immutable stage map and the prelaunch materialization receipts.
    manifest_path = stage / "stage_manifest.json"
    assert sha(manifest_path) == expected["stage_manifest_sha256"]
    manifest = read_json(manifest_path)
    staged = {str(p.relative_to(stage)): p for p in stage.rglob("*")
              if p.is_file() and p != manifest_path}
    assert set(staged) == set(manifest["files"])
    for rel, desc in manifest["files"].items():
        assert staged[rel].stat().st_size == desc["size_bytes"] and sha(staged[rel]) == desc["sha256"], rel
    materialization = read_json(ev / "remote_materialization_readback.json")
    assert materialization["status"] == "REMOTE_STAGE_AND_METADATA_PREFLIGHT_PASS_NO_WORKER_LAUNCHED"
    assert materialization["remote_payload_verification"]["all_files_passed"] is True
    assert materialization["stage_manifest_sha256"] == expected["stage_manifest_sha256"]
    assert materialization["remote_contract_sha256"] == expected["contract_sha256"]
    assert materialization["remote_payload_verification"]["source_archive_sha256"] == expected["source_archive_sha256"]
    contract_path = stage / "producer_N8_COMPACTION_F_ROOT_BOUND.json"
    freeze_path = stage / "configs/finished_sequence_compaction_source_f_20261024.json"
    contract, freeze = read_json(contract_path), read_json(freeze_path)
    assert sha(contract_path) == expected["contract_sha256"]
    assert sha(freeze_path) == expected["source_freeze_sha256"]
    assert contract["source_freeze"]["sha256"] == expected["source_freeze_sha256"]
    assert contract["expected_model_sha256"] == expected["model_sha256"] and contract["expected_step"] == 52224
    assert contract["sample_counts"] == [8] and contract["test_generation"] is False

    owner_path = run / "owner-result.json"
    owner = read_json(owner_path)
    files = owner["files_sha256"]
    assert owner["schema"] == "owned_wide_compaction_reference_parent_f_v1"
    assert owner["complete"] is True and owner["actual_exit"] == 0 and owner["worker_complete"] is True
    assert owner["cleanup"] == {"reaped": True, "group_exited": True}
    assert owner["identity"]["model_sha256"] == expected["model_sha256"] and owner["identity"]["step"] == 52224
    assert owner["concurrent_training"] is True and owner["T4_budget_qualified"] is False
    for rel, digest in files.items():
        p = run / rel
        assert p.is_file() and sha(p) == digest, rel

    out = run / "reference"
    worker_path = out / "own_wide_validation_compaction_result_f.json"
    worker = read_json(worker_path)
    parity_path = out / "actual_compaction_parity.json"
    parity = read_json(parity_path)
    assert worker["schema"] == "own_wide_validation_compaction_result_f_v1"
    assert worker["complete"] is True and worker["phase"] == "finished"
    assert worker["parity_passed_before_sampling"] is True and parity["passed"] is True
    assert worker["source_freeze_sha256"] == expected["source_freeze_sha256"]
    assert worker["contract_sha256"] == expected["contract_sha256"]
    assert worker["producer_sha256"] == expected["producer_sha256"]
    assert worker["helper_sha256"] == expected["compaction_helper_sha256"]
    assert worker["generic_a_control_source_sha256"] == expected["generic_sampler_sha256"]
    assert worker["identity"] == owner["identity"]
    assert worker["T4_budget_qualified"] is False and worker["training"] is False
    assert worker["gold_used_for_selection"] is False and worker["test_generation"] is False
    assert worker["comparison"]["control_prior_output_reproduction"]["matches_prior_actual_candidate"] is True
    assert worker["comparison"]["control_prior_output_reproduction"]["matches_prior_actual_prediction"] is True
    for key in ("control_vs_compaction_batch_count_equal", "per_batch_max_steps_equal",
                "per_batch_rng_before_equal", "per_batch_rng_after_equal"):
        assert worker["comparison"][key] is True, key
    assert worker["comparison"]["identity_or_bitwise_equivalence_claim"] is False
    assert parity["actual_unique_rows"] == 4 and parity["repeat_per_question"] == 8 and parity["expanded_rows"] == 32

    questions_path = stage / "inputs/validation_questions.json"
    parity_input = stage / "inputs/actual_first4_parity_pairs.json"
    selector_path = ROOT / ".worktrees/strong-inference-fallback-20261022/src/nlp_hw02/sampling_candidate.py"
    assert sha(questions_path) == expected["questions_sha256"]
    assert sha(parity_input) == expected["parity_pairs_sha256"]
    assert sha(selector_path) == expected["selector_sha256"]
    questions = read_json(questions_path)
    qhashes = [hashlib.sha256(q.encode()).hexdigest() for q in questions]
    assert len(qhashes) == 1792 and len(qhashes) == len(set(qhashes))
    pair_rows = read_json(parity_input)
    assert len(pair_rows) == 4 and [r["question"] for r in pair_rows] == questions[:4]
    refs_path = ROOT / "reports/evidence/primary_reference_consensus.json"
    assert sha(refs_path) == expected["reference_sha256"]
    refs = read_json(refs_path)["groups_by_question_sha256"]
    assert sum(refs[q]["status"] == "supported_consensus" for q in qhashes) == 1529

    timings = {
        "control": read_json(out / "validation_n8_control_timing.json"),
        "compaction": read_json(out / "validation_n8_compaction_timing.json"),
    }
    progress = {
        "control": read_json(out / "validation_n8_control_progress.json"),
        "compaction": read_json(out / "validation_n8_compaction_progress.json"),
    }
    assert all(t["rows"] == 1792 and t["samples_per_question"] == 8 and t["total_candidates"] == 14336
               for t in timings.values())
    assert all(t["temperature"] == 0.8 and t["top_k"] == 50 and t["question_batch_size"] == 32
               and t["seed"] == 20261002 and t["length_penalty"] == 0.6 and t["max_new_tokens"] == 192
               for t in timings.values())
    expected_order_sha = hashlib.sha256(json.dumps(questions, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    assert all(t["ordered_questions_sha256"] == expected_order_sha for t in timings.values())
    assert all(p["completed_rows"] == 1792 and p["requested_rows"] == 1792 and p["complete"] is False
               for p in progress.values())
    # The progress files keep their stale false marker; owner and worker are the terminal authorities.

    pools, predictions, codes = {}, {}, {}
    for arm, suffix in (("control", "control"), ("compaction", "compaction")):
        pool_path = out / f"validation_n8_{suffix}_candidates.jsonl"
        pred_path = out / f"validation_n8_{suffix}_predictions.jsonl"
        pools[arm], predictions[arm] = read_lines(pool_path), read_lines(pred_path)
        assert len(pools[arm]) == len(predictions[arm]) == 1792
        assert sha(pool_path) == timings[arm]["candidate_sha256"]
        assert sha(pred_path) == timings[arm]["prediction_sha256"]
        selected = []
        caps = selected_caps = sample0_caps = support_rows = cap_questions = 0
        for i, (row, pred) in enumerate(zip(pools[arm], predictions[arm])):
            q = qhashes[i]
            assert row["row_index"] == i and row["question_sha256"] == q and row["samples"] == 8
            assert len(row["candidates"]) == len(row["scores"]) == len(row["capped"]) == 8
            selected_index, info = wide.select_plurality(row["candidates"], row["scores"])
            assert selected_index == row["selected_index"]
            assert info == {k: row[k] for k in info}
            assert set(pred) == {"code"} and pred["code"] == row["candidates"][selected_index]
            selected.append(pred["code"])
            caps += sum(row["capped"])
            selected_caps += row["capped"][selected_index]
            sample0_caps += row["capped"][0]
            support_rows += info["supported_candidates"] > 0
            cap_questions += any(row["capped"])
        assert caps == timings[arm]["all_candidate_capped_rows"]
        assert selected_caps == timings[arm]["selected_candidate_capped_rows"]
        assert support_rows == 1792 - timings[arm]["without_supported_candidate"]
        codes[arm] = selected

    # Verify exact control reproduction and per-batch RNG/admission traces.
    assert sha(out / "validation_n8_control_candidates.jsonl") == "9841337527d8629245e083c372691358e3951e9633033966c0b463b68ff77246"
    assert sha(out / "validation_n8_control_predictions.jsonl") == "ad69ae770c2c1723260313ed26cb2c21d704696b5224b49cb6c3448b17bdcf04"
    control_batches, compact_batches = timings["control"]["rng_state_by_question_batch"], timings["compaction"]["rng_state_by_question_batch"]
    assert len(control_batches) == len(compact_batches) == 56
    for control, compact in zip(control_batches, compact_batches):
        assert all(control[k] == compact[k] for k in ("question_offset", "question_rows", "global_max_steps",
                                                       "cuda_rng_before_sha256", "cuda_rng_after_sha256"))

    baseline_report_path = ROOT / "reports/evidence/cooldown_fixed24_root_quality_20261023.json"
    baseline_path = ROOT / "artifacts/kaggle/cooldown-paired-n24-v1-a-diagnostics/run-cooldown-half/validation_n24_predictions.jsonl"
    assert sha(baseline_report_path) == expected["baseline_report_sha256"]
    assert sha(baseline_path) == expected["baseline_predictions_sha256"]
    baseline_codes = [row["code"] for row in read_lines(baseline_path)]
    assert len(baseline_codes) == 1792
    assert read_json(baseline_report_path)["scores"]["cooldown-half"]["selected_step"] == 415656

    metric_results = {}
    for metric in ("legacy", "augassign_fixed_output", "augassign_additive1e6"):
        metric_rows = {}
        for arm, output_codes in {**codes, "half415656_n24": baseline_codes}.items():
            metric_rows[arm] = wide.counts(output_codes, refs, qhashes, metric)[1]
        counts = {arm: sum(outcomes.values()) for arm, outcomes in metric_rows.items()}
        pairs = {}
        for left, right in (("compaction", "control"), ("compaction", "half415656_n24"),
                            ("control", "half415656_n24")):
            left_only = sum(metric_rows[left][q] and not metric_rows[right][q] for q in metric_rows[left])
            right_only = sum(metric_rows[right][q] and not metric_rows[left][q] for q in metric_rows[left])
            pairs[f"{left}_vs_{right}"] = {"left_only_correct": left_only,
                                             "right_only_correct": right_only,
                                             "net_left_minus_right": left_only - right_only}
        metric_results[metric] = {"correct_counts": counts, "paired": pairs}

    receipt = {
        "schema": "finished_sequence_compaction_actual_quality_independent_audit_v1",
        "status": "PASS_CPU_ONLY",
        "actual_run": {
            "owner_sha256": sha(owner_path), "owner_schema": owner["schema"], "complete": owner["complete"],
            "actual_exit": owner["actual_exit"], "worker_complete": owner["worker_complete"],
            "cleanup": owner["cleanup"], "elapsed_seconds": owner["elapsed_seconds"],
            "worker_sha256": sha(worker_path), "model_sha256": worker["identity"]["model_sha256"],
            "step": worker["identity"]["step"], "architecture": worker["identity"]["architecture"],
            "gpu": worker["hardware"]["gpu"], "concurrent_training": owner["concurrent_training"],
            "T4_budget_qualified": worker["T4_budget_qualified"], "parity_passed": parity["passed"],
            "parity_pairs": parity["actual_unique_rows"], "parity_expanded_rows": parity["expanded_rows"],
            "source_freeze_sha256": worker["source_freeze_sha256"], "contract_sha256": worker["contract_sha256"],
            "producer_sha256": worker["producer_sha256"], "compaction_helper_sha256": worker["helper_sha256"],
            "ordered_questions_sha256": timings["compaction"]["ordered_questions_sha256"],
            "batch_count": len(control_batches), "rng_states_match_per_batch": True,
            "global_max_steps_match_per_batch": True,
            "control_prior_outputs_reproduced": worker["comparison"]["control_prior_output_reproduction"],
            "identity_or_bitwise_equivalence_claim": worker["comparison"]["identity_or_bitwise_equivalence_claim"],
            "progress_marker_stale_false": True,
        },
        "arms": {
            arm: {k: timings[arm][k] for k in ("rows", "samples_per_question", "total_candidates",
                                               "all_candidate_capped_rows", "selected_candidate_capped_rows",
                                               "without_supported_candidate", "complete_path_seconds",
                                               "complete_path_within180", "peak_allocated_bytes",
                                               "candidate_sha256", "prediction_sha256", "ordered_questions_sha256")}
            for arm in ("control", "compaction")
        },
        "quality": metric_results,
        "provenance": {
            "stage_manifest_sha256": sha(manifest_path), "stage_file_count": len(manifest["files"]),
            "stage_payload_bytes": materialization["remote_payload_verification"]["verified_total_bytes"],
            "source_archive_sha256": expected["source_archive_sha256"],
            "caller_sha256": expected["caller_sha256"], "source_freeze_sha256": expected["source_freeze_sha256"],
            "compaction_candidate_pool_sha256": sha(out / "validation_n8_compaction_candidates.jsonl"),
            "compaction_predictions_sha256": sha(out / "validation_n8_compaction_predictions.jsonl"),
            "control_candidate_pool_sha256": sha(out / "validation_n8_control_candidates.jsonl"),
            "control_predictions_sha256": sha(out / "validation_n8_control_predictions.jsonl"),
            "parity_sha256": sha(parity_path), "baseline_report_sha256": sha(baseline_report_path),
            "baseline_predictions_sha256": sha(baseline_path), "reference_consensus_sha256": sha(refs_path),
        },
        "limits": [
            "The supported-reference consensus is a fixed noisy diagnostic, not official pass@1 or an unbiased estimate.",
            "No test input, question, or output was read or scored; no model was executed locally.",
            "This concurrent A100 run is not a one-T4 qualification.",
            "The generation progress files retain complete=false at 1,792/1,792 rows; completed parent/worker records, flushed outputs, hashes, and process-group cleanup are separate completion evidence.",
            "Equal per-batch RNG state hashes/global maxima do not assert bitwise output identity: the FP16 compacted row layout can change trajectories.",
            "The root stage review was a prelaunch static/source review; this audit scores only recovered runtime outputs and does not certify course submission quality.",
        ],
    }
    target = ROOT / "reports/evidence/finished_sequence_compaction_launch_20261024f/independent_quality_audit.json"
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"path": str(target.relative_to(ROOT)), "sha256": sha(target),
                      "quality": metric_results,
                      "seconds": {a: timings[a]["complete_path_seconds"] for a in timings},
                      "caps": {a: {k: timings[a][k] for k in ("all_candidate_capped_rows", "selected_candidate_capped_rows", "without_supported_candidate")} for a in timings}}, indent=2))


if __name__ == "__main__":
    main()
