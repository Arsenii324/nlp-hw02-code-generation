#!/usr/bin/env python3
"""Audit completed wide-model N8 validation pools with the frozen selector.

This scores only the fixed 1,792-question validation projection. Generated code
is passed to the repository's restricted arithmetic evaluators; it is never
executed. The script does not load a model or inspect the published test set.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nlp_hw02.arithmetic import evaluate_code as legacy_eval
from nlp_hw02.arithmetic_augassign_audit import evaluate_code_augassign as aug_eval

QUESTIONS_SHA = "83ea3ea9f449619e015d98e21b1c35ec2cc6d2edd01db0c026301e718014f04e"
REFERENCE_SHA = "deb96602c88b983ecfbf23738d2a64192057273798548ab61a7cdf7acfc7c119"
SELECTOR_SHA = "250d34bee2dc60847a65a7d5a784fac261a77db153a4d49f0687a82fe5a98fe5"
BASELINE_SHA = "cc52e3d2c138c00f16676225bf022ce1bb82f32689d164867a6508f477d400b3"
BASELINE_REPORT_SHA = "5860dd764f42e9b710ed9cc16e647634ccd400ae970b000771a40f266db4867f"
BASELINE_MODEL_SHA = "edb1b328116abb82240b094093108e533a0ce026a283f7f5463602631f253089"


def sha(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def read_lines(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def equal(a, b):
    return a == b if isinstance(a, int) and isinstance(b, int) else math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def metric_equal(a, b, additive=False):
    return abs(a - b) <= 1e-6 + 1e-6 * abs(b) if additive else equal(a, b)


@lru_cache(maxsize=None)
def eval_legacy(code):
    return legacy_eval(code)


evaluate_code = eval_legacy


@lru_cache(maxsize=None)
def eval_aug(code):
    return aug_eval(code)


def select_plurality(codes, scores):
    if not codes or len(codes) != len(scores) or not all(math.isfinite(s) for s in scores):
        raise ValueError("Nonempty candidates and finite scores required")
    groups = []
    supported = 0
    for i, code in enumerate(codes):
        result = evaluate_code(code)
        if not result.supported:
            continue
        supported += 1
        group = next((g for g in groups if equal(g["value"], result.value)), None)
        if group is None:
            group = {"value": result.value, "indices": []}
            groups.append(group)
        group["indices"].append(i)
    if not groups:
        return 0, {"supported_candidates": 0, "winning_votes": 0, "numeric_groups": 0}
    best = max(groups, key=lambda g: (len(g["indices"]), max(scores[i] for i in g["indices"])))
    index = max(best["indices"], key=lambda i: scores[i])
    return index, {"supported_candidates": supported, "winning_votes": len(best["indices"]), "numeric_groups": len(groups)}


def selector_ast_matches(selector_path):
    original = ast.parse(selector_path.read_text())
    local = ast.parse(Path(__file__).read_text())

    def node(tree, name):
        return next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)

    return all(ast.dump(node(original, n), include_attributes=False) == ast.dump(node(local, n), include_attributes=False)
               for n in ("equal", "select_plurality"))


def score_one(code, reference, metric):
    if reference["status"] != "supported_consensus":
        return None
    if metric == "legacy":
        value = eval_legacy(code)
        ok = value.supported and equal(value.value, reference["consensus_value"])
    elif metric == "augassign_fixed_output":
        value = eval_aug(code)
        ok = value.supported and equal(value.value, reference["consensus_value"])
    else:
        value = eval_aug(code)
        ok = value.supported and metric_equal(value.value, reference["consensus_value"], additive=True)
    return bool(ok), bool(value.supported)


def counts(codes, refs, qhashes, metric):
    out = Counter()
    per_question = {}
    for q, code in zip(qhashes, codes):
        ref = refs[q]
        result = score_one(code, ref, metric)
        if result is None:
            continue
        correct, supported = result
        out["correct" if correct else "wrong_numeric" if supported else "unsupported"] += 1
        per_question[q] = correct
    if len(per_question) != 1529:
        raise ValueError("The fixed 1,529-row reference population changed")
    return dict(out), per_question


def oracle_summary(rows, refs, qhashes, metric):
    total = Counter()
    for row, q in zip(rows, qhashes):
        ref = refs[q]
        supported = correct = False
        for code in row["candidates"]:
            evaluator = eval_legacy if metric == "legacy" else eval_aug
            value = evaluator(code)
            supported |= bool(value.supported)
            if ref["status"] == "supported_consensus" and value.supported:
                tol = metric == "augassign_additive1e6"
                correct |= metric_equal(value.value, ref["consensus_value"], additive=tol)
        total["oracle_correct_on_1529"] += correct and ref["status"] == "supported_consensus"
        total["questions_with_supported_candidate_all_1792"] += supported
        total["scored_rows"] += ref["status"] == "supported_consensus"
    return dict(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--source-freeze", type=Path, required=True)
    ap.add_argument("--expected-source-freeze-sha", required=True)
    ap.add_argument("--selector", type=Path, required=True)
    ap.add_argument("--baseline-predictions", type=Path, default=ROOT / "artifacts/kaggle/cooldown-paired-n24-v1-a-diagnostics/run-cooldown-half/validation_n24_predictions.jsonl")
    ap.add_argument("--baseline-report", type=Path, default=ROOT / "reports/evidence/cooldown_fixed24_root_quality_20261023.json")
    ap.add_argument("--expected-model-sha", required=True)
    ap.add_argument("--expected-step", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not args.output.is_absolute():
        args.output = ROOT / args.output

    owner_path = args.run_dir / "owner-result.json"
    owner = read_json(owner_path)
    worker_path = args.run_dir / "reference/own_gelu768_validation_result_a.json"
    worker = read_json(worker_path)
    timing_path = args.run_dir / "reference/validation_n8_timing.json"
    timing = read_json(timing_path)
    progress = read_json(args.run_dir / "reference/validation_n8_progress.json")
    parity_path = args.run_dir / "reference/actual_parity.json"
    parity = read_json(parity_path)
    pool_path = args.run_dir / "reference/validation_n8_candidates.jsonl"
    pred_path = args.run_dir / "reference/validation_n8_predictions.jsonl"

    if not (owner["complete"] and owner["actual_exit"] == 0 and owner["worker_complete"] and owner["cleanup"]["reaped"] and owner["cleanup"]["group_exited"]):
        raise ValueError("Owner, exit, or owned-process-group cleanup is incomplete")
    if not (worker["complete"] and worker["phase"] == "finished" and worker["parity_passed_before_sampling"] and parity["passed"]):
        raise ValueError("Reference worker or actual parity is incomplete")
    if not selector_ast_matches(args.selector) or sha(args.selector) != SELECTOR_SHA:
        raise ValueError("Frozen selector source/hash mismatch")
    for rel, expected in owner["files_sha256"].items():
        path = args.run_dir / rel
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Owner-mapped file SHA mismatch: {rel}")

    identity = worker["identity"]
    if identity["model_sha256"] != args.expected_model_sha or identity["step"] != args.expected_step:
        raise ValueError("Checkpoint identity differs from expected actual model")
    if worker["sample_counts"] != [8] or timing["rows"] != 1792 or timing["samples_per_question"] != 8 or timing["total_candidates"] != 1792 * 8:
        raise ValueError("N8 pool shape/count differs")
    if worker["hardware"]["T4_budget_qualified"] is not False:
        raise ValueError("A100 reference may not claim T4 qualification")
    if owner["identity"] != identity or owner["contract_sha256"] != worker["contract_sha256"]:
        raise ValueError("Parent/worker identity or contract binding differs")
    source_freeze = read_json(args.source_freeze)
    if sha(args.source_freeze) != args.expected_source_freeze_sha or worker["producer_sha256"] != source_freeze["source_sha256"]["scripts/probe_own_gelu768_validation_a.py"] or worker["adapter_sha256"] != source_freeze["source_sha256"]["scripts/own_gelu768_validation_contract_a.py"]:
        raise ValueError("Worker producer/adapter differs from reviewed source freeze")

    questions = json.loads(args.questions.read_text())
    qhashes = [hashlib.sha256(q.encode()).hexdigest() for q in questions]
    if len(qhashes) != len(set(qhashes)) or len(qhashes) != 1792 or sha(args.questions) != QUESTIONS_SHA:
        raise ValueError("Question input/order pin mismatch")
    if timing["ordered_questions_sha256"] != hashlib.sha256(json.dumps(questions, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest():
        raise ValueError("Worker ordered-question digest differs")
    parity_path_input = args.input_root / "actual_first4_parity_pairs.json"
    selector_source = args.selector
    parity_pairs = read_json(parity_path_input)
    expected_input_hashes = {"validation_questions": sha(args.questions), "parity_pairs": sha(parity_path_input),
                             "legacy_selector_source": sha(selector_source)}
    if len(parity_pairs) != 4 or [x["question"] for x in parity_pairs] != questions[:4] or worker["inputs_sha256"] != expected_input_hashes:
        raise ValueError("Worker question/parity/selector input binding differs")

    consensus_path = ROOT / "reports/evidence/primary_reference_consensus.json"
    if sha(consensus_path) != REFERENCE_SHA:
        raise ValueError("Reference consensus source changed")
    refs = read_json(consensus_path)["groups_by_question_sha256"]
    if sum(refs[q]["status"] == "supported_consensus" for q in qhashes) != 1529:
        raise ValueError("Reference pool is not the fixed 1,529-row projection")

    pool = read_lines(pool_path)
    predictions = read_lines(pred_path)
    if len(pool) != 1792 or len(predictions) != 1792:
        raise ValueError("Ordered output row counts differ")
    selected_codes, sample0_codes = [], []
    total_caps = selected_caps = sample0_caps = support_rows = cap_questions = 0
    for i, (row, prediction, q) in enumerate(zip(pool, predictions, qhashes)):
        if row["row_index"] != i or row["question_sha256"] != q or row["samples"] != 8 or len(row["candidates"]) != 8 or len(row["scores"]) != 8:
            raise ValueError(f"Candidate row identity/count mismatch at {i}")
        replay_index, replay_info = select_plurality(row["candidates"], row["scores"])
        if replay_index != row["selected_index"] or replay_info != {k: row[k] for k in replay_info}:
            raise ValueError(f"Frozen selector replay mismatch at row {i}")
        if set(prediction) != {"code"} or prediction["code"] != row["candidates"][replay_index]:
            raise ValueError(f"Selected output is not code-only/selector result at row {i}")
        selected_codes.append(prediction["code"])
        sample0_codes.append(row["candidates"][0])
        total_caps += sum(row["capped"])
        selected_caps += row["capped"][replay_index]
        sample0_caps += row["capped"][0]
        support_rows += replay_info["supported_candidates"] > 0
        cap_questions += any(row["capped"])
    if total_caps != timing["all_candidate_capped_rows"] or selected_caps != timing["selected_candidate_capped_rows"]:
        raise ValueError("Candidate cap totals differ from worker timing")

    baseline_report = read_json(args.baseline_report)
    if sha(args.baseline_report) != BASELINE_REPORT_SHA or sha(args.baseline_predictions) != BASELINE_SHA:
        raise ValueError("Baseline receipt/prediction hash mismatch")
    baseline = read_lines(args.baseline_predictions)
    if len(baseline) != 1792 or any(set(x) != {"code"} for x in baseline):
        raise ValueError("Baseline is not ordered code-only 1,792 output")
    half = baseline_report["scores"]["cooldown-half"]
    if half["model_sha256"] != BASELINE_MODEL_SHA or half["selected_step"] != 415656 or half["prediction_sha256"] != BASELINE_SHA:
        raise ValueError("Baseline is not the accepted half415656 fixed24 model")

    metrics = ("legacy", "augassign_fixed_output", "augassign_additive1e6")
    summary = {}
    baseline_outcomes = {}
    for metric in metrics:
        chosen_counts, chosen_outcomes = counts(selected_codes, refs, qhashes, metric)
        first_counts, first_outcomes = counts(sample0_codes, refs, qhashes, metric)
        baseline_codes = [x["code"] for x in baseline]
        base_counts, base_outcomes = counts(baseline_codes, refs, qhashes, metric)
        oracle = oracle_summary(pool, refs, qhashes, metric)
        evaluator = eval_legacy if metric == "legacy" else eval_aug
        sample0_support_all = sum(bool(evaluator(row["candidates"][0]).supported) for row in pool)
        selected_support_all = sum(bool(evaluator(code).supported) for code in selected_codes)
        wins = sum(not base_outcomes[q] and chosen_outcomes[q] for q in chosen_outcomes)
        losses = sum(base_outcomes[q] and not chosen_outcomes[q] for q in chosen_outcomes)
        summary[metric] = {"selected_n8": chosen_counts, "sample0": first_counts, "oracle_n8": oracle,
                           "baseline_half415656_n24": base_counts,
                           "support_all_1792": {"sample0_supported": sample0_support_all,
                                                "selected_n8_supported": selected_support_all,
                                                "any_of_8_supported": oracle["questions_with_supported_candidate_all_1792"]},
                           "paired_vs_half_n24": {"n8_only_correct": wins, "half_only_correct": losses,
                                                  "net": wins - losses}}
        baseline_outcomes[metric] = base_outcomes

    out = {
        "schema": "wide_actual_n8_validation_quality_audit_v1",
        "status": "PASS",
        "source_identities": {"auditor_script_sha256": sha(Path(__file__)), "owner_sha256": sha(owner_path), "worker_sha256": sha(worker_path), "selector_sha256": sha(args.selector),
                              "questions_sha256": sha(args.questions), "parity_pairs_sha256": sha(parity_path_input),
                              "source_freeze_sha256": sha(args.source_freeze), "reference_sha256": sha(consensus_path),
                              "pool_sha256": sha(pool_path), "predictions_sha256": sha(pred_path),
                              "baseline_predictions_sha256": sha(args.baseline_predictions),
                              "baseline_quality_receipt_sha256": sha(args.baseline_report)},
        "actual_run": {"model_sha256": identity["model_sha256"], "step": identity["step"], "architecture": identity["architecture"],
                       "complete": worker["complete"], "actual_exit": owner["actual_exit"], "process_group_cleanup": owner["cleanup"],
                       "parity_passed": parity["passed"], "gpu": worker["hardware"]["gpu"], "concurrent_training": owner["concurrent_training"],
                       "T4_budget_qualified": worker["hardware"]["T4_budget_qualified"], "rows": timing["rows"], "samples_per_question": timing["samples_per_question"],
                       "candidate_count": timing["total_candidates"], "selector_replay_rows": 1792,
                       "all_candidate_capped_count": total_caps, "questions_with_any_capped_candidate": cap_questions,
                       "selected_capped_count": selected_caps, "sample0_capped_count": sample0_caps,
                       "rows_with_supported_selected_candidate": support_rows, "selected_candidate_no_support_rows": timing["without_supported_candidate"],
                       "progress_marker": {"complete": progress["complete"], "completed_rows": progress["completed_rows"], "requested_rows": progress["requested_rows"]},
                       "complete_path_seconds": timing["complete_path_seconds"], "complete_path_within180": timing["complete_path_within180"],
                       "peak_allocated_bytes": timing["peak_allocated_bytes"]},
        "scoring_population": {"validation_questions": 1792, "fixed_supported_reference_rows": 1529, "test_questions_read": False},
        "quality": summary,
        "limits": ["Fixed noisy supported-reference consensus is diagnostic, not official pass@1 or an unbiased estimate.",
                   "No test input, test questions, or test outputs were read or scored.",
                   "The measured A100 generation path is recorded from the actual timing artifact; it is not cold-load or a 1,919-question T4 proof.",
                   "The progress JSON retains complete=false despite 1,792/1,792 rows; owner exit, worker complete flag, parity, and output hashes provide separate successful completion evidence. Treat progress as a stale diagnostic marker.",
                   "Oracle N8 uses references to define an upper bound; it is not a deployable selector.",
                   "Sample0 and oracle candidate results are validation diagnostics; no model promotion is implied."]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output.relative_to(ROOT)), "selected": {m: summary[m]["selected_n8"] for m in metrics},
                      "paired": {m: summary[m]["paired_vs_half_n24"] for m in metrics},
                      "seconds": timing["complete_path_seconds"], "sha256": sha(args.output)}, indent=2))


if __name__ == "__main__":
    main()
