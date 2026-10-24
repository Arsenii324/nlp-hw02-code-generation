#!/usr/bin/env python3
"""Score recovered final inference probe outputs without executing generated code.

Validation scoring uses the fixed supported-consensus reference population. Test
outputs are checked for count, schema, and byte/hash consistency only; no test
labels are read. The arithmetic evaluator is a bounded AST interpreter.
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
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nlp_hw02.arithmetic import evaluate_code

VALIDATION_QUESTIONS = ROOT / "artifacts/paid-gelu768-validation-20261024a/final400k/inputs/validation_questions.json"
CONSENSUS = ROOT / "reports/evidence/primary_reference_consensus.json"
KNOWN_N8 = ROOT / "artifacts/antigravity_gelu768_fullpop531k_eval"
V13_BUNDLE = ROOT / "artifacts/kaggle/mixture1166-t4-proof-v13-20261024a/tools/candidate_bundle_v8.zip"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def verify_evaluator_binding():
    if not V13_BUNDLE.is_file():
        raise FileNotFoundError(f"Pinned v13 source bundle missing: {V13_BUNDLE}")
    with zipfile.ZipFile(V13_BUNDLE) as bundle:
        package_arithmetic = bundle.read("src/nlp_hw02/arithmetic.py")
    local_sha = sha_file(ROOT / "src/nlp_hw02/arithmetic.py")
    package_sha = sha_bytes(package_arithmetic)
    if package_sha != local_sha:
        raise ValueError("Workspace arithmetic evaluator SHA differs from frozen v13 package")
    return local_sha, package_sha


def equal(a, b):
    if type(a) is int and type(b) is int:
        return a == b
    if type(a) not in (int, float) or type(b) not in (int, float):
        return False
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


@lru_cache(maxsize=100_000)
def eval_code(code: str):
    return evaluate_code(code)


def select_plurality(codes, scores):
    """Replay the frozen numeric plurality plus likelihood tie break."""
    if not codes or len(codes) != len(scores):
        raise ValueError("Candidate and score arrays must be nonempty and aligned")
    if not all(type(s) in (int, float) and math.isfinite(s) for s in scores):
        raise ValueError("Candidate scores must be finite numbers")
    groups = []
    supported = 0
    for i, code in enumerate(codes):
        result = eval_code(code)
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


def row_question_hashes(questions):
    return [hashlib.sha256(q.encode("utf-8")).hexdigest() for q in questions]


def validate_pool(pool, qhashes, label):
    if len(pool) != len(qhashes):
        raise ValueError(f"{label}: expected {len(qhashes)} pool rows, found {len(pool)}")
    for i, row in enumerate(pool):
        if row.get("row_index") != i or row.get("question_sha256") != qhashes[i]:
            raise ValueError(f"{label}: row/question order mismatch at row {i}")
        codes, scores = row.get("candidates"), row.get("scores")
        if not isinstance(codes, list) or not codes or len(codes) != len(scores or []):
            raise ValueError(f"{label}: malformed candidate/score arrays at row {i}")
        if not all(isinstance(code, str) for code in codes):
            raise ValueError(f"{label}: non-text candidate at row {i}")
    return True


def score_predictions(predictions, qhashes, refs):
    if len(predictions) != len(qhashes):
        raise ValueError("Prediction count differs from validation rows")
    counts = Counter()
    scored = 0
    syntax_errors = 0
    per_row = []
    for i, (row, qhash) in enumerate(zip(predictions, qhashes)):
        if not isinstance(row, dict) or set(row) != {"code"} or not isinstance(row["code"], str):
            raise ValueError(f"Malformed code-only prediction row {i}")
        code = row["code"]
        try:
            ast.parse(code)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            syntax_errors += 1
        result = eval_code(code)
        is_supported = bool(result.supported)
        counts["supported" if is_supported else "unsupported"] += 1
        ref = refs.get(qhash)
        if ref is None:
            raise ValueError(f"Missing reference for validation row {i}")
        if ref["status"] == "supported_consensus":
            scored += 1
            correct = is_supported and equal(result.value, ref["consensus_value"])
            counts["correct" if correct else "wrong_numeric" if is_supported else "unsupported_on_scored"] += 1
            per_row.append(bool(correct))
        else:
            per_row.append(None)
    if scored != 1529:
        raise ValueError(f"Expected fixed 1,529 scored rows, found {scored}")
    return {
        "rows": len(predictions),
        "scored_rows": scored,
        "correct": sum(per_row[i] is True for i in range(len(per_row))),
        "accuracy": sum(per_row[i] is True for i in range(len(per_row))) / scored,
        "selected_supported_rows_all": counts["supported"],
        "selected_unsupported_rows_all": counts["unsupported"],
        "syntax_errors": syntax_errors,
        "scored_correct": counts["correct"],
        "scored_wrong_numeric": counts["wrong_numeric"],
        "scored_unsupported": counts["unsupported_on_scored"],
    }


def validate_test_arm(arm_dir: Path, parent: dict, arm_key: str, test_qhashes, ordered_test_sha):
    out = arm_dir / "candidate_output"
    pred_path = out / "predictions.jsonl"
    pool_path = out / "mixture_candidates.jsonl"
    timing_path = out / "timing.json"
    if not all(p.is_file() for p in (pred_path, pool_path, timing_path)):
        raise FileNotFoundError(f"Missing test arm outputs under {arm_dir}")
    predictions = read_jsonl(pred_path)
    pool = read_jsonl(pool_path)
    timing = read_json(timing_path)
    parent_arm = parent.get("arms", {}).get(arm_key, {})
    backend = timing.get("backend", {})
    if len(predictions) != 1919 or len(pool) != 1919:
        raise ValueError(f"{arm_key}: expected 1,919 test prediction and candidate rows")
    for i, row in enumerate(predictions):
        if not isinstance(row, dict) or set(row) != {"code"} or not isinstance(row["code"], str):
            raise ValueError(f"{arm_key}: malformed test code-only row {i}")
    syntax_error_rows = []
    for i, row in enumerate(predictions):
        try:
            ast.parse(row["code"])
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            syntax_error_rows.append(i)
    if timing.get("syntax_errors") != syntax_error_rows:
        raise ValueError(f"{arm_key}: recomputed test syntax-error indices differ from timing sidecar")
    for i, row in enumerate(pool):
        if row.get("row_index") != i or row.get("question_sha256") != test_qhashes[i]:
            raise ValueError(f"{arm_key}: test candidate question/order mismatch at row {i}")
        if not isinstance(row.get("candidates"), list) or not row["candidates"] or len(row["candidates"]) != len(row.get("scores", [])):
            raise ValueError(f"{arm_key}: malformed test candidate arrays {i}")
        idx, _info = select_plurality(row["candidates"], row["scores"])
        if row.get("selected_index") != idx or predictions[i]["code"] != row["candidates"][idx]:
            raise ValueError(f"{arm_key}: test plurality replay mismatch at row {i}")
    actual_sha = sha_file(pred_path)
    pool_sha = sha_file(pool_path)
    timing_match = (timing.get("rows") == 1919 and timing.get("prediction_sha256") == actual_sha
                    and timing.get("candidate_pool_sha256") == pool_sha)
    executor = parent_arm.get("executor", {})
    clocks = {
        "core_seconds": timing.get("generation_selection_serialization_seconds"),
        "executor_seconds": executor.get("elapsed_seconds"),
        "notebook_seconds": timing.get("notebook_seconds"),
        "parent_inclusive_seconds": parent_arm.get("parent_inclusive_seconds"),
    }
    clocks_positive_finite = all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in clocks.values())
    clocks_within_budget = clocks_positive_finite and all(v <= 180 for v in clocks.values())
    backend_valid = backend.get("visible_gpu_count") == 1 and "T4" in str(backend.get("gpu", ""))
    if not backend_valid:
        raise ValueError(f"{arm_key}: timing sidecar does not attest one visible T4")
    if parent.get("children_reaped") is not True:
        raise ValueError("Parent receipt does not confirm all children reaped")
    if not clocks_positive_finite:
        raise ValueError(f"{arm_key}: one or more clocks are missing, nonpositive, or nonfinite: {clocks}")
    if timing.get("ordered_test_questions_sha256") != ordered_test_sha:
        raise ValueError(f"{arm_key}: ordered test question digest differs from pinned projection")
    if not timing_match:
        raise ValueError(f"{arm_key}: prediction/pool bytes do not match timing SHA/count metadata")
    return {
        "rows": len(predictions),
        "prediction_sha256": actual_sha,
        "syntax_error_rows": len(syntax_error_rows),
        "candidate_rows": len(pool),
        "candidate_pool_sha256": pool_sha,
        "plurality_replay_matches_all_rows": True,
        "timing_sidecar_rows_and_hash_match": timing_match,
        "expected_output_sha256": timing.get("expected_output_sha256"),
        "output_reproduced_diagnostic": timing.get("output_reproduced"),
        "actual_output_equals_expected_hash": actual_sha == timing.get("expected_output_sha256"),
        "clocks_seconds": clocks,
        "all_four_clocks_present_positive_finite": clocks_positive_finite,
        "all_four_clocks_le_180": clocks_within_budget,
        "runtime_qualified": bool(clocks_within_budget and parent_arm.get("exit") == 0 and timing_match),
        "backend_one_visible_t4": backend_valid,
        "backend": backend,
        "ordered_test_question_digest_matches": True,
        "parent_children_reaped": True,
        "parent_exit": parent_arm.get("exit"),
        "parent_status": parent_arm.get("status"),
        "test_labels_not_read_confirmed": timing.get("test_labels_read") is False and parent.get("test_labels_read") is False,
    }


def score_root(root: Path):
    questions = read_json(VALIDATION_QUESTIONS)
    if len(questions) != 1792 or not all(isinstance(q, str) for q in questions):
        raise ValueError("Pinned validation projection is not 1,792 question strings")
    qhashes = row_question_hashes(questions)
    reference_doc = read_json(CONSENSUS)
    refs = reference_doc["groups_by_question_sha256"]
    if sum(refs.get(q, {}).get("status") == "supported_consensus" for q in qhashes) != 1529:
        raise ValueError("Pinned reference consensus no longer has 1,529 supported rows")
    parent_path = root / "parent_result.json"
    parent = read_json(parent_path) if parent_path.is_file() else {}
    evaluator_sha, package_evaluator_sha = verify_evaluator_binding()
    with zipfile.ZipFile(V13_BUNDLE) as bundle:
        test_questions = json.loads(bundle.read("test_questions.json"))
    if len(test_questions) != 1919 or not all(isinstance(q, str) for q in test_questions):
        raise ValueError("Frozen v13 test question projection is not 1,919 strings")
    test_qhashes = row_question_hashes(test_questions)
    ordered_test_sha = sha_bytes(json.dumps(test_questions, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    arms = {}
    for arm in ("standalone", "mixture"):
        val_dir = root / f"{arm}_validation"
        test_dir = root / f"{arm}_test"
        val_out = val_dir / "candidate_output"
        pred_path = val_out / "predictions.jsonl"
        pool_path = val_out / "mixture_candidates.jsonl"
        timing_path = val_out / "timing.json"
        predictions = read_jsonl(pred_path)
        pool = read_jsonl(pool_path)
        validate_pool(pool, qhashes, f"{arm}_validation")
        if len(predictions) != len(questions):
            raise ValueError(f"{arm}: validation prediction row count differs")
        timing = read_json(timing_path)
        if timing.get("rows") != 1792:
            raise ValueError(f"{arm}: validation timing metadata mismatch")
        pool_sha = sha_file(pool_path)
        if timing.get("candidate_pool_sha256") != pool_sha:
            raise ValueError(f"{arm}: validation candidate pool SHA mismatch")
        # The final pool is the exact ordered plurality input, including each
        # arm's sampled-candidate count. Replay selection and bind output bytes.
        selected = []
        sample_counts = Counter()
        for i, row in enumerate(pool):
            idx, _info = select_plurality(row["candidates"], row["scores"])
            if row.get("selected_index") != idx:
                raise ValueError(f"{arm}: plurality selector mismatch at row {i}")
            selected.append({"code": row["candidates"][idx]})
            sample_counts[len(row["candidates"])] += 1
        if selected != predictions:
            raise ValueError(f"{arm}: predictions do not match selected plurality candidates")
        pred_sha = sha_file(pred_path)
        if timing.get("prediction_sha256") != pred_sha:
            raise ValueError(f"{arm}: validation prediction SHA mismatch")
        arms[arm] = {
            "validation": {
                **score_predictions(predictions, qhashes, refs),
                "prediction_sha256": pred_sha,
                "candidate_pool_sha256": pool_sha,
                "candidate_count_distribution": dict(sample_counts),
                "plurality_replay_matches_all_rows": True,
                "ordered_question_rows_match": True,
                "timing_validation_generated_flag": timing.get("validation_generated"),
                "timing_seconds": timing.get("generation_selection_serialization_seconds"),
            },
            "test": validate_test_arm(test_dir, parent, f"{arm}_test", test_qhashes, ordered_test_sha),
        }
    return {
        "schema": "final_inference_probe_scoring_v1",
        "root": str(root.resolve()),
        "test_labels_read": False,
        "validation_questions": {"path": str(VALIDATION_QUESTIONS.relative_to(ROOT)), "sha256": sha_file(VALIDATION_QUESTIONS), "rows": 1792},
        "reference_consensus": {"path": str(CONSENSUS.relative_to(ROOT)), "sha256": sha_file(CONSENSUS), "supported_rows": 1529},
        "evaluator": {"path": "src/nlp_hw02/arithmetic.py", "sha256": evaluator_sha, "frozen_package_sha256": package_evaluator_sha, "generated_code_executed": False},
        "test_question_projection": {"path_in_package": "test_questions.json", "package_sha256": sha_file(V13_BUNDLE), "ordered_question_sha256": ordered_test_sha, "rows": len(test_questions), "labels_read": False},
        "parent_result_sha256": sha_file(parent_path) if parent_path.is_file() else None,
        "arms": arms,
    }


def check_known_n8():
    evaluator_sha, package_evaluator_sha = verify_evaluator_binding()
    questions = read_json(VALIDATION_QUESTIONS)
    qhashes = row_question_hashes(questions)
    refs = read_json(CONSENSUS)["groups_by_question_sha256"]
    pred_path = KNOWN_N8 / "validation_n8_predictions.jsonl"
    pool_path = KNOWN_N8 / "validation_n8_candidates.jsonl"
    predictions, pool = read_jsonl(pred_path), read_jsonl(pool_path)
    validate_pool(pool, qhashes, "known_fullpop768_n8")
    if len(predictions) != len(questions):
        raise ValueError("Known N8 prediction row count mismatch")
    selected = []
    for i, row in enumerate(pool):
        idx, _ = select_plurality(row["candidates"], row["scores"])
        if row.get("selected_index") != idx:
            raise ValueError(f"Known N8 stored selector mismatch at row {i}")
        selected.append({"code": row["candidates"][idx]})
    if selected != predictions:
        raise ValueError("Known N8 predictions are not the selected plurality outputs")
    return {
        "schema": "known_n8_scorer_check_v1",
        "prediction_sha256": sha_file(pred_path),
        "candidate_pool_sha256": sha_file(pool_path),
        "rows": len(predictions),
        "selector_replay_matches_predictions": True,
        "scoring": score_predictions(predictions, qhashes, refs),
        "generated_code_executed": False,
        "model_lineage": "fullpop768 GELU, selected 528000 from continuation endpoint 531072; artifact directory name fullpop531k",
        "evaluator_sha256": evaluator_sha,
        "frozen_package_evaluator_sha256": package_evaluator_sha,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Recovered final_probe directory")
    parser.add_argument("--output", type=Path, help="JSON score receipt path")
    parser.add_argument("--check-known-n8", action="store_true", help="Replay and score the existing selected-plurality 768 N8 validation artifact")
    args = parser.parse_args()
    if args.check_known_n8:
        result = check_known_n8()
    else:
        if args.root is None:
            parser.error("--root is required unless --check-known-n8 is used")
        result = score_root(args.root)
    data = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data, encoding="utf-8")
    print(data, end="")


if __name__ == "__main__":
    main()
