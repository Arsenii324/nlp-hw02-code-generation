"""Build one internal test-output pool from root-verified fullpopulation models.

The input files are per-question sampled candidate caches; this script does not
load a model or read test labels. Official course scoring and cold-T4 timing are
not performed here.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

from nlp_hw02.arithmetic import evaluate_code

ROOT = Path(__file__).resolve().parents[1]
POOLS = {
    "768_fullpopulation": ROOT / "artifacts/antigravity_gelu768_fullpop531k_eval/test_n8_candidates.jsonl",
    "512_new": ROOT / "artifacts/antigravity_gelu512_long262k_eval/test_n8_candidates.jsonl",
}
OUTPUT = ROOT / "artifacts/mixture_fullpop768_n8_512new_n3_test_predictions.jsonl"
COUNTS = (8, 3)


def equal(left, right):
    return left == right if type(left) is int and type(right) is int else math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def select_plurality(codes, scores):
    groups = []
    for i, code in enumerate(codes):
        result = evaluate_code(code)
        if not result.supported:
            continue
        group = next((g for g in groups if equal(g["value"], result.value)), None)
        if group is None:
            group = {"value": result.value, "indices": []}
            groups.append(group)
        group["indices"].append(i)
    if not groups:
        return 0
    best = max(groups, key=lambda g: (len(g["indices"]), max(scores[i] for i in g["indices"])))
    return max(best["indices"], key=lambda i: scores[i])


def read_pool(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 1919:
        raise ValueError(f"Expected 1919 ordered test candidates: {path}")
    for row in rows:
        if not isinstance(row.get("candidates"), list) or len(row["candidates"]) < max(COUNTS):
            raise ValueError(f"Candidate count is short: {path}")
        if len(row.get("scores", [])) != len(row["candidates"]):
            raise ValueError(f"Candidate score count differs: {path}")
    return rows


def main():
    if not all(path.is_file() for path in POOLS.values()):
        raise FileNotFoundError("A root-reviewed test candidate pool is missing")
    pools = {name: read_pool(path) for name, path in POOLS.items()}
    output = []
    for row in range(1919):
        codes = pools["768_fullpopulation"][row]["candidates"][: COUNTS[0]] + pools["512_new"][row]["candidates"][: COUNTS[1]]
        scores = pools["768_fullpopulation"][row]["scores"][: COUNTS[0]] + pools["512_new"][row]["scores"][: COUNTS[1]]
        code = codes[select_plurality(codes, scores)]
        ast.parse(code)
        output.append({"code": code})
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output).encode()
    OUTPUT.write_bytes(data)
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)), "rows": len(output), "ast_valid": True, "test_labels_read": False, "sha256": hashlib.sha256(data).hexdigest()}))


if __name__ == "__main__":
    main()
