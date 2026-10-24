"""Audit all supplied train references for the v2 primary validation groups.

Reads only the pinned training parquet shards plus the primary validation
cache and its metadata. It does not open test data or execute supplied code.
The separate consensus output is evaluation-only diagnostic metadata, never a
training target and never the course's official pass@1 score.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pyarrow.parquet as pq

from nlp_hw02.arithmetic import (
    MAX_ABS_VALUE,
    MAX_AST_NODES,
    MAX_INT_BITS,
    MAX_LOCALS,
    MAX_POWER_EXPONENT,
    MAX_ROUND_DIGITS,
    MAX_SOURCE_BYTES,
    MAX_STATEMENTS,
    evaluate_code,
)


DATASET_REVISION = "b2724689ff079e9c22efd548fe134bfb29036dbb"
TRAIN_PARTITIONS = tuple(f"part-{index:05d}.parquet" for index in range(60))
# Predeclared before inspecting group return values. Integers compare exactly
# when both sides are ints; if either value is a float, this tolerance applies.
FLOAT_REL_TOL = 1e-9
FLOAT_ABS_TOL = 1e-9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_list(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("ascii")).hexdigest()


def iter_train_rows(path: Path) -> Iterator[tuple[str, str]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["question", "code"], batch_size=8192):
        questions = batch.column("question").to_pylist()
        codes = batch.column("code").to_pylist()
        for question, code in zip(questions, codes):
            if isinstance(question, str) and isinstance(code, str):
                yield question, code


def value_equal(left: int | float, right: int | float) -> bool:
    if type(left) is int and type(right) is int:
        return left == right
    return math.isclose(
        float(left), float(right), rel_tol=FLOAT_REL_TOL, abs_tol=FLOAT_ABS_TOL
    )


def value_buckets(values: list[int | float]) -> list[dict[str, Any]]:
    """Group supported returns; comparisons are exact for int/int, tolerant otherwise.

    Sorted values are compared to the first value in each bucket, avoiding
    transitive chaining where a sequence of near-equal floats spans a wider
    range than the declared tolerance.
    """
    buckets: list[dict[str, Any]] = []
    for value in sorted(values, key=lambda item: (float(item), type(item).__name__)):
        for bucket in buckets:
            if value_equal(value, bucket["representative"]):
                bucket["reference_count"] += 1
                if type(value) is float:
                    bucket["contains_float"] = True
                break
        else:
            buckets.append({
                "representative": value,
                "reference_count": 1,
                "contains_float": type(value) is float,
            })
    for bucket in buckets:
        bucket["comparison"] = (
            "float_tolerance" if bucket.pop("contains_float") else "exact_integer"
        )
    return buckets


def load_metadata(repo: Path, validation_path: Path, data_root: Path) -> tuple[
    list[dict[str, Any]], dict[str, Any]
]:
    manifest_path = data_root / "manifest.json"
    pinned_manifest_path = repo / "assignment" / "dataset_manifest.json"
    stats_path = validation_path.parent / "data_stats.json"
    cache_manifest_path = validation_path.parent / "cache_manifest.json"
    prepare_result_path = validation_path.parent.parent / "prepare_result.json"

    manifest = json.loads(manifest_path.read_text())
    pinned_manifest = json.loads(pinned_manifest_path.read_text())
    stats = json.loads(stats_path.read_text())
    cache_manifest = json.loads(cache_manifest_path.read_text())
    prepare_result = json.loads(prepare_result_path.read_text())
    for label, source in (("local dataset manifest", manifest),
                          ("assignment manifest", pinned_manifest),
                          ("primary data stats", stats),
                          ("preparation result", prepare_result)):
        revision = source.get("revision", source.get("dataset_revision"))
        if revision != DATASET_REVISION:
            raise RuntimeError(f"{label} revision mismatch: {revision!r}")
    if cache_manifest.get("complete") is not True:
        raise RuntimeError("primary cache manifest is incomplete")

    for rel_path, path in (("validation.json", validation_path),
                            ("data_stats.json", stats_path)):
        expected = cache_manifest.get("checksums", {}).get(rel_path)
        actual = sha256_file(path)
        if not expected or expected != actual:
            raise RuntimeError(f"primary cache checksum mismatch for {rel_path}")

    pinned_train = {
        item["path"].split("/")[-1]: item["sha256"]
        for item in pinned_manifest.get("files", [])
        if item.get("path", "").startswith("data/train/")
    }
    local_train = {
        item["path"].split("/")[-1]: item["sha256"]
        for item in manifest.get("files", [])
        if item.get("path", "").startswith("data/train/")
    }
    expected_names = set(TRAIN_PARTITIONS)
    if set(pinned_train) != expected_names or set(local_train) != expected_names:
        raise RuntimeError("expected exactly 60 pinned train partitions")
    if pinned_train != local_train:
        raise RuntimeError("local data manifest differs from assignment's pinned train manifest")

    metadata = {
        "dataset_revision": DATASET_REVISION,
        "primary_cache": {
            "validation_path": str(validation_path.relative_to(repo)),
            "validation_sha256": sha256_file(validation_path),
            "validation_rows": stats.get("validation_rows"),
            "source_groups": stats.get("sampling"),
            "split_rule": stats.get("split"),
            "seed": stats.get("seed"),
            "train_rows_used_for_preparation": stats.get("train_rows"),
            "data_stats_sha256": sha256_file(stats_path),
            "cache_manifest_sha256": sha256_file(cache_manifest_path),
            "prepare_result_sha256": sha256_file(prepare_result_path),
        },
        "evaluator": {
            "module": "src/nlp_hw02/arithmetic.py",
            "module_sha256": sha256_file(repo / "src" / "nlp_hw02" / "arithmetic.py"),
            "audit_script": "scripts/audit_primary_references.py",
            "numeric_calls": ["int", "float", "round", "abs", "min", "max"],
            "resource_limits": {
                "source_bytes": MAX_SOURCE_BYTES,
                "ast_nodes": MAX_AST_NODES,
                "statements": MAX_STATEMENTS,
                "locals": MAX_LOCALS,
                "integer_bits": MAX_INT_BITS,
                "absolute_numeric_magnitude": MAX_ABS_VALUE,
                "absolute_power_exponent": MAX_POWER_EXPONENT,
                "absolute_round_digits": MAX_ROUND_DIGITS,
                "print_statements_and_arguments": "ignored by the return-value proxy",
            },
            "integer_comparison": "exact when both values are integers",
            "float_rel_tol": FLOAT_REL_TOL,
            "float_abs_tol": FLOAT_ABS_TOL,
            "tolerance_caveat": (
                "If either return is a float, math.isclose is used; the values are "
                "synthetic references and floats may reflect rounding or binary representation."
            ),
        },
    }
    return [manifest, pinned_manifest], metadata


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation",
        default=str(repo / "artifacts" / "kaggle" / "prepare-v2" / "cache" / "validation.json"),
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("NLP_HW02_DATA_ROOT", str(repo.parent / "data" / "tinysms_qa_12m")),
    )
    parser.add_argument(
        "--output-dir", default=str(repo / "reports" / "evidence")
    )
    args = parser.parse_args()
    validation_path = Path(args.validation).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    _manifests, metadata = load_metadata(repo, validation_path, data_root)
    primary = json.loads(validation_path.read_text())
    expected_count = metadata["primary_cache"]["validation_rows"]
    if len(primary) != expected_count:
        raise RuntimeError(f"validation row count {len(primary)} != metadata {expected_count}")

    validation_rows_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    primary_rows_by_source_shard: Counter[str] = Counter()
    for row in primary:
        question = row.get("question")
        supplied_hash = row.get("question_sha256")
        if not isinstance(question, str) or not isinstance(supplied_hash, str):
            raise RuntimeError("primary validation row lacks question/hash")
        actual_hash = sha256_text(question)
        if supplied_hash != actual_hash:
            raise RuntimeError("primary validation question hash mismatch")
        validation_rows_by_hash[supplied_hash].append(row)
        source_shard = row.get("shard")
        if not isinstance(source_shard, str):
            raise RuntimeError("primary validation row lacks source shard metadata")
        primary_rows_by_source_shard[Path(source_shard).name] += 1
    question_hashes = sorted(validation_rows_by_hash)
    if len(question_hashes) != len(primary):
        raise RuntimeError("primary cache contains repeated exact question groups")

    train_hashes: dict[str, str] = {}
    references: dict[str, list[dict[str, Any]]] = defaultdict(list)
    wanted = set(question_hashes)
    for filename in TRAIN_PARTITIONS:
        parquet_path = data_root / "data" / "train" / filename
        actual_file_hash = sha256_file(parquet_path)
        train_hashes[filename] = actual_file_hash
        for question, code in iter_train_rows(parquet_path):
            qhash = sha256_text(question)
            if qhash not in wanted:
                continue
            result = evaluate_code(code)
            references[qhash].append({
                "code_sha256": sha256_text(code),
                "supported": result.supported,
                "value": result.value,
            })

    # Verify partition checksums after scanning; no test path is opened.
    pinned_train = {
        item["path"].split("/")[-1]: item["sha256"]
        for item in json.loads((repo / "assignment" / "dataset_manifest.json").read_text())["files"]
        if item.get("path", "").startswith("data/train/")
    }
    if train_hashes != {name: pinned_train[name] for name in TRAIN_PARTITIONS}:
        raise RuntimeError("a scanned train shard failed its pinned SHA-256")

    consensus: dict[str, dict[str, Any]] = {}
    validation_code_match_rows = 0
    validation_code_not_found_rows = 0
    total_reference_rows = 0
    duplicate_reference_groups = 0
    duplicate_reference_rows = 0
    duplicate_code_rows = 0
    supported_reference_rows = 0
    unsupported_reference_rows = 0
    groups_with_any_reference = 0
    groups_with_supported_reference = 0
    groups_conflicting = 0
    groups_unsupported_only = 0
    reference_counts: list[int] = []

    for qhash in question_hashes:
        refs = references.get(qhash, [])
        total_reference_rows += len(refs)
        reference_counts.append(len(refs))
        if refs:
            groups_with_any_reference += 1
            if len(refs) > 1:
                duplicate_reference_groups += 1
                duplicate_reference_rows += len(refs) - 1
        code_counts = Counter(ref["code_sha256"] for ref in refs)
        duplicate_code_rows += sum(count - 1 for count in code_counts.values() if count > 1)
        values = [ref["value"] for ref in refs if ref["supported"]]
        supported_reference_rows += len(values)
        unsupported = len(refs) - len(values)
        unsupported_reference_rows += unsupported
        buckets = value_buckets(values)
        if values:
            groups_with_supported_reference += 1
        if len(buckets) > 1:
            groups_conflicting += 1
            status = "conflict"
            consensus_value = None
        elif buckets:
            status = "supported_consensus_with_unsupported_refs" if unsupported else "supported_consensus"
            consensus_value = buckets[0]["representative"]
        elif refs:
            status = "unsupported_only"
            consensus_value = None
            groups_unsupported_only += 1
        else:
            status = "no_matching_reference"
            consensus_value = None

        ref_code_hashes = {ref["code_sha256"] for ref in refs}
        val_rows = validation_rows_by_hash[qhash]
        val_matches = sum(
            sha256_text(row["code"]) in ref_code_hashes for row in val_rows
        )
        validation_code_match_rows += val_matches
        validation_code_not_found_rows += len(val_rows) - val_matches

        consensus[qhash] = {
            "status": status,
            "consensus_value": consensus_value,
            "reference_rows": len(refs),
            "supported_reference_rows": len(values),
            "unsupported_reference_rows": unsupported,
            "supported_numeric_value_buckets": buckets,
            "duplicate_code_rows": sum(count - 1 for count in code_counts.values() if count > 1),
            "primary_validation_rows_with_exact_code_in_train": val_matches,
        }

    statuses = Counter(entry["status"] for entry in consensus.values())
    summary = {
        "primary_validation_rows": len(primary),
        "primary_unique_question_groups": len(question_hashes),
        "primary_rows_by_source_shard": dict(sorted(primary_rows_by_source_shard.items())),
        "primary_question_hashes_sha256_sorted_newline_joined": hash_list(question_hashes),
        "exact_question_hashes": question_hashes,
        "train_partition_count": len(train_hashes),
        "train_reference_rows_for_primary_groups": total_reference_rows,
        "reference_coverage_groups": groups_with_any_reference,
        "reference_coverage_fraction": groups_with_any_reference / len(question_hashes),
        "groups_with_supported_numeric_reference": groups_with_supported_reference,
        "supported_reference_row_count": supported_reference_rows,
        "unsupported_reference_row_count": unsupported_reference_rows,
        "groups_with_supported_reference_conflict": groups_conflicting,
        "groups_with_only_unsupported_references": groups_unsupported_only,
        "groups_with_no_matching_reference": statuses["no_matching_reference"],
        "groups_with_duplicate_reference_rows": duplicate_reference_groups,
        "duplicate_reference_rows_beyond_one_per_group": duplicate_reference_rows,
        "duplicate_code_rows_beyond_one_per_group": duplicate_code_rows,
        "primary_validation_code_rows_found_exactly_in_train": validation_code_match_rows,
        "primary_validation_code_rows_not_found_in_train": validation_code_not_found_rows,
        "reference_rows_per_primary_group": {
            "min": min(reference_counts, default=0),
            "median": (
                (sorted(reference_counts)[len(reference_counts)//2 - 1] +
                 sorted(reference_counts)[len(reference_counts)//2]) / 2
                if len(reference_counts) % 2 == 0 and reference_counts else
                sorted(reference_counts)[len(reference_counts)//2] if reference_counts else 0
            ),
            "max": max(reference_counts, default=0),
        },
        "consensus_status_counts": dict(sorted(statuses.items())),
        "all_train_partition_sha256": train_hashes,
        "consensus_rule": (
            "exact integer equality when both returns are ints; otherwise math.isclose "
            f"(rel_tol={FLOAT_REL_TOL}, abs_tol={FLOAT_ABS_TOL}); consensus only when all "
            "supported returns fall in one bucket; conflicts are never majority-relabeled"
        ),
    }
    audit = {
        "purpose": "evaluation-only supplied-reference audit; not training targets or official pass@1",
        "dataset_metadata": metadata,
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
        "summary": summary,
        "limitations": [
            "Rows are synthetic supplied references; agreement does not establish mathematical correctness.",
            "The arithmetic interpreter ignores standalone print statements and their arguments.",
            "Unsupported code is reported separately and never assigned a numeric label.",
            "Float tolerance is a predeclared engineering convention, not a guarantee of semantic equivalence.",
        ],
    }
    consensus_output = {
        "purpose": "evaluation-only supported-reference consensus; never a training target",
        "dataset_revision": DATASET_REVISION,
        "primary_validation_sha256": metadata["primary_cache"]["validation_sha256"],
        "evaluator_module_sha256": metadata["evaluator"]["module_sha256"],
        "float_rel_tol": FLOAT_REL_TOL,
        "float_abs_tol": FLOAT_ABS_TOL,
        "exact_integer_equality_when_both_ints": True,
        "group_count": len(consensus),
        "groups_by_question_sha256": consensus,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "primary_reference_audit.json"
    consensus_path = output_dir / "primary_reference_consensus.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n")
    consensus_path.write_text(json.dumps(consensus_output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "audit_path": str(audit_path),
        "consensus_path": str(consensus_path),
        "summary": {key: value for key, value in summary.items() if key != "exact_question_hashes" and key != "all_train_partition_sha256"},
    }, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
