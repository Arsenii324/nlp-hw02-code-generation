"""Reproduce a bounded arithmetic/reference audit on pinned TRAIN shards only.

The script never opens the question-only challenge test parquet and never
prints question text. Programs are parsed and interpreted by
``nlp_hw02.arithmetic``; no Python source from the dataset is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pyarrow.parquet as pq

from nlp_hw02.arithmetic import evaluate_code


REVISION = "b2724689ff079e9c22efd548fe134bfb29036dbb"
VALIDATION_SHARD = 59
TRAIN_FILTER_SHARDS = (0, 1, 2, 3)
OVERLAP_SHARDS = tuple(range(4, 59))
REFERENCE_SHARDS = tuple(range(60))
VALIDATION_SAMPLE_SIZE = 2048
REFERENCE_SAMPLE_SIZE = 256


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def question_digest(question: str) -> str:
    return digest_bytes(question.encode("utf-8"))


def data_root_from_args(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    if os.environ.get("NLP_HW02_DATA_ROOT"):
        return Path(os.environ["NLP_HW02_DATA_ROOT"]).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "data" / "tinysms_qa_12m").resolve()


def partition(root: Path, index: int) -> Path:
    return root / "data" / "train" / f"part-{index:05d}.parquet"


def rows(path: Path) -> Iterator[tuple[str, str]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["question", "code"], batch_size=8192):
        questions = batch.column("question").to_pylist()
        codes = batch.column("code").to_pylist()
        for question, code in zip(questions, codes):
            if isinstance(question, str) and isinstance(code, str):
                yield question, code


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_validation(root: Path) -> tuple[list[tuple[str, str, str]], set[str]]:
    seen_in_train = set()
    for shard in TRAIN_FILTER_SHARDS:
        for question, _ in rows(partition(root, shard)):
            seen_in_train.add(question_digest(question))

    sample: list[tuple[str, str, str]] = []
    seen_validation = set()
    for question, code in rows(partition(root, VALIDATION_SHARD)):
        qhash = question_digest(question)
        if qhash in seen_in_train or qhash in seen_validation:
            continue
        seen_validation.add(qhash)
        sample.append((qhash, question, code))
        if len(sample) == VALIDATION_SAMPLE_SIZE:
            break
    if len(sample) != VALIDATION_SAMPLE_SIZE:
        raise RuntimeError(f"Expected {VALIDATION_SAMPLE_SIZE} unique validation questions")
    return sample, seen_in_train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="raw dataset root; defaults to NLP_HW02_DATA_ROOT")
    args = parser.parse_args()
    root = data_root_from_args(args.data_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("revision") != REVISION:
        raise RuntimeError(f"expected pinned revision {REVISION}; got {manifest.get('revision')!r}")

    sample, _ = select_validation(root)
    for qhash, _question, _code in sample:
        if not qhash:
            raise AssertionError("empty question hash")
    results = [evaluate_code(code) for _, _, code in sample]
    supported = sum(result.supported for result in results)

    reference_ids = {qhash for qhash, _, _ in sample[:REFERENCE_SAMPLE_SIZE]}
    refs: dict[str, list[str]] = defaultdict(list)
    overlap_rows = 0
    overlap_questions: set[str] = set()
    for shard in REFERENCE_SHARDS:
        for question, code in rows(partition(root, shard)):
            qhash = question_digest(question)
            if qhash in reference_ids:
                refs[qhash].append(code)
                if shard in OVERLAP_SHARDS:
                    overlap_rows += 1
                    overlap_questions.add(qhash)

    multi_ref_groups = 0
    supported_ref_groups = 0
    conflict_groups = 0
    modal_agreements: list[float] = []
    for qhash in reference_ids:
        codes = refs.get(qhash, [])
        if len(codes) > 1:
            multi_ref_groups += 1
        values = [result.value for code in codes
                  if (result := evaluate_code(code)).supported]
        if not values:
            continue
        supported_ref_groups += 1
        counts = Counter(values)
        if len(counts) > 1:
            conflict_groups += 1
        modal_agreements.append(max(counts.values()) / len(values))

    files = [partition(root, shard) for shard in sorted(set(
        REFERENCE_SHARDS + (VALIDATION_SHARD,)
    ))]
    file_hashes = {path.name: file_sha256(path) for path in files}
    manifest_hashes = {
        item["path"].split("/")[-1]: item["sha256"]
        for item in manifest.get("files", [])
        if item.get("path", "").startswith("data/train/")
    }
    mismatches = [name for name, sha in file_hashes.items()
                  if manifest_hashes.get(name) != sha]
    if mismatches:
        raise RuntimeError(f"manifest SHA-256 mismatch: {mismatches}")

    sample_hash = digest_bytes("\n".join(qhash for qhash, _, _ in sample).encode("ascii"))
    reference_sample_hash = digest_bytes(
        "\n".join(qhash for qhash, _, _ in sample[:REFERENCE_SAMPLE_SIZE]).encode("ascii")
    )
    report = {
        "dataset_revision": REVISION,
        "scope": "train shards 0-59 only; test parquet not opened",
        "sample_selection": {
            "order": "first unique exact question strings in shard 59 row order",
            "exclude_question_hashes_in_shards": list(TRAIN_FILTER_SHARDS),
            "validation_shard": VALIDATION_SHARD,
            "validation_n": len(sample),
            "validation_question_hashes_sha256_newline_joined": sample_hash,
            "first_256_question_hashes_sha256_newline_joined": reference_sample_hash,
        },
        "evaluator": {
            "module": "src/nlp_hw02/arithmetic.py",
            "source_sha256": file_sha256(Path(__file__).resolve().parents[1] /
                                           "src/nlp_hw02/arithmetic.py"),
            "audit_script_sha256": file_sha256(Path(__file__).resolve()),
            "allowed_numeric_calls": ["int", "float", "round", "abs", "min", "max"],
            "resource_limits": {
                "source_bytes": 65536, "ast_nodes": 4096, "statements": 512,
                "locals": 256, "integer_bits": 256, "absolute_value": "1e100",
                "absolute_power_exponent": 12, "absolute_round_digits": 12,
            },
            "validation_numeric_returns": supported,
            "validation_unsupported": len(sample) - supported,
        },
        "reference_audit_first_256": {
            "question_groups": len(reference_ids),
            "groups_with_multiple_reference_rows": multi_ref_groups,
            "groups_with_at_least_one_supported_reference": supported_ref_groups,
            "groups_with_conflicting_supported_numeric_returns": conflict_groups,
            "mean_within_group_modal_numeric_agreement": (
                sum(modal_agreements) / len(modal_agreements) if modal_agreements else None
            ),
            "median_within_group_modal_numeric_agreement": (
                sorted(modal_agreements)[len(modal_agreements) // 2]
                if modal_agreements else None
            ),
            "matching_question_groups_in_shards_4_58": len(overlap_questions),
            "matching_rows_in_shards_4_58": overlap_rows,
        },
        "train_partition_sha256": file_hashes,
        "warning": "Reference agreement is not correctness or official pass@1.",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
