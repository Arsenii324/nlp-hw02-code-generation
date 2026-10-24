"""Render measured B034 inference diagnostics; no model/data-code execution.

Run with the existing Anaconda Python (matplotlib). Inputs remain immutable;
the JSON sidecar records exact evidence hashes and JSON pointers for each point.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reports/evidence"
OUT = ROOT / "reports/figures"


def main() -> None:
    names = {
        "strong": "strong_probe_quality_review_20261022.json",
        "scaling": "sample_scaling_quality_review_20261022.json",
        "replication": "temperature_replication_quality_review_20261022.json",
    }
    docs = {key: json.loads((EVIDENCE / name).read_text()) for key, name in names.items()}
    sources = {
        key: {"path": f"reports/evidence/{name}",
              "sha256": hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest()}
        for key, name in names.items()
    }

    def at(key: str, pointer: str):
        value = docs[key]
        for part in pointer.strip("/").split("/"):
            value = value[part]
        return value

    specs = [
        ("K=8", "strong", "/full_test_runtime/seconds", "replication",
         "/scores/augassign1e9/seed1_temp08_K8/correct", 20261002),
        ("K=16", "scaling", "/full_test_runtime/16/seconds", "scaling",
         "/augassign_fixed_output_categories/16lane_prefix16/correct", 20261002),
        ("K=24, seed 1", "scaling", "/full_test_runtime/24/seconds", "scaling",
         "/augassign_fixed_output_categories/24lane_prefix24/correct", 20261002),
        ("K=24, seed 2", "replication", "/full_test_runtime/temp08-seed20261023/seconds",
         "replication", "/scores/augassign1e9/temp08-seed20261023_K24/correct", 20261023),
    ]
    points = []
    for label, tk, tp, qk, qp, seed in specs:
        points.append({"label": label, "seconds": at(tk, tp), "agreement": at(qk, qp),
                       "seed": seed, "time_source": tk, "time_pointer": tp,
                       "agreement_source": qk, "agreement_pointer": qp})
    for key, doc in docs.items():
        n = doc.get("strict_reference_rows", doc.get("strict_rows"))
        if n != 1529:
            raise ValueError(f"Unexpected denominator in {key}: {n}")
    if not all(isinstance(p["agreement"], int) and 0 <= p["agreement"] <= 1529
               and 0 < p["seconds"] < 180 for p in points):
        raise ValueError("Invalid measured point")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(8.4, 5.7))
    fig.subplots_adjust(left=.12, right=.95, bottom=.27, top=.81)
    for i, p in enumerate(points):
        ax.scatter(p["seconds"], p["agreement"], s=85, zorder=3,
                   marker="D" if i == 3 else "o", color="#ce6a2a" if i == 3 else "#256b91")
        offsets = [(10, -18), (-16, 13), (-92, -16), (-94, 12)]
        ax.annotate(f'{p["label"]}\n{p["agreement"]}/1529; {p["seconds"]:.2f}s',
                    (p["seconds"], p["agreement"]), xytext=offsets[i],
                    textcoords="offset points", fontsize=9, linespacing=1.4)
    ax.set(xlim=(50, 139), ylim=(1048, 1106),
           xlabel="All 1,919 test questions: generation and selection (seconds)",
           ylabel="Validation numeric agreement (count / 1,529)")
    ax.grid(axis="y", color="#dde3e8", linewidth=.8)
    ax.set_axisbelow(True)
    fig.suptitle("B034: measured sampling cost and numeric agreement", x=.12, y=.96,
                 ha="left", fontsize=15, fontweight="bold")
    fig.text(.12, .89, "Same 17.5M-parameter checkpoint; one NVIDIA T4 per measurement",
             fontsize=10, color="#45515c")
    fig.text(.12, .15,
             "Temperature 0.8, top-k 50, question batch 32. AugAssign diagnostic on fixed outputs.\n"
             "Generation clocks exclude model load, warmup and file writing: no final 180s proof.\n"
             "Reused validation with noisy references; this is not official pass@1.\n"
             "Different candidate counts/seeds use different RNG streams; points are not a matched curve.",
             fontsize=9, color="#45515c", linespacing=1.55, va="top")
    OUT.mkdir(exist_ok=True)
    stem = OUT / "b034_inference_tradeoff_20261022"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix("." + suffix), dpi=180, facecolor="white")
    plt.close(fig)
    sidecar = {"schema": "inference_tradeoff_figure_v1", "sources": sources,
               "points": points, "denominator": 1529,
               "checkpoint_sha256": docs["replication"]["identities"]["checkpoint_sha256"],
               "metric": "AugAssign fixed-output numeric agreement, not official pass@1",
               "timing": "Historical generation/selection only; excludes load/warmup/writing",
               "artifacts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (stem.with_suffix("." + s) for s in ("png", "pdf", "svg"))}}
    stem.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(json.dumps({"output": str(stem), "points": len(points), "model_runs": 0}))


if __name__ == "__main__":
    main()
