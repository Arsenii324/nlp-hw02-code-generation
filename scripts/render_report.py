"""Build a working report PDF from maintained Markdown using installed tools.

This renders actual report content; it does not run models, tests or cloud jobs.
The PDF is a draft until its factual and visual reviews are complete.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WARNINGS: list[dict[str, str]] = []


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str]) -> str:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                            timeout=60, check=False)
    if result.returncode:
        raise RuntimeError(f"{Path(command[0]).name} exited {result.returncode}: "
                           + result.stderr[-4000:])
    if result.stderr.strip():
        WARNINGS.append({"tool": Path(command[0]).name, "stderr": result.stderr[-4000:]})
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "reports/REPORT_DRAFT_20261022.md")
    args = parser.parse_args()
    source = args.source.resolve()
    if source.parent != ROOT / "reports" or "DRAFT" not in source.stem:
        raise ValueError("This renderer accepts working drafts in reports/ only")
    tools = {name: shutil.which(name) for name in ("pandoc", "typst", "pdfinfo")}
    if not all(tools.values()):
        raise RuntimeError("Required installed report tool missing; no install attempted")
    style = ROOT / "reports/report_style.typ"
    layout = ROOT / "reports/report_layout.json"
    typ = source.with_suffix(".typ")
    pdf = source.with_suffix(".pdf")
    provenance = source.with_suffix(".build.json")
    versions = {name: run([path, "--version"]).splitlines()[0]
                for name, path in tools.items() if name != "pdfinfo"}
    installed_fonts = set(run([tools["typst"], "fonts"]).splitlines())
    config = json.loads(layout.read_text())
    required_fonts = [config["mainfont"], *config["mathfont"], *config["codefont"]]
    if not all(font in installed_fonts for font in required_fonts):
        raise RuntimeError("Configured font missing; refusing silent font fallback")
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    commands = [
        [tools["pandoc"], str(source), "--standalone", "--to=typst",
         "--from=markdown+tex_math_single_backslash",
         "--include-in-header=" + str(style), "--output=" + str(typ),
         "--metadata-file=" + str(layout)],
        [tools["typst"], "compile", "--root", str(ROOT), str(typ), str(pdf)],
    ]
    inputs = {str(source.relative_to(ROOT)): sha(source),
              str(style.relative_to(ROOT)): sha(style),
              str(layout.relative_to(ROOT)): sha(layout),
              str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__))}
    figure = ROOT / "reports/figures/b034_inference_tradeoff_20261022.png"
    if figure.exists():
        inputs[str(figure.relative_to(ROOT))] = sha(figure)
    for command in commands:
        run(command)
    info = run([tools["pdfinfo"], str(pdf)])
    pages = next(int(line.split(":", 1)[1]) for line in info.splitlines()
                 if line.startswith("Pages:"))
    record = {"schema": "working_report_pdf_build_v1", "started_utc": start,
              "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "tools": versions, "commands": commands, "inputs_sha256": inputs,
              "outputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (typ, pdf)},
              "pages": pages, "draft_not_final_report": True,
              "configured_fonts_verified": required_fonts, "tool_warnings": WARNINGS,
              "visual_and_factual_review_still_required": True,
              "no_model_test_cloud_or_dataset_execution": True}
    provenance.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"pdf": str(pdf), "pages": pages, "build_record": str(provenance)}))


if __name__ == "__main__":
    main()
