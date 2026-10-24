"""Fetch pinned raw Parquet once, without a second HF/Arrow cache."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

PROJECT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def valid(path: Path, item: dict) -> bool:
    return (path.is_file() and path.stat().st_size == item["size"]
            and sha256(path) == item["sha256"])


def main() -> None:
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get(
        "NLP_HW02_DATA_ROOT", PROJECT.parent / "data" / "tinysms_qa_12m")))
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    manifest = json.loads((PROJECT / "assignment/dataset_manifest.json").read_text())
    missing = [x for x in manifest["files"] if not valid(root / x["path"], x)]
    print(f"Verified {len(manifest['files']) - len(missing)}/{len(manifest['files'])} files", flush=True)
    if args.check_only:
        if missing:
            raise SystemExit(f"Missing/corrupt files: {len(missing)}")
        return
    root.mkdir(parents=True, exist_ok=True)
    required = sum(x["size"] for x in missing)
    reserve = 5 * 1024**3
    if shutil.disk_usage(root).free < required + reserve:
        raise SystemExit("Insufficient free space: preserve at least 5 GiB after download")
    base = f"https://huggingface.co/datasets/{manifest['repo_id']}/resolve/{manifest['revision']}/"

    def fetch(item: dict) -> None:
        dest = root / item["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_suffix(dest.suffix + ".partial")
        for attempt in range(3):
            try:
                request = urllib.request.Request(base + item["path"], headers={"User-Agent": "nlp-hw02-setup"})
                with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
                    shutil.copyfileobj(response, output)
                if not valid(partial, item):
                    raise ValueError(f"Size/SHA-256 mismatch: {item['path']}")
                partial.replace(dest)
                print(f"Verified download: {item['path']}", flush=True)
                return
            except Exception:
                partial.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, missing))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Dataset ready: {root}", flush=True)


if __name__ == "__main__":
    main()
