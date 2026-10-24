"""Stage only the explicitly selected source package; never walk secrets/data."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    files = sorted((ROOT / "src").rglob("*.py"))
    files += [ROOT / name for name in ("pyproject.toml", "uv.lock", ".python-version")]
    missing = [p.name for p in files if not p.is_file()]
    if missing:
        raise SystemExit(f"Missing required source files: {missing}")
    output = ROOT / "artifacts/source.zip"
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path.is_symlink():
                raise SystemExit(f"Refusing source symlink: {path.name}")
            archive.write(path, path.relative_to(ROOT))
    print(output)


if __name__ == "__main__":
    main()
