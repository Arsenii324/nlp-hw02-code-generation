"""Explicit allowlist: no dataset, checkpoint, environment or credential upload."""
from pathlib import Path
import hashlib
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def package_source(out=None):
    files=sorted((ROOT/'src/nlp_hw02').glob('*.py'))
    files += [ROOT/p for p in ['scripts/colab_job.py','scripts/check_baseline.py','scripts/prepare_representative.py','scripts/check_data_v2.py','assignment/dataset_manifest.json']]
    for name in ['check_arithmetic.py','kaggle_dual_run.py','check_batched_muon.py','profile_muon.py','profile_decoding_gpu.py','execute_inference_notebook.py','kaggle_eval.py','check_progress.py','check_training_arrays.py','check_training_eligibility.py','check_kaggle_eval.py','check_training_budget.py']:
        if (ROOT/'scripts'/name).exists(): files.append(ROOT/'scripts'/name)
    out=Path(out) if out is not None else ROOT/'artifacts/baseline_source.zip'
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:
            if p.is_symlink(): raise RuntimeError(f'Refusing symlink {p}')
            z.write(p,p.relative_to(ROOT))
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        for p in files: assert z.read(str(p.relative_to(ROOT)))==p.read_bytes()
    return {'path':str(out),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}


if __name__=='__main__': print(package_source())
