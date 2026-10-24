# Reproduction notes

This is a source hand-in archive, not a model bundle. Model weights, optimizer/RNG states, token arrays, cached data, raw Parquet shards, `.venv`, Git metadata, credentials, and the canonical submission notebook are intentionally excluded. The final candidate choice was still pending at archive assembly; model metadata for both finalists is included without weights.

## Selected candidate lineages

The private v13 model manifest identifies two trained candidates:

- `768fp_n8`: GELU768 full-population continuation; selected checkpoint step **528,000**, validation loss 0.21149580623436748; training endpoint 531,072. These are distinct: the selected checkpoint is the CE-best model, not the endpoint.
- `512_new_n3`: GELU512 full-population continuation; selected checkpoint step **662,144** according to its bound terminal metrics and v13 manifest.

Both use seed 42 and the pinned dataset revision `b2724689ff079e9c22efd548fe134bfb29036dbb`. Their exact training contracts and the source archives referenced by those contracts are stored under `training_lineages/`. The raw dataset and precomputed full-population token arrays are not included.

## Code and dependencies

`src/`, `scripts/`, and `configs/` contain the project source snapshot. `training_lineages/{gelu768,gelu512_new}/runtime_source.bin` is the exact small code/config ZIP archive whose SHA is bound by each training contract; `runtime_source/` is its verified expansion. `frozen_inference_source/src/nlp_hw02/` contains the Python modules from the private v13 candidate bundle. Its `inference.py` is the current local override used by the D notebook builder; all other `.py` files are copied from v13.

The recorded paid training runtime was PyTorch 2.10.0+cu128, CUDA 12.8, NumPy 2.0.2, on one NVIDIA A100-PCIE-40GB. The project GPU extra in `pyproject.toml` currently requests PyTorch >=2.11,<2.12, so it does not reproduce that historical runtime; `uv.lock` is not a complete lock of the paid runtime. Python patch version and every remote package wheel were not recovered in this package. The original wall-clock wrappers also contain an admission cutoff at 2026-10-24 12:00 UTC; that historical gate has passed and is preserved as provenance rather than a ready-to-launch command.

## Limits on exact retraining

The archive contains the actual trainer/wrapper source, frozen runtime archives, full bound configs, and lineage contracts. It does **not** permit an end-to-end exact retraining run by itself: the parent checkpoint/optimizer/RNG states and the full-population token arrays/cache are absent, and no training was rerun while packaging. Obtain the original data at the pinned revision and matching parent state/cache referenced by the contracts before attempting continuation. A source archive SHA proves the code/config bytes; it does not prove a fresh run would reproduce the trained weights bit-for-bit.

## Experiments and evaluation

`experiments/LOG.md` records hypotheses, changes, failures, and measured results; `reports/AGENT_USE.md` records agent involvement; `reports/BASELINE.md` and `reports/TASK_AND_JUDGE.md` cover the existing qualified fallback and evaluation limits. Internal supported-consensus agreement is a proxy, not official pass@1. The exact final inference notebook, test predictions, and proof receipts are delivered separately.
