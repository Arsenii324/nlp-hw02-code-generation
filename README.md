# NLP HW2 — Code generation

Start at [START_HERE.md](START_HERE.md), the maintained documentation head.

Selected solution: own **30,705,728-parameter GELU512 encoder-decoder at step525952**, eager fixed16 numeric plurality. A fresh exact-byte one-T4 proof reproduced all1,919 code-only predictions; four clocks120.792/137.024/137.028/143.031s were below180s. Internal1135/1529 (74.23%) is reused noisy-reference agreement, not official pass@1. The older qualified398336/eager16 and half415656/fixed24 solutions remain preserved. Nothing submitted; improvement work continues.

[Method, hashes and proof](reports/BASELINE.md), [selection receipt](reports/evidence/own_fullpop512_n16_selection_20261024.json), [current state](reports/HANDOFF.md).

## Run inference

Put [artifacts/candidate_bundle.zip](artifacts/candidate_bundle.zip) (~109MiB) in the notebook working directory **as candidate_bundle.zip**, then run [submit/inference.ipynb](submit/inference.ipynb) on one visible T4. The identical archive is also stored at artifacts/selected_bundle.zip. No credentials or network are required. Output: `candidate_output/predictions.jsonl`,1,919 ordered `{"code":"..."}` records. Timings and candidate pools are retained separately. [Run instructions](submit/README.md).

User solution deadlineOctober24 2026,23:59 Moscow; official report deadlineOctober25,23:59 with source timezone unspecified. [Task/judge constraints and unknowns](reports/TASK_AND_JUDGE.md), [official assignment](https://github.com/ashaba1in/hse-nlp/blob/2026/homeworks/hw2_code_generation.md).

## Development and reproduction

The selected own portable-D runtime and exact proof are documented in [selected method](reports/BASELINE.md) and [proof report](reports/OWN_FULLPOP512_N16_ONE_T4_PROOF_20261024.md). The following B031 beam commands are historical reproduction, not the current selected runtime:

```bash
.venv/bin/python scripts/package_recovered.py \
  --root artifacts/kaggle/b031-b032-v1 --run b031 \
  --output artifacts/reproduced_greedy_b031.zip
.venv/bin/python scripts/package_inference.py \
  --input artifacts/reproduced_greedy_b031.zip \
  --output artifacts/reproduced_beam_b031.zip \
  --profile-root artifacts/kaggle/b031-b032-v1/decoding-b031 \
  --validation-report reports/evidence/b031_b032_root_numeric.json \
  --validation-label b031_beam4
# Follow the Colab/Kaggle CLI guides for remote reproduction.
```

Use [notebooks/work.ipynb](notebooks/work.ipynb) to inspect metrics and examples.
Reusable code lives in `src/nlp_hw02/`; VM stages are in `scripts/colab_job.py`:
`prepare`, then `train`, then `test`. Optional controlled runs use `train --variant
xsa`, `muon`, or `relu2` and reuse the baseline's prepared arrays. Measured results
and failures belong in the experiment log.

HW2 `.venv` links to the submitted NLP HW1 environment. Use `.venv/bin/python` or
`uv run --no-sync`; **never `uv sync` or plain `uv run` while linked**. Install
additively with `uv pip install --python .venv/bin/python ...`. `uv.lock` is HW2
metadata, not the entire shared environment. No local shared packages were changed
for these runs; CUDA work runs on authorized Kaggle/Colab or the owned paid node. Remote versions are in
`docs/guides/requirements-colab.txt`. DL2 venvs are separate.

## Layout

| Path | Purpose |
| --- | --- |
| `assignment/` | Official snapshot, provenance, pinned data manifest/card |
| `src/nlp_hw02/` | Tokenizer, model, data, training, inference, optimizer |
| `scripts/` | Packaging, Colab stages, checks, notebook generation, profiling |
| `notebooks/work.ipynb` | Working notebook for run inspection |
| `submit/inference.ipynb` | Canonical inference notebook, no submission automation |
| `artifacts/` (ignored) | Recovered weights, tokenizer, metrics, predictions, bundles |
| `experiments/LOG.md` | Hypotheses recorded before launch, results and failures |
| `reports/` | Results, problem log, agent disclosure, executed evidence |
| `docs/guides/` | Agent Colab/Kaggle operations, decoding and dependency records |
| `docs/research/` | Cited architecture/optimizer and NanoGPT transfer research |

## Data and credentials

The sole shared local raw-Parquet copy is `../data/tinysms_qa_12m/`; use
`NLP_HW02_DATA_ROOT` for another existing copy. The pinned dataset has 60 train
shards plus a question-only test file (~1.485 GiB compressed). B001 used four
shards; B005/B006 used2M rows. The selected model starts from the original6M eligible-row400k run and uses a completed131072NEW continuation on all11.81185M eligible original rows across all60shards. All retain the grouped held-out split. Tokenizer-fitting questions
are excluded from primary validation; legacy-validation questions are excluded
from both new training and primary validation. This addresses a measured shard-distribution bias; it does
not establish template independence. Large token arrays stay remote. No second
full dataset/HF/Arrow/CSV copy was created locally.

```bash
.venv/bin/python scripts/fetch_dataset.py --check-only
```

W&B is optional tracking, not submission. Its local key is at
`../secrets/wandb_key.txt`, or injected remotely as `WANDB_API_KEY`. Never print,
embed, commit or package credentials. This run used JSON logs and did not upload
to W&B or Telegram. Private Kaggle source/cache uploads and cloud launches were
explicitly authorized here; no credential was packaged or uploaded.
