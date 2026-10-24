# Reproduction and provenance

`inference.ipynb` is the exact I test notebook (SHA in `model_metadata.json`). It uses private Kaggle input `arsen4ikvar/nlp-hw02-mixture1166-proof-20261024a-tools`, version1, under `candidate_bundle_v8/`; required checkpoint and tokenizer hashes are listed in model metadata. This package contains no weights.

The method combines GELU768 selected step528000 (training endpoint531072) N8 with GELU512-new selected step662144 N3, using temperature0.8/top-k50/length penalty0.6/seed20261002 and numeric plurality. The exact output SHA is `f1c740584ce7d15b342fb69593390e614eb956b65fc864e9426216cb2dd143d0`. Two-model configs/contracts and training provenance are recorded in `model_metadata.json` and the included source archive.

I's cold one-T4 test clocks were118.209/151.291/151.211/160.023 seconds (core/executor/notebook/parent); all were <=180s. It ran on one visible Tesla T4, exited0 and reaped its child. G's actual validation output and I's test candidate pool/output match exactly by SHA; G scored1170/1529 on the internal supported-consensus set. I did not generate validation itself.

The user reports708/960=73.75% public score for this exact JSONL (0.1% execution errors, 0% syntax errors). The original user score receipt is `proof/user_reported_public_score.json`. The recommendation and comparisons against the qualified C/H fallback and J checkpoint are in `proof/final_candidate_selection.json`. The public service was not independently queried; no per-question feedback was used for tuning.

Exact retraining is not possible from this package alone because it excludes original parent checkpoints and token cache. The canonical old512 submission notebook/checkpoint remains preserved and was not overwritten. No agent push or official course submission occurred.
