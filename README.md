# GELU768 + GELU512-new mixture hand-in — 73.75% public result

This separate code-only candidate package binds the exact I test notebook and JSONL to the user-reported public score. It includes both model lineages, the shared training/source archive, and raw T4 proof. **No checkpoint weights, raw training data, secrets, or credentials are included.** The alternative C/H 72.50% package remains at `../github-72p50-20261024/`.

## Candidate and evidence

- Models: full-population GELU768 selected step528,000 (endpoint531,072; SHA `ea4f5bc5e72d418987bc8fe1439e0afc736a7ae51447aab5363a70d154bd6972`) plus GELU512-new selected step662,144 (SHA `fd5bbeeadc8f451a2d47c740d73a38e75d3e5cdb65b72194e5065a138c161462`). Exact configs and training contract hashes are in `model_metadata.json`. Shared tokenizer SHA: `0fa06854a98e82cd520c13f4c670040c772fec38f2cd2cacb5de11130bc42d74`.
- Method: 768 N8 plus 512-new N3; temperature0.8, top-k50, length penalty0.6, seed20261002; grouped encoder-once decoding with q64/q128 per-model question batches and numeric plurality over all11 candidates.
- Predictions: `predictions.jsonl`, 1,919 ordered code-only rows, SHA-256 `f1c740584ce7d15b342fb69593390e614eb956b65fc864e9426216cb2dd143d0`.
- **User-reported public score: 708/960 =73.75%**, with0.1% reported execution errors and0% syntax errors. The saved receipt binds the result to this exact prediction hash. No per-question test feedback was used for tuning; the public service was not independently queried in this package build.
- Internal supported-consensus agreement is1170/1529, from G's separate validation run. I itself was test-only, but its candidate pool and output hashes exactly match G's validation/test method. This internal score is not official pass@1.
- Cold one-T4 qualification for I: core118.209s, executor151.291s, notebook151.211s, parent160.023s; model load15.052s. One visible Tesla T4, exit0, children reaped, all1,919 rows replayed, zero syntax errors; every clock is <=180s.
- The older C/H fullpopulation768 N8 alternative is also qualified and scored user-reported696/960=72.50%, with H exact-output parent clocks142.017/142.016s and C internal1152/1529. The newer GELU512 step662144 N16 J run qualified at162.015s parent and scored1142/1529 internally, but has no reported public score. See `proof/final_candidate_selection.json`.
- This exact direct N8+N3 file is distinct from older A100 prefix-mixture output SHA `99e3d451…`.

## Reproduction

`inference.ipynb` is the exact I notebook; its SHA is recorded in `model_metadata.json`. Attach private Kaggle input version1 `arsen4ikvar/nlp-hw02-mixture1166-proof-20261024a-tools` as `candidate_bundle_v8` with both model files and `tokenizer.json`. Weights are not in this hand-in. The notebook masks to one visible GPU before CUDA initialization.

`solution-code-20261024.zip` contains training and inference source/contracts for both lineages; exact retraining still requires omitted parent checkpoints and token cache. I was a test-only run; G's quality receipt supplies the separate validation result and the exact pool/output hash link. Raw I output, timing, parent, executed notebook, pool and recovery receipts are in `proof/`.

This is the recommended current local candidate, not an agent-made official submission. The canonical older GELU512 notebook/checkpoint remains preserved and untouched. `MANIFEST.json` hashes every package file except itself.
