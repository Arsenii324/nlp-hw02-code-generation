# Selected solution: GELU512 step525952, eager fixed16

Selected October24 2026 after actual training recovery, held-out output audit, exact cold one-T4 proof and independent readback. No agent submission. The earlier half-cooldown415656/fixed24 solution is preserved at `artifacts/selected-half415656-fixed24-preserved-20261023/`; see its [historical report](BASELINE_HALF415656_20261024.md). Work continues toward the deadline.

## Run the selected inference

Put [candidate_bundle.zip](../artifacts/candidate_bundle.zip) in the notebook's working directory **with that exact filename**, then run [submit/inference.ipynb](../submit/inference.ipynb) on one visible NVIDIA T4. On Colab the usual working directory is `/content`. The same ZIP bytes are retained at [selected_bundle.zip](../artifacts/selected_bundle.zip). The offline notebook loads packaged source/weights/tokenizer and the ordered question-only1,919-row JSON, then writes `candidate_output/predictions.jsonl`, with one `{"code":"..."}` record per question. It also retains candidate and timing diagnostics. Use a fresh output directory for another run; preserve prior results.

| Selected artifact | SHA-256 |
|---|---|
|Bundle (114,412,183 bytes)|`22a91d853126a017b67a1a2cd290ff5c05794ee7a3764a9dd8250c4796e27918`|
|Notebook|`9b13a8a9f2e971a2df1de8ef30fa803a1c43db662a3b0b928e8aa6b7ca0e9730`|
|Model|`8f72f22a383bfe97aa251f5c5dabf895c118b375530bc73fb99893d66272131e`|
|Expected1,919-row code-only JSONL|`67990faa733f8adfb52d6891d733586599ce041d9c54f6b61080c2ce54501504`|

[Promotion receipt](evidence/own_fullpop512_n16_selection_20261024.json) records old/new hashes, preserved alternatives, the companion filename and exact proof chain. Canonical source/weight bytes were copied unchanged after qualification. No credentials, training targets, pretrained model or submission automation are packaged.

## Model and lineage

30,705,728 parameters: vocabulary2048, width512,8heads,4encoder+4decoder blocks, GELU FF2048, learned positions and64 learned self-attention projection/XSA coefficients. Cross-attention is ordinary. All weights were trained from scratch for this project. The original400,000-step run used6M eligible original rows; its full optimizer/scaler/TorchRNG endpoint then initialized a fixedNEW131,072-step linear continuation on all11,811,850 eligible original rows, with explicit new population sampler seeds. No extra data, targets, filtering or distillation.

Selected CE-best **525952**, loss0.22442179358905842; completed continuation endpoint **531072** is distinct. Training sourceac069fca/archive27a4c82d, root CPU/lineage acceptance and actual T4 reference are retained in the[candidate payload](../artifacts/own-fullpop512-n16-candidate-20261024a/payload/) and[actual T4 audit](evidence/fullpopulation512_count_pair_local_stage_20261024a/actual_quality_independent_review.json). A separate longer196608NEW fork remains experimental. Old398336 qualified artifacts are preserved in artifacts/selected-own512398336-eager16-preserved-20261024a/ and[historical report](BASELINE_OWN512398336_20261024.md).

## Inference and proof

Eager cached decoding,16 sampled programs/question, temperature0.8,top-k50, questionbatch32, seed20261002, length penalty0.6 and cap192. The frozen restricted AST interpreter groups supported scalar results; duplicate samples vote and ties use retained sampler score. Unsupported-all rows retain the original first candidate. This is reference-free numeric plurality, not arbitrary Python execution. The model does not receive external tools during generation. No RL, speculative decoding, test-time training or remasking is used.

The exact cold notebook reproduced all1,919 outputs on one Tesla T4 with Torch2.10.0+cu128/CUDA12.8. The four clocks were **120.792058 /137.023654 /137.027575 /143.030739 seconds**, including package verification, loading, warmup and output readback in the outer clocks. [Full proof and limits](OWN_FULLPOP512_N16_ONE_T4_PROOF_20261024.md). Production records its observed backend rather than rejecting a different version; qualification applies to the measured backend. The precise external judge timer/entrypoint/tolerance remains partially unconfirmed; see [task and judge](TASK_AND_JUDGE.md).

## Quality and remaining uncertainty

Internal reused noisy-reference agreement: **1135/1529 (74.23%)** legacy,1136 augmented; auxiliary1019/1141 legacy,1020 augmented. Old398336/eager16 scored1130/1132, auxiliary1013/1015. Net primary+5 (50new-only/45old-only,p=.682), auxiliary+6 (27/21,p=.471). These are descriptive reused-proxy gains, not established true pass@1 improvements. No official score is known for these bytes; do not attribute the earlier646/960 public score with unconfirmed identity to them.

[Actual paired T4 screen](FULLPOPULATION512_T4_N8_N16_ACTUAL_20261024.md): N8=1119 legacy, N16=1135. Native A100 N16=1147 did not transfer exactly to T4. The previously chosen weighted selector improved cached A100 but lost2 T4 primary rows, so was rejected. Earlier fixed-cache graph decode was slower on T4 and remains rejected. Longer training, mature768 and explicit question-batch scaling are experiments; they cannot replace this qualified package without actual quality/reference and fresh one-T4 proof.
