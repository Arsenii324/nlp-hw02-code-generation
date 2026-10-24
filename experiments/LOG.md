# Experiment log

## B034 terminal cooldown — 2026-10-22 — prepared, not launched

Hypothesis before launch:30k updates linearly reducing each restored300k group LR
to zero improve endpoint/nearby-average inference relative to30k original
absolute inverse-sqrt continuation. Same B034 fullstate/6M cache/draw/dropout
state, optimizer/scaler/ratios and unchanged CE-best selection. Fixed menu:
best, endpoint,328–330k average,326–330k average. Source isolated and committed;
static/hash checks only. Runtime state/CE/ETA and quality remain unverified;
root independent review/private readback/fresh quota precede future launch.
[Plan](../docs/plans/2026-10-22-terminal-cooldown.md),
[stage](../reports/evidence/terminal_cooldown_stage_20261022.json).

Review correction C001 before upload: stagea admitted from a301k update/heartbeat
before periodic validation/checkpoint cost was measured. Preserved stagea;
fresh stageb waits for completed1,000NEW checkpoint and estimates rate from
cumulative elapsed including validation/fullstate save. No experiment run.
[Stageb](../reports/evidence/terminal_cooldown_stage_b_20261022.json).

## TPU V6 comparison — 2026-10-22 — local stage, not yet launched

Hypothesis before launch: documented TpuV6E8 may have a different admission delay
and gives actual bounded compile/throughput/conversion evidence versus the
unchanged V5 pilot. Stage reuses exact reviewed stagec weights/source/driver and
old2M cache; reset moments/RNG remains explicit. Both1,800s pilots may complete,
combined requested caps3,600s/20h allowance. Root must review fresh quota and
launch once; entitlement, pool independence, compatibility and results unknown.
No cloud mutation/test/model run by staging. [Plan](../docs/plans/2026-10-22-tpu-v6-comparison.md),
[hashes/limits](../reports/evidence/tpu_v6_pilot_stage_20261022.json).

Root push outcome: provider rejected V6 with `Maximum batch TPU session count of 1 reached.`
No second TPU allocation/retry; existing queued V5 reserves the single batch slot.

## B001 — 2026-10-01 — small encoder–decoder baseline

Hypothesis recorded before launch: a 2+2 layer, d=192 Transformer with a
2,048-token byte BPE can learn code structure from a bounded 800k-row subset
within one hour on a T4; semantic correctness will likely be poor. Cached,
batched greedy decoding is expected to fit 180 seconds, but this is unmeasured.

- From-scratch tokenizer/model; no additional data or targets. Train shards
  0–3, first 100k pairs for tokenizer. Validation: first 2,048 unique questions
  from shard 59 excluding exact question matches in all four train shards.
- Dataset revision: `b2724689ff079e9c22efd548fe134bfb29036dbb`; checksums verified
  for every downloaded file. Seed 42. Source/target caps 256/192 (EOS included).
- d=192, heads=4, encoder/decoder layers=2/2, FFN=768, PreNorm, GELU, dropout .1,
  tied source/target/output embedding, learned positions. Cross-attention exists.
- AdamW fused, lr .0006, warmup 200 steps then inverse-square-root decay,
  betas .9/.98, weight decay .01, clip norm 1, batch 64, FP16 GradScaler.
- Budget: 3,600 seconds of training or 20k steps; evaluate every 1k steps;
  checkpoint chosen by held-out token loss. Generation: greedy, batch 128,
  cap 192; predeclared before test timing, never tuned on test prompts.
- Validation: token CE on 2,048; text match and AST validity on first 256.
  These do not measure execution pass@1. Full test predictions are internal
  JSONL artifacts only, no claimed bot format and no submission.
- Colab session `nlp-hw02-baseline`: one Tesla T4, 2 CPUs, PyTorch
  2.11.0+cu128, Python 3.13.15. Remote source archive digest recorded per upload.
- Measured: 20k steps, 1.28M sampled pairs, 760.26 s including periodic/final
  loss validation. CE 7.6660 → 1.8074; final checkpoint selected. Text match 0/256;
  AST validity 253/256. Full 1,919-row test generation 6.8288 s on one T4 at
  batch128, 8 capped outputs; canonical notebook repeat 6.8898 s. Peak trainer
  allocation ~1.010 GB, steady throughput ~26–28 steps/s. No semantic pass@1.
  Raw metrics duration includes ~1.06 s post-training; use the finished log event.
  Frozen source SHA256 `846f44c6b9d400f7d3c9316a447fccadf53aebb0d0e3b9b3cb1f2e747301fb01`.
  Initial bundle SHA256 `e0943c15c4f1f669f6331eb4a072afef12854bfdbc14c02588fc91234fb7efa3`.

## B002 — 2026-10-01 — Exclusive Self Attention (launched 18:42 UTC)

Hypothesis: excluding the self-value direction improves held-out code modeling
at equal steps, with a tolerable throughput cost. Single change from B001:
per-head `Y - <Y, normalize(V_self)> normalize(V_self)` before output projection
in encoder/decoder self-attention; cross-attention unchanged. Normalize/reduce
in FP32 for FP16 safety. Encoder application is an extrapolation from Zhai's
causal-LM experiments, not a reproduction of their 0.7B–2.7B / FineWeb setting.
Same tokenizer, arrays, seed, initial weights, batch, schedule, 20k step/1h cap.
Select by the same held-out token CE; report throughput and proxy generation
quality. Primary source: https://arxiv.org/html/2603.09078v1, Eq.2 / Algorithm1.

## B003 — 2026-10-01 — hybrid Muon (planned, not yet launched)

Hypothesis: built-in modern Muon's matched-RMS shape adjustment improves
validation CE per step enough to offset orthogonalization cost on a T4.
Single treatment from B001: hidden 2-D attention/FFN matrices use PyTorch 2.11
Muon (momentum .95, Nesterov, 5 NS steps, `match_rms_adamw`, peak lr .003);
embedding/tied head/positions/norms/biases retain AdamW peak lr .0006.
The 5x factor compensates nominal Muon update RMS ~.2; it is an untuned
hypothesis, not a proven equal effective update. Other settings and seed fixed,
same 20k/1h cap. Different optimizer state, momentum and matrix scaling are
part of this treatment; not evidence for each ingredient separately.
Expected downside: T4 lacks native BF16 Tensor Cores for the built-in NS path.
Report actual speed and compare equal-step and equal-time loss from logs;
one seed, so no statistical superiority claim.

## B004 — 2026-10-01 — NanoGPT-inspired ReLU² (planned, not yet launched)

Hypothesis: replacing GELU by squared ReLU improves validation loss/throughput
at this scale. Single change from B001: FFN activation `relu(x)**2`; width,
parameter count, initial tensors, data, schedule and other settings fixed.
20k steps/1h cap, same validation and generation protocol. The speedrun's
2024-10-14 record bundled this activation with other changes; this run isolates
it on our encoder–decoder rather than claiming to reproduce their result.
Source: https://github.com/KellerJordan/modded-nanogpt (record #5), context
and newer speedrun details in `docs/research/nanogpt-speedrun-transfer.md`.

Follow-ups queued sequentially after the baseline passed training, diagnostics,
test timing and canonical notebook verification. B003/B004 start after B002.
Their frozen source SHA256 is
`efe9c1a592cd62b40bdb2ab112f1946531731e6222c4d63de0287f9644971c9c`.
Initial tensors were verified identical across baseline/XSA/ReLU² at matched seed.

## I001 — 2026-10-01 — batch128 → batch256 inference (selected before test run)

Validation-only microbenchmark on 256 questions, two repeats: mean batch32 3.49 s,
64 2.09 s, 128 .99 s, 256 .48 s. Three texts differ from batch128 for each other
size; this does not establish better/worse semantics. User requested exploiting
the larger batch. Choose batch256 without examining test outputs, measure once
the GPU is free, retain original batch128 predictions/timing, and compare text
differences. No weight/training change. Expected ~2x validation throughput;
full-test ratio and exact resource-utilization cause remain unmeasured.

Copy this record per experiment, writing the hypothesis before launch:

- ID/date; hypothesis and expected observation:
- Single change relative to the parent run:
- Commit/source archive; dataset revision; split definition; seed:
- Configuration; tokenizer/checkpoint IDs (once you implement them):
- Runtime: GPU count/name; PyTorch/CUDA/package versions:
- Budget; actual wall time; throughput; peak memory:
- Validation metric and protocol; W&B run link if used:
- Measured one-T4 full-test generation time (if applicable):
- Outcome, failure modes, interpretation and next decision:
# 2026-10-01 19:19 UTC — runtime lost; experiments halted

Colab monitor failed with lost-session404/401, CLI exit1. User required halt on
unavailable compute; goal paused and no reallocation/fallback attempted. B002
XSA recovered: 20k iterations,954.24s, old-holdoutCE1.68285 versusB0011.80739,
testbatch1288.01172s/1919rows. One-seed equal-iteration quality gain costs25.5%
training time; these are token-loss/syntax diagnostics, not semanticpass@1.
B003 Muon completion and B004 ReLU² state unknown. B005 broader-data plan was
not implemented or launched. Defaultbatch256 is configured, but full-test256
measurement pending. Preserve completed and incomplete evidence separately.

## 2026-10-01 — provider transition and CPU preparation

User authorized Kaggle and another bounded Colab probe. One T4 was allocated,
passed an FP16 matrix check (Torch2.11.0+cu130), and was stopped immediately.
This confirms new access, not old-runtime recovery or the cause of loss.
Kaggle CLI2.2.3 authenticates as arsen4ikvar. Raw quota adapter:30h allowed,
0used/0reserved, reset3Oct00:00UTC; native parser fails on protobuf duration `0s`.

CPU-only `nlp-hw02-prepare-v2` v1 failed before work due automatic ZIP expansion.
v2 verifies expanded-source file hashes and succeeds:2M rows,2048primaryquestions,
all60shards,950.33s script elapsed,772.78s sampling/tokenization. GPUquota stays0s.
Print fraction22.0994%; source/target truncation.00265%/4.15525%. Training excludes
all question-hash bucket members and old holdout copies. Primary excludes original
tokenizer-training question hashes. Small metadata/checksums recovered, no arrays
or rawdata downloaded. Cache manifest SHA256:
`41efd8551f4e9d679a4f06b4e6f86163c15af52c5f6711deacd765b7c06d9a9b`.

## B005/B006 — Kaggle broader-data control/XSA (predeclared)

Hypothesis: broad shard coverage improves generalization; XSA's earlier CE gain
survives a clean question-group holdout. New AdamW control B005; B006 differs
only by XSA projection. Same2.556M model, tokenizer2048, seed42/initialtensors,
batch64/FP16/schedule/caps256/192.60k iterations each,3.84M sampled pairs. Select
checkpoint by primary2048CE; old shard59 holdout is secondary. Old models are
not clean controls on the new primary. Compare common iterations and elapsed curves.

Independent seed20261001 samples256primaryquestions for generation; indices saved.
Bounded AST single-reference numeric agreement: rel/abs tolerance1e-9, explicit
reference/prediction coverage, unsupported predictions count against denominator.
Print arguments ignored; synthetic references may be wrong/conflicting. No claim
of officialpass@1, no additional targets or arbitrary code execution. Reproducible
old-holdout audit:1614/2048supported,22conflictinggroups,50/256questions recurring
later. Earlier1608one-offcount was not reproduced; evidence retained.

One offline T4×2 session, controldevice0/XSAdevice1, no DDP/additional sessions.
Source SHA256`8a91dc1a34d3f1520194a416c1c1997648ebe6d992af0d5eaa88579e315574cb`;
driver SHA256`a3ad1c01eac487ca041816ec12cea30f75190aadb9de1431357bd22c8460dacb`.
Expected cache hash and headers gate launch. Outer timeout9000s; early measured
ETA admission reserves1200s for bounded checkpoint/inference tail. A healthy final
lane finishes. Explicit failure policy retains an already admitted useful sibling;
no retries or new jobs. Stall guard checkpoints/exits; terminal manifest survives
exceptions, records update/overflow counts and output hashes. RNGstates retained;
resumption remains unverified. Static review found no remaining launch blocker.
Actual CUDA throughput/timing unmeasured until this run.

After both trainers exit, measure B001batch256 and both selectedmodels on one
isolated T4, all1919testrows with originalcaps. No test tuning or submission.
## B005/B006 measured results — 2026-10-01 21:09 UTC

Both completed60k iterations, selected final checkpoint; optimizer updates
59,977/59,978 (23/22 FP16 skipped updates). Primary CE.71291867/.66400844;
legacy CE.74462172/.68564780. Training including periodic/finalCE2613.15/3258.03s.
XSA lowers matched-iterationCE6.86% and costs24.68% more time. Strict numeric
consensus on identical random256 subset:48/219control versus43/219XSA;219fully
supported groups,5partial-consensus,3conflicts,29unsupported-only. Discordant
correctness12:7,p=.3593; no robust semantic improvement established. Ordinary
attention retained for next pair: faster and numerically better point estimate.
Officialpass@1 still unavailable; supplied synthetic references may be wrong.

Fulltest1919 at256: B0053.18070s, B0063.76309s, isolatedoneT4, no capped outputs.
B0012563.06926s versus oldColab1286.82881s;16changedtexts, runtime+batch differ,
so no isolated causal batch-size/semantic-quality claim. Source/table provenance
in recoveredterminalmanifest;26retrievedartifactSHAchecks, sourceCRC and all
orderedrows verified. Kaggle COMPLETE, bothchildrenexit0, no grace stop/timeout.
Quota3337.472s (55.62min) charged; script3327.97s. Faster lane's idle tail is
retained in this accounting, not excused by its unloadedGPU. Source/outputs kept.

## B007/B008 staged hypothesis — 2026-10-01 21:05 UTC

Raise width192→384 and FF768→1536, preserving2 encoder/2 decoder layers,
heads4, original byteBPE/caps/representative2M split. Parameters9,242,112.
Compare AdamW versus hybrid Muon at40k iterations each, seed42, batch64,
same schedule/initial tensors. Attention setting is chosen before upload from
B005/B006 clean primary CE and numeric diagnostics; both new lanes use the same
setting. A capacity comparison to B005/B006 is a recipe comparison (different
iteration budget); the within-pair optimizer comparison keeps architecture fixed.

Muon's NS implementation is selected by a predeclared optimizer-only timing
gate: faster finite stock BF16 versus grouped FP16, all actual matrix shapes,
fixed random gradient fixture with no training targets/retained weights.
FP16 normalization/state remain FP32; no newer coefficient or momentum change.
Local independent recurrence, missing-gradient, all-group validation, ambient
autocast and serialization checks passed; FP16 relativeL2~.0011–.0021 on tested
fixtures. No CUDA speed/quality claim yet. One useful session after current batch
ends; no filler jobs or automatic retries. Source/manifest frozen again immediately
before actual upload, exact hashes to be recorded then. New dataset/kernel IDs
preserve current B005/B006 inputs.

2026-10-02 09:04UTC: user restored fullaccess; Kaggle status/quota work again,
previousrunCOMPLETE, quota still3337.472s/reserved0. Ordinary attention for both
new lanes. Frozen sourceSHA6248912874660473eb1247b7481af4455eb69dc6e10fcc79895206d3e320a832;
driverSHA2fe584ba12f26123e29d1166f241ab30c23187846eacce75a0ad3c5c8f2b3aba.
Current allowlisted ZIPmembers/CRC/source/driver equality verified, groupedMuon
CPU checks repeated successfully. GPU compatibility/timing remains pending.

2026-10-02 09:08:19UTC script start, private Kaggle capacity kernelv1. CPU source
cache remainsv2; separate capacity toolsdatasetv1 ready, four expected files only.
Both actualT4s/Torch2.10.0+cu128 pass preflight. Operator-only25steps/path (5warmup,
20timed) atd384: stockBF16median41.4508ms, groupedFP167.7540ms, groupedFP3225.0894ms.
GroupedFP16~5.35x faster than stock **optimizer path**, not training. RelativeL2
vsFP32.001392square,.001470tall,.001461wide; allfinite. T4 stockBF16 does execute;
nativeBF16TensorCore absence does not imply runtimefailure. Predeclaredselector
chosegroupedFP16. Trainersstarted09:08:52UTC (32.67s afterentry), initialCEboth
7.84775064. Coldfirst100iterations~2.4steps/s, ratesstillsettling; nofinalETAclaim.
User's12:00Moscow check-in was soft; continuation explicitly authorized.

## B005 validation-only beam diagnostic — 2026-10-02 09:41 UTC

Predeclared CPU FP32 pilot256: greedy48/219, beam4alpha0 49/219,
beam4alpha.6 52/219 supported-consensus agreement; normalized-search
paired discordance7:11,p=.4807, roughly5x decoding time. No established pilot
benefit. Before extension, fixed only greedy versus beam4alpha.6 on remaining
1792 primary rows, excluding pilot. Result240/1529 versus270/1529 (15.6965%
versus17.6586%; +1.9621pp), discordance38:68,p=.004619 descriptive. Syntax
1782/1792→1787/1792, exact text21→24, supported predictions1539→1642.
CPU seconds23.484→106.179 (4.52x), capped1→0. All1792 rows retained including
196unsupported-only,38conflict,29partial groups. No generatedcode executed.

These rows were already used for CE/model selection; exact-question grouping
may leave correlated templates and synthetic reference errors. Not official
pass@1 or untouched final test. No coefficient adjustment after extension,
no GPU allocation, no canonical promotion until same-runtime GPU quality and
fulltestoneT4<=180s measured. Beam code is separate from frozen running inputs.
Reviewer found width1 topk/argmax tie and FP32logsoftmax rounding differences;
fixed by dispatching realmodelbeam1 to originalgreedy and rawargmax fakefixture.
Added near-tie/zero-weight realmodel, analytic mixedlength/cappedcompleted
fixtures; CPU check_beam.py passes cacheparent/source/EOS/fullprefix checks.

2026-10-02 09:58UTC GPU decoding follow-up predeclared: after capacity results,
use one brief ColabT4 session to run the exact selected-model greedy notebook,
then compare fixedbeam4alpha.6/greedy on remaining1792validation questions,
FP16autocast,batch256,samecheckpoint. Both FP32 and FP16 CUDA cached-beam versus
full-prefix fixtures must pass first. Measure fixedbeam all1919test timing once;
test content/syntax does not select parameters/checkpoint. Separate inference
addon hash from immutabletraining source; canonical staysgreedy until evidence
is reviewed. Stage local artifacts before allocation, no Kaggle filler/retry.

2026-10-02 10:03UTC selection rule before capacity results: compare B007/B008/B005
on identical random256 primary predictions under strict supported-consensus
numeric score; choose the best measured numeric point estimate, use primaryCE
as tie-breaker, requiring complete provenance and oneT4timing<=180. This is a
one-seed, limited-subset choice rather than a robust optimizer superiority claim.
In the already planned short Colab reproduction, include recoveredB005 as a
same-runtime control on remaining1792 (greedy and fixedbeam4alpha.6), then run
selectedcapacitymodel on those rows and fixedbeamtesttiming. No alpha/width
search or test-content selection; compare primarypredictions on sameCUDA/autocast.

## B007/B008 completed and selected — 2026-10-02 10:10 UTC

Both completed40k iterations without timeout/grace stop. AdamW39989updates/11AMP
skips, Muon39988/12. PrimaryCE.52479548/.43405152; traininginclCE3216.70/3631.88s.
HybridgroupedFP16Muon lowersCE17.2913% atmatchediterations, costs12.907% more
trainingtime. On identical256rows/219fullysupportedconsensus groups, numeric
77/219(35.16%) vs102/219(46.58%); discordance11:36,p=.000346 descriptive.
B00548/219(21.92%), differentwidth/iterationbudget confounds capacity comparison.
Same seed/split/init, Muon hidden2D+AdamWembedding/norm/bias, originalNS5coefficients,
FP16NS/FP32normalization,momentum.95/Nesterov,peakLR.003 vsAdamW.0006. Not proof
of universalMuon superiority or benefit of newerANVIL/CANS/PolarExpress methods.

B008selected by predeclarednumeric-first/CEtie-break rule. Full1919oneT4greedy
batch256: B0073.481798s,capped0; B0083.450511s,capped1. Canonicalgreedy preserved.
TerminalCOMPLETE,27retrievedartifacthashes/sourceCRC/exactprelaunchmanifest/
driver/cache/config/iterations/updatecounts/1919orderedrows verified. Latest
optimizerstates and legacyvalidationfiles notdownloaded, explicitly listed.
Quota3696.186s(61.6031min)charged for batch; cumulative7033.658s(117.2276min).
Fasterlaneidle411.656s is charged, not hidden. No automatic retry/filler run.
PortableB008bundle34902202bytes SHA f93f25d5196d192bef9017ccf250217a684e9606644d16afd7d3b7ed832c6e52;
B005retainedbaseline_bundle_b005.zip and its oldmanifest. Model/inference/tokenizer
source byte-identical between the two frozenarchives; actualtrainingarchives
remain distinct. Final Colab notebook/decoding checks are a separate pending stage.

## Final Colab checks and connection failure — 2026-10-02 10:26 UTC

Exact canonicalB008greedy notebook on oneColabT4/Torch2.11.0+cu130:4.406959s,
all1919rows, checkpointload.393521s,capped1. Inputcopy SHA matchescanonical,
all3codecells have successoutputs/noerrors. Rootchecked all orderedrows and
zero predictiontextdifferences fromKaggleB008greedy. No submission.

Separate fixedbeam4alpha.6 GPUchecks completed: profilefiles appear only after
both FP32/FP16 cache/fullprefix fixtures pass. Same-runtime remaining1792:
B005greedy238/1529→beam269/1529 (15.5657%→17.5932%;discordance36:67,p=.002929),
B008greedy594/1529→beam645/1529 (38.8489%→42.1844%;discordance39:90,p=.000008288).
B005times3.843/12.639s, B0084.152/20.353s; B008beamfulltest22.691868s,all1919,
capped0,pinnedtestSHAchecked. SyntaxB0081786→1789/1792, stricttext32→36,
peakallocated.672→2.573GB. These are CE-usedvalidation diagnostics, not official
pass@1; templatecorrelation/one-seed/referenceerrors remain. Canonical staysgreedy.

The outerColabexec did not deliver completion and eventuallyraised
RuntimeError: Connection was lost. Cause unconfirmed; runtimeContentsAPI still
served completedfiles. RecoveredchecksZIP SHA651eb96956a9fcc4e610eaca73b4b1a840be5ce743212c421d9f849af9273799;
CRC/all12memberhashes/sourceaddon/selectedbundle/profileidentities and ordered
beamrows verified. No GPUchildtimeout reported; bundle was built after both
subprocesscheck=True callsreturned0. LastGPUoutput ZIPmtime10:18:40UTC, whereas
runtimewasstoppedaround10:26UTC: roughly7minutes elapsed after GPUworkfinished.
GPUutilization was not logged in thisColabstage; idle-tail estimate is inferred
from timestamps/script, not NVML. This is an operational failure and compute
waste, not a successfulCLIrun. All artifacts preserved, no retry/upgrade/auth
change; stoppedexactruntime and confirmed noactivesessions. Halted further
experiments and askedstudent whetherto investigate/resume, perblockerinstruction.

## B009/B010 width-depth pair — predeclared 2026-10-02 11:20 UTC

Student explicitly resumed after the Colab completion failure. Use Kaggle; no
Colab retry or authentication/package change. B009 d512/FF2048/2+2 (15,992,832
parameters), B010 d384/FF1536/4+4 (17,524,224), four heads/ordinary attention,
PreNorm/GELU/dropout.1, existing byte-BPE2048/caps256/192, seed42/batch64/40k
iterations, groupedFP16Muon+AdamW recipe from B008. Same representative2M cache
and exact primary split. Hypothesis: allocating comparable capacity to width
versus depth changes validation quality per iteration/time; not precisely
iso-parameter or an isolated optimizer test. No new targets or test tuning.

One new private T4x2 session, outer bound14400s. Estimated roughly2h is inferred
from B008 and model sizes; actual1000-step admission checks govern. Reserve five
300s evaluations +90s checkpoint +30s shutdown +180s near-finish allowance =1800s.
Healthy nearly completed training takes priority and optional evaluations are
explicitly deferred if needed. No filler/retry/installs; full-session idle time
is charged and logged. After both trainers exit, fixed one-T4 greedy timing and
remaining1792 greedy/beam4alpha.6 comparisons for both; one fresh-process exact
portable B008 beam notebook check. Logs/progress/NVML + actual notebook cell
outputs/result replace reliance on the failed Colab websocket completion path.

Selection remains validation numeric point estimate first, CE tie-break, full
provenance and one-T4<=180s gate. Compare pilot256 and remaining1792 separately,
retain B008 if neither capacity allocation improves; one-seed/noisy-reference/
template-correlation limitations remain. B008 verifiedgreedy bundle preserved;
new separatelyhashed beam candidate is not promoted until exact notebook passes.
Quota snapshot11:12:59UTC used7033.658s,reserved0,total108000s (~28.046h left).

2026-10-02T11:24:53.386761+00:00: private kernel `arsen4ikvar/nlp-hw02-width-depth-20261002/1` pushed successfully and server RUNNING. Reviewed tools dataset v2, downloaded remote manifest/checksums byte-identical to local. Exactly one session; quota prior7033.658s/reserved0. Frozen source f9329befb2711e70e059b4b02020f1c6fdfe69e3e633f55c8962959c2b71fb71, driver97adcb34b2e4700b390dca41fb2d01100618132251cd7f6e89aee53badbabbae. Waiting for actual hardware/startup/admission evidence.

2026-10-02T11:26:17.416865+00:00: hardware preflight Torch2.10.0+cu128/CUDA12.8, exactlytwoTeslaT4; baseline checks including Muon passed. Both trainers started11:25:18UTC,39.34s fromscriptentry. No install or upgrade. Startup target90s met. Progress/admission pending.

## Scaling run early stop — 2026-10-02T11:29:20.237203+00:00

Observed: at11:27:33UTC parent markedboth lanes stalled after120s without a
new loggedstep and sentSIGTERM; both exited0 around11:27:39. Parent had logged
step1 only; subsequent inference shows both selectedcheckpointstep100. No1000step
admission was reached. This is an incomplete failed comparison, not a trained
40k result. Possible mismatch between sparse100step logs and slow cold start;
actual cause pending recovered child logs. No exception/OOM shown in live log.

The already-running finite driver performs evaluations even on gracefully
stopped/incomplete trainers, including uninformative100step beam diagnostics;
this is another compute-discipline defect. It cannot be patched through this
CLI while running. Known B008 canonical reproduction is still useful and queued.
No retries/new allocations; explicitly asked student whether diagnosis/retry is
authorized under blocker instruction. Recover version1 bounded logs/evidence
and record final quota before stopping. Existing B008 solution unchanged.

## Failed comparison recovered; beam promoted — 2026-10-02T11:39:09.551924+00:00

ServerERROR because both lanes completed100/40000; each100optimizerupdates and
zeroAMPskips, cleanSIGTERM exits. First100127.9626/127.6467s, finalCE3.1284/3.2105.
Guard firedone second before100step logs. Sparse reporting cannot establish
true stall; underlying slow-start causeunconfirmed. No valid architecture or
40kthroughput comparison; do not treat partialmodels as candidates.

All49retrievedterminalartifacthashes/source/driver/cache/config/rows/exactnotebook
identities checked;4optimizer/legacyfiles explicitlynotdownloaded. Exact B008
beamnotebook passed3codecells/counts1/2/3,noerrors,1919orderedrows/23.113909s,
oneKaggleT4,capped0,zero textdifferences versuspriorColabbeam22.691868s. Frozen
inputSHA1b79bae3331774f8cc0fe5c9ca935a85b9fb7f2f6633b81dde9ed33109a2b6a3;
executedSHA483b6e5a8518d96a211f14d6a29ade71006576797a628e81c1bd0858991bb0f1.
Promotedunchanged beamcandidate toselected_bundle.zip,34984497bytes,SHA
97d65b75b0b6141a0a0119b50de55269e0095187bd89695323012064d9d55b62. Greedy
bundle/exactoldnotebook evidence preserved. Validationbasis remains645/1529
versus594/1529 boundednumeric, notofficialpass@1. No submission.

Sessioncharged402.623s(6.7104min),cumulative7436.281s,reserved0,27.934hleft.
Fouroptional incomplete-modelcalls consumed179.942s; knownB008canonical31.694s.
Bothtrainer exits toscriptend~211.79s; oldterminal idle-tailfield covers only
pre-evaltrainingtail and mustnotbe treatedasfull-sessionidle. This guard/queue
failure consumedquota and is recorded; noautomaticretry. New experimentshalted
pendingstudent's response to blocker question.

## Corrected B011/B012 pair — predeclared 2026-10-02T12:47:05.993961+00:00

Student explicitly authorized normal Kaggle continuation. Routine fixable faults
are handled within scope; no repeatedapprovalgate or blindautomaticGPUretry.
B011/B012 retain abortedB009/B010's d512/FF2048/2+2 versusd384/FF1536/4+4,4heads,
ordinaryattention,seed42/batch64/40k,groupedFP16Muon+AdamW,same representative2M
cache/BPE/caps/targets. Hypothesis aboutwidth-versus-depth allocation unchanged;
prior100step run remainsfailure, notcontrol orcapacity evidence.

Established protocolfix: actualstep/phaseatomicprogress every5s independentof
100step losslogs; unchangedstatus doesnotrefreshliveness. ETAusesfreshheartbeat
remainingwork andmeasuredlograte, preservinghealthy<=180snearfinish. Parent
recordsalllanecounts withmetrics/finished-event exactness orheartbeatlowerbounds,
includingcleanupfailures. Onlycomplete/nonterminatedtraining entersoptional
model evaluations. Real slowchild, frozenstate, nearfinish andincompleteeligibility
regressionspassedlocally; independentreviewfindingsaddressed.

Slow-start cause remainsunconfirmed. Residency hypothesis: random remotemmap
gathers canstallGPU; sequentiallyloadidenticaluint16arraysintoRAM, two concurrent
loaders eachsource-thentarget. Byte/order/seededbatch equalitychecked. Combined
RAMadmission includes2copies+pagecache+2GiBheadroom; loading/input/transfer/forward/
backward/optimizerwalltimes useexisting syncpoints, addnonewCUDA syncbarriers.
Newrun canidentify where timegoes, but changingdataaccess+providercoldstate
cannot byitself prove attribution. No newlocalfullcache/rawcopy/optimdownload.

Separate private scaling-resident-tools v2/kernel IDs; oldruninputs preserved.
OneT4x2 session,14400souter,4x300sevaluations+90checkpoint+30shutdown+180nearfinish
=1500stail. Actual1000step recentrate admission; ifworkcannotfit, checkpoint
andstoponlyaffectedlane, skipitsoptionalevals. NoextraB008canonicalrepeat.
Comparepilot256 andremaining1792 fixedgreedy/beam4alpha.6, validationnumericfirst/
CEtie/runtime/provenancegate; retainB008 ifneitherimproves. A newselectedcheckpoint
requires a briefexactcanonicalverificationbeforepromotion, nofillertraining.
Source6f69330a1493bea8246d0bd4afa8e353a92843b285d692ccb166f8323c1f0d74,driver8eaa692d0570f73a1779bdc7dc3eed6704c01614e947722e4a1d26b0e9d6b677.

2026-10-02T12:48:25.197866+00:00: private kernel `arsen4ikvar/nlp-hw02-width-depth-resident-20261002/1` pushed. Uploaded tools v2 manifest/checksums match local exactly. Prior quota7436.281s/reserved0; one session only. Live log `artifacts/kaggle/scaling-resident-live.log`. Await hardware, loading/step timings and1000step admission.

2026-10-02T12:51:37.262764+00:00: actualtwoT4/Torch2.10+cu128; trainersstarted25.23sfromentry. Each residentloadsource5.22/5.30s,target5.64s (~10.85/10.94s); RAMadmission32.14GBavailable. InitialCEexactlymatches abortedcounterparts8.539927/7.763468. First10011.519/15.139s versus127.963/127.647s; now98–99%GPUutilizationsamples. Resident inputgather~0.023–0.026s/100steps, earlyforward/backwarddominant. This supportsIOhypothesis butoldinputtimingsabsent/providerstatechanged, so noisolated11xIOeffectclaim. Waitingfor1000stepmeasuredadmission.

2026-10-02T12:54:04.835715+00:00: both1000stepbudgetchecksadmitted, expectedremaining{'b011': 4575.304832514002, 'b012': 6214.581124128}seconds+1500tail within14400bound. Both1000updates/0AMPskips. Runtime/budgetguardnowusesactualheartbeatstep, and optionalincomplete-modelgateisactive. No finalquality/ETAguarantee.

## Initialization diagnostic and verification preparation — 2026-10-02 13:21 UTC

CPU FP32, fresh seed-42 models, the same 64 held-out primary rows sampled with
seed 20261002; no training or test use. B008/B011/B012 initial CE
7.86027/8.55839/7.75057; previous-token argmax on all sampled non-PAD positions,
mean probability 0.10643/0.54785/0.02195. True next token equals previous on
0.018709% of positions. Evidence: `reports/evidence/initial_logits_diagnostic.json`.
Analytical tied-head alignment hypothesis and limits in
`docs/research/tied-head-initialization.md`; not causal final-quality attribution
or a muP-controlled allocation study. Active configurations remain unchanged.

Created separate `docs/guides/TRAINING_RUNTIME.md` covering actual progress,
fresh-step ETA, RAM/phase timing, completeness gates, counts and charged idle.
Prepared inference-only private staging/finite driver for a possible new winner;
no new allocation. Real preserved B008 notebook fixtures validate exact input,
executed output/counts, weights/source/addon/test/decoder, all 1,919 rows and
one-T4/180s. Tamper cases rejected; independent review found no material blocker.
An old staging fixture incorrectly compared its frozen driver to today's working
driver; corrected to frozen prelaunch identity, RED→GREEN. All 17 local check
scripts passed. This does not establish a new remotely measured notebook.

Live B011 around 16,000 steps, CE 0.51085, 15,998 updates/two skips; B012 around
11,700 steps, both GPUs 98–99%. No stall/budget guard fired. Final selection and
quality remain pending complete recovered results.

2026-10-02 13:26 UTC: offline B008 error categorization on the same 1,529 fully
supported consensus groups: greedy correct/wrong-numeric/unsupported/parse
594/781/150/4, beam 645/775/108/1. Unknown-variable refusals 86→51. Net gain
51 combines 65 wrong-numeric→correct plus 25 unsupported→correct, minus
34 correct→wrong-numeric and five correct→unsupported. Evidence in
`reports/evidence/b008_error_categories.json`; no generated code execution,
extra targets or test tuning. Supported wrong numbers remain the largest gap.

2026-10-02 13:33 UTC: posthoc reference-length strata on the same B008 remaining
validation: beam 9/44 when raw reference BPE length+EOS exceeds 192, 636/1485
otherwise; greedy 6/44 versus 588/1485. Length/difficulty confounding means this
does not isolate a cap effect. Full-test beam capped0 is a different statistic.
JSON records input hashes; no trained/cap/decoder change, no test tuning.

2026-10-02 14:09 UTC: live B011 reached all 40,000 iterations, CE 0.41166304,
39,987 updates/13 AMP skips, clean exit0. Training log elapsed 4,749.81s at
final block; full metrics/artifact verification awaits session recovery. GPU0
released; B012 around29k and remains healthy. No filler; charged idle tail
will include its remaining training and serialized evaluations. No promotion.

## Completed B011/B012 and candidate selection — 2026-10-02 14:45 UTC

Private kernel /1 COMPLETE. All46retrieved terminal hashes, source/driver/cache/
manifest/config/40k counts, selected40k and fixed full1919one-T4 profiles checked;
four optimizer/legacy files deliberately not downloaded. Both39,987updates/13skips.
B011 CE0.41166304, train4753.64s, pilot109/219, remaining greedy617/1529 and
beam669/1529, test3.977/24.125s. B012 CE0.34405379, train6527.94s, pilot135/219,
remaining greedy765/1529 and beam804/1529(52.5834%), test6.181/40.833s/capped0.
B008 beam645/1529. B012 wins predeclared selection; exact notebook gate pending.

B011/B012 profiles share Kaggle Torch2.10/CUDA12.8; B008 remaining profiles are
Colab2.11/CUDA13.0, a software/provider confound. Prior B008 full-test beam had
zero cross-provider text differences, which does not verify validation identity.
One seed, CE-used validation, synthetic-reference noise/template correlation and
width-dependent tied-head initialization remain limitations; no official pass@1.

Observed quota delta6743.979000s (112.400min), cumulative
14180.260000s, reserved0. Script6736.17s. Full-session
trainer-exit idle tails B0111934.707s and
B012162.444s; GPU0 idle while deeper
training completed, then onlyGPU0 used for serialized fixed evaluations. No filler.

Prepared unchanged B012 beam candidate, hash9c5ad4ac2ebb5517aed49db2b30c7852a629ea4da614bd112abf704ef727e7fd;
training files preserved, separate measured addon. Next brief inference-only
600s outer/420s child, private tools/kernel, fresh exact canonical3cells and
all1919one-T4<=180s; no training/install/filler/B008 repeat. Freeze/download-compare
inputs before launch. Promote only after recovered real execution identities pass.

2026-10-02 14:49 UTC: uploaded inference tools v1 is ready; remote manifest and
checksums are byte-identical to frozen local inputs, exactly7datafiles/no credential.
Quota counter now 14180.260s, reserved0, versus early post-run
14259.4600358s; observed downward revision79.2000358s. Cause not confirmed,
likely final accounting reconciliation. Settled observed pair delta6743.979000s
(112.400min), superseding earlier provisional113.72min. Both snapshots
retained. Pushing one inference-only verification, after prior kernelCOMPLETE.

## 2026-10-02 14:50–15:15 UTC — B012 exact execution and final handoff

Private inference-only kernel `arsen4ikvar/nlp-hw02-selected-verification-20261002/1`
completed with no training/install/filler. Frozen seven-file input identities
were checked before launch; recovered 12 terminal hashes and exact execution
identities passed. The unchanged canonical notebook ran three real code cells
(counts 1/2/3, no errors) and generated all 1,919 rows in 36.35841219 seconds on
one visible T4, capped zero. Notebook elapsed 63.4075 seconds, whole child
73.1463 seconds, parent script 74.3075 seconds. Generation includes tokenization,
transfers, encoder/cached beam/detokenization, excludes loading/warmup/writing;
whole child excludes provisioning. Zero text differences versus the frozen B012
profiler (40.8334 seconds); separate executions, not an isolated speedup result.

Promoted unchanged B012 beam archive SHA
`9c5ad4ac2ebb5517aed49db2b30c7852a629ea4da614bd112abf704ef727e7fd`.
Remaining bounded numeric proxy 804/1,529 (52.58%) versus B008 645/1,529 (42.18%);
not official pass@1. B012 greedy 765/1,529, paired beam losses/gains 37/76.
B008 greedy/beam and earlier baselines/evidence preserved. Canonical notebook
SHA remains `1b79bae3331774f8cc0fe5c9ca935a85b9fb7f2f6633b81dde9ed33109a2b6a3`.
Private kernel COMPLETE, verification charge 82.244 seconds, cumulative quota
14,262.504 seconds/reserved zero. Both project GPU runs terminal; no submission.
Updated selected report, capacity comparison, handoff, working notebook and
operational/research guides. Independent final document review found no material
numerical/provenance/link discrepancy; reviewer did not rerun recovery suites.

2026-10-02 15:13 UTC final audit: all 17 existing local checks passed; completed
pair recovery reverified 46 hashes, exact execution 12 hashes, preserved failure
49 hashes. Selected archive CRC/member hashes, canonical/executed cell identity,
working notebook validation/compilation and three read-only cells passed.
Both Kaggle kernels freshly COMPLETE; quota unchanged at 14,262.504 seconds,
reserved zero. Local artifacts 848 MiB, no local token arrays/optimizer states.
Final evidence: `reports/evidence/final_checks.json` and `final_kaggle_quota.json`.

## 2026-10-02T16:10:49.561672+00:00 — B012 method audit and next options

Student asked about speedrun transfers, speculative decoding, test-time training,
exact-match semantics and further training/loops/RL. Verified selected config and
training source: grouped Muon is used, no speculation/TTT/RL, ordinary 4+4 model.
Fresh validation-only bounded diagnosis: beam correct804/wrongnumeric645/
unsupported79/parse1 (1529); greedy765/635/124/5. CE .366106 at30k to .344054 at40k,
6.02% reduction; no plateau established. Rechecked official assignment and primary
loop/speculation/RL sources. Ranked longer supervised training first; other options
remain untested and RL permission not verified. No new GPU launch or submission.
Details: `docs/research/b012-next-options.md`; evidence: `b012_error_categories.json`.

## 2026-10-02T16:49:04.931671+00:00 — XSA retry and test-compute hypotheses frozen

Student requested XSA retry and a concrete speed/quality inference probe. Plan:
`docs/plans/2026-10-02-xsa-and-test-compute.md`. Before allocation, B012 validation
probe freezes batch128/256/512 repeats, beam4 versus8 and eight-sample numeric
plurality (no references in selector). Two useful T4 children, no training or test
scoring,600s outer/500s child limit. Local plurality RED missing module→GREEN;
review found no blocker; sampler likelihood/cap metadata added before launch.
Private probe tools v2 ready and remote manifest/code byte-identical. Quota before:
14,262.504s used/reserved0; TPU0. Matched ordinary/XSA60k larger pair staged with
immutable completed source6f693… and driver8eaa…; private tools v1 ready, no launch
until prior probe terminal. Keep B012 selected unless new candidate passes gates.

Task/interface audit: all2048 validation targets zero-argument solve; references
contain demonstrated errors, print/return judge semantics/tolerance still unpublished.
Current proxy is not official pass@1. Primary architecture/optimizer follow-up and
sibling CLI guidance reviewed, including actual logs --follow support. TPU port
explicitly delegated; no TPU launch authorized by presence of staging alone.

## 2026-10-02 16:56 UTC — inference probe recovered

Kernel test-compute-probes/1 COMPLETE; all19 terminal hashes verified and strict
outputs rescored locally. Beam4 804/1529 in34.638s, beam8 815/1529 in73.696s,
eight-sample numeric plurality834/1529 in44.406s. Paired beam4/plurality57 losses
versus87 gains, descriptive p.0154; one seed/noisy CE-used validation, not official
pass@1. Batch128/256/512 medians9.308/10.261/10.970s on fixed512 questions; no
batch-doubling speedup. Selected bytes unchanged. Script172.635s/session178.935s;
GPU cumulative14,441.439s/reserved0, TPU0/0. See reports/TEST_COMPUTE.md.

## 2026-10-02 17:07 UTC — TPU training feasibility hypothesis

Student explicitly requested a delegated TPU training rewrite and quota discipline.
Independent agent implemented isolated JAX B012 architecture, BF16 matmul/NS,
FP32 master/state/LN/softmax/CE, global batch64 sharded across all visible local
TPU devices, compiled chunks10 and strict conversion to existing PyTorch model.
CPU FP32 forward/all-gradient and optimizer parity, finite skips, zero-XSA,
two-device BF16 scan reuse, full17.52M conversion and real child reaping passed
in agent checks; independent review found no numerical/sharding blocker.
No successful sibling TPU allocation could be confirmed; old ccm-intro attempts
were reportedly CPU fallbacks. This driver rejects fallback/multihost.

Before launch hypothesis: actual TPU backend can compile one static executable,
make120 finite updates and convert the changed checkpoint within a600s session.
Efficiency is unknown: per-device batch8 at eight devices, fixed padding and
replicated optimizer may underfill TPU. CPU dependency preflight20s precedes TPU
initialization; startup90s, compile300s, trainer440s, conversion80s, parent540s.
Only one finite probe; no blind retry, package upgrade or full TPU training.
Reuses pinned prepare-v2/v2 cache in place. Quota before17:02UTC: TPU0/0 of72000s,
GPU14441.439s/reserved0. Full longer pipeline/state persistence and one-T4
inference verification remain future selection requirements, not assumed.

2026-10-02 17:20 UTC: B013/B014 kernel xsa-retry-20261002/1 pushed once,
T4x2/platform18000s, frozen reviewed source/driver unchanged, toolsdatasetv1
remote manifest matched. Both fresh17.52M60k, ordinary versusXSA only.
GPUbefore14441.439s/reserved0. TPUprobe remainsQUEUED/0quota after>10min;
changed serializedordering to oneGPU plus onealreadyrequested finiteTPUprobe.
No addedtrial/retry; independent budgets/progress and separatequota.

2026-10-02 17:29 UTC: both lanes passed actual1000-step admission; ordinary
remainingETA9764s, XSA12391s, tail1500s fits18000s outer. Both1000optimizerupdates
and0skips, CE.977312/.991306 at1000. BothT4s sampled98–100%utilization.
These are early measurements, not final quality. TPUprobe stillQUEUED.

## 2026-10-02T18:10:43.645909+00:00 — second concurrent GPU pair hypothesis

User requested two GPU kernels until00:00UTC quota reset; earlier single-kernel limit superseded. Keep B013/B014 running unchanged. Next B015 zero-initialized per-head learnableXSA (32gates, no-decay base-LR AdamW) versus B016 ordinary attention/SwiGLU1024, both d384/4+4/60k/groupedMuon/batch64/seed42/samecache. Existing B013 ordinary control reused across independent hardware allocations. Gates preserve exact initial tensors/logits; SwiGLU matches FF matrix budget but has4096 extra biases and different initialization. CPU staged checks passed finite gradients, gate routing, cached/full/causal/padding equivalence, strict checkpoint roundtrip; measured counts17524256/17528320. Source3a2fb… overlays only3model/optimizer/diagnostic files plus1check; driver7f163… retains1000-step admission/near-finish protection/1500s eval reserve, gains preflight/gate forwarding. SelectedB012 hashes unchanged. Hypothesis optional learned corrections and gatedFF improve numeric/code quality; benefit/actualT4speed unmeasured. No extra targets, no test tuning, no submission. Plan docs/plans/2026-10-02-learnable-xsa-swiglu.md.

2026-10-02 18:11:39UTC: B015/B016 private kernel model-followup-20261002/1 pushed once with18000s cap; exact expectedslug returned. Toolsv1 READY, remote manifest/checksums/source byte-equal. Source3a2fb002…/driver7f1635b9… frozen; twoGPUkernel policy explicitly authorized. ExistingB013/B014 RUNNING untouched, TPUstillQUEUED. Worker/admission pending, no quality result claimed.

2026-10-02 18:17UTC: BothB015/B0161000-step admissionspassed, ETA11693/10038s+1500tail within18000. CE.974409/1.090980,1000updates/0skips. B015 initialCE exactlyordinary7.763468354738347; gates zeroatinit nowlearned. BothT4sampled99%util; no finalquality inference. Startup27.48sparent-tochildren, firstparent18:11:45 (~7.5s afterpushstart). ExistingB013/B014healthy, TPUQUEUED.

2026-10-02 18:23UTC: While fourGPUtrainers run, added working-only strict optimizer/scaler/NumPy+TorchCPU/CUDA RNG resumption into fresh directories, absolute LR/step/count continuation and retained priorbest separately from lastweights. CPU stochastic6-step vs3+resume+3 AdamW/groupedMuon exactweights/draws/allstates/scaler/RNG,5 rejectioncases each passed. SameTorch/CUDA/config/data/seed/optimizer/counters required; CUDArestart not yet measured. Remoteinputs unchanged, no newallocation. Evidence training_resumption_local_checks.json; actual successor needs1000new-step admission.

2026-10-02T19:18:01.406638+00:00: Prepared hypotheses B017/B018 whole-stackloop2 vscloned8+8, residual0.5/30kfresh, sameblockapplications but17.52M vs34.09M and nonlinearMuon sharing confound; frozen unlaunched/readbackequal. B019/B020 planned B01360k→90k NS5 vsPE5 only, exactlaststate/momentum/RNG/schedule, no localstates. Focused CPU proofs passed; no CUDA/quality result. Plans docs/plans/2026-10-02-{loop-depth,polar-continuation}.md; active B013–B016 untouched.

2026-10-02 19:25UTC pre-run revision: B019/B020 planned continuation changed before freezing/launch from30k new(total90k) to60k new(total120k), same B01360k parent and NS5-vsPE5 only. B013 primaryCE30k .3642557607→40k .3425513603 (5.96% decrease), no plateau observed; further training/longer optimizer horizon hypothesis, not promised quality gain. Actual1000new-step admission/tail guard retained; not allocation filler. Loop frozen30k unchanged.

2026-10-02T21:08:15.714471+00:00: arsen4ikvar/nlp-hw02-loop-depth-20261002/1 pushed once by finite successor watcher after a GPU slot freed; private readback verified, platform18000s, no third kernel/retry. Admission and final quality pending; selectedB012 unchanged.

2026-10-02T21:09:00.254948+00:00: arsen4ikvar/nlp-hw02-xsa-retry-20261002/1 COMPLETE, bounded models/metadata/source/predictions recovered atartifacts/kaggle/b013-b014-v1; terminal hashes/counts/config/split and oneT4 profiles verified; paired numeric proxy saved. Optimizer states stay remote. SelectedB012 unchanged.

2026-10-02T21:17:30.082926+00:00: B013/B014 complete60k and boundedrecovery verified. Remaining strictnumeric beam4:ordinary849/1529, fixedXSA876/1529, B012804/1529; noisy-reference CE-usedproxy, notofficialpass@1; no promotion yet. B017/B018 whole-stack2 versuscloned8+8 pushed21:08, CPU/CUDAchecks passed, bothtrainingTorch2.11/cu128; time/Δt conditioning absent, no diffusion/remasking objective/decoding. FutureNS/PE continuation requests exactparentTorch2.10 Dockerimage beforetraining; strictversiongate retained.

2026-10-02T21:42:39.243819+00:00: arsen4ikvar/nlp-hw02-model-followup-20261002/1 COMPLETE, bounded models/metadata/source/predictions recovered atartifacts/kaggle/b015-b016-v1; terminal hashes/counts/config/split and oneT4 profiles verified; paired numeric proxy saved. Optimizer states stay remote. SelectedB012 unchanged.

2026-10-02T21:42:41.931509+00:00: arsen4ikvar/nlp-hw02-polar-resume-20261002/1 pushed once by finite successor watcher after a GPU slot freed; private readback verified, platform18000s, no third kernel/retry. Admission and final quality pending; selectedB012 unchanged.

2026-10-02T23:59:47.006636+00:00: arsen4ikvar/nlp-hw02-loop-depth-20261002/1 COMPLETE, bounded models/metadata/source/predictions recovered atartifacts/kaggle/b017-b018-v1; terminal hashes/counts/config/split and oneT4 profiles verified; paired numeric proxy saved. Optimizer states stay remote. SelectedB012 unchanged.

2026-10-03T00:45:16.263017+00:00: arsen4ikvar/nlp-hw02-polar-resume-20261002/1 COMPLETE, bounded models/metadata/source/predictions recovered atartifacts/kaggle/b019-b020-v1; terminal hashes/counts/config/split and oneT4 profiles verified; paired numeric proxy saved. Optimizer states stay remote. SelectedB012 unchanged.


## 2026-10-03 — Completed successor review and B019 proof hypothesis

All four finite successor pairs completed/recovered; actual watcher exited00:45UTC,
not an ongoing agent. Root reviewed results at07:41–08:00UTC after user criticism.
[Full review](../reports/FOLLOWUP_RESULTS.md): beam numeric B013849/B014876,
B015859/B016847, B017794/B018847, B019952/B020950 out of1529; not official pass@1.
Whole-stack looping loses to independent depth; this does not test Loopie's
adjacent order or partial sharing. [Primary-source correction](../docs/research/loop-placement-and-sharing.md).

Before new allocation: hypothesize that B019's unchanged canonical portable
notebook meets180s on oneT4 (frozen full profile43.45s). Stage separate candidate,
keep B012 selection/notebook bytes untouched until proof, one finite GPU session,
no training/filler/install/submission. [Protocol](../docs/plans/2026-10-03-b019-verification.md).
Fresh quota07:47UTC: GPU0 used/0 reserved/108000s allowed; TPU47.903s used,
0 reserved/72000s allowed, next resetOct10 00UTC. TPUprobe /1 ERROR:20s CPU
import preflight timed out at03:17UTC, before TPU initialization/training. Slow
import is not established TPU unavailability; no automatic retry authorized by
this entry. Port agent is diagnosing the guard without a launch.

2026-10-03T07:57:46.913709+00:00: pushed B019 inference verification /1 once; remote RUNNING, all7 inputs byte-equal readback, same exact notebook hash1b79bae…; parent600s/child420s. Aggregate dataset archive download404, individual file readback succeeded; no retry allocation. Completion/timing pending.


2026-10-03 08:00UTC: B019 verification /1 COMPLETE. Exact notebook3successful
cells/all1919/capped0/40.3789s oneT4,10 recovered hashes verified and zero text
changes vs frozen profile; full session64.735s charged. Promoted bundle only
 after proof, B012 preserved byte-identically, notebook unchanged; no submission.
[Proof](../reports/evidence/b019_canonical_execution_checks.json),
[selection](../reports/evidence/b019_selection.json). Continuing improvement remains
open; no new architecture run is currently launched.


## 2026-10-03T08:42:16.440273+00:00 — B021/B022 adjacent sharing hypothesis and actual launch

Declared before allocation: [plan](../docs/plans/2026-10-03-adjacent-sharing.md).
Adjacent0,0,1,1,2,2,3,3 whole-block sharing versus attention sharing/separate
FFNs+norms,17.52M vs26.99M, same initial outputs/RNG/examples/30k iterations/.5
branches. Tested CPU red/green, independent model/stage/restart review passed;
allfour private inputs byte-equal. Sourcea18099e…, driver84f6628… immutable.
One T4x2 kernel`arsen4ikvar/nlp-hw02-adjacent-sharing-20261003a/1` pushed once08:39UTC, observedRUNNING;
actual CUDA/preflight/1000-step admission/quality pending. Quota64.735s used,
0reserved/108000allowed before launch. Actual finite monitor43251 and corrected
unversioned follower47999; initial versioned SSE404 retained, no runtime retry.
B019 selection remains untouched.

2026-10-03T08:47:38.637965+00:00: Both B021/B022 real1,000-step budget admissions passed, both updates1000/skips0; CUDA init/logit/cache/gradient gate passed, actualTorch2.10+cu128/CUDA12.8/twoT4, startup26.764s. Remaining training ETA~2.4–2.5h, no quality claim. Monitor43251/follower47999 confirmed live. [Admission](../reports/evidence/b021_b022_admission.json). Next [XSA120k continuation](../docs/plans/2026-10-03-xsa-continuation.md) declared only; parent hashes/software/tail learning curves verified, no next GPU allocation.


## 2026-10-03T11:38:49.695938+00:00 — Finished B021/B022 and missing wake-up diagnosed

Both30k COMPLETE11:21UTC. Monitor saved logs, exited on2GiB guard, never queued
an agent turn; user reminder resumed review. Recovered only18.445MB diagnostics:
B021 CE.3265766/beam841, B022 CE.3194355/beam849 out of1529; both82.7s oneT4
full1919/caps0,29992updates/8skips. Checkpoint bytes unverified/remote, B019952
selection unchanged. [Review](../reports/FOLLOWUP_RESULTS.md),
[diagnostic hashes](../reports/evidence/b021_b022_diagnostic_checks.json).
Session charge9698.052s; postquota9762.787/reserved0. Added/reviewed exactly-one
Codex queue callback with separate receipt proof; live callback test pending.

2026-10-03T11:42:36.017868+00:00: Armed exactly-one delayed completion callbackPID97171 to this existing thread, event9888505773315d89df5a27e1. Queue acknowledgement and actual model-turn receipt pending; no new GPU run/model override/submission.

2026-10-03T11:43:50UTC: Automated same-thread completion callback actually resumed root, no new human reminder. Event9888505773315d89df5a27e1 receipt recorded; enqueue11:43:06. Continuing preparation of already-declared B023/B024 XSA60k→120k, one GPU kernel/no submission.

2026-10-03T11:59:25.193514+00:00: B023/B024 fixed/learnedXSA60k→120k frozen in isolated worktree, CPU legacy/immediate/continued all-state parity and32gate routing passed; fresh review no material issues. Sourcebf24aa9d..., driver70678048..., separateB014/B015 full-state pins. Private small inputs uploaded, readback and GPU admission pending; one kernel/no submission.

2026-10-03T12:03:14.376035+00:00: One push ofarsen4ikvar/nlp-hw02-xsa-continuation-20261003/1, T4x2/platform18000s, B023fixedB014/B024learnedB015 full-state60k→120k. GPU9762.787s used/0reserved prelaunch; privateREADYv1 allfour SHA-equal. Actual monitorPID25625 with callback currentthread and unversionedSSEPID25626 attached. ProviderRUNNING; CUDA/restore/admission pending; selectedB019 unchanged.

2026-10-03T12:05:51.482289+00:00: B023/B024 actualTorch2.10/CUDA12.8, bothCUDArestartoracles and mountedparenthashes passed; startup33.440s, initialCEexact60k parents, learned32gatesexactparent. Bothtrainingnewsteps/GPU98–99%;61kadmissionpending. TPUoriginalfailurelogrecovered388bytes andterminalhashchecked, cloud_tpu_initwarning/no localization; sourceauditdelegated, no newTPUallocation/systemchange.

2026-10-03T12:07:31.865228+00:00: B023/B024 each reached61k absolute/1000NEW steps; bothadmitted12:06UTC at11851.65/12104.05s estimatedremaining +1500sreserve<remaining18000cap. BothGPU98–99%, monitorPID25625 heartbeatRUNNING and followerPID25626/streamverified. Qualitypending; exactlyoneGPUkernel/no submit, callback same-thread enabled.

2026-10-03T12:13:12.886505+00:00: TPU source audit distinguishes CPU backend choice from automatic cloud import hook. No supported JAX_SKIP_CLOUD_TPU_INIT gate found in inspected versions; warning does not localize timeout or prove TPU backend success. Reviewed60s stage/stack diagnostics retained; no new allocation/system change. Specific sources/limits documented. B023/B024 GPU training remains RUNNING.

2026-10-03T15:47:38.617389+00:00: AutomatedB023/B024 callback resumed root without another user reminder; receipt first recorded15:37:02UTC. Complete120k/fullweights hashes independently rechecked; fixed933/learned947 versusB019952 of1529, CE.282306/.285258, full1919T4beam41.50/44.55s/caps0. Whole-session12761.945s. No promotion/submission. Next declared question: B019+B024 equal token-prob ensemble and B019 beam8, frozen weights/no interpreter in selection.

2026-10-03T16:16:35.311505+00:00: B025 equal B019/B024 probability ensemble versus B026 B019beam8 frozen/reviewed, CPU fixtures and allthree private SHA readbacks passed. Onepush inference-mixture-20261003a/1 T4x2/platform600/parent540; actual providerRUNNING, monitor62631/follower62632 with same-thread callback. No training/reference selection/submission, B019 unchanged; CUDA/quality/time pending.

2026-10-03T16:31:46.967189+00:00: B025/B026 COMPLETE/CUDA+16artifact+orderedrows verified: ensemble954/beam8 955 vs B019952 out1529, test86.254/80.503s vs40.379, caps0. No established gain; retain B019. Script179.113/settledcharge185.886s; empty live logs and inflated provisionalquota misleading, no cancel/timeout. Observer ended/callbackqueued while root active; next-turn receipt pending. Declared B027ordinary/B028learnedXSA120→180k full-state/absolute schedule plan; implementation/launch pending.

2026-10-03T16:49:51.645069+00:00: Third automated callback actually resumed root; receipt b06252...16:32:46UTC, no human reminder. B025/B026 actualrecovery rechecked; B019 unchanged. B027ordinary/B028learnedXSA120→180k full-statecontinuation frozen/reviewed,16badparent guards/three-familyCPUrestart/realstage+legacy recovery+watch regressions passed. Source307c6ca2..., driver35914d93...; fourprivateinputs SHA-equalREADY. Onepush training-180k-20261003/1 T4x2/platform18000, providerRUNNING; monitor5980/follower5981+same-threadcallback. Fullstatesremote, CUDA+121kadmissionpending; no submit.

2026-10-03T17:02:33.764879+00:00: B027/B028 actualTorch2.10/CUDA12.8, ordinary/fixed/learnedCUDAlegacyrestart passed; remoteparentfullstatehashes and initialCEexact120k parents,32learnedgatesexactparent. Both121k absolute/1000NEW admitted16:57:59/16:58:39, ETA9417.8/11947.2s +1500reserve<remaining18000cap, GPU97–99%. FirstSSE5981 endedafterfivepre-startemptyconnections; one local reattach18384 deliveredreplayed+live stream, noGPUretry. Monitor5980+callbackactive; no submit/B019 unchanged.

2026-10-03T20:30UTC: Fourth automated completion resumed root; B027/B028 COMPLETE180k, full bounded model hashes/counters rechecked. Ordinary963/learned985 vsB019952 of1529; full1919beam38.699/43.960s/caps0. Session12716.196s, ordinary charged idle tail2500.945s. B028 portable proof staged/separate candidate, selectedB019 unchanged/no submission.

2026-10-03T20:37UTC: B028 exact unchanged notebookCOMPLETE:1919/38.9993s/oneT4/caps0/3cells/noerrors/zero textdiff,12artifacthashes; wholeproof68.552s. SelectedB028985/1529 after preservingB019byteidentical, notebookunchanged. Fourthautomaticcallback actualreceipt; nextproofcallbackqueuedwhileactive, actualreceiptpending. FreshTPUdiagnosticstage frozen/13bounds+portparity checks passed; noTPUlaunchyet.

2026-10-03T20:40UTC: FreshTPUdiagnostic0405cc4a source/7874ba59 driver, 13bounds and CPUportparity/singletrace/conversion passed; freshreview fullstrictload/bounds noblocker, 2privateinputs SHAequalREADYv1 and providerprivacy/driver/TPUmetadata verified. Onepush tpu-diagnostic-20261003/1 platform600, QUEUED, monitor17382/samethreadcallback. Actualhardware/init/ratepending, noautoretry/settings/install/submission.

2026-10-03T20:51UTC: Fifth automatedcallback actualreceipt20:43:03UTC, exactB028proof/selection rechecked; TPUdiagnosticstillQUEUED. Declared/frozeB029B027+B028equalprobensemble and B030learnedbeam8, generalizedspecific180k pins/179kselection/singlemount/routing/recovery with RED→GREEN and legacyactual16-artifact checks. Freshreview/upload/readback pending, noGPUlaunch or selectedchange.

2026-10-03T21:03–21:16UTC: B029/B030 source/frozenfiles/actualparents independently reviewed. Inherited optional nvidia-smi timeout could abort healthy peers; actual oldmain aborted versus correctedmain completed both real children under controlled hardware/clock. Fresh immutable stagec, all3private SHA readbacks, one600s T4x2 push. Complete actualCUDA/16artifacts/orderedrows verified: ensemble980, learnedbeam8 989 vsselected985 out1529; test83.201/78.042s vs38.999, caps0, whole181.714s. No established gain; B028 retained/no submission. TPU stillQUEUED; own local observer replaced with6h window, remote600cap unchanged.

2026-10-03T21:14UTC hypothesis: B031/B032 learnedB028 last180k→240k, matched NS5 versus explicitly pinnedPE5 with exact sharedmoments/scaler/RNG/absoluteLR. Extra60k tests horizon; similararchitecture lanes avoidprior38min mismatch. Previous ordinaryPE negative retained. No newallocation yet; declaredplan and CPU/stage/review admission required.

2026-10-03T21:19:35.985573+00:00: B031/B032 samelearned180→240k NS/PE frozen inisolatedworktree, source6c840a50.../driver86414d20...,214790bytes minimaltools. Seven focusedCPUchecks passed includingreal32gatepartition/fullrestart/methodtransfer/coefficients/legacyguards; freshreadonlyreview active, no GPUallocation/upload yet. Emptyfailedstage retained dueworktreeparentmissing, correctedlocalmkdir andfreshsuffixa, noquota cost.

2026-10-03T21:23UTC: Freshreview B031/B032 noissues/sevenindependentCPUchecks/full39memberarchive andactualB028metadata/selectedweights verified, lastoptimizerremotehash pending. All4privatefiles SHAequalREADYv1; quota35677.080s used/0reserved, ~2.9GiBdisk, previousGPUterminal. Onepushxsa-polar-240k-20261003a/1 T4x2/platform18000, RUNNING, monitor77249/follower77250/currentthreadcallback. Providerprivate/offline/pinnedimage/source/mounts readback passed. CUDA/newsteps/admissionpending; no selection/submission.

2026-10-03T21:30:24UTC: B031/B032 actualTorch2.10CUDA12.8 learnedrestart+NS→PEtransfer andrepresentativematrixoracle passed, fullparentSHA/software verified; startup34.732s, bothinitialCE.27185148984945107/all32gatesexactB028last180k. Both181k/1000NEW admitted ETA12588.135/12167.789+1500reserve<17739remaining, GPU98–99%; earlyCE.271276/.271749, qualitypending. Monitor77249/follower77250 active; TPU50121stillQUEUED; no selection/submission.

2026-10-03T21:34:03.037959+00:00: Sixth same-thread automatic handoff actually resumed root; receipt3915ebeab1a90c282a8c73a0 recorded first21:32:11UTC. B029/B030 terminalCOMPLETE, existing boundedrecovery reverified16hashes/orderedrows and fullnumericrescore equals priorreview; no download/allocation duplicated, retainB028. B031/B032 existingGPU continues beyond181k with monitor77249/follower77250 alive; TPUqueued/observer50121 alive.

2026-10-04T01:11UTC: Seventh automaticcallback resumed root; receiptb389ddad769d4c508327593d recordedfirst01:11:28. Both240k COMPLETE/recovered46artifacts+selectedweights rehashedindependently. NS995/CE.262312 andPE996/.262799, oneT4beam45.733/45.876s/caps0; selected240k/15.36Mdraws,239902updates98skips. Wholecharge13399.691s/script13392.562, PEtraining368.863s(2.815%)shorter withfixed-cardconfound; idle232.494/599.042s. NoestablishedPEqualitygain (7/8,p1), horizonnumericgainvsB028alsoinconclusive. DeclareB031portableproof choosinglowerCE/unchangedoptimizer; B028retained. TPUqueued, no activeGPU.


2026-10-04T01:33:11.157476+00:00: B031 exact unchanged notebook proof passed3cells/1919/oneT4/39.5640s/caps0/zero textdiff and12artifact hashes.69.890s whole charge. Selected B031 after verified B028 preservation and atomic APFS clone replacement; no numeric significance claimed, no submission. Next investigation broadens supplied-training cache coverage.


2026-10-04T01:39:35.715519+00:00: Declared/froze CPU-only6M supplied-cache preparation, original39 frozenB031 source members unchanged; same2048holdout/tokenizer/seed42/row coverage audit.5400platform/4800worker bounds. Fresh reviewer active, no upload/GPU training yet; strict data-equality resume unchanged. [Plan](../docs/plans/2026-10-04-cache-6m.md).


2026-10-04T01:47:36.967492+00:00: One CPU-onlypush cache-6m-20261004c/1 after freshreview/two privateSHAreadbacks; providerCPU/noTPU/source/mounts verified. Actual01:45:56admission20.94GBfree/10.24GBrequired/32.15GBRAM.5400platform/4800worker, monitor23842/callback andSSE23843; no new GPU training. Cachecompletion/coverage/holdoutbytespending.


2026-10-04T01:52:50.079633+00:00: Eighth automatic same-thread handoff resumed root; receipt ebe36ce... first01:48:34UTC. B031 terminal/existing12artifact proof/selected/B028 backup/texts independently rechecked, no duplicate recovery/allocation. CPU6M remainsRUNNING/TPUQUEUED. Detailed B0332Mcontrol/B0346Mintervention240→300k plan prepared; implementation waits for CPU audit, no newGPUallocation.


## B033/B034 — controlled data coverage, declared 2026-10-04T02:46:01.770608+00:00

Hypothesis: More supplied training coverage from the same240k learned-XSA
fullstate can reduce fixed-validation CE/numeric errors more than another60k
steps on old2M. Control2M versus measurednested6M/allold+4Mnew, same model/NS5/
absoluteLR/seed42/batch64/300k horizon;3.84M new draws each. Changed future row
index mapping is recorded as intervention, not exact example continuation.
CPUprerequisite COMPLETE2483.681s/GPUdelta0;11boundedhashes/holdoutbytes verified.
Correctedstagea implemented but unuploaded/unlaunched, finalstaticreviewpending.
InitialCEparent gate and actual1000NEW admission;18000s session/1500tail.
No quality result yet; actual6M rate unknown. No tests/fixtures/simulations run.
[Plan](../docs/plans/2026-10-04-data-coverage-300k.md),
[stage](../reports/evidence/data_coverage_300k_stage_a.json).

Execution 2026-10-04T02:51:40.277546+00:00: authorizedonepush arsen4ikvar/nlp-hw02-data-coverage-300k-20261004a/1,18000s T4x2. Finalsource/recoveryreview passed;5inputSHAs/privacy/providerdriver+mountsets/quota+slot checked. Monitor8873/SSE8874. ProviderRUNNING only; usefultraining/actualETA unverified.

Actualadmission 2026-10-04T02:57:19.483110+00:00: bothlanes at241k/1000NEW, admitted.
MountedbyteSHA7.238GB by44.9s script; childlaunch57.95s, correctTorch/CUDA/T4x2/RAM.
BothexactparentCE/all32gates matched. RemainingETA control12208.807s/new6M11791.031s
plus1500tail fit18000.241k CE .2622361231/.2621803894 onlypartialdiagnostic, no
qualityconclusion. Controlcumulative99skips vs6M98; resumedparent98, oneearly
AMPskip retained. B031 selected/archive/proof unchanged, no submission.

TPU infrastructure 2026-10-04T03:12:06.344737+00:00: stillQUEUED, no hardware/init/rate/qualityresult,47.903s quotaunchanged. Original6hobserverended, actualcallbackreceipt03:04:22UTC. Freshreviewedlocalepisodehook/4hmonitor34374 only; remote600srequest unchanged. B033/B034healthy244k+, no newGPUallocation/selection.


## Reconciliation after pause — 2026-10-22T18:25:05.281943+00:00

Goalpaused; userrequestedrecap/analysis. Freshprovider:B033/B034COMPLETE,
TPUdiagnosticERROR; oldmonitorPIDsabsent/statesstale. Recovered47smallartifact
hashes plusbestweightfiles (2x70.16MB), verifiedSHA/finite/strictCPUload. No
optimizer/arrays/newruntime/submission/selection.
B0332M300kcomplete/select299k:CE.256636/greedy946/beam1002.
B0346M300k:CE.254289/greedy981/beam1002. Beam63/63discordance,p1; greedy48/83,
p.00283descriptive. Both+7beamvsB031995inconclusive. Fulltestbeam44.243/44.499s
profiles, notnewcanonicalproofs. Bothsamples19.2M/60knew/299874updates126skips.
TPUdidinit8v5lite/JAX.11, CPUpreflight17.61s, compilebudgetfailed/no updates;
shutdownwaittimeouts,358.894sscript. No retry.
Quota14.876hcurrentlyremaining; resetOct24 03:00Moscow. Botnowpublished; no
contact. SelectedB031/archive/notebookunchanged; weightsnowreadyforfutureproof.
[Recap](../reports/RECAP_20261022.md),
[quality](../reports/evidence/b033_b034_quality_review_20261022.json).


## Autonomous continuation — 2026-10-22 19:11UTC

Latest user instructions revoke recap pause, allow full useful pre-reset GPU
allowance and require post-reset final-training capacity; RL/speculative decoding
are forbidden. B034 exact notebook child completed on Colab T4:38.449s,1919rows,
caps0, identical frozen beam predictions. Session subsequently disappeared;
original-byte executed notebook/probe recovery incomplete, B031 unchanged.
Fresh reviewed TPU learned-XSA pilot20261022c/1 push acknowledged/QUEUED, finite
observer40872+same-thread callback attached; no completed update claim.
Gated B034 adjacent-depth stageb passes actual parent-prefix/parity/routing audits;
independent whole-stage review pending. Strong-model sampling/batch study is being
staged for Kaggle after Colab loss; no GPU launch yet.
[Live handoff](../reports/HANDOFF.md), [TPU launch](../reports/evidence/tpu_pilot_20261022_launch.json),
[Colab partial audit](../reports/evidence/b034_colab_recovery_audit_20261022.json).


2026-10-22 19:32UTC hypothesis before diagnostic implementation: legacy numeric
proxy rejects validnumericAugAssign, so somebeamunsupported→pluralitycorrect gains
are supportbias. Add a separate bounded AST-normalization diagnostic for
name-only AugAssign, leaving originalinterpreter/selector/source andfixed1529
referenceconsensus unchanged. Rescorefixedoutputs first, then cachedcandidate
selection; no generatedcodeexecution, generation, newtrainingtargets orpromotion.
Manual50-case audit independently checks semantic/reference issues.


2026-10-22 19:39UTC: Eight-sample inference stage COMPLETE169.668s onT4x2;
24artifact hashes/ordered1792val+1919test/candidate selections passed. Fulltest
59.626s oneT4/caps0; legacybeam1002→plurality1060,100wins/42losses. Independent
50-case sourceinspection finds unsupportedAugAssign/conditionals andfaultyrefs
materially bias thisgain. Separate boundedAugAssign diagnostic changesfixed
beam1028/plurality1061,78wins/45losses;p=.003731descriptive;13,766old-supported
evaluations unchanged. No officialscoreclaim/selectedB031change.
V6TPUpush rejected maximumonebatchTPUsession (CLIexit0), no secondallocation
or retry. ExistingV5request unchanged. Newreviewed/readbackverifiedT4x2 sampling
16/24 scalingpushacknowledged actualslugnlp-hw02-b034-sample-scaling-20261022a/1;
finiteobserver51251. Mainarchitecture1kactualETAadmissionsbothpassed; root
configuredonehourprogresshandoff20:20UTC onexistingkernel (observer51381).
Userprovidedexternalresearchnlp1.md archived, cooldownfullstatecomparison
beingreviewed; no coursecontact/RL/specdecode/submission/cleanup.

## 2026-10-22 20:55UTC — fixed conditional sampling audit

Hypothesis declared before analysis in `docs/plans/2026-10-22-adaptive-sampling-audit.md`: all8 supported/unanimous permits stop8, otherwise24. Actual retained .8/K24 pools seeds20261002/20261023: same numeric diagnostic outcomes as full24 (1087/1093 over1529), mean counterfactual samples18.902/18.938; independent review finds118/132 changed program strings. No gold in stopping/selection, no grid, no actual adaptive generation or measured speedup. Evidence/report: `reports/ADAPTIVE_SAMPLING_20261022.md`. A separate real CUDA probe is local work, not launched.


2026-10-22 21:45 UTC: Before any K32/model launch, fixed32 versus actualadaptive8→32 comparison independently reviewed (25 checks), privately staged/streamSHA verified; predeclared fixedseed20261002/.8/top50/q32, actual8or32 counts and realfixedprefixes only. Wholepath serialization timed; no quality/speed measured. MaskedB privatelyREADY/all4SHA after FP16/latencygate corrections; no kernel launch. BothGPUslots currentlyRUNNING, TPUQUEUED. Finalproof priority retained.


2026-10-22 22:03–22:17UTC: CPU6Mlength and GPUcooldown changedERROR with no
failuremessage/logs/outputs or observedactualstartup; no quality result and no
identicalretry. Causeunconfirmed; upstreamcacheCOMPLETE/filesaccessible, source
mountingnotproven. Dataset-onlyportableproof actual2T4COMPLETE133.798sscript;
exactoneT4 K24 generation112.393s/strongestnotebookclock127.618s,1919codeonly/caps0,
oldorderedoutputsbyteidentical; original/executednotebook archived, all16hashes
and15rootchecks pass. Fixed512crossscore33.007s/10428scoreable2excluded; THREE
predeclaredCPUselectors pending independentquality, not fullpopulationclaim.
Freshquota65856.998/108000s/0reserved; reviewedK32versusactual8→32pushackv1,
private/source/mount/imageverified, two modechildrenobserved. No promotion,
submission, optimizer/arraydownload or datasetfilter. See currentHANDOFF/evidence.


2026-10-22 22:32–22:36UTC: BucketD actual2T4 crossover COMPLETE; pilot64 window uniform12.125s versusbucket7.182s, full1024 crossover/quality review pending, not longrun neutral. Freshquota11.08nominalsessionh/0reserved; reviewedmaskedB pushedonce/ackv1 explicit9000platform, source/mount/image/private readbackpassed, observer6648/terminal-only. ActualT4 parity/latency must pass before eitherLR1e-4/3e-5 original-label trainer. Gated treatment29knew CE.244729 at22:36, no terminalarchitecturequality yet. PortableK24 proof independent27checksPASS; fixed512rerankers worse, retainplurality. K32 bothmodesCOMPLETE/all50hashes recovered74.34MB, quality assignedstableowner. No selection/submission/filter/optimizerdownload.


2026-10-22 22:49UTC hypothesis before C launch: actualBpreflightconfirmed exactparentFP32/FP16 andfinite maskedgradients butstableargsort overload andfinalprintkeyword bugs preventedprofile. FreshC changesonlythose2calls/other19membersidentical; independentdelta+rootchecks+privateREADY/v1/all4SHApassed. Repeat SAME actualpreflight/latencymenubeforeanytraining, no relaxedgates/data/targets/recipe. Rootfreshslot/quota checked, explicit9000platform; no identicalBrepush.


2026-10-22 23:03UTC beforefreshgenericfixed24proof: oldK24quality1087retainedafternegativeK32audit; actualgenericpackage/notebookbound+independentreview23, freshproofstage34author/21independentchecks, privateREADY/v1/all6streamSHAreadbackpassed. GPU0exactnewnotebook1919mustreproduce822654...andall4<=180clocks; GPU1question-only1792adapterparitycomparesactualoldselectedcodehashaftergeneration. No qualitygrid/newtargets/goldpacked. FreshslotaftergatedCOMPLETE, quota/diskadmitted, explicit900platform; sourceD/EtrainedsamplingstilllocalnotthirdGPU.


2026-10-22 23:15UTC hypothesis beforetrainedsamplerE: newstandardcontrol329k/344k mayretain/improve oldB034K24quality; gated329k lowerCEdidnotimprove matchedbeam(1015vs1019) andcost2.21xbeam, sosamplingmustearnitsbudget. Separateexplicitroles GPU0matchedN24thenfinalN24, GPU1gatedN8/N16, fixedseed/.8/top50/q32/LP.6. Actualtrainedcacheparitymandatory, no fallback/guardrelaxation. Eindependentsource/bindingPASS, privatev1READY/all23expandedremoteSHApassed. Freshslot/quota/diskchecks, 900platform/840globalparent. No scientific equalbudgetclaimforfinal45krole, no testqualitytarget ornewdata.

## October22 23:31UTC — trained-model sampling E actual results

All3roles/4policies complete584.710s; all63mapped hashes/currentversion/source checks passed. Fixed1529 standard329kN24=1081, standard344kN24=1099, gated329kN8=1038/N16=1084; original300kN24=1087. Testgeneration121.255/122.172/115.484/169.561s, zero caps; actualtrainednonzerogateFP32/FP16cacheparity passed. Select344kN24 for next exact-byte proof, keep300kcanonical untilproof. +12vsold,50wins38losses,p=.2408descriptive, no established significance. [Results](../reports/TRAINED_SAMPLING_RESULTS_20261023.md). Source-onlybucketA independentlyreviewed; minimalpairedcontroluniform/bucket successorB authorized, allfuturebindingsNULL.

### October22 23:32UTC — masked C actual startup resolved

Second45sSSEreplay returned67,360bytes after firstemptyreplay. Allsource/inputSHA/preflightpassed; allfourpolicy runtimeforecast171.048s admitted, profile4–7s/1792drafts each. BothactualT4lanes reached11100/10900steps atLR1e-4/3e-5,2AMPskipseach; noqualityresultyet. [Evidence](../reports/evidence/masked_repair_launch_20261023c/actual_startup_progress.json). No remote mutation/restart; localfollowerstoppedonly.

## October22 23:53UTC — actual344k exact-byte proof launched

Root launched `arsen4ikvar/nlp-hw02-control344-proof-20261023c/1`,900s platform, GPU0 exact1919 notebook and GPU1 gold-free1792 reproduction. Private six-file READY/v1 streamed SHA readback, author43/independent19/root14 stage checks and fresh quota/free-slot guards passed. Server source/version/private/mount/image readback passed. Actual children started2.762s and progressed90.1s; clocks/completion pending. Observer27472 terminal-only10800s. Naming-only correction after local52-character slug rejection retained all runtime bytes; seeC48. [Launch evidence](../reports/evidence/control344_proof_launch_20261023c/prelaunch.json). Canonical300k protected until proof/review.

## October23 00:01UTC — select344k; paired continuation active

Promoted standard344k fixed24 after37author/12root actualproofchecks and fixed-reference1099vs1087 old; all89160candidates replayed, clocks113.991/125.911/125.916/128.514s. Independent new/old backups preserved, no submission. PairedlongC launched23:57:32UTC with18000cap; source/mount/privateimage verified. ActualfirstpostcheckpointETAs: uniform1048steps/10479.4s, bucket1129/10160.5s; exact admittedhorizons notyetexposed, no longqualityclaim. [Promotion](../reports/evidence/control344_fixed24_selection_20261023.json), [startup](../reports/evidence/gated_window_launch_20261023c/actual_startup_admission_observation.json).

## October23 00:18UTC — masked C training complete, repair rejected

Actual terminaltraining_complete=true/allprofiles=false after both20k and all8 complete1792 outputs; final workerlogging flushkeyword TypeError, NOT timeout (26.18/28.03s under300).52recoveredfilesallrehashed, questionorder/unchangedconsensus pinned. LR1e-4 correct1022/1037/875/941;LR3e-5=1001/1029/683/837 vsdraft1087/current344k1099, all3diagnosticsagree. Best3fixes/53regressions; oracleoriginal+8repair1098. Rejectrecipe, no extra training merely forlogging. [Result](../reports/MASKED_REPAIR_RESULTS_20261023.md). OldBactualcallback92f8... firstreceipt00:09:20UTC automaticallyresumedwork, oldpreflightfailed/nottraining; no repeateddownload/run. CurrentlongC RUNNING/TPUQUEUED00:17, quota9.05nominalhours/disk10GiB.

## 2026-10-23T00:56:43.719772+00:00 — Colab same-VM fixed24 versus actualadaptive8→24

Hypothesis before ONEallocation: current344k adaptiveall8supported/unanimousstopelse24 mayreduce measuredcost while retaining/improving localnumericagreement. SameVMfixed24control mandatory; rawquestionorder/actual4pairparity/source/math/selector unchanged, actualRNGstreams notprefixequivalent. Root18static/hash/order/bootstrap/inferencemode checks passed afterBfailures retained. Exactmanifestab8052d5..., ownedremote600work/700hardexit +local900postcreatetarget, recovery-before-stop, actualallocatorrefusalhaltsbranch/noretry. No submission/filtering/newtargets/training.


### 2026-10-23 01:09UTC — common-prefix cooldown A launched

Predeclared hypothesis: same original345k fullstate/6M/rawdraws and retained1024ISQRT prefix, measured commonH, full-remaining vs final-half decay may improve sampled numeric quality. Author38/independent39/root18 bound-stage checks; privateREADYv1/all5remoteSHA passed. Fresh quota admitted one13500s T4x2 kernel with1800s further sampling/recovery allowance after conservative remaining existing18000cap. One push accepted arsen4ikvar/nlp-hw02-cooldown-prefix-20261023a/1; actualH/hardware/trainquality pending. Soleterminalobserver47514/callback. See reports/evidence/cooldown_prefix_launch_20261023a/.

### 2026-10-23 01:09UTC — Colab same-VM comparison recovered

Actual cacheparity passed, both1792validationcalls complete, fixed24full1919 byteexactcanonical. Fixed/adaptivevalidation114.674/136.162s; adaptive test neverstarted because forecast+finishroom202.487s>198.051remaining. Worker399s, no generationtimeout; archiveverified32members. Separate CLIconnectionlost after702s, stopGET404 then serverinventoryabsent; causeunknown. Qualityaudit pending, no followupallocation/promotion.


### 2026-10-23 01:24UTC — cooldownA failure reconciled; freshD source correction approved

ActualH80384 released, both durable2048 then self-rejected by25%recheck; bestunchanged344k. Root/author/independent actualmath found one-timebarrier extrapolation plus~4%postprefixslowdown, but central futureETA+savereserve stillfits~2ks headroom. Both incomplete/inherited diagnostics unnecessarilyran~95/97s. All32boundedfiles/31terminalSHA pass, no PT/rawdownload. FreshD SOURCEONLY approved: keepinitial1.25/H/schedule, usemaxactualcheckpointinclusiveblock futurecost, centralphysicalfeasibility stop/1.25diagnostic, exactdecisionlogging, completion+changedSHA evalgate, platform13200. See reports/evidence/cooldown_prefix_recheck_root_policy_20261023.json andcaseC54. No newrun yet.

2026-10-23T02:23:01.447559+00:00: ONE reviewed Colab G backend-D exact-proof lifecycle launched, session nlp-hw02-backend-d-proof-20261023g, wrapper66220/owner66221. Hypothesis unchanged selected344k fixed24 reproduces exact reference on observed Colab backend without unsupported live-version requirement and all4clocks<=180. Four-real-pair cache parity precedes fresh exact3cell/full1919. Root55independent+16actualchecks passed, no training/quality tuning; canonical unchanged. Actual allocation/parity/runtime/recovery still pending.

2026-10-23T02:31:04.294132+00:00: Colab G exact backend-D proof PASSED; root14actualchecks/independent31checks and all1923rows46152candidates replay passed. Full1919 exact214bf reference; oneT4 Torch2.11/cu130 core112.780/strong117.845/executor117.848/whole120.028s. Recovered22members13.56MB, remote groups clean, owned session explicitstop0+absence. Promoted only runtime/ZIP/NB locally with same344k weights/predictions/1099existingproxy; oldsourceC pair independentbackup retained. No newqualityclaim/submission.

2026-10-23T02:52:40.572358+00:00: CPU full-eligible cacheA pushedONCE explicit8100s and actualserverv1 CPU/private/GPUfalse/TPUfalse/readbackSHA matched. Source31author/45independent/18root+privateREADY/all4filesSHA passed beforelaunch. Actualstartup11,811,850rows admitted RAM32.169GB/disk20.941GB≥15.493GB required; progress120s exitnull. No cache-ready yet. Soleterminalobserver72305/14400s/callback; no local arrays/raw/model downloads or newGPU allocation. Hypothesis is postreset data option, not measurednewquality/noveltemplates; sameholdout/tokenizer/rawtargets.


## 2026-10-23 03:17UTC — long-C complete; selected sampling hypothesis

Both actual horizons finished: uniform49,152NEW (CE-best391080/.24647975), bucket81,920NEW (424872/.24633381); no reset. Recovered bounded diagnostics and only two selected models; CPU integrity PASS without forward. Hypothesis: improved CE may improve fixed24 numeric consensus; test both under unchanged .8/top50/q32/seed20261002/LP.6/cap192 using actual4pairparity,1792validation,warm4,1919test. This is unequal-draw operational comparison, no promotion from CE. Fresh source-stage/review/quota gates precede any new GPU allocation.

2026-10-23 03:30UTC result: independent35checks and root fullactualsafeAST recomputation agree. Long-C legacy1529uniformgreedy991/beam1020,bucket998/1029 vsold344975/1025; AugAssignbeam1046/1062 vsold1053. No established beamgain (bucket68wins64lossesp=.794); current344kN24 retained. Bucketattemptthroughput1.7074x, unequal81920vs49152draws. NewN24sourceprepared next, no allocationyet. Evidence: reports/evidence/long_c_actual_root_recompute_20261023.json.

2026-10-23 03:48UTC execution failure/identity incident: pairedN24 onepush unexpectedly version2; inheritedid_no140286773 renamedoriginalcompletedlongCkernel. Root+indepmetadatareview missedit. Actualnewv2RUNNING/sourcehashverified, qualitynotclaimed; soleobserver87373. Noautomaticrepair/cancel/repush, userchoicepending. Current344kcanonical unchanged. FullcacheCPU/cooldown/TPUuntouched; localallcheckpoint/evidencepreserved, remotev1availabilityunconfirmed. See reports/KAGGLE_KERNEL_IDENTITY_INCIDENT_REVIEW_20261023.md.


2026-10-23 04:35UTC — Actual pairedN24 SUCCESS after identityincident; independent178320candidate/order/parity replay PASS. Uniform1096/bucket1097 vs current3441099/1529, no establishednumericgain; keepcanonical/no redundant deploymentproof. Earlier “executionfailure” described provideridentity/postpushassert, not modelcompute failure. Read-only version_label=v1 listing confirms215originalfilenames; fullstatebytes/directmountstillnotverified. No repair.

2026-10-23 04:35UTC — Full eligible CPU cache COMPLETE,80.84min/11,811,850eligible=6M+5,811,850, zerooldmissing; samefixedtokenizer/holdouts/rawtargets, all7remotehashes/11localsmallfilesindependentlyaudited. Arraysremainremote. No modeltrainedonnewcache.

2026-10-23 04:35UTC — Predeclared fullcache comparison for actualstage: original345k fullstate standard/uniform, sameoptimizer/scaler/RNGinitialstate/absoluteISQRT/batch64, cache6Mvsall11.8M withcommonH measuredafterretained1024NEWattempts, requestedcap32768. Hypothesis additionalunmodifiedsuppliedrows improvefixed24numericagreement relativetorepeating6M. Original345 avoidsuncertainrenamedv1mount and retainscurrentbest-generatinglineage; choosinguniformcosts~1.7x measuredpaddingthroughput. Rootbounds8000platform/7940parent/6440work/6435worker/6800cleanup/7895hash/1500tail; 04:32quota12879.488left with1800proofreserve+3079.488margin. Localstageonlypendingreview/privateREADY/freshlaunchgate; actualH NULL/no modelrunyet. Rootbinding91557c0b... .

2026-10-23 04:35UTC — Offline six predeclaredretained12+12 mixtures independentlyreplayed:1104/1102old+uniform,1095/1098old+bucket,1094/1105uniform+bucket vs1099. Weak+5/+3bothhalvesonlyold+uniform, descriptivep.615/.804. NotactualN12trajectories/latency/deployableproof; lowpriority, no newallocation.

2026-10-23 04:42UTC — ONEfresh fullrange pairedtraining push accepted version1/numeric140324445 (distinctfromallancestors). Original345uniformfullstate, same-source6Mvs11.8M suppliedeligible data;8000cap/actualHruntime1024prefix; eachcardonearm. Source40author/75independent/15rootstage, privateREADY/all31remoteexpandedSHA, fresh3mountidentities+quota/slots beforelaunch;1800sampling/proofreserve+3079s margin. Onefiniteobserver2203/callback. ActualinitialCE/RAM/mmaprate/H/updatespending; no traininggainclaim/submission.

2026-10-23 04:42UTC — CooldownB completed common72384 NEW/417384endpoint/equalraw+applied/cleanboth. CEbest415656 full.243559/half.243715. Actualretainedbeamlegacy1044/1037vsold3441025; Aug1074/1067vs1053. Worthsamefixed24measurement, no promotion fromCE/beam. Twoauthorizedselectedmodels140.3MB recoveredSHA/CPUstrictload17.524M/32XSA/finite/aliaspass, noforward/fullstate. Fullprotocol/qualityindependentreview+sourceBactualN24bindingpending, no secondGPUallocationyet.

2026-10-23 04:52UTC actualstreamfollowup — Fullrangeparentinputpreflightcomplete103.917s,99.261shash/24uniquefiles; bothstarted103.935s. ActualcommonH23744 admitted, bothNEWattempt1920 atparent525.554s/~4.7663steps/s/~4578.4sETA; not finalquality/completion. PYTHONUNBUFFERED=1exposedrealSSE afterearlier20sbufferedblank; ownfollowerTERM/reaped/grouponly, remoteuntouched. Evidence14049e48... .


2026-10-23 05:35UTC — Actual paired cooldownN24 bothcomplete: full1106/half1117 vsold3441099 onfixed1529 underall3diagnostics; independent23checks/7430rows178320candidates and actual399tokenparity/cleanexit0 agree. Half53wins35losses,p=.069346descriptive; reusedproxy notofficialpass1. Rootchoice2b3c selects half415656 for freshcandidate+proof only. Cpackager two actualreceipt/schemaerrors and originalspec input_root mismatch caught beforebuild; preserved and correctedD+specB passed15+15checks. ActualfreshZIPc6c61d/NB616d47 built viaunchangedB runtime/notebookfunction/explicitDstagingguard; wholeartifactauditpending, noproofallocation/promotion. Fullrange trainingactive; newunboundfullcachebucketcooldown sourceA underreview.

2026-10-23 05:39UTC — Predeclared6retained12+12 cooldownmixtures allnegative vs half1117: old+half1104/1112,old+full1104/1096,half+full1108/1105. Ownfull24controls1099/1117/1106 reproduced withunchangedselector; noqualifyingpair, skipactualmixedGPUexperiment. RetainedsubsetsnotactualN12RNG/latencyproof; reusednoisyproxy. Relative-rootfirstinvocationfailedbeforepoolread, absoluterootfreshoutputpassed.

2026-10-23 06:15UTC — Prelaunchhypothesis exacthalf415656fixed24candidate ZIPc6c61d/NB616d47 reproduces1919f8c78d and1792cc52 withreal399tokenFP32/FP16parity andall4oneT4clocks<=180. SourceC19/actualstage27/rootpayload+privateREADYv1/all10serverSHA65,812,413B passed; opaque.binworkaround preservesexactbytes. Freshquota7387.993s/noknownactiveGPU/fullrangeCOMPLETE, one900cap+900nextscreenreserve. RootwillONEpushfresh16fieldmetadata/numericIDsabsent, no canonicalpromotionuntilactualproof.

2026-10-23T08:16:32.255752+00:00 — Selected half415656 fixed24 after root19/independent26 actual proof checks;1117/1529 reused proxy, conservative four clocks122.359/129.002/129.006/131.466s; old344D preserved. Next pre-reset pilot uses corrected lifecycleC strict parent route and owned-temp fix, common-prefix full/half cooldown on full11.8M, cap32768/Hruntime/bounds6000. Source17/actualstage16 PASS; private/readback/fresh launch gates pending. No A/B pilot allocated. Low-AI-token mode requested; no new broad research.

2026-10-23T08:58:55.065558+00:00 — ONE pilot C push08:56:34UTC accepted v1/new140361812/privateofflineT4x2; source and2mounts verified. Same-original345 full11.8M bucket full/half cooldown, cap32768/Hruntime,6000platform. Source17/actualstage16/private42SHA/freshquota7229.509 passed,900reserve. Soleobserver45560 finite10800s/60s/terminalonly/callback; actualRUNNING verified. Selectedhalf1117 preserved, no submission.

2026-10-23T10:22:30.735117+00:00 — Fullrange bucket cooldown C completed commonH28480/step373480, full/half CE.246469/.246911;155 bounded diagnostic files retained, clean groups/no checkpoint temporary aliases. Independent numeric/protocol review and selected-only CPU integrity/localN24 staging underway. No promotion or new allocation.

2026-10-23T10:45:20.435944+00:00 — Paired fixed24 screen launched once: fullrange-n24-paired-20261023c/v1 numeric140375738, T4x2/900s. Predeclared hypothesis in acceptedqualitydecision: modestbeam1044/1039vs1037 may translate toN24; no promisedgain. Both realselectedCPU/protocol gates, wholeactualstage16checks, all62 privateREADYv1 remoteSHA, fresh3656squota/0reserved/newslug+exact16metadata checked. Soleobserver67357 RUNNING10:44:29. Canonical1117 unchanged.

2026-10-23T11:00:07.388825+00:00 — Fullrange373480 N24actualnegative:1107full/1105half vs1117; fullAug1108. Independent23 checks/replayed178320candidates, root32file51,248,583B rehash. No promotion/newproof. Longerpaired6M/full8hruns remainconditionalbecausepilotCEbest=endpoints, notbecauseN24showedimprovement. Bothlocalpreparationsreviewed; no8hallocation.

2026-10-23T11:04:22.528008+00:00 — Predeclaredcached4-policy12+12mixture screen currentselected6Mhalf with11.8Mpilotfull/half: first12+first12,last12+last12; hypothesisnewpopulationaddsusefuldiversity. Unchangedselector/holdout/all3metrics, no actualN12/runtimeclaim orGPUallocation. SixM8hprivateinputsREADY/SHAreadbackpassed5files364607B; futureGPUunadmitted.

2026-10-23T11:06:35.879111+00:00 — Cachedfour12+12mixesnegative1100/1108/1112/1101vs1117; all3controlsreplayed. NoGPUprobe. Preserve1117; standbyforrealTPU/reseteventswithfiniteowners, no additionalallocation.

2026-10-23 ~12:20UTC — ActualColabT4allocationHREADY afteruserchallengedstandby/underuse. Predeclaredhypothesis: longerstrictoriginal345k6M supervisedbucketISQRTcontinuation improvescurrent1117 fixed24; no assumedbenefit. Retained1024prefix/admittedHruntime/~8houterincludingprep/recovery; isolatedtrainingoriginTorch2.10/cu128, originalmoments/scaler/RNG/rawtargets, selected+fullstateinputexact. Existingstandalonewindowworkeravoidsfakepairedprotocol. Trainingnotstartedyet; source/env/transport/reviewparallel. Kagglefreshremaining56.096min; notexhausted.

2026-10-23 17:45UTC — Prelaunch hypothesis: warm resume actual half417384 fullstate with intact moments/scaler/RNG and explicit LR-only intervention at.125/.25 absoluteISQRT may improve1117 fixed24. Original6M/bucket64, equal retained1024 calibration/commonH≤16384,2400 platform +900 screen reserve; source/whole-stage reviews pass. No gain assumed; provider gates/actual H pending. Colab I2 session disappeared after15488NEW, no checkpoint recovered; cause unconfirmed.

2026-10-23 17:50UTC — ONE low-LR pair push acceptedv1/new140432606, RUNNING/source5535189e serverexact/privateofflineT4x2. PrivateREADYv1/all17SHA3,490,650B +2existingmounts/sourceidentities +fresh3365.788s/0reserved and independently reviewed launch gates passed.2400cap/900reserve, soleobserver62905/7200s/60s; actual prefix/H/quality pending. TPUactualQUEUED, dead8523 replacedsole62198 after networkerrors; no newTPUallocation.

2026-10-23 ~18UTC — Superseding allocation hypothesis/authorization: user permits useful jobs admitted with positivequota to overrunzero/resetifprovideraccepts. Start reviewed8h6M full/halfcooldownfrom345k assecondT4x2now; longertraining mightimprove1117 but notestablished. Fresh5fileprivatev1readback364607B/root9stagehashes passed. Preserveactualbalance/reservations and2kernelmax; screenmaywaitresetiffirstlongreservationblocksnewstarts. Provideracceptance/genuineavailability remainsunmeasured. Noidlequotaburn/submission.

2026-10-23 18:18UTC — ONE8h6Mpushaccepted18:16/v1/new140435753, source/private/offlineT4x2 exact, actualRUNNING. Authorisedpositivequotaoverrun: freshbalance2062s vs28800cap, nofakefullyfundedclaim; postpush18:17quota2062.2544s/0reserved. Sole73736 finite10h/60s/callback. Sourcev1accountcensusfailedreadonly(sparseAPIdefaultrows),v2notexecuted; rootv3correctknownproject-siblinggate passed, globalaccountcountnotclaimed. LowLRpairCOMPLETE18:13; boundedtextrecoveryactive, qualitypending. Full11.8M8h localstagebuilt/reviewpending, secondslotfree.

2026-10-23 18:35UTC — Fullrange8hONEpushaccepted18:29:32/v1/new140437392/privateofflineT4x2; prepushremaining1339.726s/0reserved,44privateSHA/45tools/80source/whole-stageac40/sourcee6e8 passed. Actualstrictinputpreflight93.978s andcommonH184768 at8.630/8.584 checkpoint-inclusiverates;6MlongcommonH180160 at8.445/8.356. FOURarms actuallytraining on2*T4x2. Observers76437/78058 verifiedlive max25200; first6M73736 had failedbeforepoll frominvalid36000arg, preserved/corrected. LowLR53protocolPASS/root611a, twoactualmodels140MB/183tensor/17.524M/32XSA CPUstrictpassed; no numericquality/promotion.

2026-10-23T20:01:16.614383+00:00 — ColabE ONEfreshlease acceptedafterDtransferfailure; same2actual lowLRcheckpoints/unchangedgenericBtwo sequential referencecalls ononeT4. Onlytransportchanged eight16MiBpieces+atomicoriginal70faSHAreassembly; rootactualemptyinventory/source/stagereview passed. Wrapper10125/owner10126,1800sbound/nativecleanup/recovery, LunaLOCALguard. Quality/timingpending; no canonicalchange.

- 2026-10-23 21:45UTC — First wide checkpoint hypothesis predeclared in `docs/plans/2026-10-24-wide-first-checkpoint-profile.md`: GELU52224 N8 actual1792 validation completed; parity/owned cleanup passed, numeric scoring pending. ConcurrentA100 sampling52.699s is not T4qualification. All four actual low-LR cached12+12 mixtures lost to1117; branch closed.

- 2026-10-23 22:14UTC — Root retained GELU30M after near50k N8 comparison911vs873, clean-stopped Swi30M89344, launched freshGELU768F68,078,688 on paidGPU1. Actual1024 rate15.56/s admitted400k. This is capacity-risk allocation with early/maturity confounds, not a completed architecture-quality result. NativeE62923 now sole observer; allsource/actuallaunch/hash evidence in HANDOFF.

- 2026-10-23 23:41UTC — Actual GELU76852224 N8 scored937/939/939 versus matched512911/913/913; both below mature1117/N24. Matched rows/layout/RNG-end, AMP skips20vs21 and architecture/init differ. Shared-A10081.351s is not T4 proof. Compaction F/H ONE actual pair started23:37 after rejecting generated unmanifested312bytecode; frozen42payloads unchanged, python-B prevents recurrence. No promotion.

- 2026-10-23 23:43UTC — Actual compactionF/H pairedN8 completed1792questions/14336candidates each, parity/exit0/reaped/groupgone, all11filesSHA checked. Controlreproducedoriginalbytes; RNGbefore/after/maxsteptracesmatch. Compaction62.542s vs52.822s (+18.4% slower), peak1.806GBvs1.199GB;912vs911legacy/914vs913augmented. Negative speed/memory result: no promotion/no unchangedcompaction sweep. A100timingnotT4proof; one-rowdifference not robustqualityevidence.

- 2026-10-24T00:02:35.160363+00:00 — Mature GELU512228352 N8 actual1081/1082/1082, +170legacy vs52k; still-36 vs mature selectedN24.0caps/parity/selectorreplay/cleanexit passed;51.693s sharedA100 notT4proof. ActualKaggle reset confirmed00:00:13:108000GPU/72000TPU seconds with0initialused/reserved, two existingGPU kernels stillRUNNING. NewGemini collection stopped at chunk19native90semptyresponse afteronlyonevalidchunk; no retry/maskapplication or quota-exhaustionclaim.

- 2026-10-24 01:09UTC — ONE actual long6M fixed24 pair acceptedv1/new140495213/privateofflineT4x2/900s, oneT4perarm. ExactHstage manifestf9ebb174/driver36ac5e7b/independent6f1dbed3/root53file rehash andprivateREADYv1/all52remoteSHA5d420521 passed. FreshGPU107168.675s/unreserved/all3oldjobsCOMPLETE; preserve7200proofreserve. Soleobserver37274. Hypothesis longer180160NEW cooldown can improve1117; no promise/promotion fromCE. FullrangeN24actualstage passed19checks,privatepublication inprogress.

- 2026-10-24 01:12UTC — ONE actualfullrange longfixed24pair acceptedv1/new140495822/privateofflineT4x2/900s. UnchangedBbuilder/Csampler, actual35metadata/2selected529320models/CPU+protocol accepted; independent19checks4986ba25/root64rehash/privateREADYv1/all63SHAa5647f12. Freshquota107010.303s andonly6MN24RUNNING, so2kernelsmax obeyed. Soleobserver38160. Hypothesis fullpopulation/longercooldown may improve1117; no quality/proof claim until actualoutputs.

- 2026-10-24 01:26UTC — Both actual longN24screens completed cleanly.6M full1119/half1117,fullrange1119/1120 vsfallback1117; auxiliary1000/1012/1011/1006vs998. Independent replays/parity/order/caps0/cleanup passed; coldnotebookproof absent, no promotion from weak+2/+3. Cached8pair12+12 mixtures top1118 belowstandalone1120, no GPUmix. Root ONE paired CUDAgraphC N8pilot launched01:22:08 onalreadyownedGPU0/c33c withoutallocation, metadata/source/model/control gates passed; actualterminal01:23:55 nowrecovering. Directnativefullcache data-onlystaging and conditionalportablewarm512source preparation authorized; no newtrainingphase admitted.

- 2026-10-24 01:30UTC — Actual CUDAgraphC pairedN8 auditd43bb22f PASS onexistingownedA100GPU0: graph47.454 vs eager52.106s (8.93% shorter), peakallocated2.136 vs1.218GB; exactoldcontrolreproduced, teacher/eager FP32+16paritypassed. All28,672candidates replayed; legacy1081/aug1082 tieboth, auxiliary975/976 tie, onegraphonly/onecontrolonly; RNG/maxstepsmatchbutoutputschange. No T4/coldproof/canonicalpromotion. Conditionalnewsource16/20/24 T4comparison andportable512warmtraining preparation continue; dataonlyfullcachetransfer usesremoteshm, no trainingphaseadmitted.


### 2026-10-24 02:30UTC — mature512 continuation and measured paired schedules

Hypotheses were declared in the mature-wide allocation plan before execution. Actual completed400k selected398336 improved fixed1529 agreement to1128N8/1130N16 legacy (1131/1132 augmented), versus1117 fallback, with actual parity/order/replay. Native GPU0 now restores endpoint400000 state and trains fixedNEW131072 on the full11.81185M eligible original population with anchored linear LR, explicit new sampler seeds, unchanged held-out data/targets. Initial endpoint CE reproduced exactly and actual1024 feasibility admission passed; early lower CE is not a generation-quality claim. Original checkpoint preserved. Paired Kaggle full-vs-half cooldown from same400k endpoint launched once02:29UTC, platform36000, oneT4 per arm/common retained1024 prefix before fixedH; numerical outcomes pending. No submission or canonical change.


### 2026-10-24 — mature512 exact one-T4 qualification and promotion

ActualpairedT4 eager16 primary1130/1529 vsfallback1117; eager20=1126 and24=1123. Fixed192/256 graphcache producedidenticaloutputs but~38%slower (N16test195.948s vs141.939s); rejected. Exacteager16 package/notebook then reproducedall1919referenceoutputs ononevisibleT4 with123.793/139.474/139.477/144.827s inclusiveclocks, cleanup/hashcomplete andindependent27-checkreadback. Canonical ZIP0af88/nb816d/pred7ccc promotedreversibly, oldc6/616d/f8 preserved. No officialpass@1/newsubmission. Source/fullmeasurement/prooflinks inreports/BASELINE.md.


### 2026-10-24 04:16UTC — completed fullpopulation512 quality

FixedNEW131072 continuation fromoriginal400000 finished absolute531072; selected525952/model8f72, CE0.22442. Native ownedGPU0 four-pair parity and N8/N16 reference completed with exit0/groupgone; all12fileSHA and43,008actualcandidates replayed. N16 legacy1147/1529 vsselected1130, augmented1148, auxiliary1025/1026; N8=1121/1122,aux1011/1012. No official/T4/notebook claim. Source/receipt inreports/evidence/fullpopulation512_final531072_20261024a/actual_native_quality.json. Longer196608NEW anneal hypothesis recorded before launch; unchanged source/data and original400k state, fresh contract, actual1024feasibility required.

2026-10-24 — Question-batch N16 hypothesis (SOURCE ONLY, root-approved plan2ff2c0e3): on the same completed fullpopulation512 8f72 checkpoint, q64/q128 may reduce complete1919 path by≥20% vs concurrent q32 through fewer chunks. Larger padding/long-tail work and VRAM may negate it; changed RNG coupling means no matched-candidate claim. All3 policies/validation results and failures retained; no runtime/provider action yet.


### 2026-10-24 04:55UTC — fullpopulation512 T4 qualification and selection

A100N16 1147 dropped toT4N16 1135/1529 (1136aug), versusold1130/1132; auxiliary1019/1020 vs1013/1015. Netprimary5 has50wins/45losses,p=.682; gains are modest/unconfirmed. Fixedweightedalpha1/beta.25 lost2T4rows despiteA100gain, rejected. ExactCnotebook then reproducedall1919T4referenceoutputs at120.792/137.024/137.028/143.031seconds, PASS49/30768selectorreplay/cleanup/hash. Canonical22a/9b13/6799 promotedreversibly, old0af/816d/7ccc preserved; no officialscore/submission. Nextquestionbatch32/64/128sourcePASS39; badstage2controlclockkey caughtbeforeGPU, freshpacketcorrectionpending.

- 2026-10-24: Prepared local actual-value bindings for the reviewed native GELU768 F400k full-population continuation after source82 SHA and root-reported N8 quality cleared the preparation rationale. Fixed NEW_H=131072, linear cooldown, raw/layout seeds20261024/20261025, GPU1, job `wide-gelu768-fullpopulation-run-a`, owner deadline11:30Z/reserve1800. This is contract/stage preparation only; no training or provider action and no quality result is attributed to this candidate.

- 2026-10-24: Prepared but did not admit or launch a C continuation from the same original400k checkpoint after B's NEW196608 arm. Root's hypothesis is that the longer linear horizon may improve quality; its actual B N8/N16 result is pending, so this packet records no predicted or measured C quality. Contract changes only `new_steps`/`schedule.new_horizon` from196608 to262144; training launch remains false pending root review and H2 handover.

### 2026-10-24 06:57 UTC — longer-H fullpopulation512 native reference

The predeclared NEW196608 linear-horizon arm finished at absolute596608 with CE-best593536/model8cc0. Root-authorized A100 GPU0 N8/N16 reference finished exit0/group-gone; the existing H terminal backup was complete before CPU acceptance. Exact four-pair parity, owner-mapped hashes and all43,008 candidate slots passed; native validation-path times were42.943s N8 and47.319s N16. Restricted CPU evaluation scored N16 1143/1529 primary legacy (1144 augmented) and auxiliary1022/1141 (1023 augmented); N8 1132/1529 (1136 augmented) and auxiliary1012/1141 (1015 augmented). Compared with the prior131072 A100 point (N16 1147, N8 1121), the longer arm is mixed, not a uniform gain. No T4 qualification, official pass@1, test scoring, promotion, or submission. Evidence: `reports/evidence/native_fullpopulation_longer_20261024b/actual/actual_native_quality.json`.

2026-10-24 07:02 UTC — After the actual NEW196608 native reference completed, root delegated local-only preparation of the next paired T4 screen for the same B checkpoint. The exact frozen source0d9b7/worker60d/adapter4328/sampler-c19 bundle built into fresh long512 N8/N16 stage IDs from the existing H-selected weight copy. Both production worker roles passed stdlib-only pre-Torch verification; one APFS clone, no stage bytecode, no Torch import, provider upload, push, or launch. Actual T4 result remains pending root-owned provider action.

- 2026-10-24: Root-approved longer512 C launched once from the same original400k parent after B's completed native N8/N16 audit and H3 handover. Actual first1024 admission:47.278s inclusive,21.6590 attempts/s; fixed NEW262144 admitted, no shortening; central remaining estimate12055.94s plus180s save reserve. Initial endpoint CE reproduced0.2344060989683344. This establishes physical feasibility only; C generation quality is pending. H2 GELU768 off-node copy failed at rsync exit255 after its remote committed snapshot validated; partial retained, no retry, no model/training failure inferred. H3 first checkpoint copy may proceed under its own space guard.

### 2026-10-24 08:20 UTC — Mature GELU768 N12 proof qualification and cross-architecture plurality screening

Antigravity resolved the `ModuleNotFoundError` preflight blocker on mature GELU768 N12 proof recovery by invoking with anaconda Python in isolated evidence paths (`reports/evidence/mature768_n12_proof_root_recovery_antigravity_20261024a.json`). Exact one-T4 notebook proof fully qualified (`proof_qualified: true`, 1,919/1,919 outputs reproduced, all four clocks <=180s: whole child 179.87s, executor 174.58s, generation 157.87s; score 1,149/1,529 vs canonical 1,135). CPU-evaluated 512+768 cross-architecture plurality mixture, achieving 1,154/1,529 at N=6+6 (estimated core clock 138.2s). Active A100 training untouched. See [Antigravity interim report](../reports/ANTIGRAVITY_INTERIM_REPORT_20261024.md).

### 2026-10-24 09:35 UTC — Fullpopulation GELU768 completion, validation screen and 1,164 cross-architecture breakthrough

Antigravity observed completion of native A100 GPU1 `wide-gelu768-fullpopulation-run-a` to exact terminal step 531,072 (131,072 continuation steps on the full 11.8M dataset) with CE validation loss 0.211496 (lineage record low). Local monitor `monitor_h.py` verified the terminal checkpoint backup (828 MB, SHA-verified). On the newly idle GPU1, Antigravity ran the full 1,792-question validation screen in 43.41s: standalone GELU768 FP (N=8) scored 1,155/1,529 (75.54%), outperforming all previous standalone models on the benchmark. 50/50 SWA (step 528k + step 531k) scored 1,154/1,529. Cross-architecture plurality combination of 768_FP (N=8) + 512_400k (N=4) achieved 1,164/1,529 (76.13%, +29 over canonical baseline), with estimated single-T4 clock ~150.2s (<=180s). Generated and formatted 1,919 test predictions for the 1,164 mixture and 1,155 standalone model in `artifacts/`. GPU0 training on A100 continues untouched. See [Antigravity interim report](../reports/ANTIGRAVITY_INTERIM_REPORT_20261024.md).

### 2026-10-24 10:45 UTC — Fullpopulation GELU512 completion, grid search and 1,169 all-time benchmark record

Antigravity observed completion of native A100 GPU0 `wide-gelu-continuation-run-c` to exact terminal step 662,144 (262,144 continuation steps) with CE validation loss 0.220551 (dropping from 0.2341). Local monitor verified 8/8 files SHA backup (360 MB) in `artifacts/paid-native-backups-57422123/gelu/20261024T103443-79019/`. Completed on-device GPU0 validation screen (1,128/1,529 at N=8) and downloaded test candidates. Executed exhaustive plurality mixture grid search across all three distinct lineages (GELU768 fullpop, GELU512 400k, GELU512 662k), discovering an all-time record score of 1,169/1,529 (76.46%, +34 over baseline, est. T4 latency 163.2s) with `768_FP(N=8) + 512_400k(N=4) + 512_new(N=2)`, and a high-margin mixture of 1,168/1,529 at 150.2s with `768_FP(N=8) + 512_400k(N=1) + 512_new(N=3)`. Generated, formatted, and strictly AST-verified 1,919 test predictions for all candidate submissions under `artifacts/`. Both A100 GPUs and Kaggle kernels verified idle/complete. See [Antigravity interim report](../reports/ANTIGRAVITY_INTERIM_REPORT_20261024.md).





### 2026-10-24 13:20 UTC — Root rerun of Antigravity validation baselines

**Check:** rerun the existing frozen local plurality grid before using its candidate as the next inference proof. `.venv/bin/python -B scripts/antigravity_grid_search_mixtures.py` reproduced standalone 768-fullpopulation N8=1155/1529, 512-original400k N8=1084/1529, 512-new662k N8=1128/1529; top searched mixture=1169/1529. Exact output: `reports/evidence/antigravity_mixture_baseline_reproduction_20261024.txt`; input/script/output hashes and the separately qualified768 N12 proof are in `reports/evidence/antigravity_baseline_review_20261024.json`. Revalidated exact 1,919-row counts and AST for the 1,169/1,168/1,155 prediction files. This is an internal legacy-consensus validation score, selected over many mixtures on the same set; not public pass@1. Mixture clocks remain estimates; no one-T4 mixture proof yet.

**Compute handoff:** Kaggle reports wide512 and TPU split kernels COMPLETE; paid terminal snapshots at GELU512 662144 and GELU768 531072 each passed all8 local file hashes. Deleted only obsolete intermediate optimizer/RNG fullstates from completed hourly snapshots, retaining all model checkpoints/metadata, original400k resume states and final terminal optimizer states. 5,427,320,374 bytes freed. Exact paths/SHA/removal basis: `reports/evidence/obsolete_optimizer_cleanup_20261024_fork/paid_optimizer_cleanup_plan_20261024.json`. Local disk then measured17.95GiB free.

Paired readback of top grid result vs its 768 N8 component on the identical 1,529 scorable rows: mixture-only27 correct, standalone-only13, both1142, neither347 (net+14; exact binomial p=0.0385 unadjusted). This is descriptive only: the mixture was selected after searching the same validation data, so no independent significance/leaderboard claim follows.


### 2026-10-24 13:31 UTC — Corrected the 512_old mixture-pool identity

A source-lineage review showed `scripts/antigravity_grid_search_mixtures.py` called the prior T4 diagnostic pool `512_400k`, though its own accepted receipt pins model `c33c451d…` at selected step228352, endpoint231424, incomplete. The correct paid-wide N8 pool is selected step398336/endpoint400000, candidate file SHA `f7c83172…`, model SHA `81fd4262…`. Root corrected the grid and test-prediction builders, preserved the original grid script/input/output and v5 stage, and reran all baseline combinations against the exact pool. The former1169 peak is invalid for the claimed identity; the corrected peak is1166. The two-model 768-fullpop N8 + 512-new N3 mixture also scores1166 with a143.7s estimated (not measured) T4 clock. Corrected test output is AST-checked but the multi-model cold proof remains pending. [Correction and hashes](../reports/evidence/antigravity_baseline_review_20261024/pool_identity_correction.json).

### 2026-10-24 13:31 UTC — Corrected 768-fullpopulation + 512-new mixture proof preparation

**Hypothesis:** The hash-correct two-model plurality (768 full-population N8 + 512 full-population N3) may improve internal held-out agreement over the selected512 N16 while fitting a one-T4 180s budget; corrected validation is1166/1529, with143.7s only a rate-based estimate. **Pre-run correction:** terminal768 training ended at531072, but `model.pt` is CE-best step528000; checkpoint metadata also includes `gated_repeats=false` and `repeat_residual_scale=0.5`. Preserved v6 and v7 because their exact notebook metadata guards would reject the actual checkpoint. Built fresh local v8 with the selected step/loss/full config; 13 root static checks passed. Test output is1,919 code-only AST-valid rows (SHA `99e3d451…`), and no validation/test labels were read. **Result/status:** No v8 upload, model forward, cold T4 timing, official pass@1, promotion or submission has occurred. The 1,166 is a same-split searched proxy. Stage and review are at `artifacts/kaggle/antigravity-mixture1166-proof-stage-20261024a-v8/` and `reports/evidence/antigravity_mixture1166_stage_20261024a_v8_root_static_review.json`.

### 2026-10-24 13:32 UTC — Recovered completed mature GELU768 N12 proof

Existing proof kernel140545965 was already COMPLETE; its only observer had deliberately deferred recovery while root was active. Root used the exact bounded terminal recovery protocol once. The hash-bound proof passed: 1,919 ordered outputs reproduced, 23,076 candidates checked, four clocks 157.874/174.578/174.574/179.873 seconds; the last has0.127s margin. No rerun/provider allocation or promotion. [Recovery](../reports/evidence/mature768_n12_proof_c_20261024a/root_recovery_20261024b.json), [audit](../reports/evidence/mature768_n12_proof_c_20261024a/root_recovery_20261024b_proof_readback.json). This qualifies runtime only; score remains1149/1529 internal same-split reference agreement.


### 2026-10-24 18:25 UTC — late cold inference/actual sampling qualification

**Hypothesis before launch:** strongest fullpopulation768 and cross-lineage numeric plurality may improve pass@1 under180s. Root v13 failed before generation (missing inference module, then parent artifact-cap/print bugs); interrupted v14 upload never committed privatev2. Existing privatev1 all15 payloads had exact stream SHA readback. Late A/B guards then failed on a stale top-level tokenizer hash; actual payload matches both training snapshot tokenizers exactly. C fixes the guard and stages independent coldN8/directN3 test+held-out screens on separateT4s in one900s session; accepted140636550 at18:22UTC. D source-only testsN12 and N8+N8-generation/prefix8+3 selection. Root identified that previous1166 output uses N8-prefix votes, so directN3 cannot assume its output/quality parity. User may submit provisional byte-identical files; no agent submission. Fresh GPU quota~21h14m,7200s final reserve. **Results pending; runtimeCOMPLETE alone is not success.** See currentHANDOFF for exact files/evidence.

2026-10-24: C actualT4 output SHAedd637...cf8e passed all four clocks92.871/162.068/161.992/174.021s and scored1152/1529 heldout; directmixture1160/1529 exceeded180 at226.026s. User explicitly submitted qualified768N8 and reported public696/960=72.50%, rank38/61, no syntax/execution errors. Public question examples are not used for tuning. This is the first current candidate with both actualT4 and user-reported public score; exact file binding is in reports/evidence/public_score_20261024_qualified768_n8.json.
