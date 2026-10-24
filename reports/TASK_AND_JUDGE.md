# What the model is asked to do, and what we know about the judge

## Confirmed course/bot clarification — 22 October 2026

The user supplied primary bot/course text snapshots, read completely and
[archived unchanged with hashes](../assignment/clarifications/README.md).
These supersede earlier uncertainty about serialization, stated tolerance and
candidate selection. No bot contact or course submission was performed.

| Item | Confirmed statement | Primary snapshot |
| --- | --- | --- |
| Output | `.jsonl`, one object `{"code":"<program>"}` per task,1919 programs in the order of `pass1_questions.jsonl` | [bot](../assignment/clarifications/2026-10-22-bot.txt) |
| Score | One program per task executes in a sandbox; its numeric result is compared with the reference; missing code, error, hanging, nonnumeric or wrong answer scores zero, with no partial credit | [bot](../assignment/clarifications/2026-10-22-bot.txt) |
| Tolerance | Stated `rtol=atol=1e-6`; comparator formula/operand convention not supplied | [bot](../assignment/clarifications/2026-10-22-bot.txt) |
| Length | At most 64 generated tokens is recommended, not a mandatory cap | [bot](../assignment/clarifications/2026-10-22-bot.txt) |
| Candidate selection | Generating several alternatives and choosing one is allowed; the submitted result is still one program per task | [task3](../assignment/clarifications/2026-10-22-task3.txt) |
| Preprocessing | Answer-preserving code removal, including the discussed comment/print removal, is allowed after generation and in training targets; dropping existing training rows is allowed | [task2](../assignment/clarifications/2026-10-22-task2.txt), [task3](../assignment/clarifications/2026-10-22-task3.txt) |
| External LLM filtering | Allowed for filtering existing training rows; target correction/new replacement code is forbidden | [task2](../assignment/clarifications/2026-10-22-task2.txt) |
| Architecture | Encoder and decoder mandatory; decoder-only disallowed | [task3](../assignment/clarifications/2026-10-22-task3.txt) |
| Speculation | Speculative decoding forbidden | [task2](../assignment/clarifications/2026-10-22-task2.txt) |
| Reference repair | Staff report most wrong test answers fixed and results recomputed; training data was left unchanged | [task2](../assignment/clarifications/2026-10-22-task2.txt) |

Remaining unknowns are the exact entry point and arguments, return-versus-stdout
extraction, sandbox libraries, per-program timeout, numerical comparator formula
and the 180 second measurement boundary. Do not infer these from the usual `solve()`
training targets. The named `pass1_questions.jsonl` was not supplied with the
three snapshots; equivalence of its ordered questions with the pinned test
Parquet remains a final input-order check. Current internal JSONL artifacts can
contain `row_index`; they are not automatically the code-only bot payload.

These statements authorize method classes, not arbitrary semantic rewrites.
Removing comments/prints must actually preserve the answer; prints can have
side effects or exceptions, and naive line deletion can break strings/code.
Teacher filtering cannot repair a wrong target. No such preprocessing or
filtering has been applied to the frozen runs. Do not reduce 192-token generation
to 64 solely because it is recommended: truncation/return completeness, validation
quality and full runtime must justify a changed inference policy.


Task/interface audited2026-10-02 from pinned supplied training/validation data
and local evaluation code; assignment and user-supplied clarifications integrated
2026-10-22. Historical data audits did not inspect published test answers. The new
course transcript includes a user-reported judge example, archived solely as
clarification evidence and excluded from training/tuning. B012 examples below
are historical.

## Task and interface

Input is a math word problem as text; output is Python source. A usual target is
`def solve():` with the quantities from that one question embedded as constants,
intermediate calculations and a scalar return. An AST-only scan of all 2,048
primary validation references found 2,048 zero-positional/zero-keyword argument
`solve` functions, no varargs or kwargs. This proves the observed convention on
that split, not an undocumented course API. The dataset card says targets were
renamed to `solve` and docstrings stripped; comments remain.

Example supplied prompt: Michael needs tickets for himself and five friends,
$40 each. A robust generated program is:

```python
def solve():
    tickets = 1 + 5
    return tickets * 40
```

The program delegates multiplication to Python when an evaluator executes it;
the neural model must still map the wording to the correct operation and count.
B012's beam output on this validation example counted only five tickets and
returned 200 rather than 240. This is a model interpretation error.

If the judge only observes the return for this fixed zero-argument problem,
`return 240` is behaviorally equivalent. Comments are not executed. The statement
currently does not establish a requirement for a particular derivation, variable
names or textual match. The bot now confirms numerical-result comparison, but does not define the
entry-point/argument contract or constant-return requirements. Producing the
actual calculation is consistent with supplied targets and avoids requiring
all numerical computation inside token generation.

## Why 'straightforward' is not a complete data description

A supplied question says skirt + blouse + jacket cost 950; blouse=1.7×skirt,
jacket=3.5×blouse. Its target arbitrarily sets skirt 100, blouse 170, jacket 595:
these sum to 865, contradicting 950. Algebra gives skirt=950/(1+1.7+5.95),
jacket=950*5.95/8.65≈653.4682. The reference program is syntactically simple
and internally computes 595, yet does not solve the stated question. A reference
consensus agreement metric can call 595 correct even when the mathematical
question disagrees. The internal-value calculation was checked arithmetically;
no dataset program was executed.

Another reference sets Jake's age to 77−11=66 for a sum 77/difference 11 problem.
The two ages must instead be 33 and 44; which is Jake is underspecified unless
an age ordering is supplied. Duplicate synthetic references can also conflict.
Some supported Python imports, such as `math.ceil`, are rejected by our deliberately
restricted interpreter; unsupported is not identical to semantically wrong.

Some targets print marker-wrapped values and also return a value. The local
proxy ignores print expressions, including exceptions they might raise. A normal
Python execution judge might reject those same programs. This is why direct
optimization of the present proxy would require further verifier work.

## Reasoning and tools during generation

B012 is ordinary autoregressive generation from encoder representations and
previous code tokens. It has no separate scratchpad, hidden-depth loop, tool
calls, execution feedback, external calculator or judge inside decoding. It can
generate explanatory code comments; those are output tokens, not a special
reasoning channel. There is latent computation in Transformer layers, but no
measured additional deliberation mechanism to claim.

The new eight-sample experiment uses a bounded arithmetic interpreter only to
cluster generated candidates by their numeric return. It does not feed execution
results back to the model, see the gold answer or verify interpretation of the
question. It is test-time selection, not RL or test-time weight adaptation.

## Official scoring and the separate local diagnostic

The [assignment](../assignment/hw2_code_generation_20261022.md) requires all test
predictions within 180 s on one T4 and the public threshold pass@1>.33. The newly
archived [bot instructions](../assignment/clarifications/2026-10-22-bot.txt)
confirm the code-only JSONL and numeric sandbox comparison above. No official
submission was made, so official pass@1 remains unmeasured.

The historical B012 52.58%=804/1529 is a strict supported-consensus numeric-return
diagnostic on CE-used validation. Its integers compare exactly and floats use
relative/absolute 1e-9; unsupported predictions fail and conflicting/partially
supported reference groups are excluded. These frozen choices differ from the
bot's stated 1e-6 tolerances and its broader execution semantics. They are retained
for controlled comparisons, not silently relabeled as official evaluation.
Text exact match and syntax validity remain secondary measurements.

Staff's test-reference repair does not correct the unchanged training-derived
validation references. Local reference mistakes, ignored prints and unsupported
valid Python remain relevant. The clarification snapshots do not provide a new
test label file, and their example must not be used for tuning.

## Evidence

- Pinned revision `b2724689ff079e9c22efd548fe134bfb29036dbb`, provenance in
  `assignment/SOURCE.md` and `assignment/dataset_manifest.json`.
- `artifacts/kaggle/b011-b012-v1/run-b012/validation.json` rows 0,2–6 and
  `decoding-b012/beam4-alpha06/validation_predictions.jsonl`.
- `reports/evidence/primary_reference_consensus.json`, B012 score/error reports.
- `src/nlp_hw02/arithmetic.py` and `scripts/score_primary_predictions.py`.


## October22 model/metric update

Selected B031 is995/1529; latestunselectedB033/B034both1002/1529. These retain
all verifier/split limitations above, notofficialpass1. Newillustrativesamples
contain real modeloperation/undefinedvariable errors and wrongsyntheticreferences;
consensus doesnot guarantee mathcorrectness.
[Freshsamples](evidence/b034_error_samples_20261022.json),
[recap](RECAP_20261022.md),
[latestassignment snapshot](../assignment/hw2_code_generation_20261022.md).

## Question-file source check (22 October2026)

The recorded complete recursive course GitHub `2026` tree search found only
`homeworks/hw2_code_generation.md` among the relevant paths and no question JSONL.
The API response was not truncated; tree SHA
`a40420cc00bd27c4a45651bd650f495db077bb81` and response SHA are retained in
[the search evidence](evidence/course_github_question_search_20261022.json).
The current archived assignment links the Hugging Face dataset. This negative
repository-tree search does not establish equivalence between the bot-named
`pass1_questions.jsonl` ordering and the pinned Parquet ordering.

Course permission for external LLM KEEP/DROP filtering is broader than the
student’s current experiment choice: no filtering by us or Luna; only a bulk
Gemini3.8 low-effort/minimal-boilerplate filtering path or no filtering. The
frozen training data and portable inference candidate remain unchanged.
