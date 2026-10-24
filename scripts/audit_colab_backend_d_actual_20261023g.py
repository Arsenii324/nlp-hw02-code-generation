"""Read-only audit of retained Colab G artifacts; no provider/model/generated-code execution."""
from pathlib import Path
import ast
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
import sys
import time
import zipfile

R = Path(__file__).resolve().parents[1]
B = R / 'artifacts/colab/selected-backend-d-proof-20261023g'
V = B / 'recovered-result'
W = R / '.worktrees/strong-inference-fallback-20261022'
S = W / 'artifacts/colab/selected-backend-d-proof-tools-20261023g'
P = W / 'artifacts/colab/backend-d-proof-source-snapshot-20261023g'
STARTED = time.monotonic()
CHECKS = {}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def rows(path):
    result = []
    with Path(path).open() as stream:
        for line in stream:
            if not line.endswith('\n'):
                raise ValueError('Partial JSONL row: ' + str(path))
            result.append(json.loads(line))
    return result

def ordered(values):
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def check(name, condition):
    CHECKS[name] = bool(condition)
    require(condition, name)

owner = load(B / 'owner_result.json')
terminal = load(B / 'session-result.json')
receipt = load(B / 'result-receipt.json')
manifest = load(V / 'source/colab_backend_d_manifest.json')
archive_path = B / 'result.zip'
check('manifest_exact_reviewed_G', sha(V / 'source/colab_backend_d_manifest.json') == sha(S / 'colab_backend_d_manifest.json') == owner['stage_manifest_sha256'] == terminal['input_manifest_sha256'] == receipt['input_manifest_sha256'] == '9c6dd7ecb12268d537dbc64ab28ccda380bcd174178ed2536a473f68a257427f')
check('receipt_terminal_owner_complete_crossbound', owner['complete'] is owner['recovery_verified'] is terminal['complete'] is terminal['receipt_complete'] is receipt['complete'] is True and terminal['terminal_written'] is True)
check('archive_SHA_and_size_crossbound', sha(archive_path) == receipt['archive_sha256'] == terminal['archive_sha256'] == owner['archive_sha256'] == 'b3b375bc3514bfcb49842e049d6ec149446e90897447eb5836d7539e186db297' and archive_path.stat().st_size == receipt['archive_bytes'] == terminal['archive_bytes'])
check('receipt_SHA_crossbound', sha(B / 'result-receipt.json') == terminal['receipt_sha256'] == owner['receipt_sha256'])
check('standalone_terminal_SHA_owner', sha(B / 'session-result.json') == owner['remote_terminal_sha256'])
check('receipt_terminal_22membermap_identical', len(receipt['members']) == 22 and receipt['members'] == terminal['artifact_sha256'])
check('all22recovered_member_SHA', all(sha(V / name) == digest for name, digest in receipt['members'].items()))
with zipfile.ZipFile(archive_path) as archive:
    check('all22actual_archive_members_SHA_and_CRC', len(archive.namelist()) == len(set(archive.namelist())) == 22 and set(archive.namelist()) == set(receipt['members']) and archive.testzip() is None and all(hashlib.sha256(archive.read(name)).hexdigest() == digest for name, digest in receipt['members'].items()))
    check('archive_outputonly_safe_allowlist', all(not Path(name).is_absolute() and '..' not in Path(name).parts and Path(name).suffix.lower() in ('.json', '.jsonl', '.log', '.py', '.ipynb') for name in archive.namelist()) and not any(Path(name).name in ('model.pt', 'test.parquet', 'actual_first4_parity_pairs.json', 'selected_bundle.bin') for name in archive.namelist()))
check('archive_total_uncompressed_size', sum((V / name).stat().st_size for name in receipt['members']) == owner['recovery']['uncompressed_bytes'])
check('actual_archived_sources_exact_stage', all(sha(V / 'source' / name) == manifest['payload_sha256'][name] for name in manifest['source_allowlist']))
check('all8local_payloads_still_exact', all(sha(S / name) == digest for name, digest in manifest['payload_sha256'].items()))
snapshot = load(P / 'snapshot_manifest.json')
check('actual_local_owner_snapshot_sources_exact', all(sha(P / name) == digest == manifest['local_source_sha256'][name] for name, digest in snapshot['sha256'].items()) and snapshot['stage_manifest_sha256'] == owner['stage_manifest_sha256'])
with zipfile.ZipFile(S / 'selected_bundle.bin') as archive:
    candidate_bytes = archive.read('candidate_manifest.json')
    candidate = json.loads(candidate_bytes)
check('exact_model_source_and_candidate_manifest', hashlib.sha256(candidate_bytes).hexdigest() == manifest['candidate_manifest_sha256'] and candidate['member_sha256'] == manifest['candidate_member_sha256'] and candidate['member_sha256']['model.pt'] == manifest['model_sha256'] == 'eed1fdbb4f7e056536b322c08fad8660f5df24460f716a4d8e5382eb58ce8dc9' and candidate['member_sha256']['src/nlp_hw02/model.py'] == manifest['model_source_sha256'] == '07e58176c39eed8ed03134074282d74589be97120e6d6569a15f27eee1538b26')
check('candidate_model_package_notebook_provenance_consistent', terminal['candidate_model_sha256'] == manifest['model_sha256'] and terminal['candidate_bundle_sha256'] == manifest['candidate_bundle_sha256'] == '4c389ba89cb1351f9728d4aa0db279e2a909107b116ae957d9c0a42ae1b6505e' and terminal['candidate_notebook_sha256'] == manifest['candidate_notebook_sha256'] == '7489641d70580d41e86b3f1d60a3d80e62678fd281061b004d7afaeed9c6ffe4')
parity_worker = load(V / 'results/parity/parity_worker_result.json')
parity = load(V / 'results/parity/parity.json')
check('parity_exact_actual_fourpair_receipts', parity_worker['complete'] is parity_worker['passed'] is parity['complete'] is parity['passed'] is terminal['parity_passed'] is True and parity_worker['parity'] == parity and parity_worker['manifest_sha256'] == owner['stage_manifest_sha256'] and parity_worker['actual_parity_pairs_sha256'] == manifest['actual_parity_pairs_sha256'] == '1fbf9051c99ed84e816123b561f691cd942f8c742cf19b4a2850c58f4e7190ed' and parity['actual_rows'] == 4 and parity['prefix_width'] == 192 and parity['active_target_tokens'] == 399 and parity['target_used_for_generation_or_selection'] is False)
tolerances = {'fp32': {'atol': 1e-3, 'rtol': 1e-4, 'max_abs': 1e-3}, 'fp16': {'atol': .0625, 'rtol': .002, 'max_abs': .125}}
check('parity_tolerances_and_cache_finite_actual_results', parity['tolerances'] == tolerances and set(parity['results']) == set(tolerances) and all(row['passed'] is row['finite'] is row['elementwise_close'] is True and row['max_abs_logit_error'] <= tolerances[name]['max_abs'] and row['cache_depth'] == row['distinct_self_K_buffers'] == row['distinct_self_V_buffers'] == 4 and row['active_argmax_disagreements'] == 0 for name, row in parity['results'].items()))
check('parity_pinned_model_and_source', parity_worker['model_sha256'] == manifest['model_sha256'] and parity_worker['model_source_sha256'] == manifest['model_source_sha256'])
original_nb = load(S / 'inference.ipynb')
executed_nb_path = V / 'results/notebook/inference_output.ipynb'
executed_nb = load(executed_nb_path)
execution = load(V / 'results/notebook/inference_output.json')
a = [cell for cell in original_nb['cells'] if cell['cell_type'] == 'code']
b = [cell for cell in executed_nb['cells'] if cell['cell_type'] == 'code']
check('exact_three_executed_original_sourcecells', len(a) == len(b) == 3 and all(''.join(x['source']) == ''.join(y['source']) for x, y in zip(a, b)) and [cell['execution_count'] for cell in b] == [1, 2, 3] and not any(o.get('output_type') == 'error' for cell in b for o in cell['outputs']))
check('executor_input_output_SHA_complete', execution['complete'] is True and execution['executed_code_cells'] == 3 and execution['input_sha256'] == manifest['candidate_notebook_sha256'] and execution['output_sha256'] == sha(executed_nb_path) and terminal['notebook_child_returncode'] == 0)
qs = load(W / 'artifacts/colab-adaptive24-tools-20261023c/test_questions.json')
check('existing_questiononly_projection_pinned_order', len(qs) == 1919 and all(isinstance(q, str) for q in qs) and ordered(qs) == manifest['expected_ordered_questions_sha256'] == '6dd86866b2ed86a09781b7f471fc2fce5c00975503c4a85f1ec959dae0b93279')
# Execute only two reviewed selector function ASTs; generated programs go solely
# through the pinned restricted arithmetic AST interpreter, never Python exec/eval.
selector = W / 'src/nlp_hw02/sampling_candidate.py'
arithmetic = R / 'src/nlp_hw02/arithmetic.py'
check('safe_selector_and_arithmetic_source_pins', sha(selector) == '250d34bee2dc60847a65a7d5a784fac261a77db153a4d49f0687a82fe5a98fe5' and sha(arithmetic) == 'e01078307ec395b41afe03b588e014b7498ee968ed6ed2f3d13f8105c45663d6')
sys.path.insert(0, str(R / 'src'))
from nlp_hw02.arithmetic import evaluate_code
@lru_cache(maxsize=None)
def bounded_eval(code):
    return evaluate_code(code)
functions = [node for node in ast.parse(selector.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in ('equal', 'select_plurality')]
require(len(functions) == 2, 'Two reviewed selector functions required')
namespace = {'math': math, 'evaluate_code': bounded_eval}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(selector), 'exec'), namespace)
select = namespace['select_plurality']
base = V / 'results/selected_predictions'
phases = {}
for label, questions, prediction_name in [('warmup_n24', qs[:4], 'warmup_n24_predictions.jsonl'), ('test_n24', qs, 'predictions.jsonl')]:
    pool_path = base / (label + '_candidates.jsonl')
    pred_path = base / prediction_name
    pool, preds, timing = rows(pool_path), rows(pred_path), load(base / (label + '_timing.json'))
    require(len(pool) == len(preds) == len(questions), 'Actual row counts')
    capped = selected_capped = unsupported = 0
    codes = []
    for index, (row, pred, question) in enumerate(zip(pool, preds, questions)):
        require(row['row_index'] == index and row['question_sha256'] == hashlib.sha256(question.encode()).hexdigest() and row['samples'] == 24, 'Ordered actual question/sample identity')
        require(all(len(row[k]) == 24 for k in ('candidates', 'scores', 'capped', 'generated_lengths')), 'Actual vector widths')
        require(all(isinstance(c, str) for c in row['candidates']) and all(math.isfinite(s) for s in row['scores']) and all(type(c) is bool for c in row['capped']) and all(type(n) is int and 0 <= n <= 192 for n in row['generated_lengths']), 'Actual vectors finite/types/caps')
        selected, diagnostics = select(row['candidates'], row['scores'])
        require(selected == row['selected_index'] and all(row[k] == v for k, v in diagnostics.items()) and set(pred) == {'code'} and pred['code'] == row['candidates'][selected], 'Exact selector replay / raw code projection')
        codes.append(pred['code'])
        capped += sum(row['capped']); selected_capped += row['capped'][selected]; unsupported += diagnostics['supported_candidates'] == 0
    require(timing['candidate_sha256'] == sha(pool_path) and timing['prediction_sha256'] == sha(pred_path) and timing['ordered_questions_sha256'] == ordered(questions), 'Actual phase hashes')
    require(timing['rows'] == len(questions) and timing['total_candidates'] == 24 * len(questions) and timing['samples_per_question'] == 24 and timing['all_candidate_capped_rows'] == capped and timing['selected_candidate_capped_rows'] == selected_capped and timing['without_supported_candidate'] == unsupported, 'Actual phase counts')
    require(all(timing[k] == v for k, v in {'temperature': .8, 'top_k': 50, 'question_batch_size': 32, 'length_penalty': .6, 'seed': 20261002, 'max_new_tokens': 192}.items()), 'Frozen generation policy')
    phases[label] = {'rows': len(pool), 'candidate_count': 24 * len(pool), 'all_capped': capped, 'selected_capped': selected_capped, 'without_supported_candidate': unsupported, 'pool_sha256': sha(pool_path), 'prediction_sha256': sha(pred_path), 'ordered_codes_sha256': ordered(codes), 'ordered_questions_sha256': ordered(questions), 'complete_path_seconds': timing['complete_path_seconds'], 'peak_allocated_bytes': timing['peak_allocated_bytes']}
check('all1923actualrows_46152candidates_selector_order_projection_replayed', sum(v['rows'] for v in phases.values()) == 1923 and sum(v['candidate_count'] for v in phases.values()) == 46152)
check('full1919exact_original_reference', phases['test_n24']['prediction_sha256'] == manifest['expected_code_only_sha256'] == '214bf02531a70cccc75a722c3a710d2466899d664c6d2d30c70a1c86f6eb7f29' and phases['test_n24']['ordered_codes_sha256'] == manifest['expected_ordered_codes_sha256'] == '16ba397472d2ade3f0dc93cbfcb14af62f8e224ff21b1eb39ef55412201f0fa2')
timing = load(base / 'candidate_timing.json')
proof = terminal['proof']
clocks = {'core': timing['generation_selection_serialization']['complete_path_seconds'], 'strongest_notebook': timing['notebook_timing_seconds']['notebook_start_through_output_and_initial_sidecar'], 'executor': execution['elapsed_seconds'], 'whole_notebook_child': terminal['notebook_executor_seconds']}
check('all4independent_actual_proof_clocks180', all(isinstance(x, (int, float)) and math.isfinite(x) and 0 < x <= 180 for x in clocks.values()) and clocks == {'core': proof['core_seconds'], 'strongest_notebook': proof['strongest_notebook_seconds'], 'executor': proof['executor_seconds'], 'whole_notebook_child': proof['whole_notebook_child_seconds']})
check('actual_output_sidecar_no_reference_training_selection', timing['output_sha256'] == phases['test_n24']['prediction_sha256'] and timing['ordered_codes_sha256'] == phases['test_n24']['ordered_codes_sha256'] and timing['reproduction']['matches'] is timing['runtime_gate_passed'] is timing['notebook_total_within180'] is True and timing['training'] is timing['selection_uses_reference'] is timing['validation_generated'] is False)
check('actual_backend_fingerprint_matches_parity', timing['software'] == parity_worker['software'] == proof['software'] and timing['hardware'] == proof['hardware'] == {'gpu': 'Tesla T4', 'visible_gpu_count': 1} and terminal['hardware'] == [{'name': 'Tesla T4', 'driver_version': '580.82.07'}])
check('training_provenance_distinct_from_actual_backend', timing['training_torch_cuda_match'] is False and timing['deployment']['required_software'] == {'torch': '2.10.0+cu128', 'cuda': '12.8'} and timing['software'] == {'torch': '2.11.0+cu130', 'cuda': '13.0', 'python': '3.13.15'})
check('all_remote_owned_groups_reaped_and_exited', len(terminal['cleanup']) == 2 and all(row['verified'] is row['reaped'] is row['group_exited'] is True and not row['errors'] and not row['signals'] for row in terminal['cleanup']))
check('local_exec_group_terminated_after_recovery_and_verified', owner['exec_process']['verified'] is owner['exec_process']['group_exited'] is owner['exec_process']['leader_reaped'] is True and owner['exec_process']['terminated_by_owner'] is True and owner['exec_process']['returncode'] == -15)
stop_events = [row for row in owner['events'] if row['label'] in ('stop_exact_owned_session', 'sessions_after_stop')]
check('explicit_owned_stop_success_and_inventory_absence', owner['stop']['attempted'] is owner['stop']['verified_absent'] is True and owner['stop']['command_returncode'] == 0 and len(stop_events) == 2 and all(row['returncode'] == 0 and row['timed_out'] is False for row in stop_events) and owner['session'] not in stop_events[-1]['stdout'])
report = {'schema': 'colab_backend_d_actual_g_independent_review_v1', 'utc': datetime.now(timezone.utc).isoformat(), 'reviewer': '/root/gated_stage_review_20261022', 'status': 'PASS_ACTUAL_OBSERVED_BACKEND_1919_PROOF', 'checks': CHECKS, 'check_count': len(CHECKS), 'all_checks_pass': all(CHECKS.values()), 'owner_sha256': sha(B / 'owner_result.json'), 'terminal_sha256': sha(B / 'session-result.json'), 'archive_sha256': sha(archive_path), 'receipt_sha256': sha(B / 'result-receipt.json'), 'archive_members': len(receipt['members']), 'archive_uncompressed_bytes': owner['recovery']['uncompressed_bytes'], 'phase_results': phases, 'parity': parity, 'actual_backend': {**timing['software'], 'gpu': 'Tesla T4', 'driver': '580.82.07', 'visible_gpu_count': 1}, 'training_origin': timing['deployment']['required_software'], 'four_clocks_seconds': clocks, 'worst_clock_headroom_to180_seconds': 180 - max(clocks.values()), 'lifecycle': {'remote_script_seconds': terminal['session_seconds'], 'owner_total_seconds': owner['elapsed_seconds'], 'owner_post_create_seconds': owner['post_create_elapsed_seconds'], 'local_exec_cleanup': owner['exec_process'], 'remote_cleanup': terminal['cleanup'], 'stop': owner['stop']}, 'source_audit_path': str(Path(__file__).resolve().relative_to(R)), 'source_audit_sha256': sha(Path(__file__)), 'limits': ['Observed Colab Python3.13.15/Torch2.11+cu130/CUDA13/driver580.82.07 oneT4 only; no guarantee for unpublished grading image or arbitrary backend.', 'Actual new proof covers four-pair FP32/FP16 parity and1919test+warmup4 generation, not new1792validation generation/quality, official judge/pass1, tolerance or task-protocol correspondence.', 'Selected program hashes reproduce exactly; this is deterministic inference/backend evidence, not evidence of higher semantic score. Canonical unchanged by this review.', 'Local owned CLI exit-15 is deliberate post-recovery cleanup, not remote failure. Remote notebook child exit0 and both remote groups exited without signals.', 'No inner terminal is archived in this design; standalone terminal is independently rehashed against owner and SHA-binds the receipt/archive/member map. Receipt and terminal uncompressed byte totals have different scopes: terminal excludes copied source/log duplicates; owner full-member sum verified.', 'Retained complete_path description says all8-depth; observed actual model is standard4+4,17,524,256params,32XSA/cache4. Stale text does not alter architecture.', 'No provider requests/downloads, model/PT/raw Parquet/array reads, generated Python execution, synthetic tests/fixtures, or canonical/source-stage edits. Restricted AST interpreter replay only; archive contents preserved.'], 'audit_seconds': time.monotonic() - STARTED}
out = R / 'reports/evidence/colab_backend_d_actual_g_independent_review_20261023.json'
require(not out.exists(), 'Preserve existing frozen evidence')
out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'evidence': str(out.relative_to(R)), 'sha256': sha(out), 'checks': len(CHECKS), 'clocks': clocks, 'headroom': report['worst_clock_headroom_to180_seconds'], 'replay_rows': sum(v['rows'] for v in phases.values()), 'replay_candidates': sum(v['candidate_count'] for v in phases.values()), 'audit_seconds': report['audit_seconds']}, indent=2))
