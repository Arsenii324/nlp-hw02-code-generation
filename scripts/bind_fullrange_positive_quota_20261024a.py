"""Root-delegated metadata binding only. No provider calls or model/array reads."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
WT = ROOT / '.worktrees/strong-inference-fallback-20261022'
OUT = ROOT / 'reports/evidence/fullrange_positive_quota_launch_20261023c'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')

rawpath = OUT / 'resource_snapshot_ACTUAL.json'
raw = rawpath.read_text()
start = raw.index('{')
actual = json.loads(raw[start:])
assert set(actual) == {'observed_utc', 'quotaRefreshTime', 'gpuQuota', 'tpuQuota'}
# Preserve exact SDK stdout, including its version warning; no quota value changes.
rawcopy = OUT / 'resource_snapshot_ACTUAL.raw.txt'
assert not rawcopy.exists()
rawpath.rename(rawcopy)
write(rawpath, actual)
write(OUT / 'quota_payload_normalization.json', {
    'schema': 'actual_quota_stdout_normalization_v1',
    'raw_stdout_sha256': sha(rawcopy), 'actual_json_sha256': sha(rawpath),
    'prefix_kind': 'installed_Kaggle_SDK_version_warning', 'prefix_bytes': len(raw[:start].encode()),
    'observed_payload_values_unchanged': True, 'additional_provider_requests': False,
    'signed_urls_or_credentials_present': False,
})
quota = actual['gpuQuota']
seconds = lambda value: float(value[:-1])
remaining = seconds(quota['totalTimeAllowed']) - seconds(quota['timeUsed']) - seconds(quota['timeReserved'])
assert remaining > 0

builder_freeze = WT / 'configs/fullrange_bucket_cooldown_stage_builder_source_c_20261023.json'
assert sha(builder_freeze) == '3b1768a0de2d0573bef0f818a245af150ae67a0fa634bea869ca899b9c28a561'
review_path = ROOT / 'reports/evidence/fullrange_positive_quota_builder_c_independent_20261023.json'
assert sha(review_path) == '1676479c79ffbdd4afb50c8b27703d5e1425e0f2053f4326501f3665c8fd92ff'
review = json.loads(review_path.read_text())
assert review['status'] == 'PASS_SOURCE_ONLY_UNBOUND' and review['pins']['freeze_sha256'] == sha(builder_freeze)
normalized_review = OUT / 'builder_independent_review_ROOT_NORMALIZED.json'
write(normalized_review, {
    'schema': 'root_normalized_independent_source_review_v1', 'root_owned': True,
    'root_authority': 'Root explicitly accepted independent1676479c and delegated actual local binding/build; no provider write.',
    'status': 'PASS_SOURCE_ONLY_UNBOUND', 'freeze_sha256': review['pins']['freeze_sha256'],
    'source_independent_review': {'path': str(review_path), 'sha256': sha(review_path)},
    'normalization': 'Expose the existing pins.freeze_sha256 at top level for unchanged builder receipt interface; original independent review is preserved.',
})

proposal = WT / 'configs/proposals/fullrange_bucket_cooldown_positive_quota_20261023c.prepared.json'
p = json.loads(proposal.read_text())
assert p['verified_actual_values'] is False
accept = json.loads((OUT / 'source_acceptance_PROPOSAL.json').read_text())
accept.update(root_owned=True, accepted_for_local_stage=True,
              builder_independent_review_sha256=sha(normalized_review), builder_independent_review_pass=True,
              utc=datetime.now(timezone.utc).isoformat(),
              delegated_authority='Root explicitly accepts builderC independent1676479c and freeze3b1768; one local fullrange8h build only.')
accept_path = OUT / 'source_acceptance_ROOT_AUTHORIZED.json'
write(accept_path, accept)

results = json.loads((OUT / 'current_results_decision_PROPOSAL.json').read_text())
results.update(root_owned=True, authorize_local_stage_build=True, utc=datetime.now(timezone.utc).isoformat(),
               hypothesis='Longer supervised full11,811,850-row bucket training from original345k may improve quality after the short negative pilot; fixed full-versus-half cooldown pair is exploratory, no score gain assumed.',
               decision_limits='One local8h stage build under latest-user positive-quota-overrun policy. Provider acceptance/slots/deadline/freshpositive launch quota remain root-owned external gates.',
               current_selection_fixed1529=1117,
               latest_fullrange_pilot_N24={'full':1107,'half':1105,'baseline_selected_half6M':1117,
                    'evidence_sha256':'fc12ef0ebdc3a874ca54013517f691df67a27dbac309223ff9bc4cf7674f821a'},
               latest_cached_mixtures={'all_four_below1117':True,
                    'evidence_sha256':'340cd1bae251ee20e998a241b73ef328a872a17c635764cf18e2239837ad2202'},
               temporary_alias_limit='Accepted lifecycleC/PID-owned samefile fix retained; actual pilot C terminal-map audit found zero temporary model-link aliases. No historical cleanup.',
               latest_user_policy='Useful provider-accepted running jobs may cross zero/reset; new launch still requires actualpositive quota.')
for path, expected in [
    (ROOT/'reports/evidence/fullrange_pair_n24_actual_quality_independent_review_20261023.json', results['latest_fullrange_pilot_N24']['evidence_sha256']),
    (ROOT/'reports/evidence/fullrange_selected_cached_mixtures_20261023.json',results['latest_cached_mixtures']['evidence_sha256']),
]:
    assert sha(path) == expected
results_path = OUT / 'current_results_decision_ROOT_AUTHORIZED.json'
write(results_path, results)

root = json.loads((OUT / 'root_binding_PROPOSAL.json').read_text())
root.update(verified_actual_values=True,root_owned=True,
    source_review_receipt_sha256=sha(accept_path),build_quota_snapshot_receipt_sha256=sha(rawpath),
    current_results_decision_sha256=sha(results_path),
    delegated_authority='Root authorized one actual local fullrange8h stage preparation; no provider write or actual launch.',
    actual_build_policy='Latest-user explicitly authorized positive-quota overrun; this does not assert enough quota for8h or future reset allowance.')
root['build_resource_admission'].update(
    admission_mode='user_authorized_positive_quota_overrun', snapshot_receipt_sha256=sha(rawpath),
    actual_unreserved_seconds=remaining, other_active_residual_bound_seconds=31200,
    concrete_proof_reserve_seconds=900,
    other_active_residual_bound_basis={'kind':'conservative_declared_caps_not_live_census',
        'lowLR_cap_seconds':2400,'prospective6M_cap_seconds':28800,
        'not_observed_activity_or_measured_remaining_duration':True,
        'fresh_launch_census_required':True})
root_path = OUT / 'root_binding_ROOT_AUTHORIZED.json'
write(root_path, root)

resource = root['build_resource_admission']
funded = remaining >= resource['other_active_residual_bound_seconds'] + root['bounds']['platform'] + 900
assert funded is False
policy = json.loads((OUT / 'positive_quota_overrun_authorization_PROPOSAL.json').read_text())
policy.update(root_owned=True,user_authorized=True,
    authority_note='Latest user instruction relayed and explicitly authorized by root: useful Kaggle jobs may overrun positive remaining quota across zero/reset if provider accepts; no new nonpositive-quota launch.',
    allow_running_job_to_cross_zero_or_reset=True,allow_new_job_with_nonpositive_quota=False,
    quota_numbers_are_observed_not_forecast=True,root_decision_sha256=sha(root_path),
    actual_resource_snapshot_sha256=sha(rawpath),observed_utc=actual['observed_utc'],
    actual_unreserved_seconds=remaining,current_quota_funds_declared_cap_and_reserves=funded,
    proof_quota_reserve_funded_by_current_balance=funded,
    provider_acceptance_established=False,fresh_positive_quota_slot_deadline_gate_required=True)
policy_path = OUT / 'positive_quota_overrun_authorization_ROOT_AUTHORIZED.json'
write(policy_path, policy)

p.update(verified_actual_values=True,actual_H=None,
    dataset_id='arsen4ikvar/nlp-hw02-fullrange-8h-tools-20261024a',
    kernel_id='arsen4ikvar/nlp-hw02-fullrange-cooldown-20261024a',
    source_acceptance={'path':str(accept_path),'sha256':sha(accept_path)},
    stage_builder_independent_review={'path':str(normalized_review),'sha256':sha(normalized_review)},
    root_resource_snapshot={'path':str(rawpath),'sha256':sha(rawpath)},
    root_decision={'path':str(root_path),'sha256':sha(root_path)},
    current_results_decision={'path':str(results_path),'sha256':sha(results_path)},
    positive_quota_overrun_authorization={'path':str(policy_path),'sha256':sha(policy_path)},
    preparation_status='ROOT_AUTHORIZED_ONE_LOCAL_BUILD_NO_PROVIDER_AUTHORITY')
pop_fields=['provisional_bounds','provisional_NEW_cap','future_build_output_root']
for field in pop_fields:p.pop(field,None)
packet_path = WT / 'configs/proposals/fullrange_positive_quota_actual_input_packet_20261024a.bound.json'
write(packet_path,p)
write(OUT / 'actual_binding_receipt.json',{
    'schema':'fullrange_positive_quota_actual_local_binding_v1','complete':True,
    'actual_snapshot':{'path':str(rawpath),'sha256':sha(rawpath)},'observed_utc':actual['observed_utc'],
    'actual_unreserved_seconds':remaining,'provider_acceptance_established':False,
    'current_quota_funds_declared_cap_and_reserves':funded,
    'packet':{'path':str(packet_path),'sha256':sha(packet_path)},
    'root_decision_sha256':sha(root_path),'authorization_sha256':sha(policy_path),
    'builder_review_normalization_sha256':sha(normalized_review),
    'binding_script_sha256':sha(Path(__file__)),
    'local_stage_built':False,'provider_write':False,'model_or_array_read':False,
})
print(json.dumps({'packet':str(packet_path),'packet_sha256':sha(packet_path),'actual_unreserved_seconds':remaining,
    'provider_write':False,'stage_output':str(ROOT/'artifacts/kaggle/fullrange-cooldown-20261024a')},indent=2))
