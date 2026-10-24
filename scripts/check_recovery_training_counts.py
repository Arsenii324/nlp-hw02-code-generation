"""Recovery must reject incorrectly sampled data even when step totals match."""
import json
import check_recovered_capacity as recovery

assert hasattr(recovery,'verify_training_counts'),'Recovery has no sampled-example count guard'
lane={'iterations':180000};good={'completed_steps':180000,'optimizer_updates':179950,'skipped_updates':50,'sampled_examples':11520000}
recovery.verify_training_counts(good,lane,64)
rejected=0
for bad in [{**good,'sampled_examples':7680000},{**good,'sampled_examples':11520001},{**good,'completed_steps':179999},{**good,'optimizer_updates':179949}]:
    try:recovery.verify_training_counts(bad,lane,64)
    except AssertionError:rejected+=1
    else:raise AssertionError('Accepted wrong training counters')
assert rejected==4
recovery.verify_training_counts({'completed_steps':120000,'optimizer_updates':119950,'skipped_updates':50,'sampled_examples':7680000},{'iterations':120000},64)
print(json.dumps({'event':'recovery_training_counts_passed','wrong_counts_rejected':rejected,'legacy120k_valid':True}))
