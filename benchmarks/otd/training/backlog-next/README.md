# Authorized next backlog jobs — October 1, 2026

User authorized highest-priority unblocked work, using GPUs where useful. Deployed to `/data/workspace/paper-diversity-v1/backlog-next`.

- **GPU0:** six retained-checkpoint calibration/selection audit → training-only human scoring/preparation → `hard-human-v1` preflight, training, calibration, workflow/comparison evaluation.
- **GPU3:** `objective-token-only-v1` preflight, training, calibration, workflow/comparison evaluation.
- **GPU1:** existing seed73 control, unchanged.
- **GPU2:** existing batch sweep evaluations, unchanged.

No new data generation or public-data training. Six audit checkpoints are control, boundary, RAID, MAGE, RAID curriculum, MAGE curriculum. Audit fits thresholds only on calibration and compares only selection data, at 0.5%, 1%, 2% calibration FPR. It does not change production cutoffs. Training retains only stage2-best, so this compares retained model candidates; it cannot retrospectively select unavailable epoch checkpoints. Original fixed row chunks32 and inference microbatch8 preserve batch composition. Results are recorded by token/sentence and paired/novel-human stratum.

Mining is a bounded pilot reweighting **existing approved human training windows**, not fresh unseen human documents. The paper IDs must be disjoint from selection/calibration. Rank by 90th-percentile nonwhitespace native-token score from the control. Replace at most half novel-human draws with higher-scoring windows of exactly the same native token length; all labels remain human. Added replacements are capped at two per text and at max(2, original novel-human paper draw count) per source paper per epoch. These caps apply to replacement draws; original occurrences remain. Freeze scores, changes and manifest before automatic training. Human share, draw counts and native token exposure remain unchanged. It inherits the existing corpus provenance audit; scoring is not independent human-authorship verification.

Ablation preserves stage1, data/order, seed42, token budget, schedule and original combined validation loss. Only stage2 training changes to token loss without sentence/segment/mixed losses. Synthetic-logit check verifies equality to the original token component and no auxiliary-head gradients. This is not token-only training from the first stage. Original validation criterion is deliberately retained; selection audit is separate.

All model forwards BF16; assets/checkpoints stay on Space. Existing successful or partial run directories must never be overwritten. `deploy.py` is one-shot and refuses duplicate launch. On recovery inspect live processes, individual stage logs and completed files before changing anything. `analysis_mining.py` resumes completed per-model audit files but stops if a hard-human directory already exists; inspect that run rather than overwrite/restart it.

Monitoring: existing 30-minute heartbeat includes both queues. Sources here are local; no model assets were downloaded locally.

## Executable follow-ups

`fill_queue.py` prepares isolated boundary-seed73, objective-token-sentence-v1, mage10-v1 and raid10-v1 arms and registers them with the persistent dispatcher. Do not rerun against existing arms. The initial MAGE preparation cache failure was preserved remotely as mage10-v1-prep-cache-failure; recovery used the existing control/run/tokenizer, without new model downloads. All four workers passed preflight and reached stage2 training.

`selection_followup.py <arm>` performs BF16 calibration/selection scoring only and writes auto-dispatch/audits/<arm>. Each registered `<arm>-selection` depends only on its own training/evaluation job. Standard worker evaluation retains frozen workflow/comparison profiles. No profiling updates warm-start these production runs; no test threshold optimization.
