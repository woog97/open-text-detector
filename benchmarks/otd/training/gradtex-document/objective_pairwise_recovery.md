# Ranking preparation during the Space storage incident

The root agent confirmed that authentication and no-storage execution work, but `/data` FUSE reads time out and Python processes are in disk sleep. Do not restart the Space, retry remote writes, or launch another budget worker while this persists.

The earlier budget deployment has an unknown outcome. Its intended files are `/data/workspace/paper-diversity-v1/objective-pair-ranking/budget-process.json`, `budget.log`, `budget-manifest.json`, and `prepared/`. Existing-pair pool preparation succeeded before the storage incident: 1572 pairs, 790 papers, one impure-provenance pair excluded. GPU pair jobs have not been confirmed registered.

Completed local hardening during the incident:
- Independent publication validator now checks actual compressed-file hashes and row counts, unchanged file inventory/stage1/nontraining data, the audited pair-pool hash, every replaced member against its audited source, every retained row against original control, complete/nonreused pair membership within each eight-row microbatch, source caps, fresh tokenizer lengths, per-row length tolerance, and actual token budgets.
- Both arm directories are prepared before any dispatcher job is published. Existing arms/jobs cause a deliberate stop, preventing accidental retries or overwrites after uncertain outcomes.
- Config gate requires the planned microbatch8/effectivebatch32, matching pair-boundary accounting.
- Local synthetic dataset-contract tests passed for a valid10% fixture and rejected cross-microbatch pairs, retained-row changes, source-provenance changes, role changes, and reused pair IDs. These tests use only a deterministic length-only fake tokenizer: no model inference, downloads, or Space accesses.
- Earlier real BF16 synthetic-logit ranking checks passed remotely before the incident. The local computer has no torch package; no package installation or substitute model inference was performed during this review.

Recovery after storage health is independently verified:
1. Inspect existing budget process identity/log/output with `/tmp/pangram-training-access/objective_pairwise_health.py`. Do not rerun the original budget deployment simply because its client timed out.
2. If complete, use `/tmp/pangram-training-access/objective_pairwise_register_deploy.py`. It now includes the independent validator and refuses existing arm directories. It performs no training until registration and worker BF16 preflight.
3. If partial/failed, preserve files and logs. Repair deliberately using a versioned preparation attempt; do not delete the original. Check dispatcher jobs/states before registering anything.
4. The pair-aware preflight verifies finite BF16 complete-pair gradients and frozen base weights, alongside existing longest-row memory and checkpoint roundtrip checks. Standard selection/evaluation does not include the ranking term.

This is authorized existing-paper objective work. HIP remains a separate, later document-level protocol and must never be passed through the verified-token pair integration.
