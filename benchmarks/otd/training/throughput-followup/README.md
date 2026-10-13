# Qwen3-0.6B time-to-validation-quality followup

Prepared October 1, 2026. Remote root: `/data/workspace/paper-batch-sweep-v1/throughput-followup-v1`.

Three isolated candidates bracket the original effective-batch32 learning rate: 1.4e-4, 2e-4, and 2.8e-4 (head LR remains 0.1 adapter LR). Stage 1 uses microbatch16 and stage 2 microbatch8, with checkpointing disabled; effective batch remains32. Exact original data, example order, 42k draws, seed42, optimizer, objectives and model storage are retained. Every forward uses BF16. No checkpoint warm start. Existing results and checkpoints remain untouched.

Validation always uses the original microbatch4, independently of training microbatch, to avoid changing the selection-loss definition. Larger training microbatches can still alter averaging across variable-length examples; numerical equivalence is not claimed. Epoch-end validation trajectory, cumulative supervised/input tokens, elapsed training time and time to first reaching the historical control's best validation loss are recorded. Historical timing has system-load uncertainty; epoch-end checks provide coarse time-to-quality resolution.

## Gates completed

Longest32 actual training examples: larger-microbatch gradient cosine 0.99992768/0.99999950 and relative L2 0.01203/0.001032 (stage1/stage2). Peak reserved memory25.2/26.3GB of85.1GB.

Deterministically sampled32 mixed-length examples in original relative order: cosine0.99999641/0.99999920, relative L2 0.002713/0.001376. All four mixed-length cases and all four longest-case checks verified BF16 outputs, finite gradients, unchanged frozen parameters. These are bounded diagnostics, not proof across the full training distribution.

## Dispatch

- `throughput-mixed-gradient-v1`: completed diagnostic.
- `throughput-micro16-8-lr1p4e4`
- `throughput-micro16-8-lr2e4`
- `throughput-micro16-8-lr2p8e4`
- `throughput-quality-comparison-v1`: depends on all three training jobs; picks minimum stage2 selection loss before calibration and frozen workflow/comparison evaluation of the winner. Historical control results are preserved and reused.

Workers perform the existing GPU preflight before training. No blind retries: any partial failure requires inspection. Use `/tmp/pangram-training-access/check_throughput_follow.py` through the secure remote helper for current status; GPU assignments in old reports are historical.

Source: `worker.py`, `finalize.py`, `prepare_remote.py`. Preparation expects transport-provided worker/finalizer source strings and refuses existing trial directories/registry entries. Models and assets stay on the Space; these local files contain only source and small documentation.
