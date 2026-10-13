# Matched LoRA comparison

Repeats ModernBERT-large, Qwen3-0.6B, and official Qwen3.5-4B from the same pinned base revisions. Starts only after the existing full-tuning queue finishes all training, calibration, and evaluations successfully. It does not initialize from the fully tuned checkpoints.

The prepared manifest and rows are identical: 6,000 stage-1 draws and 3 × 12,000 stage-2 draws. Same order, seeds, windows, splits, losses, batch size 32, micro-batches, optimizer type, cosine schedule, 6% warmup, and checkpoint selection. No new data, including the newly copied Pangram datasets, enters this comparison.

Following https://thinkingmachines.ai/blog/lora/: rank 128, alpha 32 (alpha/r scaling), zero B / Kaiming-uniform A initialization, dropout 0, adapters on every backbone linear projection, adapter LR 2e-4 (10× baseline), classifier-head LR unchanged at 2e-5. This is one prespecified configuration, not a claim that the rank or learning rate is optimal. The blog's experiments use generative post-training and do not guarantee identical results for token classification. We retain the baseline cosine schedule for comparability rather than reproduce its constant-LR experiments.

All attention and MLP projections are included. Qwen3.5 gated linear-attention input/output projections are included. These models have no MoE experts. Embeddings, normalization parameters, biases, and Qwen3.5 depthwise convolutions remain frozen. Task heads train fully. Actual module names and counts are saved in coverage-audit.json and run.json; gradient flow is checked in a deferred GPU preflight.

Stage 2 retains the selected stage-1 adapter and starts a fresh optimizer, matching the baseline's stage transition without introducing a second adapter's capacity. This does not reproduce Pangram's merge-and-fresh-adapter recipe.

Model assets/checkpoints remain on the Space in /data/workspace. BF16 autocast is mandatory for model forwards. Full state checkpoints include the frozen base plus adapters/heads to make strict reload and comparison reproducible. Calibration and both frozen workflow/comparison suites are identical to baseline. Test results never select the LoRA settings. GPU preflight runs only after the baseline releases the GPU; failure stops the queue.
