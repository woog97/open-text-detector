# Prepared encoder versus causal-backbone training

**Prepared, not trained.** Three flows use the same frozen raw-text windows, losses, paper splits and effective batch size. The A100 preflight performs maximum-window forward/backward checks without optimizer steps or trained checkpoints; optimizer-state allocation is checked without updating model weights.

| Flow | Backbone | Attention/context | Adaptation |
|---|---|---|---|
| encoder | `answerdotai/ModernBERT-large` | Native bidirectional attention; one copy | Full fine-tuning |
| causal | `Qwen/Qwen3-0.6B` | Causal attention retained; Repeat2 during token training and inference | Full fine-tuning |
| qwen35 | Official `Qwen/Qwen3.5-4B`, language backbone only | Hybrid recurrent/full attention retained; Repeat2 | Full fine-tuning, microbatch 1 |

Exact model commits are in `models.lock.json`. ModernBERT-large is approximately 395M parameters , Qwen3 approximately 600M, and Qwen3.5 approximately 4B; report actual loaded parameter counts, throughput, peak memory, processed tokens and training time. This is a practical backbone comparison, **not** an architecture-only or equal-compute experiment. All see equal examples/optimizer updates; Repeat2 approximately doubles Qwen's stage-2 sequence length. The original ModernBERT-base checkpoint remains an unchanged baseline.

## Report method, adapted to our labels

[Pangram 4 §4.1–4.2](https://arxiv.org/html/2607.27183v1#S4) repeats a source window as `[x, x]`, retains the causal mask, and supervises only the second copy. The first copy gives each second-copy token access to the whole source window. This is full-window access through repetition, **not** a conversion to bidirectional attention. Padding is added after repetition; original offsets refer only to second-copy predictions. Inference repeats every overlapping window too.

All flows use two stages:

1. **Segment warm-up:** one epoch / 6,000 shared draws, single-copy inputs, a 15-bin character-weighted AI-fraction head.
2. **Token training:** three epochs / 12,000 shared draws each. Binary token provenance loss, plus sentence (0.2), segment (0.2), and mixed-window (0.1) auxiliary losses. Target and context token losses receive equal weight when both exist. The causal flows switch to Repeat2; the encoder stays single-copy.

Stage 2 starts from the best stage-1 checkpoint with a fresh optimizer. Unlike Pangram, this initial comparison full-fine-tunes all backbones instead of comparing different LoRA recipes. There is no adapter merge/reset. We have reliable binary provenance, not enough gold AI-assisted or humanizer labels: unknown spans remain masked, and we do not invent those heads/classes. We also defer CRF/decoding changes to a separate ablation; deployment currently uses token probabilities and mean-token sentence scores.

## Data improvements

- Preserve the old pair-level extraction/fidelity filter: **1,573 training pairs / 3,146 contextual examples**. Most discarded source pairs have extraction problems; increasing examples by restoring them would add shortcuts.
- Add **23,585 clean, non-targeted human paragraphs**. The combined training pool spans **1,170 papers**.
- Draw 75% paired examples and 25% novel humans. Within each stratum, sample papers, targets and variants uniformly. Target-only, target-anchored and random-context views reduce dependence on paragraph position. The draw list is frozen before any model trains.
- Text windows fit **both** tokenizers (at most 510 source tokens), so neither backbone receives a different truncation of the same example. Rebuild labels from character offsets; never reuse the dataset's old token IDs.
- Split the original validation papers into **195 checkpoint-selection papers** and **195 calibration papers**. Selection has 2,100 frozen windows; calibration has 2,019. No paper crosses these partitions or training.
- Exclude exact normalized held-out text/targets and cross-split/conflicting targets. All 189 new workflow papers—including its pilot and calibration partitions—remain outside training. Original author overlaps are not fully eliminated; old splits are paper-disjoint, not claimed author-disjoint. New workflow papers were independently author-screened during acquisition.

The training generators are still predominantly Luna. This change does not establish cross-generator robustness. The workflow sentence-replacement tasks and third-party benchmarks remain held-out tests, not additional training material. Exact-hash checks do not prove absence of all paraphrase overlap or base-model pretraining contamination.

## Selection, calibration, evaluation

- BF16 autocast for every forward pass, FP32 master parameters and 8-bit Adam states, no FP32 inference fallback. AdamW8bit 2e-5, effective batch 32, 6% warm-up, cosine schedule, clipping 1, gradient checkpointing; seed 42. Encoder microbatch 8, Qwen3 microbatch 4, Qwen3.5 microbatch 1.
- Select each stage's checkpoint using only the frozen selection-set loss. The same objective weights and schedule apply to all models. No learning-rate sweep has been run; equal learning rates need not be optimal for every backbone.
- Calibrate separate token, sentence and document thresholds **after checkpoint selection**. Choose the stricter cutoff needed for at most 1% empirical human FPR in each of two cohorts: original human targets and novel human paragraphs. This avoids the previous easy-context dilution. It is not a population FPR guarantee.
- Score the frozen `workflow` suite first, including per-condition/per-view results and paper-clustered intervals. Both models project to the existing ModernBERT reference-token grid. Then run the development-exposed `comparison` profile for transfer diagnostics. Do not use those test results to adjust the completed comparison.
- Compare precision, recall at the chosen FPR, realized human FPR, sentence performance, and cost/throughput. Existing MELD/old-encoder scores used a different calibration cohort: retain their operating-point label; don't silently present them as calibrated by this new procedure.

## Model storage

All model weights and further tokenizer/config downloads stay on the Hugging Face Space, in `/data/workspace/model-cache`; checkpoints stay under its `/data/workspace` volume. Model/tokenizer-loading entry points reject local execution. Local files are source code, data and small verification/revision records. No Qwen3.5 model artifacts are downloaded locally. Existing local models from earlier work are not moved or deleted.

The official [Qwen3.5-4B release](https://huggingface.co/Qwen/Qwen3.5-4B) includes a vision encoder. The loader retains the official language backbone and releases the unused vision module; no third-party text conversion or quantization is used. Its architecture is hybrid Gated DeltaNet/full attention, so Repeat2 gets its own numerical context-visibility check.

## Prepare and verify (no training)

```bash
python resolve_models.py       # Space only: pins revisions and caches metadata/tokenizers.
python add_qwen35.py           # Extends the existing shared windows to Qwen3.5; archives changes.
python audit_prepared.py       # Validates all windows against all three tokenizers.
python -m unittest test_flows.py
python preflight.py            # A100: actual backbones, BF16 gradients, ZERO model updates.
python optimizer_preflight.py  # Allocate actual 8-bit states, backpropagate; no model updates.
python freeze_ready.py         # Freeze verified code/config/data identities before training.
```

Preparation already completed. `prepared/manifest.json` pins source files and every draw-list hash. `prepared/audit.json` and `preflight-results.json` hold validation results when complete. Preflight also checks that changing a later source token cannot alter the causal first-copy prefix, but **can** alter the supervised second-copy prefix.

## Training commands — prepared, not executed

On the existing Space, files are under `/data/workspace/paper-backbone-comparison-v1`. Run from that directory, sequentially on the A100:

```bash
python train.py --flow encoder --output runs/encoder
python train.py --flow causal  --output runs/causal
python train.py --flow qwen35  --output runs/qwen35
```

Then calibrate and evaluate **each** run:

```bash
python calibrate.py --run runs/encoder \
  --suite /data/workspace/paper-v3-modernbert-20260930/eval-suite-v1 \
  --reference /data/workspace/paper-v3-modernbert-20260930/run-01/best_model
python evaluate.py --run runs/encoder --profile workflow --output results/encoder-workflow \
  --suite /data/workspace/paper-v3-modernbert-20260930/eval-suite-v1 \
  --reference /data/workspace/paper-v3-modernbert-20260930/run-01/best_model
# Repeat for runs/causal and runs/qwen35; use --profile comparison for the older diagnostic suite.
```

Training refuses to start without the verified `READY.json` or if code/config/data identities have changed. Run directories cannot be overwritten; calibration and evaluation verify checkpoint/reference/metric identities. Training does not automatically resume a partial epoch: retain partial logs, start a new directory after resolving a failure. The older frozen benchmark runner/results are unchanged. No generated outputs, new dataset publication, or full model training is triggered by preparation. The initial FP32-Adam memory estimate for 4B was about 84 GB, too close to the A100 limit. All three flows therefore use the same AdamW8bit optimizer (weights are not quantized). The larger 4B flow has the same data budget but substantially greater compute cost; memory and preparation checks are recorded before any training is launched.

## Design review

**First critique:** Naively toggling `is_causal` may not change an implementation's mask and changes the pretrained attention regime. Implement the report's exact repetition pattern instead, with a numerical future-context test.

**Second critique:** A larger model, different context crops, extra clean humans, and different adaptation recipes could all masquerade as an architecture gain. Use identical text windows, objectives, optimizer schedule and full fine-tuning; report remaining model-size/pretraining/compute differences. A matched-size and equal-compute follow-up is still needed for a causal architectural conclusion.

**Third critique:** Human context can hide poor FPR on actual target paragraphs, and fine-tuning on the new evaluation collection would invalidate its purpose. Keep that entire collection held out, split selection/calibration by paper, and enforce the FPR cutoff separately on original targets and novel human prose. Assistance quality judgments remain outside binary training.

Qwen3.5 currently uses Transformers’ correct PyTorch fallback for its Gated DeltaNet and causal-convolution operations. Optimized optional kernels are not installed; its training throughput should not be interpreted as the best achievable throughput of this architecture.
