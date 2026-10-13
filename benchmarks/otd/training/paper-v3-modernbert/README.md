# Paper provenance: first ModernBERT baseline

Fine-tunes `answerdotai/ModernBERT-base` as a binary human/AI token classifier on
`woog/ai-paper-provenance-v3`, pinned to dataset commit
`98ca42d9ff340d8a8fddba50e880a8528aafecf2`. The resolved base-model commit and
runtime versions are recorded in each run's `config.json`.

The subsequent [wider evaluation](wide-eval-v1/README.md) covers 65,348 examples and exposes poor transfer at the frozen thresholds. Treat the narrow reconstruction results below as one task-specific baseline; the wider suite also provides 95,334 remaining human paragraphs in separate train/validation/test pools.

## Run

The initial job runs in the private `open-text-detector/training` Hugging Face
Space on its existing A100 80 GB. Persistent files are under
`/data/workspace/paper-v3-modernbert-20260930/`; `run-01/` contains checkpoints,
configuration, manifests, training history, thresholds, predictions and results.
No credentials are included in the training source or outputs.

```sh
python -m unittest test_common.py
python -u train.py --output /data/workspace/paper-v3-modernbert-20260930/run-01 --epochs 4
```

Uses the Space's installed Torch 2.12.0, Transformers 5.17.0, Datasets 4.8.5,
NumPy, PyArrow and Hugging Face Hub. Check `status.json` and the parent
`train.log` for progress. A completed output directory cannot be overwritten.

## Completed run-01

Completed on the Space's A100 in 337 seconds (training plus evaluation), with
1,573 training pairs from 790 papers and 524 validation pairs from 260 papers.
Four epochs were run; epoch 2 had the lowest validation cross-entropy and was
selected before test inference. The checkpoint independently reloads and runs
inference. The A100 is idle after completion.

Quality-filtered test: 519 pairs from 256 unseen papers. Scores below are
restricted to the target paragraphs, with surrounding context supplied to the
model. Sentence text is deduplicated within this target-only evaluation view.

| Unit | AI precision | AI recall | Human false-positive rate |
|---|---:|---:|---:|
| Target tokens | 93.40% | 73.95% | 4.91% |
| Target sentences | 95.00% | 66.75% | 3.38% |

Across all context, token precision/recall/FPR were 93.40%/73.95%/0.98%; sentence
precision/recall/FPR were 95.93%/66.75%/1.03%. The 1% validation calibration
includes easy surrounding human text, so it does not establish a 1% rate within
original target paragraphs. Sentence deduplication is independent per view and
can select different occurrences of repeated text. The target-sentence check
was added after observing this denominator effect, using saved predictions and
unchanged thresholds. No model or threshold was changed after test results.

With only the target paragraph supplied at inference, token precision was
93.15%, recall 81.51%, and human FPR 5.63%. This helps check dependence on the
fixed position of the generated paragraph. Full results, paper-bootstrap
intervals, broad-test results and fully-faithful sensitivity are in
`run-01/results.json`. Model files and hashes are recorded in
`run-01/verification.json`. In the Space, load the checkpoint from
`/data/workspace/paper-v3-modernbert-20260930/run-01/best_model` and use its
adjacent `thresholds.json`; softmax scores alone are not a calibrated decision.

## Protocol

- Use `fresh_passages`, excluding development-exposed pilots. Keep whole pairs
  only when both variants pass eligibility and paired fidelity is fully faithful
  or mostly faithful with minor differences. Exclude whole pairs whose target
  text has conflicting labels or occurs in multiple splits.
- Feed text only. Rebuild token labels from Unicode character regions using the
  new tokenizer. Whitespace-only, boundary-crossing, padding and special tokens
  are masked. Existing token IDs are not reused.
- Sample retained papers equally in expectation, then targets and variants.
  One third of training examples are target-only; one third are target-anchored
  windows; one third are random context windows. Crops change each epoch.
- Full fine-tuning, four epochs, batch 16, accumulation 2, AdamW learning rate
  2e-5, weight decay .01, 6% warmup, cosine decay, gradient clipping 1, BF16.
  Seed 42. Masked token cross-entropy; no class weighting in this baseline.
- Select the checkpoint by minimum unweighted validation token cross-entropy.
  Inference uses 512-token windows including special tokens, stride 256, and
  averages overlapping token probabilities.
- Freeze separate token and sentence thresholds at at most 1% empirical human
  false positives on validation. Sentence score is mean token probability.
  No test-based threshold, smoothing or hyperparameter changes.
- Final test reports quality-filtered, fully faithful and all-fresh subsets;
  full-context token, target-token, sentence and replacement-boundary results;
  plus isolated-paragraph sensitivity. Paper bootstrap confidence intervals
  condition on this trained checkpoint and its fixed thresholds.
- Sentence evaluation deduplicates exact text and excludes cross-split or
  conflicting sentence text. Full-context token metrics retain repeated context,
  so target-token results are an essential companion. Test identity metadata is
  inspected for split overlap exclusions before training; test predictions are
  not inspected until checkpoint and threshold selection are complete.

This is a single-seed Luna/research-paper baseline, not a replication of Pangram
4 or evidence of transfer to other generators. Automated fidelity judgments are
not expert gold. The optimizer/current-weights snapshot is retained for recovery;
auto-resume is not implemented. Do not rerun completed tests to tune the model.

## Shared baseline comparison

[BF16 comparison with MELD v5 and v8](comparison-v1/README.md): 13,751 fixed test examples across 19 groups, common token/sentence boundaries, and shared validation-only calibration. Includes complete per-example predictions, source-group confidence intervals, untouched human paragraphs with/without context, and the original v3 target test. These are diagnostic results, not a new blind holdout.
