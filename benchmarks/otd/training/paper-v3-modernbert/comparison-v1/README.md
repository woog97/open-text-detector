# Three-model comparison

[Benchmark table with highlighted winners and Pangram 4 reported references](BENCHMARK_TABLE.md).

**13,751 test examples across 19 groups**, with 1,048 separate validation examples. Same inputs, common reference-token/sentence boundaries, and validation-only 1% human-FPR operating points. No model retraining. All final inference uses BF16; the interrupted FP32 baseline attempt is excluded.

| Measure | Our ModernBERT | MELD v5 | MELD v8 |
|---|---:|---:|---:|
| v3 target token precision / recall | 93.40% / 73.96% | 79.95% / 2.14% | 92.82% / 3.79% |
| v3 target sentence precision / recall | 93.05% / 73.10% | 86.36% / 0.85% | 98.28% / 2.55% |
| v3 target human token FPR | 4.91% | 0.50% | 0.28% |
| Untouched clean human: token FPR, isolated | 4.86% | 6.70% | 4.29% |
| Untouched clean human: token FPR, with neighbors | 3.04% | 3.75% | 2.42% |
| 50-generator Arena: token recall | 0.49% | 61.72% | 64.14% |
| 50-generator Arena: sentence recall | 0.33% | 88.51% | 77.80% |
| OpAI editing trajectories: token precision / recall | 76.57% / 4.29% | 88.61% / 19.63% | 88.38% / 22.77% |

## Selection and interpretation

The earlier full evaluation took 300 seconds plus 100 seconds for the paired human-context supplement. This fixed subset preserves every available categorical label, generator, domain, attack, operation, year, conference and length bucket in each source benchmark. It includes all 390 historical test papers, all 50 Arena generators (20 shared prompts), complete small benchmarks, paired human-context checks and all 1,038 quality-filtered v3 target examples. Selection uses fixed hashes and metadata; no detector score determines inclusion. Twelve examples were added to fill rare metadata gaps before scoring.

This is a coverage benchmark for repeated comparisons, not a prevalence-weighted estimate of all available text. Sampling fractions differ, and precision depends on each benchmark’s human/AI mix. Individual rare slices can be noisy: Arena has only 20 prompt clusters, and source-group sampling changes OpAI point estimates. Use per-dataset results and the paper/source-group bootstrap intervals in `summary.json` and each model’s `results.json`; do not average all rows into one accuracy number. Confidence intervals treat the chosen suite as sampled clusters and do not correct every stratified sampling fraction.

The suite has already been inspected and is no longer a blind final holdout. Possible public-benchmark overlap with MELD training is unknown. Paper pilots are explicitly exploratory; synthetic mixed examples are proxies. Historical human labels are supported by pre-2022 provenance, not observed writing histories.

## Scoring

Our unchanged epoch-2 checkpoint uses the existing BF16 512-token sliding-window inference. MELD v5/v8 use the exact locally saved release weights, verified by SHA-256, BF16 inference, 2,046 content tokens per window and no overlap. No input is truncated. MELD is evaluated as a localization baseline on raw, unchanged text; this is not its publisher’s normalized-text, capped, top-quartile document-scoring leaderboard protocol.

All scores are mapped to our ModernBERT reference-token boundaries by character-overlap-weighted averaging, then averaged over the same sentence boundaries. Thus all models have identical human/AI denominators, checked in `summary.json`. Mixed-boundary and ignored regions are masked. Document-only labels are never invented as token/sentence truth. Existing contradictory span annotations are excluded from primary span metrics. Exact duplicate text has one vote within each reported condition.

Each model’s token and sentence thresholds are selected using the same original 1,048 validation passages, with at most 1% empirical human FPR. Document decisions separately threshold the mean reference-token score at 1% FPR on the 524 validation human originals. This is in-domain historical-paper calibration, not a promise of 1% FPR under domain shift. Our original frozen thresholds are also reported separately in `ours/original-threshold-results.json`. MELD scores are raw evidence, not probabilities, despite the retained machine field name `mean_ai_probability`.

## Document-label diagnostics

These use only unambiguous native human/AI document labels and the common validation-calibrated mean-score rule. Mixed and assisted/ambiguous labels are excluded. Human-only / AI-only groups cannot establish both precision and FPR.

Each cell is **recall / human FPR**. Detailed precision, recall, FPR and AUROC breakdowns are in the JSON reports.

| Dataset | Our ModernBERT | MELD v5 | MELD v8 |
|---|---:|---:|---:|
| human_paper_remaining | — / 19.64% | — / 22.40% | — / 32.99% |
| pelic | — / 0.25% | — / 3.75% | — / 10.25% |
| vub | 0.00% / — | 77.50% / — | 77.50% / — |
| perkins | 2.91% / 0.00% | 94.17% / 70.00% | 100.00% / 90.00% |
| liang | — / 0.00% | — / 16.29% | — / 28.65% |
| meld_eval | 7.59% / 8.82% | 99.55% / 0.00% | 98.77% / 0.00% |
| detectrl | 9.09% / 3.57% | 97.05% / 3.57% | 97.61% / 3.27% |
| epoch | 9.26% / 0.00% | 63.80% / 0.40% | 78.79% / 1.62% |
| opai | 33.33% / 1.59% | 33.33% / 0.00% | 0.00% / 0.00% |
| sem_detect | 0.00% / 0.00% | 100.00% / 0.00% | 99.67% / 0.00% |
| gede | 7.89% / 0.00% | 99.34% / 12.50% | 100.00% / 12.50% |
| saha | 0.00% / 0.00% | 100.00% / 0.00% | 89.58% / 0.00% |
| local_binary | 15.62% / 0.00% | 96.09% / 34.38% | 95.31% / 15.62% |
| local_length | 21.57% / 17.71% | 98.04% / 33.33% | 98.04% / 18.75% |
| local_mixed | — / — | — / — | — / — |
| arena50 | 1.93% / — | 95.54% / — | 93.01% / — |
| paper_pilots_exploratory | — / — | — / — | — / — |

## Artifacts and timing

`manifest.json` freezes selection and input hashes. `coverage.json` audits coverage and compares our old-threshold subset estimates with the earlier full suite; it does not tune selection. `summary.json` contains shared-denominator verification and confidence intervals. Each model directory has validation thresholds, per-example predictions and complete breakdowns. Raw reference-token score caches remain on the private A100 Space for future reaggregation.

| Model | Inference, calibration and score summarization |
|---|---:|
| ours | 146.7 seconds |
| meld-v5 | 164.5 seconds |
| meld-v8 | 254.8 seconds |

Times exclude checkpoint transfer, initial failed upstream download, final bootstrap report construction and setup. The full end-to-end log is retained on the Space.
