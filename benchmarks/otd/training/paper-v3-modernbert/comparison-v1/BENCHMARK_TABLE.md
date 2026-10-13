# Benchmark results — BF16

**Bold marks the best local score in each row; ties at displayed precision are all bold.** ↑ higher is better; ↓ lower is better. `-` means unavailable or not the same metric. Pangram 4 is a separately reported reference, excluded from local winner highlighting because its samples, operating points and verdict definitions differ. These are descriptive maxima/minima, not significance tests.

Our three models used the identical frozen 13,751-example suite and BF16 inference. Token/sentence results share boundaries and validation-only thresholds; document diagnostics use the existing validation-calibrated mean-score rule. No inference was rerun for this table. [Full local protocol and confidence intervals](README.md). Pangram values come from the [Pangram 4 technical report, v1](https://arxiv.org/html/2607.27183v1), chiefly Tables 9, 11, 18–29.

**Important:** ROC operating points below are reaggregated descriptive test-set ROC statistics, not changes to deployed thresholds. GEDE has only 8 human controls, MELD-eval 34, and Sem-Detect 161; their requested 1%/0.1% FPR points permit zero empirical false positives and have weak low-FPR resolution. Pangram used much larger cohorts. GEDE ROC uses the benchmark’s native binary labels, including improved-human positives, unlike the stricter primary provenance metrics.

PELIC and TOEFL sample sizes differ from the report; VUB is 40 vs 39 papers. Epoch has matching class counts (594 AI, 495 human), but the document verdict rules differ. Public-benchmark overlap with MELD training is unknown. Do not interpret the reference column as a controlled four-model leaderboard.

## Our common span suite

| Benchmark | Metric | Our ModernBERT | MELD v5 | MELD v8 | Pangram 4 (reported) |
|---|---|---:|---:|---:|---:|
| v3 reconstruction — tokens | Precision ↑ | **93.40%** | 79.95% | 92.82% | - |
| v3 reconstruction — tokens | Recall ↑ | **73.96%** | 2.14% | 3.79% | - |
| v3 reconstruction — tokens | Human FPR ↓ | 4.91% | 0.50% | **0.28%** | - |
| v3 reconstruction — sentences | Precision ↑ | 93.05% | 86.36% | **98.28%** | - |
| v3 reconstruction — sentences | Recall ↑ | **73.10%** | 0.85% | 2.55% | - |
| v3 reconstruction — sentences | Human FPR ↓ | 5.23% | 0.13% | **0.04%** | - |
| Untouched human paragraphs — isolated | Human FPR ↓ | 4.86% | 6.70% | **4.29%** | - |
| Untouched human paragraphs — with neighbors | Human FPR ↓ | 3.04% | 3.75% | **2.42%** | - |
| Arena, 50 generators — tokens | Recall ↑ | 0.49% | 61.72% | **64.14%** | - |
| Arena, 50 generators — sentences | Recall ↑ | 0.33% | **88.51%** | 77.80% | - |
| OpAI — tokens | Precision ↑ | 76.57% | **88.61%** | 88.38% | - |
| OpAI — tokens | Recall ↑ | 4.29% | 19.63% | **22.77%** | - |
| OpAI — tokens | Human FPR ↓ | **1.65%** | 3.17% | 3.76% | - |
| OpAI — sentences | Precision ↑ | 74.07% | **88.09%** | 86.78% | - |
| OpAI — sentences | Recall ↑ | 4.77% | 30.71% | **32.68%** | - |
| OpAI — sentences | Human FPR ↓ | **1.98%** | 4.91% | 5.89% | - |

## Public benchmark reference

| Benchmark | Metric | Our ModernBERT | MELD v5 | MELD v8 | Pangram 4 (reported) |
|---|---|---:|---:|---:|---:|
| PELIC | Document FPR ↓ | **0.2500%** | 3.7500% | 10.2500% | [0.0067%](https://arxiv.org/html/2607.27183v1#S5.T9) |
| Liang TOEFL originals | Document FPR ↓ | **0.00%** | 32.22% | 53.33% | [0.00%](https://arxiv.org/html/2607.27183v1#S5.T9) |
| VUB | Document recall ↑ | 0.00% | **77.50%** | **77.50%** | [100.00%](https://arxiv.org/html/2607.27183v1#S5.T11) |
| Epoch style imitation | Document FNR ↓ | 90.74% | 36.20% | **21.21%** | [2.86%](https://arxiv.org/html/2607.27183v1#A3.T23) |
| Epoch style imitation | Document FPR ↓ | **0.00%** | 0.40% | 1.62% | [0.00%](https://arxiv.org/html/2607.27183v1#A3.T23) |
| GEDE | AUROC ↑ | 83.10% | 98.58% | **99.43%** | [100.00%](https://arxiv.org/html/2607.27183v1#A3.T21) |
| GEDE | TPR @ 1% FPR ↑ | 51.70% | 88.64% | **96.02%** | [100.00%](https://arxiv.org/html/2607.27183v1#A3.T21) |
| MELD-eval | AUROC ↑ | 75.54% | **100.00%** | 99.99% | [99.99%](https://arxiv.org/html/2607.27183v1#A3.T26) |
| MELD-eval | TPR @ 1% FPR ↑ | 1.23% | **100.00%** | 99.78% | [99.99%](https://arxiv.org/html/2607.27183v1#A3.T26) |
| Sem-Detect | AUROC ↑ | 79.35% | **100.00%** | **100.00%** | [97.80%](https://arxiv.org/html/2607.27183v1#A3.T27) |
| Sem-Detect | TPR @ 0.1% FPR ↑ | 2.96% | **100.00%** | **100.00%** | [95.50%](https://arxiv.org/html/2607.27183v1#A3.T27) |
| DetectRL — local pooled sample | Pooled binary F1 ↑ | 16.44% | 97.77% | **98.12%** | - |

## Other local diagnostics

| Benchmark | Metric | Our ModernBERT | MELD v5 | MELD v8 | Pangram 4 (reported) |
|---|---|---:|---:|---:|---:|
| Perkins — local binary proxy | Document recall ↑ | 2.91% | 94.17% | **100.00%** | - |
| Perkins — local binary proxy | Document FPR ↓ | **0.00%** | 70.00% | 90.00% | - |
| GEDE — local binary labels | Document recall ↑ | 7.89% | 99.34% | **100.00%** | - |
| GEDE — local binary labels | Document FPR ↓ | **0.00%** | 12.50% | 12.50% | - |
| DetectRL — local pooled sample | Document recall ↑ | 9.09% | 97.05% | **97.61%** | - |
| DetectRL — local pooled sample | Document FPR ↓ | 3.57% | 3.57% | **3.27%** | - |
| Sem-Detect — local binary subset | Document recall ↑ | 0.00% | **100.00%** | 99.67% | - |
| Sem-Detect — local binary subset | Document FPR ↓ | **0.00%** | **0.00%** | **0.00%** | - |
| Saha — local sample | Document recall ↑ | 0.00% | **100.00%** | 89.58% | - |
| Saha — local sample | Document FPR ↓ | **0.00%** | **0.00%** | **0.00%** | - |
| Local binary proxy | Document recall ↑ | 15.62% | **96.09%** | 95.31% | - |
| Local binary proxy | Document FPR ↓ | **0.00%** | 34.38% | 15.62% | - |
| Local length proxy | Document recall ↑ | 21.57% | **98.04%** | **98.04%** | - |
| Local length proxy | Document FPR ↓ | **17.71%** | 33.33% | 18.75% | - |
| Local interleaving proxy | Token precision ↑ | 65.58% | **65.81%** | 59.43% | - |
| Local interleaving proxy | Token recall ↑ | 0.87% | **91.72%** | 81.32% | - |
| Local interleaving proxy | Token human FPR ↓ | **0.25%** | 26.01% | 30.30% | - |
| Development-exposed paper pilots | Token precision ↑ | **76.49%** | 16.72% | 28.42% | - |
| Development-exposed paper pilots | Token recall ↑ | **65.92%** | 2.06% | 4.57% | - |
| Development-exposed paper pilots | Token human FPR ↓ | 4.86% | **2.47%** | 2.76% | - |

## Reported categories without matching local results

| Benchmark | Metric | Our ModernBERT | MELD v5 | MELD v8 | Pangram 4 (reported) |
|---|---|---:|---:|---:|---:|
| Overall private holdout | AUROC ↑ | - | - | - | [99.16%](https://arxiv.org/html/2607.27183v1) |
| Overall private human holdout | FPR ↓ | - | - | - | [0.0041%](https://arxiv.org/html/2607.27183v1#S5.SS3) |
| Overall private AI holdout | FNR ↓ | - | - | - | [0.3396%](https://arxiv.org/html/2607.27183v1#S5.T3) |
| ELLIPSE | FPR ↓ | - | - | - | [0.00%](https://arxiv.org/html/2607.27183v1#S5.T9) |
| ICNALE | FPR ↓ | - | - | - | [0.00%](https://arxiv.org/html/2607.27183v1#S5.T9) |
| UChicago standard, full length | TPR @ 1% FPR ↑ | - | - | - | [100.00%](https://arxiv.org/html/2607.27183v1#A3.T18) |
| UChicago standard, <50 words | TPR @ 1% FPR ↑ | - | - | - | [99.70%](https://arxiv.org/html/2607.27183v1#A3.T18) |
| UChicago humanizer, full length | TPR @ 1% FPR ↑ | - | - | - | [98.93%](https://arxiv.org/html/2607.27183v1#A3.T18) |
| UChicago humanizer, <50 words | TPR @ 1% FPR ↑ | - | - | - | [73.32%](https://arxiv.org/html/2607.27183v1#A3.T18) |
| UChicago human controls | FPR ↓ | - | - | - | [0.00%](https://arxiv.org/html/2607.27183v1#A3.T19) |
| GEDE fully generated (report split) | TPR @ 1% FPR ↑ | - | - | - | [100.00%](https://arxiv.org/html/2607.27183v1#A3.T21) |
| GEDE improved human (report split) | TPR @ 1% FPR ↑ | - | - | - | [100.00%](https://arxiv.org/html/2607.27183v1#A3.T21) |
| GEDE humanized (report split) | TPR @ 1% FPR ↑ | - | - | - | [100.00%](https://arxiv.org/html/2607.27183v1#A3.T21) |
| Perkins baseline AI | Native three-method mean accuracy ↑ | - | - | - | [100.00%](https://arxiv.org/html/2607.27183v1#A3.T22) |
| Perkins manipulated AI | Native three-method mean accuracy ↑ | - | - | - | [94.10%](https://arxiv.org/html/2607.27183v1#A3.T22) |
| DetectRL native average | Mean F1 across native subsets ↑ | - | - | - | [95.30%](https://arxiv.org/html/2607.27183v1#A3.T24) |
| Private light AI polish | AI-verdict FPR ↓ | - | - | - | [0.01%](https://arxiv.org/html/2607.27183v1#S5.T4) |
| Private editing prompts | Mixed-verdict recall ↑ | - | - | - | [55.01%](https://arxiv.org/html/2607.27183v1#S5.T5) |
| Private WildChat edits | Mixed-verdict recall ↑ | - | - | - | [65.17%](https://arxiv.org/html/2607.27183v1#S5.T6) |
| Saha easy AI-BP | AI-or-Mixed TPR ↑ | - | - | - | [98.20%](https://arxiv.org/html/2607.27183v1#A3.T28) |
| Saha easy AI-EP | AI-or-Mixed TPR ↑ | - | - | - | [100.00%](https://arxiv.org/html/2607.27183v1#A3.T28) |
| Saha easy AI-HI | AI-or-Mixed TPR ↑ | - | - | - | [96.30%](https://arxiv.org/html/2607.27183v1#A3.T28) |
| Saha easy H-AI | AI-only FPR ↓ | - | - | - | [1.40%](https://arxiv.org/html/2607.27183v1#A3.T28) |
| Saha hard AI-BP | AI-or-Mixed TPR ↑ | - | - | - | [100.00%](https://arxiv.org/html/2607.27183v1#A3.T28) |
| Saha hard AI-EP | AI-or-Mixed TPR ↑ | - | - | - | [100.00%](https://arxiv.org/html/2607.27183v1#A3.T28) |
| Saha hard AI-HI | AI-or-Mixed TPR ↑ | - | - | - | [98.90%](https://arxiv.org/html/2607.27183v1#A3.T28) |
| Saha hard H-AI | AI-only FPR ↓ | - | - | - | [2.50%](https://arxiv.org/html/2607.27183v1#A3.T28) |
| Saha human | AI-or-Mixed FPR ↓ | - | - | - | [0.00%](https://arxiv.org/html/2607.27183v1#A3.T28) |
| Commercial humanizers | AI recall ↑ | - | - | - | [97.69%](https://arxiv.org/html/2607.27183v1#S5.T14) |
| Commercial humanizers | AI-or-Mixed recall ↑ | - | - | - | [98.83%](https://arxiv.org/html/2607.27183v1#S5.T14) |
| Humanizer auxiliary head | Accuracy ↑ | - | - | - | [96.82%](https://arxiv.org/html/2607.27183v1#S5.T13) |
| BLADER | FNR ↓ | - | - | - | [0.43%](https://arxiv.org/html/2607.27183v1#S5.T15) |
| OpAI v0 | Mean AI+Assisted fraction | - | - | - | [0.00%](https://arxiv.org/html/2607.27183v1#A3.T29) |
| OpAI v1 | Mean AI+Assisted fraction | - | - | - | [0.01%](https://arxiv.org/html/2607.27183v1#A3.T29) |
| OpAI v2 | Mean AI+Assisted fraction | - | - | - | [3.18%](https://arxiv.org/html/2607.27183v1#A3.T29) |
| OpAI v3 | Mean AI+Assisted fraction | - | - | - | [13.59%](https://arxiv.org/html/2607.27183v1#A3.T29) |
| OpAI v4 | Mean AI+Assisted fraction | - | - | - | [5.48%](https://arxiv.org/html/2607.27183v1#A3.T29) |
| OpAI v5 | Mean AI+Assisted fraction | - | - | - | [32.57%](https://arxiv.org/html/2607.27183v1#A3.T29) |
| OpAI v6 | Mean AI+Assisted fraction | - | - | - | [56.17%](https://arxiv.org/html/2607.27183v1#A3.T29) |
| OpAI v7 | Mean AI+Assisted fraction | - | - | - | [67.94%](https://arxiv.org/html/2607.27183v1#A3.T29) |
| OpAI v8 | Mean AI+Assisted fraction | - | - | - | [69.50%](https://arxiv.org/html/2607.27183v1#A3.T29) |

## Multilingual coverage

| Benchmark | Metric | Our ModernBERT | MELD v5 | MELD v8 | Pangram 4 (reported) |
|---|---|---:|---:|---:|---:|
| Arabic | FPR ↓ | - | - | - | [0.0000%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Arabic | FNR ↓ | - | - | - | [0.9764%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Persian | FPR ↓ | - | - | - | [0.0000%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Persian | FNR ↓ | - | - | - | [3.1715%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Chinese | FPR ↓ | - | - | - | [0.0000%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Chinese | FNR ↓ | - | - | - | [0.7133%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Polish | FPR ↓ | - | - | - | [0.0000%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Polish | FNR ↓ | - | - | - | - |
| Czech | FPR ↓ | - | - | - | [0.0000%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Czech | FNR ↓ | - | - | - | [0.2760%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Portuguese | FPR ↓ | - | - | - | [0.0078%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Portuguese | FNR ↓ | - | - | - | [0.9946%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Dutch | FPR ↓ | - | - | - | [0.0026%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Dutch | FNR ↓ | - | - | - | [0.8397%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Romanian | FPR ↓ | - | - | - | [0.0000%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Romanian | FNR ↓ | - | - | - | - |
| French | FPR ↓ | - | - | - | [0.0026%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| French | FNR ↓ | - | - | - | [1.7839%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Russian | FPR ↓ | - | - | - | [0.0052%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Russian | FNR ↓ | - | - | - | [0.4850%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| German | FPR ↓ | - | - | - | [0.0026%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| German | FNR ↓ | - | - | - | [1.3258%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Spanish | FPR ↓ | - | - | - | [0.0026%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Spanish | FNR ↓ | - | - | - | [0.8867%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Greek | FPR ↓ | - | - | - | [0.0000%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Greek | FNR ↓ | - | - | - | - |
| Swedish | FPR ↓ | - | - | - | [0.0000%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Swedish | FNR ↓ | - | - | - | - |
| Hindi | FPR ↓ | - | - | - | [0.0000%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Hindi | FNR ↓ | - | - | - | [1.3682%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Turkish | FPR ↓ | - | - | - | [0.0000%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Turkish | FNR ↓ | - | - | - | [1.1066%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Hungarian | FPR ↓ | - | - | - | [0.0000%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Hungarian | FNR ↓ | - | - | - | - |
| Ukrainian | FPR ↓ | - | - | - | [0.0361%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Ukrainian | FNR ↓ | - | - | - | [1.5214%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Italian | FPR ↓ | - | - | - | [0.0000%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Italian | FNR ↓ | - | - | - | [0.3436%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Urdu | FPR ↓ | - | - | - | [0.0000%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Urdu | FNR ↓ | - | - | - | [5.3169%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Japanese | FPR ↓ | - | - | - | [0.0000%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Japanese | FNR ↓ | - | - | - | [0.9305%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Vietnamese | FPR ↓ | - | - | - | [0.0026%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Vietnamese | FNR ↓ | - | - | - | [1.9653%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Korean | FPR ↓ | - | - | - | [0.0000%](https://arxiv.org/html/2607.27183v1#S5.T8) |
| Korean | FNR ↓ | - | - | - | [0.5805%](https://arxiv.org/html/2607.27183v1#S5.T8) |

## Private interleaving coverage

| Benchmark | Metric | Our ModernBERT | MELD v5 | MELD v8 | Pangram 4 (reported) |
|---|---|---:|---:|---:|---:|
| Interleaving N=1 | Token accuracy ↑ | - | - | - | [75.02%](https://arxiv.org/html/2607.27183v1#S5.T7) |
| Interleaving N=1 | Token precision ↑ | - | - | - | [92.82%](https://arxiv.org/html/2607.27183v1#S5.T7) |
| Interleaving N=1 | Token recall ↑ | - | - | - | [62.86%](https://arxiv.org/html/2607.27183v1#S5.T7) |
| Interleaving N=1 | Document AI-fraction MAE ↓ | - | - | - | [18.27%](https://arxiv.org/html/2607.27183v1#S5.T7) |
| Interleaving N=4 | Token accuracy ↑ | - | - | - | [92.41%](https://arxiv.org/html/2607.27183v1#S5.T7) |
| Interleaving N=4 | Token precision ↑ | - | - | - | [99.56%](https://arxiv.org/html/2607.27183v1#S5.T7) |
| Interleaving N=4 | Token recall ↑ | - | - | - | [85.43%](https://arxiv.org/html/2607.27183v1#S5.T7) |
| Interleaving N=4 | Document AI-fraction MAE ↓ | - | - | - | [6.80%](https://arxiv.org/html/2607.27183v1#S5.T7) |
| Interleaving N=8 | Token accuracy ↑ | - | - | - | [95.94%](https://arxiv.org/html/2607.27183v1#S5.T7) |
| Interleaving N=8 | Token precision ↑ | - | - | - | [99.61%](https://arxiv.org/html/2607.27183v1#S5.T7) |
| Interleaving N=8 | Token recall ↑ | - | - | - | [92.31%](https://arxiv.org/html/2607.27183v1#S5.T7) |
| Interleaving N=8 | Document AI-fraction MAE ↓ | - | - | - | [3.51%](https://arxiv.org/html/2607.27183v1#S5.T7) |
| Interleaving N=12 | Token accuracy ↑ | - | - | - | [97.12%](https://arxiv.org/html/2607.27183v1#S5.T7) |
| Interleaving N=12 | Token precision ↑ | - | - | - | [99.66%](https://arxiv.org/html/2607.27183v1#S5.T7) |
| Interleaving N=12 | Token recall ↑ | - | - | - | [94.33%](https://arxiv.org/html/2607.27183v1#S5.T7) |
| Interleaving N=12 | Document AI-fraction MAE ↓ | - | - | - | [2.58%](https://arxiv.org/html/2607.27183v1#S5.T7) |
| Interleaving N=16 | Token accuracy ↑ | - | - | - | [97.50%](https://arxiv.org/html/2607.27183v1#S5.T7) |
| Interleaving N=16 | Token precision ↑ | - | - | - | [99.63%](https://arxiv.org/html/2607.27183v1#S5.T7) |
| Interleaving N=16 | Token recall ↑ | - | - | - | [94.73%](https://arxiv.org/html/2607.27183v1#S5.T7) |
| Interleaving N=16 | Document AI-fraction MAE ↓ | - | - | - | [2.20%](https://arxiv.org/html/2607.27183v1#S5.T7) |
| Interleaving N=20 | Token accuracy ↑ | - | - | - | [98.18%](https://arxiv.org/html/2607.27183v1#S5.T7) |
| Interleaving N=20 | Token precision ↑ | - | - | - | [99.51%](https://arxiv.org/html/2607.27183v1#S5.T7) |
| Interleaving N=20 | Token recall ↑ | - | - | - | [96.31%](https://arxiv.org/html/2607.27183v1#S5.T7) |
| Interleaving N=20 | Document AI-fraction MAE ↓ | - | - | - | [1.52%](https://arxiv.org/html/2607.27183v1#S5.T7) |
| Interleaving N=Overall | Token accuracy ↑ | - | - | - | [92.02%](https://arxiv.org/html/2607.27183v1#S5.T7) |
| Interleaving N=Overall | Token precision ↑ | - | - | - | [98.35%](https://arxiv.org/html/2607.27183v1#S5.T7) |
| Interleaving N=Overall | Token recall ↑ | - | - | - | [85.45%](https://arxiv.org/html/2607.27183v1#S5.T7) |
| Interleaving N=Overall | Document AI-fraction MAE ↓ | - | - | - | [5.87%](https://arxiv.org/html/2607.27183v1#S5.T7) |

The native DetectRL mean-F1 protocol, Perkins three-method mean accuracy, three-way Saha decisions and OpAI AI+Assisted fractions have no matching results in this run. Their cells remain `-`; locally available proxy metrics appear separately. Manual/agent red-team evaluations and backbone ablations are not substituted with scalar classifier scores.

All scalar values, direction flags, provenance links and per-row caveats are retained in `benchmark_table.json`.
