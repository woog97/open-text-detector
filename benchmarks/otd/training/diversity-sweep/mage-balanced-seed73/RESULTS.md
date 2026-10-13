# MAGE balanced 10%: seed73 confirmation

Second-seed matched-control comparison. Separately calibrated thresholds; higher recall is not a matched observed FPR improvement. No promotion. Native document labels are not token gold.

All scores are percentages. Cells show **control73 → MAGE73**; bold denotes the better numerical score (lower FPR, higher other metrics), not an overall winner. Dash means unavailable or no positive support. Thresholds fitted on calibration only. Existing frozen evaluations reused without new inference.

## workflow

| Group | Unit | F1 | Precision | Recall | Human FPR |
|---|---|---:|---:|---:|---:|
| ellipse | token | — → — | — → — | — → — | **0.00** → 0.16 |
| ellipse | sentence | — → — | — → — | — → — | **0.00** → 0.19 |
| ellipse | document | — → — | — → — | — → — | **0.00** → 0.04 |
| human_paper_workflow_matched | token | — → — | — → — | — → — | **0.28** → 0.38 |
| human_paper_workflow_matched | sentence | — → — | — → — | — → — | **0.32** → 0.36 |
| human_paper_workflow_matched | document | — → — | — → — | — → — | 0.00 → 0.00 |
| human_paper_workflow_remaining | token | — → — | — → — | — → — | 0.29 → **0.22** |
| human_paper_workflow_remaining | sentence | — → — | — → — | — → — | 0.28 → **0.24** |
| human_paper_workflow_remaining | document | — → — | — → — | — → — | 0.33 → **0.20** |
| paper_workflow_reconstruction | token | 71.27 → **73.62** | **97.07** → 95.47 | 56.30 → **59.91** | **0.83** → 1.39 |
| paper_workflow_reconstruction | sentence | 68.34 → **73.12** | **97.20** → 95.27 | 52.70 → **59.32** | **0.74** → 1.44 |
| paper_workflow_reconstruction | document | 74.05 → **76.22** | 100.00 → 100.00 | 58.80 → **61.57** | — → — |

## comparison

| Group | Unit | F1 | Precision | Recall | Human FPR |
|---|---|---:|---:|---:|---:|
| arena50 | document | 0.00 → **2.40** | — → 100.00 | 0.00 → **1.22** | — → — |
| detectrl | document | 0.37 → **57.17** | 100.00 → 100.00 | 0.18 → **40.02** | 0.00 → 0.00 |
| epoch | document | 0.00 → 0.00 | — → — | 0.00 → 0.00 | 0.00 → 0.00 |
| gede | document | 0.00 → **24.28** | — → 100.00 | 0.00 → **13.82** | 0.00 → 0.00 |
| human_paper_context | token | — → — | — → — | — → — | **0.62** → 0.91 |
| human_paper_context | sentence | — → — | — → — | — → — | **0.48** → 0.86 |
| human_paper_context | document | — → — | — → — | — → — | 0.00 → 0.00 |
| human_paper_remaining | token | — → — | — → — | — → — | **0.29** → 0.39 |
| human_paper_remaining | sentence | — → — | — → — | — → — | **0.21** → 0.34 |
| human_paper_remaining | document | — → — | — → — | — → — | **0.19** → 0.26 |
| liang | document | — → — | — → — | — → — | **0.00** → 5.62 |
| local_binary | document | 1.55 → **24.66** | 100.00 → 100.00 | 0.78 → **14.06** | 0.00 → 0.00 |
| local_length | document | 0.00 → **10.23** | — → 100.00 | 0.00 → **5.39** | 0.00 → 0.00 |
| local_mixed | document | — → — | — → — | — → — | — → — |
| meld_eval | document | 0.67 → **1.11** | **100.00** → 83.33 | 0.33 → **0.56** | **0.00** → 2.94 |
| opai | document | 0.00 → 0.00 | — → 0.00 | 0.00 → 0.00 | **0.00** → 1.59 |
| paper_pilots_exploratory | token | **77.83** → 77.48 | **100.00** → 97.73 | 63.70 → **64.18** | **0.00** → 0.36 |
| paper_pilots_exploratory | sentence | 80.66 → **81.75** | **100.00** → 97.51 | 67.59 → **70.38** | **0.00** → 0.42 |
| paper_pilots_exploratory | document | — → — | — → — | — → — | — → — |
| paper_v3_target | token | 89.89 → **91.24** | **97.97** → 96.89 | 83.05 → **86.21** | **1.62** → 2.60 |
| paper_v3_target | sentence | 86.19 → **90.77** | **97.46** → 96.85 | 77.26 → **85.41** | **1.93** → 2.66 |
| paper_v3_target | document | — → — | — → — | — → — | 0.00 → 0.00 |
| pelic | document | — → — | — → — | — → — | **0.00** → 0.25 |
| perkins | document | 0.00 → **7.48** | — → 100.00 | 0.00 → **3.88** | 0.00 → 0.00 |
| saha | document | 0.00 → 0.00 | — → — | 0.00 → 0.00 | 0.00 → 0.00 |
| sem_detect | document | 0.00 → 0.00 | — → — | 0.00 → 0.00 | 0.00 → 0.00 |
| vub | document | 0.00 → 0.00 | — → — | 0.00 → 0.00 | — → — |

## Selection curves at calibration targets

These are held-out selection scores using calibration-only thresholds. The nominal target is not observed selection FPR; none is a test oracle or deployable change.

| Calibration target | Unit | Paired F1 | Paired precision | Paired recall | Paired human FPR | Novel-human FPR |
|---|---|---:|---:|---:|---:|---:|
| 0.5% | tokens | **78.97** → 77.62 | **99.48** → 99.44 | **65.48** → 63.65 | **0.32** → 0.34 | 0.28 → **0.20** |
| 0.5% | sentences | 68.94 → **71.33** | **99.54** → 98.99 | 52.73 → **55.76** | **0.23** → 0.53 | 0.22 → 0.22 |
| 1% | tokens | 81.43 → **82.03** | **99.50** → 98.90 | 68.91 → **70.07** | **0.32** → 0.72 | **0.31** → 0.40 |
| 1% | sentences | 78.64 → **81.40** | **99.38** → 98.61 | 65.06 → **69.31** | **0.38** → 0.90 | **0.33** → 0.40 |
| 2% | tokens | **87.58** → 86.58 | **99.46** → 98.53 | **78.25** → 77.21 | **0.40** → 1.08 | **0.51** → 0.54 |
| 2% | sentences | **86.32** → 85.16 | **99.47** → 98.19 | **76.24** → 75.18 | **0.38** → 1.28 | **0.44** → 0.50 |

## Training trajectory

| Model | Stage / epoch (one-based) | Selection loss | Epoch seconds | Cumulative processed tokens |
|---|---|---:|---:|---:|
| control-seed73 | 1 / 1 | 0.418185 | 193.5 | 1280845 |
| control-seed73 | 2 / 1 | 0.201682 | 430.1 | 3837850 |
| control-seed73 | 2 / 2 | 0.196343 | 430.0 | 6392989 |
| control-seed73 | 2 / 3 | 0.170038 | 414.4 | 8944904 |
| mage10-balanced-seed73 | 1 / 1 | 0.337377 | 173.4 | 1280845 |
| mage10-balanced-seed73 | 2 / 1 | 0.232604 | 381.7 | 3837850 |
| mage10-balanced-seed73 | 2 / 2 | 0.207631 | 389.6 | 6392989 |
| mage10-balanced-seed73 | 2 / 3 | 0.192483 | 370.8 | 8944904 |

## Interpretation

MAGE increases reconstruction recall and F1 in both evaluated seeds, but false-positive rates also increase. At the 1% calibration target, selection token recall improves only 68.91% to70.07% while paired human FPR rises0.32% to0.72%; sentence recall65.06% to69.31% accompanies FPR0.38% to0.90%. At the 2% calibration target, control has higher recall and lower FPR for both units. MAGE best stage2 validation loss0.192483 is worse than control0.170038. Broad DetectRL document recall rises0.18% to40.02% at zero observed human FPR, but Liang human FPR rises0% to5.62%, MELD0% to2.94%, and OPAI0% to1.59%. No candidate promotion is justified from unpaired FPR comparisons or these two seeds alone. The original parity-thinned mage10-v1 and raid10-v1 remain confounded and are not included.

JSON preserves all dataset breakdowns, support counts, existing bootstrap intervals, thresholds and source hashes. Tiny human document samples cannot establish precise 1% population FPR.

## Document discrimination and support

AUROC is threshold-independent and shown on a 0–1 scale; missing values mean the scored group lacks the required binary support. Document gold remains separate from token/span gold.

| Group | Binary documents control / MAGE | Human documents | AI documents | AUROC control → MAGE |
|---|---:|---:|---:|---:|
| ellipse | 2571 / 2571 | 2571 | 0 | — → — |
| human_paper_workflow_matched | 216 / 216 | 216 | 0 | — → — |
| human_paper_workflow_remaining | 4900 / 4900 | 4900 | 0 | — → — |
| paper_workflow_reconstruction | 216 / 216 | 0 | 216 | — → — |
| arena50 | 987 / 987 | 0 | 987 | — → — |
| detectrl | 2302 / 2302 | 673 | 1629 | 0.6393 → 0.8840 |
| epoch | 1089 / 1089 | 495 | 594 | 0.8306 → 0.7618 |
| gede | 160 / 160 | 8 | 152 | 0.9137 → 0.8684 |
| human_paper_context | 1558 / 1558 | 1558 | 0 | — → — |
| human_paper_remaining | 1558 / 1558 | 1558 | 0 | — → — |
| liang | 178 / 178 | 178 | 0 | — → — |
| local_binary | 160 / 160 | 32 | 128 | 0.7815 → 0.9407 |
| local_length | 300 / 300 | 96 | 204 | 0.6821 → 0.9273 |
| local_mixed | 0 / 0 | 0 | 0 | — → — |
| meld_eval | 930 / 930 | 34 | 896 | 0.8162 → 0.7149 |
| opai | 69 / 69 | 63 | 6 | 0.6825 → 0.9048 |
| paper_pilots_exploratory | 0 / 0 | 0 | 0 | — → — |
| paper_v3_target | 519 / 519 | 519 | 0 | — → — |
| pelic | 400 / 400 | 400 | 0 | — → — |
| perkins | 113 / 113 | 10 | 103 | 0.8476 → 0.6136 |
| saha | 56 / 56 | 8 | 48 | 1.0000 → 0.9375 |
| sem_detect | 465 / 465 | 161 | 304 | 0.9340 → 0.9436 |
| vub | 40 / 40 | 0 | 40 | — → — |
