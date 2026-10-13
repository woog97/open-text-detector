# Evaluation generation results

189 papers; 1323 generated targets; 2646 correlated views.

| Condition | Outputs | Equivalent | Minor | Material | Uncertain |
|---|---:|---:|---:|---:|---:|
| sentence | 189 | 106 | 80 | 1 | 2 |
| two_sentence | 189 | 91 | 88 | 7 | 3 |
| paragraph_v3 | 189 | 75 | 101 | 9 | 4 |
| paragraph_concise | 189 | 76 | 102 | 11 | 0 |
| proofread | 189 | 89 | 94 | 1 | 5 |
| light_polish | 189 | 93 | 90 | 1 | 5 |
| substantial_rewrite | 189 | 113 | 70 | 2 | 4 |

Quality judgments are from Luna, not independent expert annotation. All valid outputs are retained. No classifier performance has been calculated on this collection.

| API stage, including archived pilots | Input tokens | Output tokens | Cost USD |
|---|---:|---:|---:|
| outline | 1,079,973 | 622,469 | $0.218115 |
| writer | 1,983,216 | 275,547 | $0.187525 |
| judge | 1,708,389 | 436,095 | $0.208169 |

Total API charges: **$0.613810**, including $0.135239 from archived pilots.

Untouched human body pool: 9,301 paragraphs; 1,590 clean, novel controls across all splits.

See scoring/manifest.json for prepared suite hashes, frozen thresholds, and the separate ELLIPSE supplement.
