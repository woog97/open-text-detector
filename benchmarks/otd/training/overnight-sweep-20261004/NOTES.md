# Overnight sweep notes

## 2026-10-05 ~11:30 UTC: interim read and queue change

Test-half results at the final stage-2 epoch (all-edit sentence recall at 1% human-sentence FPR):

- A (2e-4 cosine, 3 seeds): 0.770 ± 0.010. Stable.
- B (5e-4 constant, 1 seed): 0.659. G (B with standard AdamW, 1 seed): 0.515, so the optimizer is not the fix.
- E (1e-3 constant): collapsed (0.017; flags every human sentence). The 9B on B collapsed too (0.000).
- Learning curves: on B, the 4B collapses after step 1,052 and stays there (about 0.01). The 9B collapses at step 1,052 and partly recovers (0.23). On A, the 4B is flat at 0.70–0.77 from step 526 on, so longer training buys little.

Change: added three LR arms ahead of the remaining third seeds; no planned run was removed.

- J: 5e-4 with cosine decay and 6% warmup. Tests whether the collapse comes from the constant, no-warmup schedule or from the LR itself.
- H: 3e-4 cosine.
- I: 1e-4 cosine. H and I bracket the stable LR range.

The E repeats (1e-3) moved to the end of the queue.

## 2026-10-05 ~13:30 UTC: the schedule caused the collapse, not the LR

All runs are Qwen3.5 4B at 20% length, cosine decay with 6% warmup. Test half, all-edit recall at 1% FPR:

- A (2e-4, 3 seeds): 0.761 / 0.780 / 0.770
- I (1e-4): 0.773
- H (3e-4): 0.768
- J (5e-4, the LoRA Without Regret LR): 0.756

LR from 1e-4 to 5e-4 barely matters once there is warmup and decay. The collapses in B, E and the learning curves come from the constant, no-warmup schedule. So arms C, D and F, which used that schedule, are confounded.

Next: K = A plus sentence-loss weight 1.0 (stable schedule), 3 seeds on the H200. It retests D's change cleanly; sentence loss was the strongest lever in the ModernBERT ablation. I seed 2 also moved to the H200, and the idle H200 GPU 0 runner was restarted.

## 2026-10-05 ~14:25 UTC: plateau on small edits, so test short-span exposure

K seed 1 (sentence-loss weight 1.0): all-edit recall 0.785, small edits 0.227, so no clear gain. Across every LR, schedule and loss change, small-edit recall at 1% FPR stays at 0.21–0.27. The full-length baseline learning curve (curve-4b-A, final) improves paper_v3 sentences (0.889) and standalone rewrites (0.704) but not small edits (0.207). That points at the training data: only 2 of 24,000 windows contain an AI span under 200 characters.

New arm L (3 seeds, H200): recipe A plus `--short-span-oversample`. Paper windows whose AI span is 600 characters or more are swapped for copies of existing windows with shorter spans. At 20% length that takes short-span windows from 347 to 585 per epoch, with the same row count and dataset mix and no new text (verified on the real data). Spans under 200 characters barely exist in the source, so this tests exposure to 1–3-sentence spans, not single sentences. Sentence-scale training data still needs new data work; that's the user's call.

## 2026-10-05 ~15:30 UTC: plateau confirmed; full-length learning curves on stable schedules

Test half, final epoch, mean ± sd over seeds (all-edit / small-edit recall at 1% FPR):

- 4B: A 0.770/0.249 (3 seeds), H 0.770/0.210, I 0.780/0.247, J 0.768/0.240, K 0.757/0.211 (3), L 0.699/0.243.
- 9B: A 0.763/0.218 (3).
- Constant-schedule arms (B, C, D, E, F, G) all trail or collapse.

L (short-span oversampling) left small edits flat and hurt paragraph edits and paper_v3. So the stable recipes all plateau, and 9B ≈ 4B at 20% length.

The one lever that moved metrics: training length. curve-4b-A final reached paper_v3 0.889 and standalone 0.704, against 0.77 and 0.51 at 20%; small edits stayed flat. Queued three full-length curves on the H200 (checkpoint every 525 steps, to protect the shared disk quota; folder was at 99 GB):

- curve-4b-A2: confirms the length effect with a second seed.
- curve-9b-A: does 9B gain more from length?
- curve-4b-J: is 5e-4 with cosine decay stable over a long run?

## 2026-10-05 ~18:30 UTC: calibrated cutoffs don't transfer

`calib_score.py` scored the calibration windows (195 papers; 4 sharing a few sentences with the evaluation suite were excluded; no ID overlap) for 16 top checkpoints. `calibrated_report.py` fits sentence cutoffs at 1% calibration human FPR and applies them to the test half.

Realized human FPR on the test half is 2–23%. By slot:

- Ordinary untouched human paper passages: 0.2–1.6%, close to the target.
- Human sentences next to AI edits: 4.5–7%.
- Matched untouched passages: 2–5%.
- Human paper_v3 sentences: 16–29%.

Calibration windows are mostly easy novel-human text, so cutoffs come out at 0.002–0.05. A deployable cutoff needs calibration data containing hard human text (neighbors of edits, paired originals). Results: `results/calibrated-operating-points.json`.

## 2026-10-05 ~20:00 UTC: final full-length curves; sweep complete

Final epoch, test half (all / small / paper_v3 / standalone / public AUROC):

- curve-4b-A: 0.789 / 0.207 / 0.889 / 0.704 / 0.976
- curve-4b-A2: 0.790 / 0.240 / 0.852 / 0.796 / 0.967
- curve-9b-A: 0.797 / 0.260 / 0.841 / 0.735 / 0.926
- curve-4b-J: 0.765 / 0.240 / 0.661 / 0.694 / 0.834
- For comparison, 4B A at 20%: 0.770 / 0.249 / 0.771 / 0.510 / 0.975

The length effect is confirmed on two seeds for paper_v3 and standalone rewrites; small edits stay flat. Long 5e-4 cosine generalizes worse. No further runs queued (user instruction); all sweep GPUs are idle.
