# Results page: verified numbers (Oct 11 2026)

Model: final-moe = `abl-noseg-nomix` (Oct 10 loss-ablation winner), checkpoint stage2-epoch0. Eval outputs on the H200:
`/workspace/woog/pangram/final-20261007/sweeps/abl-noseg-nomix/eval/` (eval_final.py, FAST_EVAL).

- Pooled small-edit recall at 1% FPR (heldout-writers + t21-heldout-v2 + wide1-heldout, each at its own control cutoff):
  `loss-ablation-20261010/pooled_heldout.py --run final-moe=<eval dir>` → 0.7462 (n=7,243), spill FPR 0.0109;
  with `--suffix=-lc` → 0.4270, spill 0.0143. Per-cohort and per-writer numbers on the page come from the same outputs
  (copies: H200 `atlas-rescore/pooled-final-moe.json`, `pooled-final-moe-lc.json`).
- 95% intervals: 1,000-draw paper bootstrap within each set, cutoffs recomputed (same definitions as pooled_heldout.py):
  standard [0.732, 0.759], -lc [0.410, 0.445].
- Larger text and stress sets: `recall_at_1pct_controls` and `auroc_vs_controls` from each `<set>-stage2-epoch0.json`
  (t21 paragraph/section cohorts, wide1 para_rewrite, fable-humanizer cohort humanized, fable-cowrite all, cross-model-s500 all);
  fable-polish has 2 AI-labelled sentences, so only `documents.auroc_mixed_vs_controls` = 0.814 is reported.
- lc-1020 comparison (71.0% vs 42.7% on the -lc sets): memory note long-context-finding-20261010 / loss-ablation results.
