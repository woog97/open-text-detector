# lc-1020 Atlas rescore — PAUSED (Alice, 2026-10-12 ~21:00 PDT)

Paused: lc-1020's only clear lead is on the long-context held-out sets, which were built with the same context-extension
procedure as its training rows (in-distribution for lc-1020). On the standard sets it ties final-moe (+0.2 [-0.9, +1.4])
and trails on paragraph and section edits (92.7 vs 94.7). Next step instead of step 1 below: a whole-paper test — splice the
held-out AI edits into real held-out human papers as full extracted papers through the site's own pipeline (not the
extension code), score end to end with each model's site setup, compare recall at each model's own calibrated cutoff.
Only if lc-1020 wins there does this rescore plan resume.

(Original plan, approved earlier the same evening:)

lc-1020: same 72,000 rows, recipe and loss terms as final-moe (abl-noseg-nomix), trained on 1,020-token windows with each
row extended by its real surrounding paper text. Ties final-moe on the standard held-out sets (74.9 vs 74.6 small-edit
recall at 1% FPR) and leads on edits inside real paper context (71.0 vs 42.7). Run folder on the H200:
final-20261007/sweeps/lc-1020 (adapter archived to the bucket; ARCHIVED.json), base assets assets/moe-final-1020.

0. Prerequisites (no GPU): restore the adapter from the bucket and check its hash; add width/stride options to
   research/atlas-rescore-moe-20261007/score_atlas_fast.py and score with 1,020-token windows, stride 510 (lc-1020's eval
   setup), token score = mean over covering windows; check it reproduces lc-1020's eval scores on a held-out set.
1. Go/no-go sample (1 GPU, ~15 min): the 600 papers in H200 atlas-rescore/year-check-20261012 (300 ICLR 2027 + 300
   pre-2023, already scored with Oct 7 and Oct 10). Compare human flag rates, ICLR 2027 rates, bibliography scores.
   Report to Alice before step 3 if lc-1020 looks worse on human papers or no different on ICLR 2027.
2. Calibration scoring (<=3 GPUs, ~1 h): 2,403 human calibration papers, 2023-2026 year sets, 120 AI manuscripts,
   held-out edit sets and AI sections. Cutoffs for all four counting modes at 0.1% and 1%, held-out check, red cutoff.
3. Full Atlas rescore (GPUs 0, 1, 3; estimate 9-11 h, benchmark first; ~3 GB output, check /workspace free space).
   Monitor on the 1, 2, 4, 8, 16, 32, then 60-minute schedule.
4. Build and check site data with research/atlas-oct10-20261012/*.py under model id qwen36-35b-a3b-lc1020.
5. Site: FLAG_MODEL, cutoffs, label, Calibration JSONs, Architecture (window length 1,020), Evaluation (lc-1020 numbers
   as main), Limitations (drop "a better model not ready"; update window and length items).
6. Publish (upload, catalogue, browse snapshot, deploy from main) only with Alice's go-ahead.
