# Mix grid (Oct 7): mixA and mixB

Two mixes built from the same pools as the Oct 7 final mix, scaled to 72,000 stage-2 rows for comparison and designed to scale up:

- **mixA** (`plan-mixA.json`): AI pools maxed, human text kept at the final mix's ratio. Full size about 144,000 stage-2 rows.
- **mixB** (`plan-mixB.json`): everything maxed. Full size about 718,000 stage-2 rows; mostly human paper text, generic human text and GRADTEX.

Files (copied from the Space's `/tmp/mixgrid-20261007/`, which a Space restart wipes):
- `orchestrate.sh`: pool-v1 → `final-20261007/mix_build.py --plan` → `fit_rows.py` → two MoE FSDP runs on A100 pairs.
- `run_fsdp.sh`: one MoE run, FSDP2 on 2 GPUs (micro-batch 16, prefetch 2), resumes up to 3 times, persists to `/data`.

Outputs: `/data/workspace/mixgrid-20261007/{mixA,mixB}`; scores in `../mixgrid-eval-20261007/`.
Trainer speedups (packing, fast MoE, Triton grouped GEMM, 2-H200 data parallel) are in `overnight-sweep-20261004/` (commit 5e4cb02); the 72k runs used only FSDP mb16 + prefetch 2.
