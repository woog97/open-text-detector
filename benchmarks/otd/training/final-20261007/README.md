# Final full-length runs (Oct 7)

Three full-length runs on the single-epoch mix: Qwen3.5 4B, Qwen3.6 35B-A3B MoE and EmbeddingGemma 2. Each run uses one seed. Decided by Alice on Oct 7. None of the runs uses DDP: the 4B and EmbeddingGemma each train on one GPU. The MoE trains as one process split across 2 GPUs by layers, or on one H200.

## Launch

From the Mac, once the mix exists as a `prepared-v2` directory on the Space:

```
uv run -q --with 'huggingface_hub>=1.0' --with httpx --with websockets python benchmarks/otd/training/final-20261007/launch.py --mix /data/workspace/<...>/prepared-v2
```

The mix needs `manifest.json`, `stage1-epoch0`, `stage2-epoch0..2` (three disjoint files of ~24k rows from `mix_build.py`; the trainer runs with `--fraction 1.0 --stage2-epochs 3`, one pass over all 72k distinct rows), `selection-windows` and `calibration-windows`.

Options:
- `--dry-run` shows which GPUs are free and where each run would go.
- `--only` launches a subset of the runs.
- `--sync-h200` copies finished H200 runs to the bucket under `workspace/final-20261007/h200/`.

## What the launcher does

1. **Fits the mix to every tokenizer** with `fit_rows.py`, on the Space. All three backbones must see the same rows. Gemma's tokenizer puts about 4% of T2 stage-2 rows and 11% of stage-1 rows over 510 tokens, though they fit Qwen's. Those rows are cropped the way `build_t21.crop510` does it: a window of up to 500 tokens around the AI span, snapped to whitespace. Document-only rows are left whole, because the trainer splits them into 510-token chunks. The fitted mix is saved to `/data/workspace/final-20261007/mix/prepared-v2`, with a crop count per file in `fit-report.json`.
2. **Checks free GPUs at launch time.** A GPU counts as free if it has no compute process and under 100 MiB in use.
   - **MoE:** 2 free Space A100s, else 1 free H200 (141 GB holds the bf16 MoE; faster than splitting it over 2 H200s, 3.7 vs 3.1 rows/s in the smoke tests), else 2 free H200s. Changed Oct 7 ~4 AM from Alice's 2-H200-first order for that reason. The MoE is placed first. Its first option in that order that still leaves room for every other run wins.
   - **4B and EmbeddingGemma:** 1 Space A100, else 1 H200.
   - Space GPUs 2 and 3 are never used; they are reserved for Alice. The H200 is shared, so the launcher only takes GPUs with nothing running on them. A run with no free GPU is reported as `NOT PLACED` and not started.
3. **Starts each run** with `run_one.sh`, which resumes once from the resume state after a crash.
   - On the Space, runs work in `/tmp/final-20261007/sweeps/<tag>`. `persist_daemon.py` mirrors that folder every 5 minutes to `/data/workspace/final-20261007/sweeps`.
   - On the H200, runs work in `/workspace/woog/pangram/final-20261007/sweeps/<tag>`, with resume state in `/dev/shm/woog-pangram/state`.
4. **Records where each run went** (host, GPUs, tag, arguments) in `/data/workspace/final-20261007/placement.json` and `placements/`.

## Run settings

All runs share these arguments: `--fraction 1.0 --stage2-epochs 3 --head-lr 2e-5 cosine --warmup 0.06 --seed 1 --on-diverge rollback`. They also save resume state every 20 minutes and a snapshot every 100 steps.

| Tag | Model | GPUs | Micro-batch | LR | Notes |
|---|---|---|---|---|---|
| final-moe | qwen36-35b-a3b | 2 (`--model-parallel`) or 1 H200 | 8 | 1e-4 | 2e-4 diverged on the H200 (moe-A-full) |
| final-qwen35-4b | qwen35-4b | 1 | 8 | 2e-4 | arm A |
| final-eg2 | embeddinggemma-2 | 1 | 32 | 2e-4 | encoder: no Repeat2; stage 1 + 2 as for other encoders |

## Code changes for these runs

- `overnight-sweep-20261004/train_sweep.py`: new `--model-parallel` option. It splits one model across all visible GPUs by whole decoder layers (accelerate, balanced) and keeps the heads on GPU 0, where batches and losses live. It cannot be combined with DDP.
- `overnight-sweep-20261004/modeling_sweep.py`: the head width is the backbone's output width. EmbeddingGemma 2 projects each token to 768 dimensions after the final norm; its hidden size is 512.
- `backbone-launch-20261003/data.py`: encoder boundary tokens fall back to BOS/EOS when a tokenizer has no CLS/SEP, as with EmbeddingGemma 2.
- EmbeddingGemma 2 loads through `AutoModel`, and its `language_model` (the text encoder) becomes the backbone. The vision and audio towers are loaded on the CPU and discarded. LoRA targets every Linear layer of the text model, including its per-layer-embedding projections and the final embedding projection. The 262k-row embedding table stays frozen. No task prefix is added, matching the other backbones.

## Hosts

- **Space:** `space_setup.py` builds `/tmp/final-20261007` and is safe to rerun after a restart. It copies:
  - the code from `/data/workspace/final-20261007/code`, pushed by `push.py`;
  - the vendor libraries (transformers 5.19.0, which has `embedding_gemma2`), from the T2 workspace;
  - the 4B and MoE weights, from the bucket's `backbone-launch-20261003/assets`;
  - EmbeddingGemma 2 at revision 914f7f8, downloaded from HF on the Space into the bucket assets.
- **H200:** `h200_setup.sh` builds `/workspace/woog/pangram/final-20261007`.
  - It uses the backbones venv (torch 2.11), with transformers 5.19.0 in `vendor/` installed over it, because the venv's 5.18.0 lacks `embedding_gemma2`.
  - The MoE weights are symlinked from `/dev/shm/woog-pangram/assets`. The 4B and EmbeddingGemma 2 are downloaded anonymously from HF (both are ungated) to `/workspace`, about 10 GB.
  - The W&B key arrives over ssh stdin. The HF token arrives over stdin only for bucket transfers; nothing is written to disk.

## Smoke tests (Oct 7)

Each smoke run used 1% of the T2/T2.1 mix, or 10% for EmbeddingGemma, with `--no-validate`. Each one trained through stage 1 and stage 2 and saved checkpoints. Throughput is for stage 2.

| Path | Host | Stage-2 throughput | Peak memory (GPU 0) |
|---|---|---|---|
| 4B, 1 GPU | H200 | 5.9 rows/s | 12.8 GB |
| MoE, 1 GPU | H200 | 3.7 rows/s | 75.4 GB |
| MoE, 2 GPUs `--model-parallel` | H200 | 3.1 rows/s | 38.7 GB |
| EmbeddingGemma 2, 1 GPU | H200 | 63 rows/s (stage-2 loss 3.5 → 0.83 over 70 steps) | 5.4 GB |

**Not tested on the Space A100s yet:** the MoE split across 2 A100s and EmbeddingGemma 2 in the Space environment (torch 2.14). At about 10:10 UTC, the ablation queue in `/tmp/pangram-ablation-20261007` took all 8 GPUs, including GPUs 2, 3, 4 and 5. I stopped my MoE A100 smoke test while it was still loading the model. The code path is the same one that passed on the H200. The MoE split on the H200 peaked at 38.7 GB on GPU 0, so it should fit an 80 GB A100.

`fit_rows.py` on the T2.1 mix cropped 1,320 of 12,000 stage-1 rows, about 980 of 24,000 rows per stage-2 epoch, and 1 calibration row. It ran in a few minutes once batched.
