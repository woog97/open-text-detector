# Final single-pass mix (Oct 7)

The training data for the final runs: 72,000 stage-2 rows, which is the T2.1 plan ×3 with DeepSeek added. They are written as three epoch files of 24,000 different windows each. `train_sweep.py` runs its usual 3 stage-2 passes, which here means a single pass over 72k windows with no window seen twice. Stage 1, the selection windows and the calibration windows are copied unchanged from T2.1 (`/data/workspace/pangram-t2-20261006/prepared/qwen35-4b-t21/prepared-v2`).

## Rerun (one command, from this folder on the Mac)

```
uv run --with requests --with pyarrow --with 'huggingface_hub>=1.0' --with httpx --with websockets python mix_run.py
```

This takes about 2 minutes plus the upload.
- **Output:** `/data/workspace/final-20261007/prepared-v2/` (manifest and files), plus `mix-report.json` and `mix-build.log` in `/data/workspace/final-20261007/`.
- **Dry run:** `mix_run.py --test` writes to `/data/workspace/final-20261007/mix-test/` instead.
- **Builder flags:** anything after `--` goes to `mix_build.py`. For example `-- --human-source clean`, `-- --pool /data/workspace/corpus-sampling-20261007/pool-early`, or `-- --workers 16`.

Rerun it after each new batch of data lands:
- the Claude wide1 ingest (local files),
- the DeepSeek and Luna wide1 outputs on the Space,
- GPT-6.1 Sol Codex rows (ingested into `llm-edits-claude-hsm-v1`).

## Pieces

- **`mix_pools.py` (Mac).** Builds the candidate pools from the ingested datasets: Claude/GPT-6.1/Luna small edits, sections, paragraph edits, unused older hosts and Claude full papers. It bundles them with every never-train list in `build_llm_edits.NEVER_TRAIN_FILES` into `mix-inputs.tgz`.
- **`mix_build.py` (Space, CPU only).**
  - **Adds sources:**
    - The wide1 API outputs (`wide1/out-{deepseek,luna}-{d,e}/accepted.jsonl`, split=train only).
    - The T2 source pools: paired papers, mirrors, GPT/Luna manuscripts, generic human text and GRADTEX. Long documents are cut into non-overlapping windows.
    - The wide clean-v2 pool: `pool-v1`, or `pool-early` until it exists.
  - **Screens every row:**
    - Crops it to 510 tokens under every backbone tokenizer: Qwen3.5, Qwen3.6, and Gemma, which stands in for EmbeddingGemma 2 and is replaced by it once its assets exist.
    - Runs `never_train_hit` against every held-out list plus `wide1-heldout-never-train.json`.
    - Drops rows on evaluation, selection or calibration papers, rows sharing a 10+-word sentence with them, and reserved manuscript families.
    - Writers never used: Fable, GPT-6 Sol, GLM, MiMo.
  - **Allocates rows per key, round-robin over papers.** If a key runs short, it fills from other keys in the same group first; only then does it repeat rows. Repeats are reported.
- **`mix_run.py`.** Uploads through the bucket API, starts the build detached on the Space, and polls once a minute until the report is written.

## Plan (rows over the whole pass)

| Group | Keys |
|---|---|
| Small edits | `edit_claude` 13,320 · `edit_luna` 6,720 · `edit_deepseek` 3,360 · `edit_gpt` 1,800 (GPT-6.1 Sol only) |
| Human paper text | `human_paper_orig` 9,000 · `human_section` 3,000 · `human_para_e` 3,000 · `human_host` 3,000 |
| Paragraph edits | `para_luna` 4,221 · `para_e` 3,480 · `para_deepseek` 999 |
| Sections | `sections_ai` 4,200 |
| Fully AI | `mirrors` 4,200 · `fullpapers` 1,500 · `claude_fp` 1,500 |
| Generic human | `human` 5,100 |
| GRADTEX | `gradtex` 3,600 |

## Human sources and the format check

New human paper text is clean v2. With `--human-source balanced` (the default):
- **`human_paper_orig`:** uses the older paired human windows first, then clean-v2 windows (up to 5 per paper) from wide-pool papers that no generation job used.
- **`human_host`:** takes clean-v2 hosts in the same proportion as the hosted AI rows in the mix that are clean v2. The rest are older PMC, peS2o and arXiv hosts.

This keeps the text format from predicting the label. `mix-report.json` → `format_check_paper_text` gives, for each text format, the AI and human row counts and the AI share. If one format's AI share is far from the others', that format correlates with the label.

`--human-source clean` takes both keys from clean v2 only.
