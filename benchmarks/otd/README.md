# Pangram 4 evaluation workbench

## Unified evaluation suite

Use [eval_suite/README.md](eval_suite/README.md) for the current frozen, resumable **BF16-only** runner covering our ModernBERT model and MELD v5/v8. It unifies the representative comparison, fresh workflow/ELLIPSE tests, assistance diagnostics and larger historical collections; the commands below describe the older workbench.

Public workflow dataset: [woog/ai-paper-workflow-eval](https://huggingface.co/datasets/woog/ai-paper-workflow-eval). Its selection index covers all local profiles without mirroring third-party corpora. [Upload receipt](eval_suite/huggingface_upload.json) pins the verified revision.


A separate command-line research suite for **our local MELD v5 and EditLens Qwen3-4B models**. Everything lives in this directory. It reuses the existing read-only model files and inference classes, but never starts the website, reads its database or credentials, or submits scans to its queue. Scoring is offline. Download commands access public data only.

See [COVERAGE.md](COVERAGE.md) for the complete evaluation inventory, source links, and what remains unavailable. These are evaluations of local baselines on released datasets and explicit proxies, **not measurements of Pangram 4**.

## ModernBERT broader evaluation

The [completed wider suite](training/paper-v3-modernbert/wide-eval-v1/README.md) scores the frozen research-paper ModernBERT baseline on 65,348 examples across 17 benchmark groups, including 50 Arena generators. It also extracts 95,334 remaining historical human paragraphs with preserved splits. Broader recall is poor at the original thresholds; these results are separate from the MELD/EditLens evaluations below. No new generation calls were made.

## Planned fresh-generation benchmark

The paper editing pilot (`paper_pilot10.py`) now defaults to `openai/gpt-6-luna` for future generation, with a separate Luna output directory. The completed GPT-6.1 Sol pilot remains an immutable, separately identified dataset. Input/output usage and cost accounting continue to include every attempt.

For cross-model evaluation, train and choose thresholds using Luna examples on the training/validation paper splits. Freeze the detector and thresholds before scoring expensive-model examples on held-out papers. Keep every variant of a paper in its original split, and report each generator separately. Existing Sol test examples can be an exploratory transfer check; use fresh unseen papers for a final evaluation after reviewing pilot examples. Report token/sentence precision, recall and human-text false positives, with uncertainty clustered by paper; ten papers are only a pilot.

To work with the historical Sol run, explicitly pass `--model openai/gpt-6.1-sol --output research/data/paper-pilot10-gpt61-sol-20260929`. Changing the generator requires a separate output directory; existing manifests are checked before any action.

[Arena-20 protocol](ARENA_20_PROTOCOL.md) documents how to select 20 in-scope conversation prompts reproducibly and send the same prompts to multiple generators. It specifies filtering, sampling, generation controls, missing-response handling, and small-sample reporting. The sample and roster are frozen; generation, local response review and offline detector scoring are implemented. See [Arena-20 results](ARENA_20_RESULTS.md): 365 successful responses, 351 eligible responses scored by both local detectors, and $0.1530 reported usage cost.

## Run

Use the existing environment from the workspace root:

```sh
# Public data acquisition; cached downloads include URL, bytes, and SHA-256 receipts.
backend/.venv/bin/python benchmarks/otd/download.py
backend/.venv/bin/python benchmarks/otd/download_saha.py
backend/.venv/bin/python benchmarks/otd/download_documents.py

# Prepare bounded, deterministic, score-independent samples.
backend/.venv/bin/python benchmarks/otd/prepare.py --per-stratum 4

# Quick pilot. Remove the limit to score every prepared example.
backend/.venv/bin/python benchmarks/otd/run.py \
  --model meld --device mps --limit-per-dataset 24 \
  --output benchmarks/otd/runs/my-meld-pilot
backend/.venv/bin/python benchmarks/otd/run.py \
  --model editlens --device mps --limit-per-dataset 8 \
  --output benchmarks/otd/runs/my-editlens-pilot

# A whole small public benchmark, rather than four rows per stratum.
backend/.venv/bin/python benchmarks/otd/prepare.py --only liang --per-stratum 0
backend/.venv/bin/python benchmarks/otd/run.py \
  --model meld --datasets liang --output benchmarks/otd/runs/liang-full

backend/.venv/bin/python -m pytest benchmarks/otd/test_metrics.py -q
```

`--datasets` accepts comma-separated prepared file stems. `local` includes binary baselines, length truncations, and mixed-authorship proxies. `--device cpu` works without the Mac GPU, but takes longer. `--resume` requires unchanged code, data, model manifest, and settings. Use a new output directory when any of these change. Failed predictions are recorded and cause a nonzero exit; three consecutive failures stop the run. Completed predictions are flushed after every document.

Each run contains `manifest.json`, `predictions.jsonl`, `metrics.json`, and a readable `REPORT.md`. New runs also archive the scoring source files under `code/`. Prediction files contain IDs and scores, not full source text. Source text stays in `data/`. Data and run artifacts are excluded from version control, while the suite and documentation can be tracked independently of the website.

## What the numbers mean

- Document metrics: strict FPR/FNR, accuracy, AI F1, AUROC, empirical TPR at 0.1%, 1%, and 5% FPR, and Wilson intervals. Missing classes produce nulls. Mixed decisions count as errors on pure human/AI data.
- Editing metrics: fully-AI false alarms on light polish, Mixed recall on mixed examples, trajectory mean predicted/target fractions, and fraction MAE.
- Localization: word-level accuracy, precision, recall, and document AI-word-fraction MAE. Alignment uses word midpoints and Unicode character offsets, making it comparable across the two local tokenizers. This is explicitly **not** the report's exact tokenization protocol.
- Breakdowns: dataset, cohort, generator, domain, language, attack, editing version, mixing block size, and length. A single-class subgroup cannot independently estimate an ROC curve; a separately named supplemental curve compares AI generator/attack/cohort subsets against deduplicated unpolished human controls from the same dataset.

MELD uses the shipped raw-score threshold and a binary decision. It cannot recover a homogeneous Mixed class or a dedicated humanizer label. Its highlighted-sentence coverage is only an exploratory fraction proxy. For short-text challenges the runner forces a research-only decision and flags examples below the model's recommended 100-word minimum. This differs deliberately from the website's uncertain verdict.

EditLens uses the existing expected-bucket score and configurable exploratory document cutoffs (Human <0.1; AI >=0.8; otherwise Mixed). These cutoffs are not Pangram's production calibration. Its score estimates editing extent; it is not a literal percentage of characters written by AI. Comparisons to OpAI provenance fractions are therefore explicitly proxy comparisons.

The installed MELD checkpoint is **v5, released after the Pangram 4 report**. Its own MELD-eval dataset may overlap its development data; performance there is a compatibility check, not independent evidence of generalization. Our long-document inference also uses the existing workbench's overlapping-window aggregation, rather than silently truncating to the first window. Both models' exact revisions and preprocessing are saved in each run.

## Sampling and interpretation

Prepared samples use the lowest SHA-256 values of a fixed seed plus source IDs within each cohort/label/generator/domain/attack stratum. `--per-stratum 0` retains every eligible row; large corpora can consume substantial memory and inference time. The secondary pilot cap cycles labels first, then cohorts within each dataset, so multiple AI cohorts cannot crowd out human controls. Pilots may omit strata when the cap is smaller than their number.

No model training or threshold tuning is performed on these examples. Do not treat repeated versions, temperatures, lengths, reviews of one paper, or overlapping DetectRL tasks as independent observations. Shared source IDs are preserved where published; exact text hashes are always retained. Wilson intervals are descriptive, not cluster-adjusted uncertainty estimates. Never combine the local human-book and AI-Q&A rates into a claim about real-world accuracy: topic and style are confounded.

At 100 human examples, even zero false positives has a roughly 3.7% upper Wilson bound. That cannot validate a one-in-tens-of-thousands claim. The empirical ROC uses indivisible tied scores without interpolation; low-FPR operating points are poorly resolved in small samples and are not held-out deployment calibration.

ICLR paper-year cohorts from the website are intentionally excluded from labeled evaluation because their authorship is unverified. The public-dataset scoring paths invoke no paid APIs, new frontier generations, commercial humanizers, or adaptive attacks. The separately authorized Arena experiments below generate new responses through OpenRouter.

## Prepared data in this workspace

The audited collection currently has **4,607 prepared examples across 12 input files**, backed by approximately **1.04 GB** of downloaded raw data. This includes all 270 selected Liang-source rows, 100 sampled PELIC answers, 114 Perkins documents, and the 40 VUB AI documents currently published in the source folder. The remaining public collections use bounded stratified samples. The complete downloaded source pools can be re-prepared with a larger cap; the Sem-Detect and OpAI downloads themselves are subsets.

`AUDIT.json` records the actual counts, labels, duplicate-text counts, and receipt checks. Run `backend/.venv/bin/python benchmarks/otd/audit.py` to verify them again. The first exploratory runs used earlier, smaller preparations; their manifests preserve the exact selection IDs and input hashes, and they should not be mistaken for runs over every currently prepared row.

## Arena-20 execution

The [20-model roster](ARENA_20_MODELS.md), [frozen prompt sample](ARENA_20_SAMPLE.md) and [sampling protocol](ARENA_20_PROTOCOL.md) are separate from the website. `arena20.py` builds/validates the sample; `arena_generate.py` runs the frozen 400-cell OpenRouter grid. Credentials belong only in ignored `.env.secrets`. Execution artifacts are in `data/arena20/` and `runs/arena20/`; completed response audits and detector results are in [Arena-20 results](ARENA_20_RESULTS.md), explicitly separated from exact Pangram replication. The original 400 cells and 40 replacement cells are preserved.

### Reproducible Arena workflow

From the project root, use the existing Python environment. `arena_acquire.py` verifies or downloads the pinned source; `arena20.py prepare` creates the full deduplicated order. Screen a contiguous prefix with the protocol, then `arena20.py freeze` checks the 20 selected prompts. `arena_select_models.py` records a roster from the saved catalog and refuses to overwrite a frozen manifest.

`arena_generate.py --dry-run` validates the frozen grid without network requests. Actual generation is `arena_generate.py`; the two availability replacements use `--manifest benchmarks/otd/data/arena20/replacement_manifest.json --output benchmarks/otd/runs/arena20-replacements`. Existing completed cells are never regenerated on resume. An interrupted cell with uncertain completion requires reconciliation.

`arena_review.py packet` produces local review packets; retain explicit decisions and response hashes in `content_reviews.jsonl`. `arena_review.py finalize` prepares the exploratory detector inputs. `arena_score.py --model meld` and `arena_score.py --model editlens` process only reviewed responses offline (MPS), preserving checkpoints, code and per-response hashes. `arena_report.py` checks the complete grids and builds the per-model report, all-cell matrix and common-prompt comparisons. The scorers refuse to overwrite existing prediction files. Create a new experiment directory for a changed seed, roster, rubric or generation setting.

Run validation with `backend/.venv/bin/python -m pytest benchmarks/otd/test_arena20.py benchmarks/otd/test_metrics.py -q`. No website database or website API credentials are used.

## Arena-100 execution

The separate [Arena-100 protocol](ARENA_100_PROTOCOL.md) and [100-prompt sample](ARENA_100_SAMPLE.md) extend the original seeded sample to 100 prompts across 12 retained Pangram-report models. The original 20 prompts are preserved. Optional reasoning is disabled, GPT-OSS uses low reasoning, and Llama exposes no reasoning control. Grok uses an asynchronous batch; the other models use concurrent requests.

`arena100_prepare.py` freezes the screened prefix and roster. `arena100_generate.py` preserves requests, responses and errors; `--resume-blocked` retries only explicit key-limit or rate-limit rejections, while `--poll` retrieves the existing batch. It never regenerates a successful cell. An uncertain transport outcome is retained rather than automatically retried. Secrets remain in ignored `.env.secrets`.

`arena100_score.py --model meld` and `--model editlens` score successful, stop-finished responses with at least 50 words. This is a mechanical-screen variant, without exhaustive substantive-prose/refusal review or a verified native token-length gate. It must remain separate from the earlier Arena-20 response-screen protocol. Scorers wait for all cells and a verified `runs/arena100/sync_complete.json` marker; `arena100_results.py` validates complete detector coverage and response hashes before producing the final report. Raw attempts and all excluded outcomes remain in `runs/arena100/`. No new datasets or weights are required.

The [completed Arena-100 generation report](ARENA_100_GENERATION_RESULTS.md) contains per-model charges and token counts; [runs/arena100/dataset.jsonl](runs/arena100/dataset.jsonl) contains all 1,200 successful prompt/response pairs. Failed-cell cleanup uses `arena100_generate.py --retry-failed --request-timeout 600 --concurrency 8`, only after explicit authorization, and retains original failed attempts.

The [completed Arena-100 detector report](ARENA_100_RESULTS.md) verifies 1,180 predictions from each local detector. `arena100_score.py --model editlens --resume` preserves existing predictions and checks the latest eligible response set before finishing; it archives previous manifests and code. The earlier cheap-roster expansion uses `arena100_cheap_prepare.py`, `arena100_generate.py --experiment arena100-cheap`, and `arena100_cheap_results.py`; see [its protocol](ARENA_100_CHEAP_PROTOCOL.md).

The [completed cheap-roster expansion](ARENA_100_CHEAP_RESULTS.md) covers all 2,000 cells, including 365 exact-match reused responses. The [combined dataset](runs/arena100-combined/dataset.jsonl) contains 3,200 successful responses across 32 model variants. The new 20-model detector inputs are prepared in `runs/arena100-cheap/detector_inputs.jsonl`; detector scores currently cover the separate original 12-model pass.

## Previously skipped models and newer GPT releases

The [skipped-model extension](ARENA_100_SKIPPED_PROTOCOL.md) includes all 14 previously omitted Table 3 generators and the requested available GPT releases. It reuses all 100 GPT-6 Luna responses and adds 1,700 calls with low/disabled reasoning and 32 concurrent requests. GPT-6 Terra is unavailable in the frozen OpenRouter catalog; its absence is recorded explicitly. See the [generation status and cost table](ARENA_100_SKIPPED_RESULTS.md) and `runs/arena100-expanded/` for the cumulative, deduplicated export. Local detector scoring remains separate.

Resume only failed/missing requests with `arena100_generate.py --experiment arena100-skipped --retry-failed --concurrency 32 --request-timeout 600`, after confirming no other generator process is active. Rebuild exports with `arena100_skipped_results.py`.

## Published Hugging Face dataset

The 4,900-response corpus is public at [woog/arena-prose-100-49-models](https://huggingface.co/datasets/woog/arena-prose-100-49-models), as a single `test` split in Parquet. It contains prompt/response text, model and route metadata, generation settings, nullable usage/cost fields, source attribution and eligibility flags. All 49 models have the same 100 prompts. The upload receipt and verified remote hashes are in `runs/arena100-expanded/huggingface_upload.json`; local export files are in `exports/arena-prose-100-49-models/`. `package_huggingface.py` rebuilds the package; `upload_huggingface.py` uploads only the five explicitly listed dataset files.

The published dataset has since been extended with **100 Kimi K3 responses**, bringing it to **50 models and 5,000 rows** at the same Hugging Face URL. The original 4,900 rows are unchanged. Kimi K3 used disabled reasoning, the same 100 prompts, a 4,096-token cap, and 32 concurrent calls; all succeeded and met the mechanical eligibility gate. Reported generation cost was $0.8986124493. The cumulative local JSONL is `runs/arena100-expanded-kimi/dataset.jsonl`; extension validation is in `runs/arena100-kimi/publication_validation.json`.
