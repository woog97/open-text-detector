# Human-source candidate collection

## Approved expansion to 100,000

The user approved continued background fetching to fill the exact existing source quotas. `expand_pool.py` now runs on the Space under `/data/workspace/human-source-mix-v2`, preserving the first release. It imports prior candidates, corrects Wikimedia article/talk routing, deduplicates before counting, and resumes broader scans from transactional SQLite checkpoints. The original per-source allocations are unchanged. Collection is not yet complete.

The chat follow-up `complete-100k-human-source-mix` runs every 10 minutes to recover collection, research and implement missing source readers, and publish verified private HF updates. Recovery details and remaining adapter work are in `EXPANSION_RUNBOOK.txt`; `expansion-job.json` records the initial deployment. Do not treat old PID or count snapshots as live state.

## First collection

The executable intake is `collect_pool.py`. It runs on the existing training Space at `/data/workspace/human-source-mix-v1`, using pinned public dataset revisions in `collection-sources.json`. It does not run models, generate synthetic examples, or start training.

This first pass is a **partial, quarantined candidate release**, not the finished 100,000-passage human training pool. All records have `training_eligible=false` and `admission_status=quarantined_candidate`. The original plan remains unchanged; unsupported sources, restricted or unresolved rights, and insufficient candidate supply leave explicit shortfalls. There are no training, calibration, or test partitions yet.

## What the pipeline does

- Preserves the proposed category and source quotas as maximum candidate counts.
- Scans a deterministic, bounded selection of pinned source shards. This is not uniform sampling over the complete source repositories.
- Checks the recorded license against the intake allowlist and rejects clearly post-2021 dates. An old claimed date does not establish the age of the exact text version.
- Samples untouched contiguous paragraphs across the planned word-length bins, retaining original text and character offsets. It does not paraphrase or repair prose.
- Excludes the 100 known PG-19 test book IDs/title strings from Gutenberg intake; further benchmark and family-overlap checks remain mandatory.
- Removes exact normalized passage duplicates across the collected sources.
- Writes source-level scan counts, rejection reasons, errors, remaining quotas, and per-file provenance.

## What is still required

Complete historical-version and record-level rights checks; implement the remaining source-specific collection and extraction routes; review genre, language, and extraction fidelity; build the full protected-data exclusion index; perform near-duplicate and author/family grouping; audit intake samples; assign family-level partitions; then produce an admitted dataset release. The candidate configuration must not be substituted for an approved human pool.

## Running another collection

On the training Space, use a new persistent output directory for a changed configuration:

```bash
python collect_pool.py --config-dir /data/workspace/human-source-mix-v1/pipeline \
  --out /data/workspace/human-source-mix-v1/intake --workers 3 --max-files 6 --max-rows 5000
```

Completed source scans can be reused when rerunning the identical configuration. Incomplete source outputs are intentionally not truncated or automatically restarted: inspect the status and preserve the partial results before recovery. A file lock prevents two collectors writing into the same output directory.

`build_plan.py --size N` changes planned quotas. Supply the resulting plan in the configuration directory and use a new output directory. The chart is a separate planning snapshot and does not automatically reflect collection progress.

## Publishing

`upload_pool.py` validates every passage against the preserved original, confirms source quota bounds, checks hashes and duplicate passage identities, and refuses a public destination. Its `upload(base, repo_id, token)` function uses a credential supplied in memory. Do not store credentials in source files or manifests.

The private repository includes Parquet candidate shards, compressed original records with attribution, the original design, pinned collection sources, pipeline code, and a file manifest. The uploader downloads files at the resulting commit and verifies their SHA-256 checksums. `upload-receipt.json` records the exact repository revision after successful verification; `collection-summary.json` records candidate counts and shortfalls.

Local validation:

```bash
python test_collect_pool.py
python test_upload_pool.py
```

## Verified first release

Private dataset: https://huggingface.co/datasets/open-text-detector/human-source-mix-v1

Commit: `ab9da452953e6ee4999cfac037c20a68b4804b59`. Contains 34,206 candidate passages from 14 source slices, with 65,794 unfilled quota slots and zero admitted passages. 44 payload files were downloaded at this commit and verified by SHA-256. Total payload: 277,896,595 bytes. See `collection-summary.json` and `upload-receipt.json` for machine-readable records.

## Per-record source tags

Every uploaded candidate row includes `category` (one of the nine broad source categories) and `source_id` (the specific source slice, such as `arxiv` or `gutenberg`). `source_dataset`, `source_revision`, `source_file`, `source_row`, and `source_url` provide more detailed provenance. The source registry defines the category/source mapping. Mirror records inherit `category` and `source_id` and retain `source_record_id` for linking to the human parent. These are source-derived category tags; per-document genre review remains part of admission.

Verified current HF revision ab9da452953e6ee4999cfac037c20a68b4804b59: all 34,206 rows have nonempty category and source tags. Checked 289 mirror records on the Space: zero missing tags. The uploader now rejects missing categories and source/category mismatches against the registry, in addition to existing provenance checks. New human collection and synthetic mirrors have not yet been published as an updated release.
