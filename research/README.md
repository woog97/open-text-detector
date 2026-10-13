# Baseline collection

For the current cross-source inventory, see [Training data inventory](TRAINING_DATA_INVENTORY.md), audited October 3, 2026. It includes the new 600 manuscripts, source counts and token estimates, existing splits, storage locations, overlap, and the filtered-mirror recovery gap.

For the October 1, 2026 Pangram-style human-source distribution research, see [curated source distribution](human-source-pool/README.md). It contains source allocations and admission rules, not an assembled training corpus; the baseline collections below remain unchanged.

Collected 22 September 2026. All source revisions, selection rules and file checksums are recorded in each collection's `manifest.json`. No private account data was used.

| Collection | Records | Usable JSONL | Including downloaded source and manifest |
|---|---:|---:|---:|
| ICLR 2023 accepted papers | 100 full papers | 8.33 MB | 8.33 MB final collection |
| ICLR 2026 accepted papers | 100 full papers | 9.07 MB | 9.07 MB final collection |
| Human: PG-19 test split | 100 complete books | 41.50 MB | 66.38 MB |
| AI: MDTA, 2025 generators only | 6,397 responses | 9.69 MB | 39.07 MB |

MB means 1,000,000 bytes. Both additional Hugging Face sources are below 100 MB **even counting retained raw downloads plus normalized output**. The ICLR download cache is approximately 1.09 GB and is separate from those two collections.

## ICLR year cohorts

Files: `data/iclr_2023/papers.jsonl` and `data/iclr_2026/papers.jsonl`.

The [official 2023 program](https://iclr.cc/virtual/2023/papers.html) and [official 2026 program](https://iclr.cc/virtual/2026/papers.html) were scraped for paper titles and poster links. Full paper text comes from [Samarth0710/reviewbench](https://huggingface.co/datasets/Samarth0710/reviewbench), revision `7d1b399bd7297318a2d6284b5349326166702530`, a public archive of OpenReview papers with OCR markdown. OpenReview's API and original PDF downloads required a browser challenge; that restriction was not bypassed. **These are full archived OCR texts, not downloaded original PDFs.** Each record includes its original OpenReview/PDF URLs and official conference page.

Eligible records match the official program after punctuation/case normalization, match the year, and contain at least 1,000 words. There were 1,571 eligible 2023 papers and 5,344 eligible 2026 papers. Select the 100 lowest SHA-256 values of `seed + forum_id`, with seed `pangram-baseline-2026-09-22-v1`. No detector scores influence selection. Full text, abstracts, titles, authors, forum IDs, acceptance metadata and source revision are retained. Peer reviews are not included in the exported papers.

2023 is labeled `historical_human_proxy`; 2026 is `contemporary_comparison`. Both have `label: null` and `authorship_verified: false`. Paper year is not evidence of AI authorship. The 2023 archive may contain revisions written after ChatGPT's launch; 2026 papers can be wholly human-written. Neither cohort supports a false-positive/true-positive calculation without verified authorship labels. OCR, equations, references, appendices, topic changes and paper length can affect scores. The archive may contain AI-assisted OCR/transcription artifacts even where the original prose predates LLMs.

License: ReviewBench states CC-BY-4.0; underlying authors retain rights in their papers.

## Human baseline: historical books

File: `data/human_pg19/books.jsonl`.

The entire 100-book test split of [emozilla/pg19](https://huggingface.co/datasets/emozilla/pg19), revision `c021754c8e01c5b1cc83a1f549c1f97fbbb756b8`, a Parquet distribution of [DeepMind PG-19](https://huggingface.co/datasets/deepmind/pg19). Publication dates are before 1919. This established literary corpus provides a strong human-authored baseline by publication provenance. Original titles, dates and Gutenberg URLs are retained. No train shards were downloaded.

The compilation is Apache-2.0; underlying book rights depend on jurisdiction. Text may contain editorial/transcription artifacts. Historical literary prose differs substantially from modern academic prose. A low false-positive rate here would not establish one for ICLR papers. Whole books can exceed the app's input limit: select passages or use a separate batch evaluation pipeline.

## AI baseline: models released in 2025

File: `data/ai_mdta_2025/responses.jsonl`.

Source: [nsp909/MDTA](https://huggingface.co/datasets/nsp909/MDTA), revision `fffb86b767e23367d09d27b32d34c3c586cf5219`, CC-BY-SA-4.0. Domains: `open_qa` and `wiki_csai`.

- 2,058 responses attributed to Gemma 3 12B, [released 12 March 2025](https://blog.google/technology/developers/gemma-3/).
- 4,339 responses attributed to Qwen2.5-VL 7B, [released 28 January 2025](https://qwenlm.github.io/blog/qwen2.5-vl/).

Only the standard `model_responses` field is used, at the three provided temperatures. Older model families, human answers and adversarial rewrites are excluded. Require at least 50 whitespace-delimited words and deduplicate exact text hashes. Preserve prompts, question IDs, generator names, temperature, source field and release evidence. Generator identity is supplied by the dataset publisher, not independently authenticated generation logs. The corpus may include refusals. Near-duplicate answers across temperatures share a question: **split evaluation by question**, not response row.

## Reproduce and inspect

From the workspace root, after installing backend extras:

```sh
cd backend
uv sync --frozen --extra models --extra research
cd ..
backend/.venv/bin/python scripts/collect_baselines.py all
backend/.venv/bin/python scripts/validate_baselines.py
```

The collector also accepts `iclr`, `human`, or `ai`. Downloads are cached; output is deterministic for the recorded inputs. Dataset cards, official program HTML and metadata are in `sources/`. Large data and model files are excluded from source control.

`validation.json` records counts, checksums, uniqueness and size checks. `live-verification.json` contains four setup checks through the real API, including a three-chunk paper. Inputs were short excerpts except the longer setup check; these are **smoke tests, not a benchmark**. No model thresholds were calibrated or trained on these collections. These datasets are not inserted into the app's plagiarism reference corpus.
