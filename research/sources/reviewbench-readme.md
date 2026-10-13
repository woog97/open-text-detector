---
license: cc-by-4.0
task_categories:
- text-generation
- text-classification
- summarization
- question-answering
language:
- en
tags:
- peer-review
- openreview
- scientific-papers
- nlp
- benchmarking
- meta-science
pretty_name: ReviewBench — A Multi-Conference Peer-Review Corpus
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: neurips
    path: data/neurips-*
  - split: iclr
    path: data/iclr-*
  - split: icml
    path: data/icml-*
  - split: tmlr
    path: data/tmlr-*
  - split: emnlp
    path: data/emnlp-*
  - split: corl
    path: data/corl-*
  - split: colm
    path: data/colm-*
dataset_info:
  features:
  - name: forum_id
    dtype: string
  - name: conference
    dtype: string
  - name: year
    dtype: int32
  - name: track
    dtype: string
  - name: venue_id
    dtype: string
  - name: paper_number
    dtype: int32
  - name: title
    dtype: string
  - name: abstract
    dtype: string
  - name: authors
    list: string
  - name: keywords
    list: string
  - name: tldr
    dtype: string
  - name: primary_area
    dtype: string
  - name: venue
    dtype: string
  - name: decision
    dtype: string
  - name: decision_comment
    dtype: string
  - name: author_rebuttal
    dtype: string
  - name: num_reviews
    dtype: int32
  - name: reviews_json
    dtype: string
  - name: markdown
    dtype: string
  - name: markdown_chars
    dtype: int64
  splits:
  - name: colm
    num_bytes: 69559593
    num_examples: 717
  - name: corl
    num_bytes: 55415874
    num_examples: 813
  - name: emnlp
    num_bytes: 175864122
    num_examples: 2009
  - name: iclr
    num_bytes: 2043218314
    num_examples: 22532
  - name: icml
    num_bytes: 387957515
    num_examples: 3257
  - name: neurips
    num_bytes: 2023212137
    num_examples: 18453
  - name: tmlr
    num_bytes: 400634995
    num_examples: 3748
  download_size: 2728439622
  dataset_size: 5155862550
---

# ReviewBench

A large, multi-conference corpus of **peer-reviewed papers + their reviews + author rebuttals + acceptance decisions**, harvested from [OpenReview](https://openreview.net) and aligned with **OCR'd full-text markdown** of every paper.

- **51,529** papers
- **196,099** reviews
- **558,785** OCR'd PDF pages (markdown inlined per row)
- **7 conferences**, **22 venue/year combinations**, **2020 – 2026**

```python
from datasets import load_dataset
ds = load_dataset("/reviewbench")
print(ds)
# DatasetDict({
#   neurips: Dataset(num_rows=...)
#   iclr:    Dataset(num_rows=...)
#   icml:    Dataset(num_rows=...)
#   tmlr:    Dataset(num_rows=...)
#   emnlp:   Dataset(num_rows=...)
#   corl:    Dataset(num_rows=...)
#   colm:    Dataset(num_rows=...)
# })
```

## Coverage

One **split per conference family**. Within a split, filter by `year`, `venue_id`, or `track` for slicing.

| Split   | Venues / years                                                                      | Tracks                                  | ≈ Papers |
|---------|--------------------------------------------------------------------------------------|-----------------------------------------|----------|
| neurips | 2021, 2022, 2023, 2023 D&B, 2024, 2025                                               | main + Datasets & Benchmarks (2023)     | ~18,400  |
| iclr    | 2020, 2021, 2022, 2023, 2024, 2025, 2026                                             | main                                    | ~22,500  |
| icml    | 2025                                                                                 | main                                    | ~3,300   |
| tmlr    | rolling (all accepted papers as of April 2026)                                       | main                                    | ~3,750   |
| emnlp   | 2023                                                                                 | main + Findings                         | ~2,000   |
| corl    | 2021, 2022, 2023, 2024                                                               | main                                    | ~820     |
| colm    | 2024, 2025                                                                           | main                                    | ~720     |

NeurIPS 2020 and earlier used CMT and have no public OpenReview reviews.

## Schema

Each row is one paper.

| Column            | Type            | Notes                                                                   |
|-------------------|-----------------|-------------------------------------------------------------------------|
| `forum_id`        | `string`        | OpenReview forum ID — primary key                                       |
| `conference`      | `string`        | `neurips` / `iclr` / `icml` / `tmlr` / `emnlp` / `corl` / `colm`        |
| `year`            | `int32`         | Conference year                                                         |
| `track`           | `string`        | `main` / `datasets_and_benchmarks` / `findings` / etc.                  |
| `venue_id`        | `string`        | OpenReview venue ID, e.g. `NeurIPS.cc/2024/Conference`                  |
| `paper_number`    | `int32`         | Submission number (nullable)                                            |
| `title`           | `string`        |                                                                         |
| `abstract`        | `string`        |                                                                         |
| `authors`         | `list<string>`  |                                                                         |
| `keywords`        | `list<string>`  |                                                                         |
| `tldr`            | `string`        |                                                                         |
| `primary_area`    | `string`        |                                                                         |
| `venue`           | `string`        | Final venue string, e.g. `"NeurIPS 2024 poster"`                        |
| `decision`        | `string`        | `Accept (poster)`, `Accept (oral)`, `Reject`, etc.                      |
| `decision_comment`| `string`        | Area-chair meta-review                                                  |
| `author_rebuttal` | `string`        | General rebuttal (≤2024); empty when per-reviewer rebuttals are used    |
| `num_reviews`     | `int32`         | Convenience count                                                       |
| `reviews_json`    | `string`        | All reviews as a JSON string — see schema below                         |
| `markdown`        | `string`        | OCR'd full text of the PDF (see *OCR* below); `""` if PDF unavailable    |
| `markdown_chars`  | `int64`         | `len(markdown)` for fast filtering                                      |

### Decoding `reviews_json`

`reviews_json` is `json.dumps(list[dict])`. Decode with:

```python
import json
df = ds["neurips"].to_pandas()
df["reviews"] = df["reviews_json"].map(json.loads)
print(df.iloc[0]["reviews"][0].keys())
```

Each review dict is a **union over all forms used by all venues across all years**, with absent fields as empty strings / `None`. The most reliably populated fields are:

| Field             | Where populated                                               |
|-------------------|---------------------------------------------------------------|
| `review_id`, `reviewer`, `rating`, `confidence`, `rebuttal` | All venues |
| `summary`, `questions`, `limitations`, `strengths`, `weaknesses` | NeurIPS 2022–24, ICLR, ICML, CoRL, COLM |
| `soundness`, `presentation`, `contribution`                | NeurIPS 2022–24, ICLR, ICML                |
| `quality`, `clarity`, `significance`, `originality`        | NeurIPS 2025, TMLR                         |
| `strengths_and_weaknesses`                                 | NeurIPS 2025 (merged form)                 |
| `was_revised`, `final_justification`                       | NeurIPS 2025 (in-place review revisions)   |
| `claims_and_evidence`, `theoretical_claims`, `experimental_designs_or_analyses`, `relation_to_broader_scientific_literature`, `essential_references_not_discussed` | TMLR |
| `paper_topic_and_main_contributions`, `reasons_to_accept`, `reasons_to_reject`, `excitement`, `reproducibility`, `ethical_concerns` | EMNLP 2023 |
| `summary_of_paper`, `summary_of_recommendation`, `technical_quality`, `clarity_of_presentation`, `potential_impact`, `robotics_focus` | CoRL |
| `extra_scores`, `extra_text` | dicts capturing any venue-specific fields not in the union |

Schema differences in detail:

- **NeurIPS 2021 D&B (track)**: older form, most numeric sub-scores absent; lives in the `neurips` split.
- **NeurIPS 2022–2024**: `soundness/presentation/contribution`; separate `strengths`/`weaknesses`; single general `author_rebuttal`.
- **NeurIPS 2025**: `quality/clarity/significance/originality`; merged `strengths_and_weaknesses`; per-reviewer rebuttals; in-place review revisions tracked via `was_revised` + `final_justification`.
- **ICLR 2020–2026**: NeurIPS-style `soundness/presentation/contribution`; per-reviewer rebuttals.
- **ICML 2025**: similar to NeurIPS-style; some venue-specific fields (e.g. `technical_quality`, `novelty`) live in `extra_scores`.
- **TMLR**: claim-evidence-style structured review; rolling acceptance.
- **EMNLP 2023**: ARR-style review form with `reasons_to_accept`/`reasons_to_reject`/`excitement`/`reproducibility`.
- **CoRL**: robotics-focused review form.
- **COLM**: language-modeling-focused review form.

## Source and collection

- **Source**: [OpenReview](https://openreview.net) — main and any track-level conferences (e.g. NeurIPS Datasets & Benchmarks).
- **Collected**: April 2026 via the [OpenReview Python API](https://github.com/openreview/openreview-py) (a mix of `openreview.api` v2 and legacy v1 for older NeurIPS/ICLR years).
- **Scraping pipeline**: parallel Modal workers (100 containers, single-token reuse to bypass the 3-req/min rate limit). Each forum was fetched with all official reviews, official comments, decision, and author rebuttals; PDFs were downloaded to a Modal volume.

## Markdown / OCR

- **Engine**: [`nvidia/nemotron-ocr-v2`](https://huggingface.co/nvidia/nemotron-ocr-v2)
- **Compute**: 10× NVIDIA L40S GPUs in parallel on Modal (~6 h wall-clock end-to-end)
- **Throughput**: ~7,200 PDFs/hour aggregate; mean **7.3 s/PDF** per worker; **558,785** pages processed
- **Failures**: 1 PDF errored out during OCR; ~76 PDFs were unavailable from OpenReview at scrape time and have `markdown == ""`. Use `markdown_chars > 0` to filter.

### OCR quality (spot-checked across NeurIPS, ICLR, ICML, TMLR, CoRL, COLM)

What works well:

- Body prose, abstracts, section headers, paragraph structure
- In-line citations like `(Author et al., YEAR)` mostly preserved
- Equations rendered linearly (variable names + structure visible)
- Page boundaries marked with `\n\n---\n\n` separator

Recurring artifacts to be aware of (consistent across venues, low impact for most NLP tasks):

- Email addresses and URLs containing repeated chars (e.g. `name@@@cmu.eed`, `https:////aaaaaaa`)
- Occasional word-doubling at line breaks (`decision-decision-making`, `Complex-Valuee Valued`)
- Citation lists missing semicolons (`(Singer 2007 Uhlhaas et al. 2009)`)
- Greek letters and super/sub-scripts often dropped or flattened
- First-page logo/header text occasionally bleeds into title (`git Cooperative …`)
- Figure caption tokens interleave with body text on figure-heavy pages

**Practical impact**: fine for dense retrieval, language modeling, review-grounding, summarization. **Not** suitable for tasks requiring exact equations or canonical citation strings.

## Suggested uses

- Train review-quality classifiers / score predictors
- Study reviewer agreement, rebuttal effectiveness, decision dynamics
- Build retrieval-augmented or grounded scientific-paper assistants
- Meta-research on the peer-review process across venues and years
- Few-shot / RAG benchmarks that require the paper full text + the reviews

## Related work

This dataset was assembled in support of an ICML 2026 Datasets and Benchmarks-track submission introducing **ReviewBench**. Citation will be updated upon publication.



## License

Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Paper full text and review text remain the intellectual property of their respective authors; this dataset redistributes them for non-commercial research consistent with OpenReview's public-access policy. If you are an author and would like content removed, please open an issue on the dataset repository.

## Acknowledgements

- The [OpenReview](https://openreview.net) team for keeping the peer-review record open.
- NVIDIA for releasing [nemotron-ocr-v2](https://huggingface.co/nvidia/nemotron-ocr-v2).
- [Modal](https://modal.com) for the GPU and storage infrastructure used to assemble this corpus.
