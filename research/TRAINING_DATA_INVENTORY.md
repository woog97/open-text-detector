# Training data inventory

Audited October 3, 2026, Pacific time. This inventory covers local project files, live Hugging Face training-Space storage, the project’s Hugging Face dataset repositories, and a shallow search of the user’s own directory on the shared server. It records what exists, which versions overlap, where the data live, and what is actually assigned to training. The initial audit did not change data. The subsequently requested mirror publication and paper transfer are recorded below; no generation, training or split changes were made.

**The four main custom collections contain about 79.1 million stored-text tokens across 150,600 records:** 100,000 human passages, 30,000 raw mirrors, 20,000 paper target paragraphs, and 600 full manuscripts. This is a gross inventory estimate, not a deduplicated or training-ready total. The 28,120 accepted mirrors are a subset of the 30,000, and the paper collection includes held-out splits. Larger public corpora and unlabeled paper archives are listed separately.

## Changes since the earlier count

- The new **600 full manuscripts add about 4.43M tokens**, including their human-supplied title/abstract sections. Final revision models: 196 GPT-6.1-Sol, 200 GPT-6-Sol, 204 GPT-6-Luna. All 600 finals are present in the consolidated local package and private HF release.
- **Human usage approval now covers all 100,000 passages.** The latest published revision includes that approval. Earlier zero-admission/audit fields remain historical evidence; they do not negate the user’s later usage decision.
- **Mirrors finished at 30,000 raw documents**, approximately 15.65M tokens. Later automated checks retained **28,120** and rejected **1,880**. These accepted records still need final family/split preparation.
- **Arena now contains 5,000 responses from 50 models**, not 4,900/49. The repository and local folder names retain the old number.
- GRADTEX is accepted as-is for the current plan. HIP is set aside. The proposed half/half paper split and its trial arms were cancelled.

## Current custom collections

| Collection | Records | Text tokens | Current use |
|---|---:|---:|---|
| Broad human source pool | 100,000 | 56.39M | User approved all passages for usage; final family/split preparation separate |
| Luna synthetic mirrors raw | 30,000 | 15.65M | Complete raw collection; filtered subset recorded separately |
| Luna mirrors accepted by automated checks | 28,120 | Not measured | Recovered byte-for-byte and published privately; final family/split preparation remains |
| Paper paragraph reconstruction pairs | 20,000 | 2.59M | Established train/validation/test split retained |
| Synthetic full research manuscripts | 600 | 4.43M | All finals included; train is storage convention, no leakage-controlled split |

The 20,000 paragraph rows are **10,000 AI targets plus 10,000 human originals**, not 20,000 AI generations. Preserve 12,000 train / 4,000 validation / 4,000 test. The target-only view is approximately 2.59M tokens; the context-preserving passage view contains exactly 8,063,003 reference tokens, including repeated surrounding human text. These are alternate views, not additive datasets. Only the training target view is about 1.57M tokens.

The current backbone-prepared training table is a derivative: **26,731 rows = 3,146 paired + 23,585 novel-human rows**. Selection has 4,459 rows and calibration 4,484. Its 6,000 stage-1 and three 12,000 stage-2 draw lists are repeated training exposure, not extra unique data. Quality filtering and novel-body extraction mean this derivative is not identical to the entire 12,000-row original training split.

## Downloaded public and copied editing corpora

| Collection | Stored records | Approximate text tokens | Split or prepared subset |
|---|---:|---:|---|
| RAID downloaded train none | 467,985 | 142.53M | train: 467,985 |
| MAGE downloaded splits | 432,682 | 103.85M | train: 319,071, valid: 56,792, test: 56,819 |
| GRADTEX | 142,214 | 38.70M | train: 79,065, validation: 23,672, test: 39,477 |
| SeqXGPT original English files | 35,904 | 9.86M | train_candidates: 28,730, validation: 3,326, test: 3,680, excluded: 168 |
| HIP human and paraphrase pairs | 10,581 | 5.21M | 10,581 human/paraphrase pairs; set aside |
| EditLens ICLR | 80,610 | 33.71M | test: 6,115, test_enron: 6,147, test_llama: 5,957, train: 59,991, val: 2,400 |
| EditLens Grammarly edits | 1,768 | 1.13M | train: 1,768 |

Historical audited donor pools: **RAID 39,858**, **MAGE 26,232**, **GRADTEX 68,647**, and **SeqXGPT 26,517**. These pools are contained in their source downloads. GRADTEX derives from MAGE; HIP derives from MAGE and RAID. Summing their raw rows/tokens would count related source material repeatedly.

GRADTEX’s training split has 39,532 human and 39,533 AI-involved rows. Its native binary label uses 1 for human and 0 for AI-involved. Editing/scenario/document labels are not token-span gold. The historical 68,647-row donor preparation imposed experiment-specific length/overlap filters; it does not override the user’s later as-is acceptance. HIP remains excluded from the current plan despite retained prepared files.

## Evaluation and source-only collections

| Collection | Size | Role |
|---|---:|---|
| Arena | 5,000 responses, 100 prompts, 50 models | Development-exposed test; 4,898 mechanically eligible; prompt-group split required before any training reuse |
| Paper workflows | 1,323 distinct outputs from 189 papers | 756 reconstructions + 567 edits; pilot/calibration/test only; 2,646 correlated views |
| Workflow human controls | 9,301 body paragraphs; 1,590 clean/novel | Controls related to those 189 papers, not new independent paper sources |
| PG19 | 100 full books; 10.30M stored tokens | Downloaded test split; legacy human controls and excerpt proxies |
| MDTA 2025 subset | 6,397 responses; about 1.30M tokens | Legacy AI evaluation, two generator families |
| Canonical extracted paper archive | 14,561 papers; about 423M tokens | Unlabeled source text, overlapping paper datasets; train filename is a storage convention |
| ReviewBench | 51,529 records; 51,453 with manuscript Markdown | 15 downloaded shards with manuscripts/reviews/metadata; no detector authorship gold |
| ICLR year controls | 100 from 2023 + 100 from 2026; 4.71M stored tokens | Historical controls; publication year does not prove human authorship |
| OpenReview catalogues and abstract seed sets | Metadata and 600 selected title/abstract seeds | Source discovery and generation inputs, not extra synthetic documents |

The frozen detector suite packages overlapping profiles: comparison 13,751 rows, workflow 8,552, assistance 972, full 65,348, plus contextual controls and separate validation files. Do not add those profiles to each other or to their source data. Prepared benchmark subsets currently on disk are:

| Prepared source | Rows | Exact unique texts within file | Source eligible pool in saved importer manifest |
|---|---:|---:|---:|
| arena20 | 351 | 351 | Not recorded |
| detectrl | 2,477 | 2,430 | 48404 |
| ellipse | 2,571 | 2,571 | Not recorded |
| epoch | 84 | 84 | 1089 |
| gede | 184 | 184 | 13217 |
| liang | 270 | 269 | 270 |
| local | 85 | 85 | Not recorded |
| meld_eval | 928 | 926 | 227998 |
| opai | 224 | 220 | 8748 |
| pelic | 100 | 100 | 18193 |
| perkins | 114 | 113 | 114 |
| saha | 65 | 65 | 72 |
| sem_detect | 36 | 36 | 1109 |
| vub | 40 | 40 | 40 |

These prepared counts are bounded evaluation selections, not total upstream dataset sizes. Sources include DetectRL, ELLIPSE, Epoch, GEDE, Liang, MELD-eval, OpAI, PELIC, Perkins, Saha, Sem-Detect and VUB. Their publisher receipts and selection rules are in [benchmark coverage](../benchmarks/otd/COVERAGE.md); retained native labels are not automatically token labels.

## Source breakdown for the broad human pool and mirrors

Human counts come from the final 100,000-row distribution report; mirror counts were read from the recovered final 30,000-row export. Accepted-mirror counts are available by category, not source, in the saved filter report.

| Source | Category | Human passages | Raw mirrors |
|---|---|---:|---:|
| gutenberg | creative | 12,551 | 3,609 |
| writingprompts | creative | 6,667 | 1,941 |
| scp | creative | 1,893 | 585 |
| wikisource | creative | 1,111 | 357 |
| pmc | scientific | 6,306 | 1,831 |
| arxiv | scientific | 5,405 | 1,586 |
| pes2o | scientific | 3,604 | 1,075 |
| acl | scientific | 2,703 | 814 |
| wikipedia | reference | 7,958 | 2,317 |
| pressbooks | reference | 3,979 | 1,178 |
| libretexts | reference | 2,387 | 725 |
| wikibooks | reference | 1,592 | 499 |
| amazon2018 | reviews | 5,306 | 1,556 |
| imdb | reviews | 1,061 | 348 |
| opinrank | reviews | 4,244 | 1,256 |
| stack_nontech | social | 5,726 | 1,680 |
| stack_tech | social | 2,082 | 641 |
| ubuntu_irc | social | 1,561 | 489 |
| wiki_talk | social | 1,041 | 343 |
| cccc | general_web | 5,017 | 1,473 |
| eff | general_web | 415 | 164 |
| foodista | general_web | 1,545 | 487 |
| wikivoyage | general_web | 1,231 | 395 |
| globalvoices | news | 1,652 | 516 |
| voa | news | 1,717 | 536 |
| wikinews | news | 991 | 329 |
| scidev | news | 926 | 305 |
| oanc_slate | news | 1,321 | 416 |
| asap2 | essays | 4,206 | 1,241 |
| persuade | essays | 699 | 241 |
| govreport | professional | 1,086 | 357 |
| fed | professional | 776 | 267 |
| worldbank | professional | 776 | 265 |
| oanc_icic | professional | 465 | 178 |
| **Total** | | **100,000** | **30,000** |

The planned ELLIPSE contribution to this pool is zero; the separate 2,571-essay ELLIPSE evaluation set is preserved. All 34 nonempty human-source groups have user usage approval. Preserve their origin/permission audit findings, source attribution, and existing protected-family metadata; usage approval and a completed leakage-controlled split are different records. No multilingual collection was added.

## Copies and historical versions

- The v3 250-pair pilot and 2,500-pair expansion are wholly contained in the final 10,000-pair paper corpus by exact passage-text hashes: 500/500 and 5,000/5,000 rows respectively. The old `ai-paper-provenance-v3` export is a smaller view/version, not an additional 5,000 rows.
- Other 50/250-target v1/v2/v4 pilots and the 10/50-paper editing pilots contain additional historical outputs, but are development-exposed and share sources. They are documented below and excluded from the main custom total; their rows are not independent new paper families.
- A separate Qwen3.5-4B mirror pilot contains 10 outputs, 9 mechanical passes and 0 admitted; it is preserved outside the Luna 30k total.
- Original Arena20, Arena100, cheap, skipped and Kimi runs contain responses reused by the final 5,000-response package. Parquet/JSONL/ZIP copies, manuscript drafts/revisions, detector predictions, API attempt logs, epoch draw lists and storage snapshots are not added to the corpus total.
- Human-pool v1/v2/recovery checkpoints and HF releases describe stages or copies of the same current 100,000-passage collection. Earlier Luna-dollar mirror runs and the 512-output throughput benchmark are already included in the 30,000 raw mirrors.

| Historical local paper run | Stored rows | Rows exactly present in current 20,000 | Unique passages outside current version |
|---|---:|---:|---:|
| `paper-gap250-luna-v2-20260929` | 500 | 250 | 250 |
| `paper-gap250-luna-v3-20260930` | 500 | 500 | 0 |
| `paper-gap250-luna-v4-20260930` | 500 | 253 | 247 |
| `paper-gap2500-v3-luna-20260930` | 5,000 | 5,000 | 0 |
| `paper-gap50-luna-20260929` | 100 | 50 | 50 |
| `paper-gap50-luna-v2-20260929` | 100 | 50 | 50 |
| `paper-luna50-20260929` | 1,000 | 0 | 947 |
| `paper-luna50-quality-v2-20260929` | 1,000 | 0 | 824 |
| `paper-pilot10-gpt61-sol-20260929` | 200 | 0 | 192 |

“Outside current version” is only an exact-text comparison with that corpus, not cross-pilot deduplication or approval for training.

## Storage and upload status

**Paper organization release:** [open-text-detector/paper-text](https://huggingface.co/datasets/open-text-detector/paper-text) is private and accessible to organization members, with no restricted resource group. The [interactive sample table](https://huggingface.co/datasets/open-text-detector/paper-text/viewer/default/train) was verified at the first and last rows: 14,561 records and all 12 columns. Original `woog/open-text-detector-paper-text` remains private and unchanged. The Space attachment now uses the organization release.

**Named dataset attachments:** both repositories are attached directly to `open-text-detector/training` as read-only dataset volumes: `open-text-detector/synthetic-mirrors-luna-28120` at `/datasets/synthetic-mirrors-luna-28120` (28,120 rows) and `open-text-detector/paper-text` at `/datasets/pangram-paper-text` (14,561 rows). Both mounted Parquet files were opened and row counts verified after activation. These are the canonical Space access paths; bucket copies remain as preserved recovery material. [Attachment verification](synthetic-mirrors/publication-20261003/dataset-attachments.json).

**Recovery completed:** the old temporary working folders were absent. All 12 raw archives were restored and the frozen filter reproduced the accepted, rejected, decision and protocol files byte-for-byte. The 28,120 accepted mirrors are now published privately at [open-text-detector/synthetic-mirrors-luna-28120](https://huggingface.co/datasets/open-text-detector/synthetic-mirrors-luna-28120), revision `ca5d52df1a20f9820ee2aaff83230542e69688a8`, and stored at `/data/workspace/datasets/synthetic-mirrors-luna-28120`. Exact original exports and checks remain at `/data/workspace/dataset-publication-20261003/recovered-filter`. The private paper archive has been copied to `/data/workspace/datasets/pangram-paper-text` and all 14,561 rows verified against the pinned source revision `a79fcf1ffcb45a20bf6c9728142d0fa974304e95`. These transfers followed the inventory and were explicitly requested by the user.

| Data | Current destination | Audit and subsequent requested transfers |
|---|---|---|
| Human 100k | Private `open-text-detector/human-source-mix-v1` | Already published; current revision verified by metadata |
| Full manuscripts 600 | Private `open-text-detector/synthetic-research-papers-600` | Already published; current revision verified by metadata |
| Paper 20k, Arena 5k, workflow evaluation | Existing `woog` dataset releases | Already published; current revision verified by metadata |
| EditLens + Grammarly | Existing private `open-text-detector` dataset copies | Already published and present on Space |
| Raw mirrors 30k | Private training-storage bucket, mounted under `/data/workspace/.../raw-30000-20261002/checkpoints` | Already backed up; export read during this audit |
| Accepted mirrors 28,120 | Private `open-text-detector/synthetic-mirrors-luna-28120` and persistent Space copy | Published; all file hashes and row count verified by remote readback |
| Paper archive 14,561 | `/data/workspace/datasets/pangram-paper-text` | Copied from private `woog/open-text-detector-paper-text`; pinned files and rows verified |
| RAID/MAGE/GRADTEX/SeqXGPT/HIP | Existing Space dataset files | No new publication planned |

Standing approval recorded October 3 in [AGENTS.md](../AGENTS.md) now allows all data transfers **to the existing training Space**. It does not itself schedule uploads, change dataset-sharing permissions, or authorize other destinations.

## Canonical source locations and decisions

### Broad human source pool

User approved all passages for usage; final family/split preparation separate.

- [https://huggingface.co/datasets/open-text-detector/human-source-mix-v1](https://huggingface.co/datasets/open-text-detector/human-source-mix-v1)
- `/data/workspace/human-source-mix-v2-recovered-20261002`

- 34 sources; nine categories; latest HF revision da73a03fdfa863d5c44418e988d50137bcb53c47
- Usage approval does not rewrite historical audit results or metadata fields. Original texts unchanged.
- Former approved-collection.sqlite3 temporary path absent at audit; use approved HF release.

### Luna synthetic mirrors raw

Complete raw collection; filtered subset recorded separately.

- `/data/workspace/synthetic-mirrors-luna-dollar-v1-recovered-20261002/raw-30000-20261002/checkpoints/00012-1790984912.tar.gz`

- Final archive streamed read-only and contained 30,000 export rows.
- Full recovery of calls/per-document files uses all 12 incremental archives; final archive contains complete raw-documents.jsonl.
- Do not add benchmark 512, prior 10k/20k exports or initial Luna-dollar outputs again.

### Luna mirrors accepted by automated checks

Recovered byte-for-byte and published privately; final family/split preparation remains.

- [open-text-detector/synthetic-mirrors-luna-28120](https://huggingface.co/datasets/open-text-detector/synthetic-mirrors-luna-28120); revision `ca5d52df1a20f9820ee2aaff83230542e69688a8`.
- Persistent Space files: `/data/workspace/datasets/synthetic-mirrors-luna-28120`.
- Original accepted/rejected evidence: `/data/workspace/dataset-publication-20261003/recovered-filter`.
- Explicit `authorship=synthetic`, `label=1`, and model metadata; join `source_record_id` to the human corpus.
- All 28,120 records and release checksums verified; no new model calls. Final family-grouped splits remain unassigned.

### Paper paragraph reconstruction pairs

Established train/validation/test split retained.

- [benchmarks/otd/exports/ai-paper-provenance-v3-10000](../benchmarks/otd/exports/ai-paper-provenance-v3-10000)
- [research/data/paper-gap10000-v3-luna-20260930](../research/data/paper-gap10000-v3-luna-20260930)
- [https://huggingface.co/datasets/woog/ai-paper-provenance-v3](https://huggingface.co/datasets/woog/ai-paper-provenance-v3)

- 10,000 generated target paragraphs plus10,000 matched human originals;2,000 papers.
- Target-only token estimate. Context passage count is exactly8,063,003 stored reference tokens, includes repeated context.
- Saved exact AI target-region tokens1,273,649. Historical eligibility flags retain6,375 passage rows / 2,739 AI targets; eligibility is not equivalent to training split.
- Fresh view excludes pilot exposures:11,700 train / 3,900 validation / 3,900 test.

### Synthetic full research manuscripts

All finals included; train is storage convention, no leakage-controlled split.

- [research/data/synthetic-papers-600-hf-20261003](../research/data/synthetic-papers-600-hf-20261003)
- [https://huggingface.co/datasets/open-text-detector/synthetic-research-papers-600](https://huggingface.co/datasets/open-text-detector/synthetic-research-papers-600)

- 196 GPT-6.1-Sol, 200 GPT-6-Sol, 204 GPT-6-Luna, labeled by final revision model.
- 600 unique final hashes and600 unique source IDs. Title/abstract are human-supplied; generated body is AI. Do not label entire Markdown as pure AI token gold.
- Includes synthetic methods/results; no experiments executed. Drafts/revisions/batch exports are copies or histories, not extra final documents.

### RAID downloaded train none

Downloaded; only audited donor subset used in diversity experiments.

- `/data/workspace/paper-diversity-v1/raid.csv`

- Historical eligible donor pool: 39,858 rows; not additive to raw source.
- RAID subset selects clean abstracts; MAGE subset selects human and topical/specified responses, excludes continuation setups and heldout overlap.
- Do not treat all downloaded rows as approved train-ready data.

### MAGE downloaded splits

Downloaded; only audited donor subset used in diversity experiments.

- `/data/workspace/paper-diversity-v1/mage-train.csv`
- `/data/workspace/paper-diversity-v1/mage-valid.csv`
- `/data/workspace/paper-diversity-v1/mage-test.csv`

- Historical eligible donor pool: 26,232 rows; not additive to raw source.
- RAID subset selects clean abstracts; MAGE subset selects human and topical/specified responses, excludes continuation setups and heldout overlap.
- Do not treat all downloaded rows as approved train-ready data.

### GRADTEX

Accepted by user as-is for current Stage 3 plan.

- `/data/workspace/paper-diversity-v1/public-source-audit/gradtex-train.parquet`
- `/data/workspace/paper-diversity-v1/public-source-audit/gradtex-validation.parquet`
- `/data/workspace/paper-diversity-v1/public-source-audit/gradtex-test.parquet`

- User explicitly requested no further deep review.
- 79,065train rows comprise39,532 human and39,533 AI-involved; upstream binary 1 = human, 0 = AI-involved.
- Derived from MAGE, with overlapping source material. Document/scenario labels do not establish token gold.
- Existing historical length/overlap-filtered donor pool 68,647 is a derivative, not additional data or a new restriction on user acceptance.

### SeqXGPT original English files

Prepared mixed-provenance training donor subset exists.

- `/data/workspace/paper-diversity-v1/public-source-audit/seqxgpt-original/en_gpt2_lines.jsonl`
- `/data/workspace/paper-diversity-v1/public-source-audit/seqxgpt-original/en_gpt3_lines.jsonl`
- `/data/workspace/paper-diversity-v1/public-source-audit/seqxgpt-original/en_gptj_lines.jsonl`
- `/data/workspace/paper-diversity-v1/public-source-audit/seqxgpt-original/en_gptneo_lines.jsonl`
- `/data/workspace/paper-diversity-v1/public-source-audit/seqxgpt-original/en_human_lines.jsonl`
- `/data/workspace/paper-diversity-v1/public-source-audit/seqxgpt-original/en_llama_lines.jsonl`

- 6,000 human plus 29,904 generated/continuation records across 5 generators. Current prepared eligible pool 26,517.
- Group split:28,730 train / 3,326 validation / 3,680 test / 168 excluded; subsequent donor checks further narrow training pool.
- Retain human prefix versus generated suffix boundary semantics; do not label prefixes AI.

### HIP human and paraphrase pairs

Set aside by user; preserve files.

- `/data/workspace/paper-diversity-v1/public-source-audit/hip-audit/train.parquet`
- `/data/workspace/paper-diversity-v1/public-source-audit/hip-audit/pair-preparation-grouped-v2`

- 10,581 pairs = 21,162 text sides, derived from MAGE/RAID; do not add originals again to those sources.
- Prepared grouped subset 5,862 train / 697 selection / 766 reserved-test pairs; native test prefix samples 256 AI + 256 human.
- No per-token provenance supplied for paraphrases.

### EditLens ICLR

Existing private copied dataset; original split semantics retained.

- `/data/workspace/open-pangram-private-copy/dataset/editlens_iclr`
- [https://huggingface.co/datasets/open-text-detector/editlens_iclr](https://huggingface.co/datasets/open-text-detector/editlens_iclr)

- Human, AI-generated and AI-edited document labels; source_text and similarity scores are metadata, not automatically gold token labels.
- Test_llama and test contain related human sources; totals are stored rows, not globally independent documents.

### EditLens Grammarly edits

Existing private copied editing dataset.

- `/data/workspace/open-pangram-private-copy/dataset/editlens_iclr_grammarly/data/train-00000-of-00001.parquet`
- [https://huggingface.co/datasets/open-text-detector/editlens_iclr_grammarly](https://huggingface.co/datasets/open-text-detector/editlens_iclr_grammarly)

- Original source_text, edit_instructions and edited text retained. Count covers edited text only.
- train filename is not evidence of completed project admission or token-label preparation.

### Arena multi-model responses

Development-exposed evaluation; future training needs prompt-group split.

- [benchmarks/otd/exports/arena-prose-100-49-models](../benchmarks/otd/exports/arena-prose-100-49-models)
- [https://huggingface.co/datasets/woog/arena-prose-100-49-models](https://huggingface.co/datasets/woog/arena-prose-100-49-models)

- 100 prompts × 50 models; 4,898 mechanically eligible. The 49-model folder name is stale.
- Earlier Arena20/100/cheap/skipped/Kimi exports contain responses reused in the current package; do not sum them. Additional historical outputs were not globally reconciled.

### Paper workflow evaluation

Evaluation-only; original assignments retained.

- [research/data/paper-eval-workflows-luna-20260930](../research/data/paper-eval-workflows-luna-20260930)
- [https://huggingface.co/datasets/woog/ai-paper-workflow-eval](https://huggingface.co/datasets/woog/ai-paper-workflow-eval)

- 189 papers: 27 pilot / 54 calibration / 108 test.
- 756 reconstruction targets + 567 assistance outputs; 2,646 correlated views + 378 matched-human views.
- 9,301 body paragraphs, 1,590 clean/novel human paragraphs are related controls, not independent generated documents.
- 567 editing outputs remain eval-only; cancelled half-split proposal must not be applied.
- Token estimate omitted because correlated full/context/target views have different counting units.

### PG19 historical books

Legacy evaluation/source pool.

- [research/data/human_pg19](../research/data/human_pg19)

- Already used in benchmark controls/proxies; no new training assignment inferred.

### MDTA selected 2025 responses

Legacy evaluation/source pool.

- [research/data/ai_mdta_2025](../research/data/ai_mdta_2025)

- Already used in benchmark controls/proxies; no new training assignment inferred.

### Canonical extracted paper archive

Unlabeled source archive, not human-authorship gold.

- [research/exports/paper-text-hf](../research/exports/paper-text-hf)
- [https://huggingface.co/datasets/woog/open-text-detector-paper-text](https://huggingface.co/datasets/woog/open-text-detector-paper-text)

- train split is storage convention. Includes overlap with research/paper datasets and extractions.
- Paper year or full text does not establish human-only authorship.

### ReviewBench source papers and reviews

Source corpus, no detector-training labels.

- [research/data/reviewbench/original](../research/data/reviewbench/original)
- [research/data/reviewbench/catalogue.sqlite3](../research/data/reviewbench/catalogue.sqlite3)

- 51,453rows have nonempty manuscript Markdown;15 downloaded Parquet shards.
- Contains paper metadata/reviews/manuscripts; overlaps source paper archive and paper-generation seeds.
- Total corpus tokens unmeasured; no corpus-wide model/token pass performed.

### ICLR 2023 and 2026 controls

Legacy controls; year is not authorship label.

- [research/data/iclr_2023](../research/data/iclr_2023)
- [research/data/iclr_2026](../research/data/iclr_2026)

- 100 papers per cohort; caches and archive may overlap.

### Qwen3.5 mirror pilot

Historical development pilot; 9 mechanical passes, 0 admitted.

- `/data/workspace/synthetic-mirrors-v1/pilot/mirrors.jsonl`

- Separate from the Luna-only 30,000 collection. Qwen3.5-4B BF16 generation; completed 10 of 10.
- Preserved pilot; no expansion or training use inferred.

## Measurement method and evidence

Token estimates use the same existing MELD-v8/ModernBERT tokenizer on both machines (SHA256 `6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30`). Text is counted without special tokens, truncation or padding. No inference, model downloads, paid generation, or full-dataset tokenization was performed for the large remote corpora. Exact stored counts are reused where available. Most estimates tokenize 500 uniformly sampled rows per file; raw mirrors use 1,000; full manuscripts use 40 per final-model group; the unlabeled paper archive uses 100. Random seed 42 and sample uncertainty are recorded in the evidence. Small local 100-row control files were counted in full.

The approximate 79.1M custom subtotal is human text + raw mirror text + target-only paper paragraphs + final full-manuscript Markdown. It includes held-out rows, raw rejected outputs and human seed text, and excludes evaluation collections, public corpora, older distinct pilots and the unlabeled archive. It must not be reported as a unique trainable-token count. A global training total requires chosen splits, text views, provenance labels and cross-source grouping/deduplication; those were not changed by this inventory.

Initial inventory checks were read-only except for documentation and the requested project-rule update. Subsequent authorized transfers are supported by [publication and copy receipts](synthetic-mirrors/publication-20261003/completion.json). HF checks verified repository revisions and file metadata, not a new full byte readback of every remote dataset. The shared-server check found no matching dataset files within three levels of `/workspace/home/woog`, excluding GPU-monitor data and caches; it is not a scan of other tenants or every remote directory. Hjarni searches returned no relevant project notes.

- [Machine-readable inventory](training-data-inventory/20261003/inventory.json)
- [Local estimates and historical counts](training-data-inventory/20261003/local-measurements.json)
- [Remote dataset estimates](training-data-inventory/20261003/remote-measurements.jsonl)
- [Remote MAGE and prepared-training details](training-data-inventory/20261003/remote-supplement.jsonl)
- [Space storage and schemas](training-data-inventory/20261003/space-datasets.json)
- [HF repositories and file metadata](training-data-inventory/20261003/huggingface-repositories.json)
- [Local directories searched](training-data-inventory/20261003/local-directory-census.json)
- [Historical exact-overlap and archive measurements](training-data-inventory/20261003/local-supplement.json)
- [Human usage approval](human-source-pool/origin-permission-audit/user-usage-approval-receipt.json)
- [Mirror filtering report](synthetic-mirrors/stage2-filter-report-20261002.json)
- [Current Stage 3 decision](stage3-edits/decision.json)
