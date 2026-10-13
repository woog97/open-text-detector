# Sampling and admission specification

This is our proposed implementation specification for collecting the human pool. Its rules are not claimed to reproduce undisclosed Pangram internals. The accompanying registry is source-level research; no row-level audit or training release has been completed.

## Preserve the distribution without hiding gaps

Choose a target number of eligible passages, N. Allocate category c as N × displayed_share(c) / 99.9. Use largest-remainder rounding with stable identifier tie-breaking. Allocate within each category using its source percentages. `build_plan.py` implements that calculation exactly using rational arithmetic.

The current example is N = 100,000. If a source cannot supply its quota, record the shortfall. Do not duplicate records, remove the source from the denominator, shift its quota into an unrelated genre, or label a partial pool as a complete match. A proposed substitution needs a revised, versioned mixture and a statement of the changed coverage.

Allocate by passages only after assigning source families to partitions. Keep separate counts for source works, documents, authors, passages, words and model tokens. The target mixture is an exposure design: ten passages from one book do not provide ten independent books. No tokenizer or model asset needs downloading for the metadata phase; any later tokenization/inference follows the project's Space-only model-asset and reduced-precision requirements.

## Admission evidence

Each record needs evidence for four independent questions:

| Question | Required record evidence | An insufficient substitute |
|---|---|---|
| Where did this text come from? | Original source ID/URL, collection revision, retrieval timestamp and raw hash | A Hugging Face dataset name alone |
| Can this version enter this project? | Applicable text license or documented rights, attribution and any exclusions | An MIT code license, general open-access flag or compilation license |
| Why is the prose considered human? | Historical text-version evidence, or documented controlled human collection | A detector's low AI score, publication year alone or a human label on a mirror |
| Is it independent of protected evaluation? | Original-family and near-duplicate exclusion record | Different filenames or a different dataset provider |

Recommended default historical cutoff: **2021-12-31 23:59:59 UTC**. This reduces modern LLM contamination but is not proof of wholly human authorship: earlier bots, template generators, translation systems and synthetic datasets existed. Explicit machine-written examples remain excluded regardless of date.

Use one of these provenance bases:

- `historical_capture`: archived bytes or a frozen release demonstrably containing the text before the cutoff.
- `historical_revision`: a specific version/revision recorded by the original source before the cutoff.
- `historical_print_verified`: scan or stable publication evidence, with transcription checked against the original and editorial additions excluded.
- `controlled_human_collection`: documentation of writing dates, permitted tools and collection conditions supporting human authorship. This is a separately tracked exception when it does not satisfy the historical rule.
- `unverified`: quarantine. A modern page with an old publication date belongs here until its exact text history is established.

Keep `source_authorship_confidence` separate from `text_transcription_confidence`. A genuinely old book processed through generative OCR can still contain recent machine-invented wording. Do not admit generative OCR/ASR output as exact human-token gold without source comparison; prefer digital text or conventional extraction with manual spot checks.

## Canonical records

Store originals immutably and normalized passages separately. A suggested record contains:

| Field group | Required fields |
|---|---|
| Identity | `record_id`, `source_id`, `original_id`, `canonical_work_id`, `source_url`, `source_revision`, `raw_sha256` |
| Attribution and rights | `title`, `author_attribution`, `license_id`, `license_url`, `license_evidence_hash`, `rights_status`, `third_party_exclusions` |
| Dates and authorship | `original_publication_date`, `text_version_date`, `capture_date`, `provenance_basis`, `provenance_evidence`, `authorship_status` |
| Content | `raw_text_ref`, `clean_text`, `language`, `category`, `subgenre`, `section`, `word_count`, `extraction_method`, `extraction_version` |
| Passage lineage | `parent_document_id`, `raw_offsets`, `clean_offsets`, `normalization_map`, `passage_sha256` |
| Independence | `document_family_id`, `author_group_ids`, `prompt_group_id`, `duplicate_cluster_id`, `protected_overlap_status`, `split` |
| Audit | `admission_status`, `reason_codes`, `reviewer`, `reviewed_at`, `sampling_seed`, `selection_weight` |

Store public author attribution for license compliance separately from the model input. Use stable pseudonymous grouping IDs for student/reviewer accounts; do not put demographic or contact fields into the text. Preserve only the metadata needed for provenance and justified aggregate coverage audits. Unknown author identity stays unknown.

Passage offsets must refer to a frozen text version. Normalize Unicode and repair deterministic extraction artifacts while recording changes. Do not grammar-correct, paraphrase, summarize, translate or stylistically rewrite human text in this step. Do not use an LLM to fill OCR gaps. Keep original typos and learner language.

## Category routing

Assign exactly one primary category to every passage. Retrieval source and genre are separate fields. A research paper found in Common Crawl belongs in scientific; a news article from a wiki belongs in news; a student essay in a general corpus belongs in essays. An excerpt found in several corpora is one work family.

Use source metadata first, then deterministic rules, then human review for ambiguous items. CCCC is the residual web source, not a tenth domain. Store secondary tags for meaningful overlap, but do not count them twice. Synthetic demonstrations inside otherwise human articles need their own exclusion spans or a quarantined document.

For scientific text, exclude reference lists and tables from prose selection but retain enough context around citations to avoid broken sentences. For reviews, select only reviewer text. For Q&A, questions are side metadata unless deliberately sampling a question-writing stratum. For educational material, separate exercise instructions from answer keys. For fiction, omit editorial introductions, translators' notes and boilerplate unless specifically classified as separate human works.

## Duplicates and split leakage

First build canonical identities: DOI/arXiv/ACL/PMC paper families; book title-author-edition families; review IDs and parent products; question threads; wiki page revisions; news syndication and translation families; student essays and assigned source prompts; agency reports and their updates.

Then use exact hashes of normalized text and approximate matching. Proposed initial candidate rules are 5-word-shingle MinHash with estimated Jaccard at least 0.8, plus containment checks for short excerpts inside long documents. These thresholds need inspection against known duplicates and genuinely different formulaic prose. They are not sufficient evidence of independence on their own. Do not discard whole papers merely because they share citations or standard methods wording.

Group related versions and derivatives before partitioning. Apply strongest feasible author separation to paper/book/essay evaluation. A strict transitive author-and-topic graph can collapse a collaborative corpus into a giant component; measure this rather than pretending random rows solve it. For reviews, construct a designated user/product-disjoint evaluation slice where possible, then document dropped cross-group records. Keep all copies of one review together in every case.

For news, separate the same story/event across partitions where feasible, not just byte-identical syndications. For essays, source passages and prompts are especially important: many students can quote the same assigned text. With only seven ASAP prompts, seven groups do not support precise prompt-generalization estimates. Do not multiply the apparent sample size by treating every quoted sentence as independent.

Existing protected project data includes:

- The original families underlying `benchmarks/otd/eval_suite/bundle/full.jsonl.gz` and its manifest, including human originals behind generated benchmark variants.
- Existing validation/test paper families from `benchmarks/otd/exports/ai-paper-provenance-v3-10000/`, and the separate `research/data/paper-eval-workflows-luna-20260930/` collection.
- The entire 100-book PG-19 test collection in `research/data/human_pg19/`, including alternate editions or mirrors of those works.
- Cached ELLIPSE, PELIC, DetectRL, MELD, EPOCH, GEDE and other evaluated sources; benchmark use must be checked against the full manifest, not guessed from a directory name.

These are starting points for a comprehensive exclusion index, not a claim that every protected family has already been enumerated. Reuse of an already approved training family must be labeled as reuse, never as new independent coverage. Existing files and benchmark definitions remain unchanged.

## Separate fitting, mining and evaluation

Create five distinct family-level partitions: initial training, hard-human mining reserve, model selection, threshold calibration and locked test. Family assignment precedes passage extraction and any later AI transformations. All generated mirrors, edits, translations and splices inherit the original family assignment when those later stages are implemented.

A planning default is 60/20/10/5/5 percent of eligible **families**, with category balancing where group constraints permit. These are our proposed starting ratios, not Pangram facts. Family-level allocation will not exactly reproduce those ratios in passage counts; report both. Expand calibration and test separately when low false-positive claims demand more independent human examples. A 5,000-passage test cannot establish a one-in-tens-of-thousands FPR, particularly when passages are correlated.

Never run hard-negative mining on the locked test or threshold-calibration pool. Keep the mining reserve naturally sampled at first; selecting difficult examples changes its distribution, so it is not an unbiased evaluation set afterward. Save every mining round and original selection probability. This step prepares the reserve only; it does not run the detector or generate mirrors.

## Length and diversity

Retain full source documents, then sample contiguous complete sentences or paragraphs. Avoid always selecting document starts. Suggested first-pass word-length targets are below; these are tunable design choices and should be checked against measured supply.

| Category | 50–149 words | 150–399 | 400–999 | 1,000–1,500 |
|---|---:|---:|---:|---:|
| Creative | 10% | 25% | 45% | 20% |
| Scientific | 10% | 35% | 40% | 15% |
| Reference | 10% | 35% | 40% | 15% |
| Reviews | 35% | 45% | 18% | 2% |
| Social/Q&A/chat | 40% | 40% | 18% | 2% |
| General web | 20% | 40% | 30% | 10% |
| News | 15% | 40% | 35% | 10% |
| Essays | 10% | 35% | 45% | 10% |
| Professional | 15% | 35% | 35% | 15% |

Store shorter natural items as a separate diagnostic reserve rather than padding them. Preserve long originals for later long-document windows. Do not concatenate independent reviews, unrelated chat turns or separate abstracts to meet a length target.

Initial diversity caps: one selected passage per short document; at most three nonoverlapping passages per paper/report; at most ten per book, spread across chapters. Cap one known author at 1% of a category and one CCCC registered domain at 5% of general web. These caps are proposals, not measured feasible bounds. If a small source cannot fill its allocation under the caps, record a shortfall; do not repeat its handful of examples. OANC ICIC may need a documented two-passage allowance for longer letters/reports or a smaller quota after measuring supply.

Within each eligible stratum use seeded random sampling independent of detector score. Preserve formal and informal writing, simple and advanced language, low and high essay scores, positive/negative/neutral opinions, and native/non-native English when the source supplies reliable aggregate metadata. Do not infer a person's native language or demographic group from their name. Do not reject profanity or unusual dialect merely because a general language-model quality filter dislikes it.

## Pilot and acceptance checks

Before filling quotas, inspect an initial 50 candidate passages per weighted source, spread over its proposed subgenres and length bins. For the 37 source slices this is a **proposed 1,850-record intake audit**, not a completed annotation set. Rights-restricted sources can be examined through their documentation until an appropriate access basis is established. Expand the sample if the first audit finds a recurring defect.

Review origin/version evidence, category assignment, text rights, extraction fidelity, suspected machine text, duplicate families, quotations, contact details and paragraph boundaries. Reject broken extraction, not inconvenient human style. Aim for at least 98% usable extraction in the reviewed sample before scaling, with no unresolved critical provenance or rights failure in admitted records. The 98% target is an operational criterion, not a statistical guarantee that the remaining source is clean.

Report the whole flow: candidate documents → license-eligible → date/provenance-eligible → extraction-valid → independent of protected sources → unique families → selected passages. Include failure reasons and retention rates by source, topic, length and subgenre. This makes visible whether filtering has erased the intended coverage.

Release a human pool only when its manifest contains exact revisions and hashes, every admitted record has an evidence basis, all identified protected overlaps are excluded, quotas and shortfalls are explicit, and the five partition manifests are frozen. No amount of source-level research substitutes for that row-level work.

## Work still required after this curation

Implement and run the source adapters; establish rights for the held slots; recover historical revisions; perform the proposed pilot; measure usable supply; build the comprehensive benchmark-family exclusion index; and produce actual partitioned passage files. Later synthetic mirroring, AI editing, hard-negative scoring and model training are outside this step and have not been started.
