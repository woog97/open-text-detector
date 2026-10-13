# Pangram 4: direct costs in execution order

**Pricing revision:** the [Luna preprocessing update](pangram4-luna-costs.md) supersedes the Haiku-priced helper calls below. Revised totals: $11,100 / $12,750 / $45,831. This document retains the previous ledger for comparison.

This budget separates the disclosed final training runtime from hypothetical dataset volumes and experiment allocations. It includes direct API calls, preprocessing compute, experiments, mining, and training. It excludes staffing, general overhead, automatic contingency, and unincurred purchases.

## Consistent scope

Use the earlier conditional 80B-total/3B-active reconstruction with **2.25M source families**, each expanded into 4.25 document variants on average: **9.56M document-equivalents**. This expansion ratio and corpus size are assumptions, not disclosures. They approximately match the central runtime inversion, but depend heavily on throughput and exposure assumptions.

Three scenarios share the same conservative compute allocations:

- **Prepared-data reuse:** 2M source families already have all usable derivatives AND Pangram-4-style clause labels; add and label 250k new families.
- **Raw-text reuse:** 2M source families have derivatives, but their edited examples need the new clause-level labeling; add 250k new families and label all 2.25M. This explicitly accounts for introducing a new supervision method.
- **Full rebuild:** generate and label all 2.25M source families from available human sources.

The two reuse cases generate 100k new held-out AI examples and reuse the remainder of their evaluation material. The full rebuild generates 520k AI evaluation examples. Each scenario additionally generates and labels 10k separate source families for mixed-authorship calibration/evaluation at $246.75, yielding about 42.5k human/AI/edited/adversarial entries; this is a proposed allocation, not a reconstruction of every published benchmark. Reuse remaining public evaluation sets. Native multilingual sources and prompts are assumed; no separate wholesale translation corpus is purchased. Commercial humanizer subscriptions are excluded: the variant row assumes token-priced synthetic transformations, not exact reproduction of each commercial service.

The [Pangram model card](https://www.pangram.com/research/model-card/pangram-4) supplies the 4-day, eight-H100 final fit: **768 H100-hours = $1,920** at the user's $2.50/hour assumption. The split between stages below is an assumed 50:50 compute allocation, not a disclosure. It includes both stages and Repeat2; those are not extra multipliers.

## Unit rates and request lengths

[Haiku 4.5](https://www.anthropic.com/claude/haiku) costs $1/M input and $5/M output with 50% batch discount. Use $0.50/$2.50 for metadata and clause splitting. Diverse generation uses an assumed blended $1/M input and $5/M output; it is not a vendor quote or an assumption that all generators are Haiku.

| API operation | Assumed input/output tokens | Cost per call |
|---|---:|---:|
| Metadata/topic extraction | 1,000 / 150 | $0.000875 |
| AI mirror | 300 / 800 | $0.004300 |
| Edit or adversarial variant | 1,000 / 800 | $0.005000 |
| Clause splitting | 1,000 / 900 | $0.002750 |

Each source family has one mirror, two edits, 0.25 adversarial variants, and three clause-splitting calls: source once, edited targets twice. Cost = **$0.016425 for generation + $0.008250 for clause splitting = $0.024675/source family**. Pure human and generated examples do not need expensive provenance alignment. Each splitting call handles a complete document, not one separately billed call per clause. Returning compact boundaries instead of full clause text could lower output tokens, but this budget retains the earlier conservative assumption.

CPU preprocessing is explicitly budgeted at an assumed $0.50 per CPU-hour; GPU resources at $2.50 per H100-hour. These are not obtained vendor quotes. All compute amounts except the final fit are allocations awaiting a small benchmark. Local CPU execution may have no incremental rental bill.

## Chronological ledger

25k source families are reserved for late hard-negative additions. They are included in the scenario totals, so they are removed from early generation/labeling and charged once at step 24. Their embedding, target-construction, and tokenization work is included in the corresponding compute rows. Adding two edited variants to each mined source is a reproduction choice beyond the report's explicit mirroring description.

| # | Operation | Prepared reuse | Raw-text reuse | Full rebuild |
|---|---|---:|---:|---:|
| 1 | Load human sources and existing synthetic data | $10.00 | $10.00 | $10.00 |
| 2 | Clean text, normalize Unicode, deduplicate | $30.00 | $30.00 | $30.00 |
| 3 | Split by parent document and assign domain/language strata | $5.00 | $5.00 | $5.00 |
| 4 | Extract topics/metadata for new mirrors | $196.88 | $196.88 | $1,946.88 |
| 5 | Generate fully AI-written mirrors | $967.50 | $967.50 | $9,567.50 |
| 6 | Generate two edited variants per new source | $2,250.00 | $2,250.00 | $22,250.00 |
| 7 | Generate humanizer/adversarial variants | $281.25 | $281.25 | $2,781.25 |
| 8 | Reject copied generations/refusals and validate outputs | $10.00 | $10.00 | $10.00 |
| 9 | Split original human texts into clauses | $618.75 | $6,118.75 | $6,118.75 |
| 10 | Split the two edited texts into clauses | $1,237.50 | $12,237.50 | $12,237.50 |
| 11 | Compute lexical n-gram similarity and retrieve source spans | $30.00 | $30.00 | $30.00 |
| 12 | Compute semantic embeddings for soft n-gram labeling | $160.00 | $160.00 | $160.00 |
| 13 | Assign clause/token labels and 15-bucket/mixed targets | $10.00 | $10.00 | $10.00 |
| 14 | Construct interleaved passages and benign corruption controls | $10.00 | $10.00 | $10.00 |
| 15 | Tokenize, retain offsets, create windows and loss masks | $20.00 | $20.00 | $20.00 |
| 16 | Prepare separate evaluation/calibration partitions | $5.00 | $5.00 | $5.00 |
| 17 | Generate fresh held-out AI and mixed/edited examples | $676.75 | $676.75 | $2,482.75 |
| 18 | Eight small backbone/LoRA experiments | $768.00 | $768.00 | $768.00 |
| 19 | Four longer loss, data, and Repeat2 experiments | $768.00 | $768.00 | $768.00 |
| 20 | Train preliminary candidate: stage 1 | $960.00 | $960.00 | $960.00 |
| 21 | Merge stage-1 adapter and initialize stage-2 adapter | $0.00 | $0.00 | $0.00 |
| 22 | Train preliminary candidate: stage 2 | $960.00 | $960.00 | $960.00 |
| 23 | Screen reserved human pool and select hard negatives | $416.00 | $416.00 | $416.00 |
| 24 | Add 25k mined source families: metadata, mirror, edits, variants, clause labels | $616.88 | $616.88 | $616.88 |
| 25 | Final training: stage 1 | $960.00 | $960.00 | $960.00 |
| 26 | Merge adapter and initialize fresh adapter | $0.00 | $0.00 | $0.00 |
| 27 | Final training: stage 2 | $960.00 | $960.00 | $960.00 |
| 28 | Score calibration and evaluation sets | $288.00 | $288.00 | $288.00 |
| 29 | Fit calibration/CRF parameters and sentence/run-length rules | $10.00 | $10.00 | $10.00 |
| 30 | Export final checkpoint and run consistency checks | $10.00 | $10.00 | $10.00 |
| | **Total** | **$13,235.50** | **$29,735.50** | **$64,391.50** |

## How to interpret the totals

The same **$6,240 GPU budget** appears in each scenario: final fit $1,920, preliminary fit $1,920, short experiments $1,536, embedding work $160, mining $416, and evaluation $288. CPU preprocessing totals **$150**. Holding these allocations fixed conservatively overbudgets preprocessing for the smallest newly processed corpus but avoids presenting unmeasured precision as throughput evidence.

The jump from prepared reuse to raw reuse is **$16,500**: clause-splitting two million existing source families and their edited targets. Embedding and lexical comparisons are relatively cheap budgeted compute; generating source-aligned clause boundaries via API is the dominant labeling charge here.

The jump from raw reuse to full rebuild is **$32,850** of extra training-data generation plus **$1,806** of extra held-out generation. These are explicit calls, not salaries or business overhead.

The $416 mining allocation is 166.4 H100-hours. For example, five million 800-token documents with a 4x duplication/overlap factor require about 26,700 processed tokens/second/H100 to fit. The $288 evaluation allocation, applied to three million similarly sized documents, requires about 23,100 tokens/second/H100. These rates have not been measured. A segment-only first-pass miner can reduce work; a heavier backbone or slow kernels can increase it. The corpus counts are therefore targets conditional on throughput, not guaranteed throughput claims.

The embedding allocation is 64 H100-hours. Embedding 6.75M 800-token source/target documents in that allocation requires about 23,400 input tokens/second/H100 before clause-boundary and batching overhead. A small multilingual embedding encoder may be practical; a larger embedding model and repeated clause context can exceed the allocation. Benchmark this stage rather than automatically equating an embedding token to a detector token.

A reusable CPU process handles n-gram matching and target derivation; there is no paid LLM classification call per token. CRF calibration also does not require another backbone training run.

For experiments, 5% or 10% means aggregate compute relative to the final run; it can come from fewer examples, shorter training, or a cheaper backbone. If the disclosed runtime already includes the preliminary candidate, remove $1,920 from these totals. If another complete final fit is necessary, add $1,920. Rejected samples and API retries add actual generation costs; none are disguised inside a blanket reserve.

**Useful planning figures:** approximately **$13.2k** if suitable labeled data already exist, **$29.7k** if old text must be relabeled, or **$64.4k** to generate the full hypothesized corpus and labels anew. The 9.56M-document corpus is only one conditional size estimate. A smaller real corpus scales the generation and labeling rows down directly. Low training rental cost by itself does not identify the size or replacement cost of the data.

The machine-readable ledger includes the per-step formulas in `pangram4-ordered-costs.json`.
