# Pangram 4 budget with GPT-6 Luna preprocessing

This revision changes only metadata extraction and clause splitting to GPT-6 Luna. Corpus sizes, experiment allocations, the final training cost, and the varied AI-generation mixture are held fixed. This is a reproduction budget at current prices, not Pangram's historical invoice.

[Official OpenAI pricing](https://developers.openai.com/api/docs/pricing) lists GPT-6 Luna short-context standard input/output at $0.10/$0.50 per million tokens and batch at $0.05/$0.25. The preceding Haiku batch assumption was $0.50/$2.50. This is a like-for-like 10x reduction for the same billed tokens. [Luna's model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna) supports reasoning effort `none`; use that as the initial setting for this bounded extraction task so unbudgeted reasoning tokens do not consume the assumed output allowance. Task-specific boundary accuracy still needs a small comparison; higher general model quality alone does not demonstrate exact label equivalence.

## Why Haiku dominated the preceding estimate

The central scenario labels 2.25M source families with one original and two edited versions each: 6.75M document-level splitting calls. Each call was budgeted at 1,000 input tokens and 900 output tokens. That gives 6.75B input plus 6.075B output tokens. Haiku batch costs $3,375 input + $15,187.50 output = $18,562.50; Luna batch costs $337.50 + $1,518.75 = **$1,856.25**. These are corpus/request-length assumptions, not reported Pangram token counts.

About 82% of this charge comes from output. A model splitting text into clauses may reproduce much of the text in a structured response; our earlier estimate assumed that behavior. A compact boundary-only representation can cost less, provided offsets or boundary markers are verified against the unchanged source.

## Roles of external models

The [Pangram report](https://arxiv.org/html/2607.27183v1) explicitly identifies Haiku 4.5 for target-clause splitting. Applying the same splitter to source documents is our reproducible implementation assumption. The labeler then uses lexical matching and embedding similarity; it does not require an LLM to adjudicate every clause or token. Pangram 4's embedding model is not named. Metadata extraction was also assigned to Haiku in our budget, but that specific choice is ours, not a disclosed Pangram choice.

For evaluation preparation the report names Mistral-Small-24B-Instruct-2501 for prompt filtering and GPT-5.5 nano for mining generic editing instructions. Its editing evaluations include Opus 4.8, Gemini 3.1, GPT-5.5 and consumer editing tools. These are evaluation roles, not proof of the exact training generator mixture. Broad training uses varied generators; exact shares are undisclosed.

Luna is a plausible replacement for the helper tasks: boundary extraction, metadata, filtering, or prompt categorization. It can also be one generator of mirrors and edits. Replacing all generators with Luna would reduce generator diversity and change the learning problem, so that separate change is not included in these totals. The original blended generation assumption of $1/M input and $5/M output is a planning mixture, not a claim that Pangram exclusively uses Haiku.

The detector backbone itself is a separately trained open MoE of unconfirmed identity. Neither Haiku nor Luna is required in the deployed classifier inference path. Offline teacher costs do not increase per-document serving cost.

## Revised totals

| Scenario | Previous | Luna helpers | Savings |
|---|---:|---:|---:|
| Prepared labeled data reused | $13,235.50 | **$11,100.25** | $2,135.25 |
| Existing text relabeled | $29,735.50 | **$12,750.25** | $16,985.25 |
| Entire hypothesized corpus rebuilt | $64,391.50 | **$45,831.25** | $18,560.25 |

For the central relabeling scenario, the revised components are:

| Component | Revised cost |
|---|---:|
| Clause splitting, training originals | $618.75 |
| Clause splitting, training edits | $1,237.50 |
| Metadata for new training mirrors | $21.875 |
| Helper calls for mixed evaluation | $9.125 |
| Diverse training/evaluation generation | $4,473.00 |
| GPU training, experiments, embeddings, mining, evaluation | $6,240.00 |
| CPU preprocessing and calibration | $150.00 |
| **Total** | **$12,750.25** |

Late hard-negative additions are included; their helper calls are repriced too. Held-out mixed-evaluation metadata/splitting is also repriced. No 10x reduction is applied to unrelated GPU jobs or the diverse synthetic-generator mixture.

## Additional savings that are not yet assumed

At 1,000 input and 150 output tokens, Luna batch splitting would cost $0.0000875/document instead of $0.000275 at 900 output tokens. Across 6.75M training calls that is $590.625 rather than $1,856.25. This depends on achieving the shorter valid output format; it is an optimization scenario, not measured usage. Avoid normalizing or rewriting source text while producing boundaries.

Metadata already available as titles or source questions may remove extraction calls. Split each original once and cache it across its edited variants. Use lexical matching to skip semantic embedding work on exact matches. Those optimizations can further reduce costs but have not been assigned speculative savings in the main total.

## Updated chronological ledger

The same 30 steps are retained below. Amounts in dollars, before display rounding; actual corpus volume and throughput remain uncertain.

| # | Step | Prepared reuse | Raw-text reuse | Full rebuild |
|---|---|---:|---:|---:|
| 1 | Load human sources and existing synthetic data | $10.00 | $10.00 | $10.00 |
| 2 | Clean text, normalize Unicode, deduplicate | $30.00 | $30.00 | $30.00 |
| 3 | Split by parent document and assign domain/language strata | $5.00 | $5.00 | $5.00 |
| 4 | Extract topics/metadata for new mirrors | $19.69 | $19.69 | $194.69 |
| 5 | Generate fully AI-written mirrors | $967.50 | $967.50 | $9,567.50 |
| 6 | Generate two edited variants per new source | $2,250.00 | $2,250.00 | $22,250.00 |
| 7 | Generate humanizer/adversarial variants | $281.25 | $281.25 | $2,781.25 |
| 8 | Reject copied generations/refusals and validate outputs | $10.00 | $10.00 | $10.00 |
| 9 | Split original human texts into clauses | $61.88 | $611.88 | $611.88 |
| 10 | Split the two edited texts into clauses | $123.75 | $1,223.75 | $1,223.75 |
| 11 | Compute lexical n-gram similarity and retrieve source spans | $30.00 | $30.00 | $30.00 |
| 12 | Compute semantic embeddings for soft n-gram labeling | $160.00 | $160.00 | $160.00 |
| 13 | Assign clause/token labels and 15-bucket/mixed targets | $10.00 | $10.00 | $10.00 |
| 14 | Construct interleaved passages and benign corruption controls | $10.00 | $10.00 | $10.00 |
| 15 | Tokenize, retain offsets, create windows and loss masks | $20.00 | $20.00 | $20.00 |
| 16 | Prepare separate evaluation/calibration partitions | $5.00 | $5.00 | $5.00 |
| 17 | Generate fresh held-out AI and mixed/edited examples | $594.62 | $594.62 | $2,400.62 |
| 18 | Eight small backbone/LoRA experiments | $768.00 | $768.00 | $768.00 |
| 19 | Four longer loss, data, and Repeat2 experiments | $768.00 | $768.00 | $768.00 |
| 20 | Train preliminary candidate: stage 1 | $960.00 | $960.00 | $960.00 |
| 21 | Merge stage-1 adapter and initialize stage-2 adapter | $0.00 | $0.00 | $0.00 |
| 22 | Train preliminary candidate: stage 2 | $960.00 | $960.00 | $960.00 |
| 23 | Screen reserved human pool and select hard negatives | $416.00 | $416.00 | $416.00 |
| 24 | Add 25k mined source families: metadata, mirror, edits, variants, clause labels | $411.56 | $411.56 | $411.56 |
| 25 | Final training: stage 1 | $960.00 | $960.00 | $960.00 |
| 26 | Merge adapter and initialize fresh adapter | $0.00 | $0.00 | $0.00 |
| 27 | Final training: stage 2 | $960.00 | $960.00 | $960.00 |
| 28 | Score calibration and evaluation sets | $288.00 | $288.00 | $288.00 |
| 29 | Fit calibration/CRF parameters and sentence/run-length rules | $10.00 | $10.00 | $10.00 |
| 30 | Export final checkpoint and run consistency checks | $10.00 | $10.00 | $10.00 |
