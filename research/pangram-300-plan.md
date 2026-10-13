# A $300 Pangram-style open replication with a $30 final fit

Planning only; no paid resources have been started. All quantities below are proposed controls, not measured throughput or accuracy forecasts. The objective is an English-prose detector preserving Pangram's supervision and training structure, with useful token-level provenance and good recall at a low false-positive rate. The budget cannot establish parity with Pangram across its full distribution.

## The decision

Start with **Qwen3.5-2B-Base's text backbone**, BF16 LoRA, and four linear task heads. Test the 4B sibling as the one capacity challenger. Pick the winner by held-out performance projected to the same $30 final training budget, not equal optimizer steps. Preserve synthetic mirroring, clause-level supervision, two training stages with adapter merging, Repeat2, one round of hard-negative mining, and calibrated structured decoding.

The [2B base](https://huggingface.co/Qwen/Qwen3.5-2B-Base/blob/main/README.md) and [4B base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) have Apache-2.0 model cards. The 2B model uses 24 hybrid mixer/attention layers and width 2,048; its feed-forward blocks are dense. Use the language backbone without image features, image encoder, generation, or vocabulary-logit materialization. Hybrid Gated DeltaNet layers require compatible training kernels and deliberate adapter targets. If they prove slow or unsupported, use a conventional Qwen causal backbone within the same throughput screen; do not spend the research budget repairing a kernel stack.

This is a deliberate change from Pangram's MoE. Sparse activation reduces arithmetic, but does not remove resident expert weights, expert-kernel overhead, or routing costs. A 2B dense model offers a pretrained representation at much smaller engineering and memory cost. The question is whether its discriminative capacity is sufficient at the operating point, not whether it can match the largest generator's reasoning ability.

The [Pangram report](https://arxiv.org/html/2607.27183v1) motivates the retained components: four task heads, two successive LoRA stages, source-aligned weak labels, second-copy token supervision, hard-negative mining, and CRF postprocessing. Exact hidden hyperparameters cannot be replicated, so publish all chosen settings and differences.

## Why this scaling makes sense

The budget buys adaptation of an already pretrained representation. It does not pay to acquire general language competence from scratch. A Chinchilla-style allocation between random-initialized parameters and language-modeling tokens therefore does not directly determine the optimum. Relevant diminishing returns come from coverage of generator/domain/edit combinations, weak-label quality, and how far low-FPR errors can be pushed with the selected representation.

Matched human/AI pairs reduce topic-label correlation. Generator diversity reduces dependence on one model family's surface habits. Hard-negative mining reallocates limited supervised tokens toward failure regions. Each can increase the usefulness of a training exposure without increasing model size. These are hypotheses to test, not a guarantee that model size stops mattering.

Binary detection and semantic provenance are not equally easy. A larger model may add little on obvious AI prose while helping distinguish preserved ideas from newly added content. The 4B comparison must include mixed text, boundary metrics, and difficult human subgroups. Conversely, more capacity can also learn dataset artifacts better; aggregate accuracy alone will not detect that failure.

Keep task geometry rich while shrinking scale. Most head and CRF parameters are negligible next to the backbone. Dropping token supervision saves compute but discards the desired function. Moving repeated-input training into a shorter second stage retains that function more cheaply.

## Hard budget

| Work | Maximum USD |
|---|---:|
| Source preparation, small artifact storage | 10 |
| Varied synthetic generation, including mined additions and held-out examples | 90 |
| Luna metadata and clause splitting | 12 |
| Semantic embedding computation | 6 |
| Label audit / difficult-case adjudication | 8 |
| R&D including preliminary candidates | 80 |
| Hard-negative screening | 20 |
| Final two-stage fit, all job charges | 30 |
| Final evaluation and calibration | 24 |
| Uncommitted allowance for measured overruns elsewhere | 20 |
| **Maximum project spend** | **300** |

The last $20 is not a requirement to spend it. It never authorizes the final fit to exceed $30. Count paid requests already queued, failed paid calls, rental startup/idle time, storage, and retries. Reserve worst-case output cost before launching a batch; stop admitting work before provider billing catches up. Use a provider-side spend/credit limit where available, application limits, and automatic instance termination. No paid training is authorized merely by writing this plan.

At the assumed $2.50/H100-hour, the final cap is 12 hours. The actual quote controls available time. Target at most $27 of scheduled job work, leaving $3 inside the $30 cap for checkpointing, merging, saving, transfer, and shutdown. If the rate is higher, shorten the schedule before launch. Precompute all labels and tokenize before starting the final rented instance.

## Data scale and organization

Target about 20k source families across six English-prose domains: student/academic writing, educational/general web, news, reviews, creative writing, and professional correspondence. Cap one website/book/author's influence. Use pre-2022 material where possible, but source provenance rather than date alone is the label basis. Include ESL and unusually polished human prose. Do not train on the existing ICLR cohorts as if paper year were an authorship label.

Allocate source families roughly 70/10/10/10 to training, development, calibration, and test before generation. Keep every derivative of each source in that partition. Reserve editing prompt families and at least one generator family for out-of-distribution evaluation. Additional independent human evaluation pools are separate from these source families.

Proposed total pool, before split and filtering:

| Text type | Target variants |
|---|---:|
| Original human prose | 20k |
| Topic/length-matched AI mirrors | 20k |
| AI edits over a spectrum of changes | 16k |
| Interleaved human/AI/edited passages | 8k |
| Humanized / adversarial AI variants | 2k |
| **Total** | **66k** |

These are targets, not promises of 66k accepted unique examples at a fixed API price. Their derivatives are correlated; splitting or stitching does not manufacture new independent source families. Ordinary length variation and multiple sampled windows provide more window exposures than document records.

Allocate generation dollars initially as $65 for seed mirrors/edits, $15 held for mining-driven refresh, and $10 for independent evaluation. Existing suitably licensed, source-paired examples may fill part of the seed pool. If API rates or output lengths exceed the cap, reduce accepted counts while preserving domain/generator cells, not by letting the cheapest generator dominate. Approximately 38k newly generated mirror/edit/attack records would require an average bill near $0.0017 under the $65 seed allocation; this is feasible only with a cheap-heavy mixture, shorter outputs, batch prices, or some reusable examples. Spend is controlled by tokens, not by requesting a fixed number of unbounded completions.

Use at least four generator families when feasible, with no one family above roughly 35% of generated training examples. Give an undercovered expensive family fewer examples rather than excluding it entirely. Temperature changes are not substitutes for family diversity. Luna can generate some examples, but its main role is standardized preprocessing. Avoid mirroring solely from detector-preferred prompts.

Use a broad range of edit prompts. Sample one light edit and one more substantive category across different sources rather than repeatedly rewriting a few sources many times. Mix same-topic continuations and replacements; naive cross-topic concatenation teaches a topic-change detector. Cap adversarial/humanizer examples around 5% initially and include benign corrupted human controls.

## Open release choices

Implement the method independently and release code, training configuration, source manifests, prompt templates, split hashes, adapters/weights, and evaluation outputs. The current workspace contains noncommercial EditLens-derived weights and code and Pangram brand assets; these are not an appropriate foundation for an unrestricted open release. Pangram's [Open Pangram announcement](https://www.pangram.com/blog/introducing-open-pangram) describes its own release as noncommercial. Keep those models as comparison baselines.

[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) exposes pre-2022 crawl partitions and ODC-BY metadata; it supplies web prose but not every target domain. Preserve source-specific attribution and redistribution conditions; use retrieval manifests rather than publishing source text where needed. Prefer public-domain or explicitly reusable sources for the shareable training bundle. Do not infer unrestricted rights from a dataset's downloadability. This is a source-selection constraint, not a reason to buy a corpus.

## Labeling and preprocessing

Use [GPT-6 Luna batch pricing](https://developers.openai.com/api/docs/pricing): $0.05/M input, $0.25/M output at short context. Disable reasoning initially for boundary extraction. Validate output as spans over unchanged source text; never silently accept a rewritten document. Cache each source split across edited targets. Return compact boundaries when reliable, with full clauses as fallback. This is teacher-based preprocessing, not distillation of the proprietary detector.

If 16k edited texts require 16k source splits, that is 32k splitting calls. At the previous conservative 1,000-input/900-output assumption they cost $8.80. Up to 20k metadata calls at 1,000 input/150 output cost another $1.75. That leaves about $1.45 of the $12 allocation for targeted additions or retries. Skip metadata calls when a title/question already suffices. If the fallback/retry budget is exhausted, reduce new requests rather than exceed the cap.

Compute word and character n-gram similarity first. Exact copied spans get Human labels without embeddings. For rewritten clauses, embed source candidates and target clauses using a small encoder such as [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B). Near-duplicate and clause retrieval should limit candidate comparisons. A large pairwise cross-encoder or a generative call per clause is not budget-appropriate.

Audit 300–500 stratified clauses manually with assistant support and selective paid adjudication within $8. Include copied quotations, paraphrases, negation, numbers, added claims, short clauses, reordered sentences, and light edits. Tune matching thresholds here and freeze them before final evaluation. Verify offset coverage and no duplicated/dropped characters automatically.

Mask uncertain tokens or downweight their loss instead of forcing every ambiguous match into a confident class. Exclude windows with too much unresolved provenance from segment/mixed supervision or compute interval-aware targets; do not silently treat unlabelled text as Human. Weak-label bias can be learned very efficiently and does not average away with more repeated epochs.

## Exact scaled training structure

Use causal last-position heads with 15 segment classes, two mixed classes and four humanizer classes; a shared three-class token head acts at each supervised position. Retain original offsets. Use 512-token source windows. Randomize training crop offsets, but keep source-family separation. For stage 2 concatenate each window to itself, allow attention from the second copy into the first, and mask first-copy token losses. Do not implement the duplication as two disconnected packed examples.

Start with BF16 frozen backbone, rank-16 LoRA, alpha 32, dropout 0.05; adapter LR 3e-5, new-head LR 1e-4, AdamW, weight decay 0.01, 3% warmup, global-norm clipping 1, effective batch 64 source windows. These are initial search settings, not recovered Pangram settings. Adapt hybrid mixer projections as well as attention and dense FFN projections; verify actual target coverage. Omit the vocabulary generation head from the forward pass. Use memory-saving recomputation only if needed; quantization and recomputation can lower throughput when memory already fits.

Stage 1: segment loss plus humanizer loss at 0.25, with the humanizer input detached. Stage 2: merge stage-1 adapters, reset optimizer, initialize a fresh adapter, preserve trained heads, add token and mixed heads. Initial loss weights: segment 1, mean valid-token loss 1, mixed 0.5, detached humanizer 0.25. Normalize task losses before weighting them. A low humanizer score should not override the main provenance output.

Target **150k stage-1 window exposures** and **50k stage-2 source-window exposures**. That is 76.8M + 51.2M = **128M processed tokens**, including repetition. Stage 2 should deliberately sample boundary-rich and edited windows but retain pure human and AI examples. Sample domains and source families with caps so long books and hard examples cannot dominate.

Measure separate throughputs t1 and t2 in processed tokens/second. Job training time is `76.8M/t1 + 51.2M/t2`. For illustration, t1=6,000 and t2=2,500 gives 9.24 hours, leaving room within the assumed 12-hour maximum. This is an example feasibility calculation, not a measured benchmark. If slower, scale exposures down together, preserving the stage-two allocation, or use the smaller validated backbone. Always reserve time for final stage-two learning rather than discover halfway through that stage one consumed the budget.

The final job starts from the selected pretrained base, not a task-trained R&D checkpoint. This avoids hiding part of the final fit's training bill in research costs. The final job includes both stages, its validation and merges. Evaluation and calibration after freezing weights have their separate published budget.

## R&D: spend to answer five questions

| Question | Maximum |
|---|---:|
| Does the exact training graph fit and meet the throughput budget? | $5 |
| Is 4B better than 2B at equal dollars, with early and later checkpoints? | $20 |
| Which of two learning rates and two adapter placements is stable? | $15 |
| Do conservative labels and boundary-focused sampling improve mixed performance? | $20 |
| Promote at most two configurations to longer preliminary fits | $20 |
| **Total** | **$80** |

Use sequential elimination, not a full Cartesian search. Reuse immutable data and matched validation examples. Do not select from a single early-loss point: compare improvement per dollar and promote only close contenders. When differences fall within bootstrap uncertainty, choose the faster/simple candidate, or spend one replication to resolve a decision that matters. A tiny benchmark difference is not worth an architecture rewrite.

The best preliminary model is the mining candidate. Screen approximately 100k–300k additional human windows, conditional on measured speed and the $20 cap, using cheap segment-only scoring first. Recheck the most suspicious few thousand with full decoding. Mix high-scoring errors, uncertain examples, and random audit samples; cluster before selection. Add roughly 1k–3k verified source families and mirrors using the reserved $15 generation allowance. Those additions replace part of the final sampling budget; they do not create an unbounded extra epoch. Use 10–20% of final sampling for mined families initially. Keep a representative calibration distribution; mining deliberately alters class/domain prevalence.

MELD already available locally can be a free initial comparator and candidate miner. Do not train only on examples it accepts or copy its labels: that would inherit its blind spots. Public EditLens models are comparison baselines, not mandatory starting checkpoints.

## Evaluation and release target

Select primarily on AI recall at calibrated 0.1% human FPR, with 1% FPR as a less noisy early-screen metric. Constrain worst-domain FPR, light-edit false positives, and mixed-text segment F1/boundary error. Report raw token performance as well as final sentence-constrained output; smoothing can disguise weak localization. Calibration uses an independent split and is frozen before test evaluation.

Aim for a 100k independent human evaluation pool divided into 10k development, 20k calibration and 70k final test, plus approximately 3k–5k held-out AI/edited/mixed outputs within the allocated generation budget. Split and cluster by author/site/source; 1,000 windows from one book are not 1,000 independent error trials. Report binomial or cluster-aware uncertainty and subgroup sample sizes. At 70k human texts, a true 0.1% FPR yields roughly 70 errors; a 0.0041% rate yields only about three. This budget can evaluate a useful operating point, but matching a rare-error claim is a different statistical task.

Use the report's overlapping windows, segment/token fusion, CRF smoothing, sentence voting and minimum-run merging in the deployed path. Tune postprocessing on calibration data, not the final test. Clearly document the actual embedding model, clause-label thresholds, generator mixture, scopes, and measured costs. Release the budget ledger and observed examples/second so the next replication can improve on evidence rather than inferred GPU FLOPs.

The intended outcome is a credible, reproducible small detector that preserves the major learning mechanisms and exposes its limits. There is no basis to promise a numerical fraction of Pangram's accuracy before running the controlled comparisons.
