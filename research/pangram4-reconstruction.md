# Pangram 4: architecture reconstruction and token-based cost estimate

23 September 2026. Public-source analysis; no weights, private training data, or internal logs were available. **The architecture can be reconstructed much more confidently than the backbone identity or training volume.** Revised cost estimate: use the disclosed training run as the compute anchor, at the user-supplied rental assumption of $2.50 per H100-hour. Excludes staffing and general business overhead. The earlier token-priced compute estimate is superseded.

## What is actually disclosed

The [Pangram 4 model card](https://www.pangram.com/research/model-card/pangram-4) identifies a pretrained causal sparse MoE backbone and four linear heads: 15 segment classes, three token provenance classes, two mixed-authorship classes, and four humanizer classes. The humanizer branch stops gradients into the backbone. LoRA training first uses segment and humanizer objectives, then merges the adapter and initializes another for joint training. Stage two duplicates each 512-token source window and supervises token labels on its second copy. Human prose spans many domains; synthetic mirrors and AI edits supply the other classes. Commercially licensed sources are used, excluding customer submissions.

Additional specifics in the [technical report, §§3–4](https://arxiv.org/html/2607.27183v1):

- Claude Haiku 4.5 splits clauses; lexical matches become Human, semantic-only matches AI-Assisted, unmatched ideas AI-Generated.
- Segment target: `(0.5 × assisted characters + generated characters) / all characters`.
- Mixed target: non-dominant provenance exceeds 15% of supervised tokens.
- Humanizer loss weight: 0.25. Sequence heads use the last supervised hidden state.
- Hard human negatives are mirrored, added, and followed by retraining from scratch.
- Inference uses 512-token windows, stride 256, Repeat2, and an end-anchored final window.
- Six token/segment features feed multinomial calibration and a three-state CRF. Switching costs are `min(0, −λ + γm)`; mixed log-odds relax smoothing. Viterbi labels undergo sentence voting and minimum-run merging. Confidence comes from pre-constraint marginals.
- Backbone ablations anonymize candidates. A/attention+dense has the best listed FNR, 0.930%; B/routed-expert adaptation reaches 85.033%.

The exact checkpoint, corpus count, stage lengths, LoRA configuration, optimizer, matching thresholds, bucket edges, calibration coefficients, and minimum-run length remain undisclosed.

## Backbone: the most defensible inference

My first model family to investigate would be **Qwen3-Next-80B-A3B or a related efficient MoE with shared experts**. Confidence in that specific identity is **low**. It is an engineering hypothesis, not an identification.

The [launch overview](https://www.pangram.com/blog/pangram-4-technical) says approximately six times the parameters of Pangram 3.3. That is a ratio with an unknown denominator. It does not establish active parameter count. Nor does using EditLens technology establish that production uses the research checkpoint.

Historical evidence cuts several ways:

- Pangram's [2025 shared-task paper](https://aclanthology.org/2025.genaidetect-1.40/) used Mistral NeMo 12B. This is evidence of historical practice, not confirmation of Pangram 3.3. The paper also describes AdamW, batch 24, one epoch, and weighting false positives more heavily for checkpoint selection.
- The best research EditLens model used Mistral Small 24B. Its recipe used one epoch, AdamW, batch 24, constant learning rate 3e-5, and QLoRA. [EditLens paper](https://arxiv.org/html/2510.03154v1)
- The public Open Pangram release separately offers a 3B Llama and a 355M RoBERTa, explicitly compared against its commercial detector. Those sizes cannot safely be substituted for the production predecessor. [Open Pangram announcement](https://www.pangram.com/blog/introducing-open-pangram)

| Candidate | What makes it plausible | What prevents identification |
|---|---|---|
| Qwen3-Next-80B-A3B | 80/12 ≈ 6.7; efficient sparse model; shared dense expert gives meaning to attention+dense adaptation | Assumes the historical 12B lineage survived; hybrid attention is not named in the Pangram report |
| Mixtral-8x22B | 141/24 ≈ 5.9; fits the Mistral research lineage | Much heavier active compute; lacks the shared-expert explanation for “dense” |
| Mixtral-8x7B | Fits a roughly sixfold jump from an 8B predecessor | No evidence that Pangram 3.3 was 8B; older design |
| Qwen3-30B-A3B / Qwen3.5-35B-A3B | Practical low-active-parameter replication candidates | No established predecessor size makes the ratio decisive |
| Another open MoE | Release date and deployment constraints permit several families | Public evidence does not close the candidate set |

[Mixtral 8x22B's official specification](https://docs.mistral.ai/models/mixtral-8x22b-0-1-0-3) gives 141B total and 39B active parameters. This illustrates why using total parameters alone can distort a training-cost estimate.

The ablation winner does **not** prove candidate A is the deployed backbone: the candidates include dense and sparse models, and identities are hidden. Likewise, one failed routed-expert run is evidence for caution, not proof that expert adaptation inherently fails.

If the Qwen3-Next hypothesis is correct, the [official configuration](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json) supplies a concrete architecture: 48 layers, hidden width 2,048, 512 routed experts with ten selected per token, shared-expert intermediate width 512, and full attention every fourth layer. Full-attention settings include 16 query heads, two KV heads, and head dimension 256. These are **Qwen specifications, not verified Pangram specifications**.

Four bias-free output projections would then contain only `(15+3+2+4)×2048 = 49,152` weights. The task heads barely affect storage or arithmetic; the backbone and data pipeline dominate. This count excludes any unreported head normalization or bias.

## A concrete replication recipe

Everything numerical below that is not explicitly attributed above is **my proposed implementation or search range**. These settings make the method reproducible without pretending the missing settings have been recovered.

### 1. Build document families before splitting

Start with one million verified human source documents for the central scenario. A workable initial allocation is 20% academic, 20% general web, 15% news, 15% essays, 10% creative writing/books, 10% reviews, and 10% professional prose. These percentages are proposed, not readings of Pangram's distribution chart. Begin with 60% English and distribute the balance across target languages; tune this using actual deployment traffic and subgroup evaluation.

Deduplicate at the original-source level. Assign train, calibration, and test partitions before making derivatives. Keep every translation, mirror, edit, continuation, and adversarial variant of a source in its parent's partition. Also reserve generator families and editing-instruction families for generalization tests. Otherwise closely related synthetic examples can make the detector look much better than it is.

For each source, budget one fully generated mirror, two edited variants, and an average 0.25 humanized/adversarial variants. This yields 4.25M training documents. Target 800 retained tokens per document on average; sample/chunk to the detector's window length rather than truncating every document to its beginning. Token counts must be measured with the chosen detector tokenizer.

For mirrors, extract a brief semantic outline, title, register, language, and approximate length; generate from those metadata, without handing the generator the full original prose. Retain original/derivative pairs and reject substantial copying. Use multiple generator families with a proposed 25% maximum share per family. Do not infer training-generator proportions from an evaluation table.

For edited documents, balance light proofreading, stylistic rewrites, shortening/reordering, substantive paraphrase, and added content. Add explicit human/AI continuations and interleaved passages so the model sees actual boundaries. Their exact construction is a replication choice. Match topic and style across boundaries to avoid learning that a topic change means an author change.

Maintain clean and corrupted human controls for OCR errors, unusual casing, encoding damage, and typos. A humanizer probe otherwise risks learning “messy text” as its target. Preserve character offsets through normalization; blindly inheriting an older lowercasing pipeline would erase useful signals for this auxiliary task.

### 2. Reconstruct the supervision engine

Ask the clause splitter for character spans rather than rewritten text. Validate that the returned spans cover the target without silently modifying it. Cache source splitting once per source and reuse it across edited variants.

For each target clause, retrieve candidate source spans, including adjacent clauses. Score candidates with word n-gram overlap, character n-gram overlap, and semantic embedding similarity. Use a multilingual embedding model for multilingual training; Linq-Embed-Mistral is a lineage-based English baseline, not an identified Pangram 4 component.

I would audit 2,000–10,000 source/target clause pairs and tune lexical and semantic thresholds on them. Useful exploratory ranges are lexical similarity 0.80–0.98 and embedding cosine 0.75–0.95, **only after defining the metrics and embedding model**. These numbers are not transferable universal thresholds. Short generic clauses, negation, named entities, copied quotations, and translations deserve separate audit strata.

Map clause spans to tokens through tokenizer offsets. Define how tokens crossing a boundary are handled—maximum character overlap is a reasonable starting rule—and mask special tokens and padding. Derive each window's segment target from its own covered characters rather than inheriting a whole-document target.

Use 15 evenly spaced anchors `b/14` as a starting reconstruction; nearest-anchor assignment is easy to reproduce. Compare that against endpoint-special buckets. The actual edges are unknown. Do not transplant EditLens's cosine-distance thresholds into a different character-fraction target.

### 3. Optimize a small set of trainable components

The [public EditLens Llama configuration](https://raw.githubusercontent.com/pangramlabs/EditLens/main/configs/llama.yaml) gives a useful starting prior: rank 8, alpha 16, dropout 0.05, learning rate 3e-5, zero weight decay, constant schedule, NF4 storage, and BF16 compute. It is not a Pangram 4 configuration.

| Setting | My replication starting point | Sensitivity check |
|---|---|---|
| Backbone | Qwen3-Next-80B-A3B; benchmark a smaller Qwen MoE first | Backbone choice remains uncertain |
| Weight precision | BF16 if feasible | QLoRA is a memory-saving alternative, not disclosed P4 behavior |
| LoRA | Rank 16, alpha 32, dropout 0.05 | Ranks 8/16/32; alpha 2r |
| Target modules | Attention/mixer projections and shared dense expert | Compare attention-only; initially freeze routers and routed experts |
| Optimizer | AdamW, β=(0.9,0.999), epsilon 1e-8 | Standard defaults, not recovered settings |
| Adapter learning rate | 3e-5 | 1e-5, 3e-5, 1e-4 |
| New-head learning rate | 1e-4 | Compare shared 3e-5 |
| Weight decay | 0 | Compare 0.01 |
| Gradient clipping | Global norm 1.0 | Examine rare high-loss batches |
| Schedule | Constant after 2% warmup | Compare inherited constant/no-warmup |
| Effective batch | 128 source windows | 32–256; tune by tokens and gradient stability |
| Stage 1 exposure | One source-token-equivalent pass | Early stopping on clean held-out objectives |
| Stage 2 exposure | Half a source-token-equivalent pass | 0.25–1 pass; emphasize boundary-rich windows |

Inspect module names rather than applying an unrestricted `all-linear` target rule: a router “gate” and a feed-forward gate are different modules. For a hybrid backbone, attention-only can accidentally mean adapting only a minority of layers unless mixer projections are deliberately included.

My reconstructed objectives are:

```
L1 = CE(segment) + 0.25 × CE(humanizer(detach(h_last)))
L2 = CE(segment) + mean_valid_tokens CE(provenance)
     + 0.5 × CE(mixed) + 0.25 × CE(humanizer(detach(h_last)))
```

The token weight 1 and mixed weight 0.5 are proposals. Normalize token loss before combining it with sequence losses; summing hundreds of token losses changes the effective balance drastically. Mask missing auxiliary targets. Preserve segment/probe heads between stages and initialize the new heads separately. Reset optimizer state when introducing the fresh adapter.

Do not add language-model next-token loss or a generator-identification objective merely because earlier systems used related ideas. Keep the pretrained LM vocabulary projection out of the classifier forward path. Freeze router parameters initially; if router regularization is used, document that additional objective explicitly.

### 4. Mine failures, then calibrate independently

For the central budget, score 20M reserved human documents, audit high-scoring cases, and supplement each selected human example with a matched AI mirror. A candidate detector's false-positive set may contain mislabeled or contaminated sources; verify provenance before treating every error as truth.

Interpret “from scratch” here as restarting task adaptation from the original pretrained backbone, not repeating foundation-model pretraining. Re-run the adaptation stages on the augmented corpus. Retain broad random examples so a small set of difficult domains does not monopolize the final training distribution.

Use a separate calibration split to tune token/segment fusion and smoothing. Start with minimum-run values 8, 16, and 32 tokens and select against annotated boundaries. Sweep smoothing and class-prior offsets jointly with document length and language. Those values are proposed; there is no universal smoothing penalty independent of logit scale.

Production-level false-positive claims need large independent human sets. At a true rate around 4×10^-5, 100,000 examples yield only about four expected errors; one million yields about forty. Also report per-domain results and confidence intervals. A matching architecture cannot guarantee matching low-tail error rates if the human-source distribution differs.

## Revised cost: measured run as the anchor

The [model card](https://www.pangram.com/research/model-card/pangram-4) reports four days on eight H100s. That is 4 × 24 × 8 = **768 H100-hours**. At the user-supplied $2.50/H100-hour planning rate, the disclosed run costs **$1,920**. At $2–$3/hour the range is $1,536–$2,304. These hourly rates are assumptions, not newly obtained rental quotes.

The previous calculation used managed per-token training prices as a proxy for custom GPU training. That produced $14,960 for a final run, roughly 7.8 times this anchor. The proxy was inappropriate once the actual runtime was available. The old screening proxy and unverified corpus sizes compounded the overestimate. Staffing, contractor audits, generic infrastructure allowances, and blanket contingency have been removed.

### Compute budget

Use final-run equivalents to budget other jobs. These are proposed experiment allocations, not recovered Pangram run logs. A 5% experiment means 5% of the aggregate compute, achieved through fewer tokens, fewer steps, or a cheaper model—not necessarily 5% of elapsed time.

| Item | Compute relative to disclosed run | Cost at $2.50/hour |
|---|---:|---:|
| Final training run | 1.00 | $1,920 |
| Preliminary candidate for mining | 1.00 | $1,920 |
| Eight short backbone/adapter experiments | 8 × 0.05 = 0.40 | $768 |
| Four longer loss/data experiments | 4 × 0.10 = 0.40 | $768 |
| Hard-negative screening and embedding work | 0.30 allowance | $576 |
| Held-out scoring and calibration | 0.15 allowance | $288 |
| **Total compute** | **3.25 runs / 2,496 H100-hours** | **$6,240** |

Treat the disclosed runtime as covering the complete final fit, including its two stages. Do not add another factor for Repeat2, validation, or the two stages on top of this measured runtime. The release does not explicitly delimit all jobs included in that number; if it already includes the preliminary candidate, the table double-counts $1,920 and should be reduced accordingly.

The screening and evaluation rows are spending allocations that must be checked with a throughput benchmark. They are not evidence that a particular pool size fits. As an illustration, retaining the earlier 20M-document pool at 800 source tokens each and approximately four processed tokens per source token requires 64B processed tokens. Fitting that into 230.4 H100-hours requires about 77,000 processed tokens/second/H100. That rate must be measured; if unattainable, reduce the pool, screen with the cheaper segment-only stage, or increase this row. Do not quietly claim the same large-scale evaluation within a smaller budget.

| Experiment scope | Run equivalents | Compute cost |
|---|---:|---:|
| Lean: shortened preliminary candidate and fewer short trials | 2.05 | $3,936 |
| Central: the itemized plan above | 3.25 | $6,240 |
| Broader: more experiments and screening | 5.00 | $9,600 |

One additional full retraining adds $1,920. An isolated 10%-compute experiment adds $192. A smaller backbone can lower costs further, but parameter count alone is not a reliable runtime scaling factor for sparse models.

### New data generation is separate

Runtime constrains training compute; it does not establish how much training data exists or what it cost to generate. The million-source corpus used earlier was a scenario, not a finding. It should not be treated as mandatory for a reproduction.

For a transparent optional data budget, preserve the earlier token assumptions: per human source, one metadata call (1,000 input/150 output), one mirror (300/800), two edits (1,000/800 each), an average 0.25 adversarial variants (1,000/800), and three clause-splitting calls (one cached source and two targets, 1,000/900 each).

Use $0.50/M input and $2.50/M output for batch Haiku metadata/splitting, based on [Anthropic pricing](https://www.anthropic.com/claude/haiku). For the heterogeneous generators, retain the explicit blended assumption of $1/M input and $5/M output. The resulting cost is **$0.024675 per human source family**:

| New human source families | New training documents at 4.25 per source | Generation and clause-labeling cost |
|---|---:|---:|
| 100,000 | 425,000 | $2,467.50 |
| 250,000 | 1,062,500 | $6,168.75 |
| 1,000,000 | 4,250,000 | $24,675 |

These totals exclude API retries and specialized commercial humanizer fees; pay those only if incurred. They assume no separate wholesale regeneration of a huge evaluation corpus. Embedding and screening GPU work is already in the compute budget. Existing usable labeled data can reduce incremental generation spend; it cannot be assumed equivalent to Pangram's private corpus.

Combining the central compute allocation with 100k–250k newly generated source families gives **$8,708–$12,409**. This conservatively retains a full-size final training allocation even for the smaller fresh-data scenario; actual training may be shorter, or existing data may fill the rest. It does not predict identical accuracy from less data.

If we keep the earlier million-source all-new data assumption, the corrected central total is **$30,915**, not $128k. Lowering training cost alone does not justify erasing genuine generation calls.

### Working budget

**About $6k for training, experiments, and GPU-based evaluation; about $9k–$12.5k including a targeted fresh synthetic dataset.** A practical initial compute range is $4k–$10k. Exact Pangram-quality performance and the size of the required data refresh remain unknown.

No team, salaries, general business overhead, or automatic contingency is included. This estimates direct incremental resources for an agent-assisted implementation. No separate agent subscription or API usage charge is assumed.

Machine-readable arithmetic: [pangram4-cost-scenarios.json](pangram4-cost-scenarios.json).
