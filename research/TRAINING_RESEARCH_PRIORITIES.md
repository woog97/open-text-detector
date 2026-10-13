# Research priorities for improving AI-paper detection

Updated: October 1, 2026. This is the evidence and historical research agenda, not authorization to launch experiments.

**Current prioritization:** see the [complete ranked backlog](EXPERIMENT_BACKLOG.md). Its ordered ratings supersede the preliminary ratings below. In particular, difficult-human mining moves to 5/5, clean Arena training to 4/5, and new generation is deferred to 2/5. Completed audits and pilots are evidence, not pending jobs. Historical preparation/running statements below describe their respective snapshots; use the backlog and live Space records for current state.

## Priority scale

Ratings reflect expected benefit to **paper token/sentence detection per unit of cost and time**, with uncertainty included.

- **5/5 — Next:** resolves a critical uncertainty or tests a cheap, plausible route to substantial improvement.
- **4/5 — High:** strong next candidate once its prerequisite is met.
- **3/5 — Conditional:** worthwhile only if a diagnostic identifies the corresponding failure.
- **2/5 — Defer:** limited expected gain, weak evidence, or a more expensive way to test the same idea.
- **1/5 — Avoid for now:** likely redundant, misleading, or poor value.

A high rating is not evidence that the intervention works. Every proposal below needs a control and a measurable success criterion.

## Objective and decision rules

Improve localization of AI-written text in research papers, especially unfamiliar edits and generation workflows, while keeping false accusations against human writing low. Broad-writing benchmarks are supporting diagnostics, not the primary optimization target.

Primary measurements: paper token and sentence recall/F1, precision, human false-positive rate (FPR), and performance by editing operation, text length, generated fraction, and generator. Report both paired human originals and untouched body paragraphs. Compare recall at matched false-positive operating points; separately calibrated thresholds do not guarantee equal FPR on unseen papers.

Use separate training, model-selection, threshold-calibration, and final-test data. Keep related papers, prompt groups, response variants, and near-duplicates together. Use confidence intervals grouped by paper/prompt and confirm promising gains with another seed. Do not treat correlated tokens as independent evidence.

## Initial priorities — superseded by the ranked backlog

| Rating | Question / intervention | First useful experiment | Decision it should enable |
|---|---|---|---|
| **5/5** | Is a failure caused by scoring, the cutoff, or poor learned separation? | Finish the saved-score audit and verify repeat-inference consistency; inspect paper failure slices. | Choose between calibration work and a targeted training-data change. |
| **5/5** | Can we reuse diverse AI text already available? | Inventory unused Arena responses and candidate Hugging Face training splits; audit provenance, human counterparts, and overlap before training. | Establish a clean, low-cost diversity pool without new paid generation. |
| **5/5** | Does that pool improve paper generalization? | Paper-only control versus broad-first/paper-second training, with the same backbone and total example budget; also report processed tokens and time. | Determine whether diversity transfers to papers rather than merely improving broad benchmarks. |
| **4/5** | Are our prompts too narrow? | With Luna, vary realistic requests and how content requirements are expressed while holding source papers and generated scope fixed. | Test workflow diversity without paying for another model family. |
| **4/5** | Does the scope of generated text matter? | Separately vary sentence edits, multi-sentence replacements, paragraphs, and longer spans; include realistic mixed human/AI context. | Improve short-span detection and robustness to the amount of AI text. |
| **4/5** | Are we selecting the right checkpoint? | Compare loss-based selection with a predeclared paper recall/FPR objective on model-selection data, with thresholds fitted separately. | Avoid choosing a lower-loss model that is worse for deployment. |
| **3/5** | Which human negatives actually matter? | Sample difficult, verified human paragraphs from training/selection papers; add targeted negatives, not just a larger global human fraction. | Reduce specific false-positive modes without unnecessarily suppressing AI recall. |
| **3/5** | Does domain or request diversity help? | Compare paper-adjacent technical prose with broad prose at the same mix weight, balancing human/AI sources within domain. | Separate useful stylistic coverage from irrelevant domain expansion. |
| **2/5** | Do newly generated examples from other model families add unique value? | Only after reuse and Luna variations: a small, matched comparison against an equally sized Luna addition. | Justify incremental generation expense with paper gains. |
| **2/5** | More backbone, rank, human-ratio, or batch/LR sweeps? | Finish interpreting existing runs; pursue only a specific failure or promising result. | Avoid spending GPUs on changes smaller than unresolved data/measurement effects. |

## AI-example diversity: cheapest routes first

“More diversity” has several meanings. Our results motivate testing them; they do **not** establish that generator family is the dominant cause of failure.

### 1. Reuse Arena responses — 5/5 for inventory; conditional for training

We already have **3,200 successful responses, 32 model IDs, and 100 prompt groups** in `benchmarks/otd/runs/arena100-combined/dataset.jsonl`; 3,141 are mechanically eligible. This is valuable diversity already paid for.

**Split requirement:** an exact normalized-text audit found all 3,141 eligible responses, spanning all 100 prompt groups, in `eval_suite/bundle/full.jsonl.gz`. The comparison profile also contains 635 matching responses from 20 groups. We can reuse this collection by creating a new, versioned train/development/test split; we cannot continue using the old full Arena collection as an independent benchmark for models trained on its new training partition. Exact matching alone also misses paraphrases and related conversations.

**Agreed direction: partition the existing collection before using it for training.** Start with a proposed 70/15/15 split of the 100 prompt groups, approximately stratified by request category. This allocation is a planning default, not a finalized split. Keep every model response, regenerated variant, related conversation, and near-duplicate prompt in the same partition. Freeze a seed, group assignments, source hashes, and split manifest before training. Model count does not multiply the number of independent prompts.

Use development groups for mix/model selection and test groups only for the final comparison. Keep threshold calibration on its existing separate data; if we need broad-domain calibration, obtain a separate adequately sized human/AI calibration set. Fifteen test prompt groups are a small exploratory test, especially for generator-specific comparisons. Credible matched human examples must be split by their source document too; AI responses alone cannot measure human FPR.

Version the Arena evaluation profile and exclude **all** new training/development groups from every profile used for final reporting. Retain the old suite and results as historical artifacts. Rescore baseline models on the new test partition for an apples-to-apples comparison. Because the existing collection has already been evaluated and inspected, describe the new test as held out from training but previously exposed during research—not a pristine blind benchmark. Do not choose group assignments based on detector scores.

Next, inspect unused conversations and assistant responses from the wider cached Arena source to expand the training pool and obtain a stronger, untouched final test. The source manifest records 33,000 conversations; the number actually eligible after excluding related prompts and near-duplicates is **not yet established**. Original assistant replies may provide this expansion without fresh generation.

Human user prompts are **not** matched human answers. Pair AI responses with credible human prose of similar topic, genre, length, and formatting. Otherwise the classifier may learn “assistant answer versus short question” instead of authorship. Keep refusals, code, lists, quotation, and boilerplate identifiable; do not let them become easy class shortcuts.

### 2. Mix in Hugging Face datasets — 5/5 for screening, 4/5 for a clean pilot

Look for existing, separately designated training data with multiple generation styles, source models, and domains, plus credible human source text. Hosting on Hugging Face is not itself evidence of quality, provenance, or independence from our benchmarks.

Screen candidates for:

- Stable dataset revision, usable terms, traceable text origin, model/request metadata where available, and explicit split definitions.
- Human examples that are genuinely human, and AI labels that distinguish fully generated from edited, quoted, or mixed text.
- Exact and near-duplicate exclusion against all frozen evaluation profiles, calibration/selection sets, and related source documents or prompts.
- Adequate human/AI balance **within** domain and source, not only across the pooled dataset.

Existing DetectRL, EPOCH, MELD, and other benchmark examples must remain evaluation data. An independent training split from the same dataset family would require its own source-level overlap audit.

Document labels are not automatically token labels. Use document-only supervision for ambiguous provenance; use token/sentence supervision only where the text origin supports it. For broad pretraining, add an appropriate document objective rather than labeling every token in a mixed document AI. Do not blindly reuse our 15-bin AI-fraction head for uncertain document labels.

### 3. Vary requests, prompts, and scopes with Luna — 4/5

Keep generator identity fixed initially so we can identify the effect of the task:

- Requests: clarify an argument, explain a mechanism, describe results, summarize limitations, compress prose, or expand notes.
- Prompt format: content bullets, experiment notes, reviewer-requested revision, or drafting from surrounding context. For held-out reconstruction, continue hiding the original paragraph from the generator.
- Scope: individual sentences, several sentences, a paragraph, or longer contiguous text.
- Domain: different technical subfields and paper sections before unrelated prose.

Change one dimension at a time in the first pilot. Preserve factual fidelity and realistic quality; deliberately awkward rewrites can teach shortcuts. A minimal edit does not make every surviving human-written token AI: retain operation-aware provenance and ambiguous labels where necessary.

### 4. New model-family generation — 2/5 until cheaper routes are tested

Defer large Claude/Gemini or other paid-family batches. The user-reported costs are roughly 40× and 8× Luna Flex, respectively; these are planning inputs, not independently verified current prices. Existing cross-model responses should be used first where splits permit.

Escalate only if held-out model-family failures persist after prompt/scope/data-mix interventions. Compare a small new-family addition with an equal-budget Luna diversity addition; cost per improvement matters more than model count.

## Concrete shortlist requiring no new generation

Screened October 1, 2026. Dataset-card screening is complete; row-level provenance and cross-suite overlap audits are **not** complete. These are candidates, not clean training pools yet.

| Priority | Dataset / intervention | Why test it | Important limitation |
|---|---|---|---|
| **5/5** | [RAID](https://huggingface.co/datasets/liamdugan/raid), clean training abstracts first | Human scientific abstracts and outputs from multiple generators let us test diversity close to our target domain. `source_id` connects related examples. | Start with `attack=none`; split by source, not generated row. Abstracts are not full paper bodies. Preserve official test data. |
| **5/5** | Recombine existing training-paper human/generated passages | Place existing generated paragraphs in their actual human context; vary crop position and AI fraction, and include short contiguous generated spans where joins remain coherent. Tests localization coverage with exact inserted-span provenance. | Audit existing augmentations first to avoid a redundant experiment. Artificial joins can become shortcuts; evaluate on independently generated workflows, not just splices. Copied or ambiguous content retains its existing label treatment. |
| **4/5** | [MAGE](https://huggingface.co/datasets/yaful/MAGE), balanced subset of official train | Roughly 319k training rows; broader human/AI sources provide a cheap test of transfer from general writing. | Its displayed human examples have label **1**, opposite our internal convention. Verify/remap labels. Source metadata is less convenient than RAID's explicit source linkage; audit related texts across splits. |
| **4/5** | Existing Arena outputs after agreed group split | Adds already-paid-for model/request variety. | All eligible rows in the audited 32-model collection appear in the old full benchmark. Version the suite, rescore controls, and supply credible human counterparts. Wider inventory may contain additional models; do not confuse one collection's count with all available responses. |
| **3/5** | [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) | Human and ChatGPT answers to shared questions reduce topic mismatch between classes. | Old, narrow generator coverage. Keep every answer to one question together; do not count component files and `all.jsonl` twice. |
| **3/5** | [GRADTEX](https://huggingface.co/datasets/elisabeth-pl-pl/GRADTEX) | Card describes seven generators and thirteen editing/generation scenarios, with training/validation and held-out scenarios. Relevant to workflow diversity. | Derived from MAGE: jointly audit source overlap. Graded document labels do not establish exact AI token spans. Inspect provenance before deciding its training objective. |

M4/SemEval Task 8 is another paper-adjacent lead: arXiv/PeerRead sources and a human-to-machine boundary task. The [HF copy found](https://huggingface.co/datasets/d0rj/SemEval2024-task8) explicitly calls itself an unofficial mirror of the [authors' M4 project](https://github.com/mbzuai-nlp/M4). Verify upstream files and splits before prioritizing it over RAID. Do not pool mirrors or derivative datasets as independent evidence.

### Recommended next GPU comparisons

Use ModernBERT-large LoRA for the first inexpensive screening round, with the same initialization, seed, optimizer and a fixed processed-token budget. Run a fresh matched control rather than comparing different amounts of training against an old checkpoint.

1. **Paper-only control.** Repeat the selected paper recipe under the experiment's budget.
2. **Paper-adjacent diversity.** Replace a predeclared 20% of training exposure with clean RAID abstracts, balanced by human/AI class and capped per generator/source.
3. **Broader diversity.** Replace the same 20% with a balanced MAGE training subset. Use the same scheduling as arm 2 so curriculum does not confound the comparison. This compares useful data packages, not a pure domain effect: generators and prompts also differ.
4. **Localization coverage.** Keep paper sources and generator pool fixed; change only the training crop/context composition to cover short AI regions and varied human/AI boundaries. Run after confirming this meaningfully extends the current augmentation.

The 20% share is a starting hypothesis, not a tuned optimum. Start with tens of thousands of eligible unique examples if available, rather than millions or a ratio grid. Documents with uncertain mixed authorship require document-only supervision; introduce any new objective in a matched control too. Use token gold only where provenance supports it.

Split before sampling or augmentation; pin revisions, hash texts, group source documents/prompts and near-duplicates, and exclude all selection/calibration/test relatives. The source-card audit alone does not establish independence from DetectRL, EPOCH or other suite components.

Select using paper validation performance at a separately calibrated low-human-FPR cutoff; report realized test FPR and uncertainty alongside recall/F1. Keep broad scores secondary. Confirm a meaningful winner with another seed before paying for a Qwen3.5 replication. A cheap secondary experiment is training-only hard-human-negative mining, replacing random negatives at the same human share; never mine final-test failures into training.

User authorized these trials later on October1. The control, RAID and MAGE arms are being prepared/queued on GPUs0,1,3; a contiguous short-AI boundary arm follows the control onGPU0. See [execution protocol](../benchmarks/otd/training/diversity-sweep/README.md). Existing mixed-window coverage contains only18–25 examples per12,000draw epoch with under200AI characters, motivating the fourth arm. No OpenRouter calls. Live status remains in the Space records.

## Minimal next training comparison

First finish the score audit and freeze the clean training pool. Use one inexpensive, established backbone and the same LoRA configuration:

1. **Control:** current paper recipe.
2. **Broad-first, then paper:** allocate an initial portion of the fixed training budget to the audited broad pool, then specialize on papers.
3. **Optional interleaved mix:** only if useful to distinguish curriculum from data composition; keep the broad/paper exposure identical to arm 2.

Choose one modest broad-data share in advance, not a large ratio grid. Equal example counts are not equal compute: report supervised tokens, processed tokens, optimizer updates, and wall time. Use a token-budget-matched follow-up if text lengths materially differ.

Advance an intervention only if it improves predeclared paper recall/localization at acceptable FPR, without a material regression in a major paper workflow. Select on validation; calibrate separately; evaluate finalists once. Better broad scores alone do not justify adopting a paper detector.

## Findings from experiments so far

### Human sampling mix: a tradeoff, not a universal improvement

ModernBERT-large LoRA tested 25%, 50%, and 75% **novel untouched-human sampling shares**. These are not total human-label percentages: the paired stratum also contains human originals. All arms used the same number of examples/epochs, but their token counts differed.

The 50% arm won validation loss: 0.1720 versus 0.1785 at 25% and 0.1977 at 75%.

| Frozen-test metric | 25% | 50% |
|---|---:|---:|
| Paper target token F1 | 88.01% | 87.75% |
| Paper target sentence F1 | 83.02% | 85.32% |
| Untouched-paper token FPR | 0.31% | 0.11% |
| Reconstruction token recall | 53.10% | 49.55% |

**Update:** do not keep increasing human share indiscriminately. Investigate targeted negatives and selection criteria. Single seed; no significance claim.

### Qwen3.5: LoRA is promising, but operating points differ

| Frozen-test metric | Full tuning | LoRA |
|---|---:|---:|
| Paper target token F1 | 55.20% | 93.84% |
| Paper target token recall | 38.13% | 90.53% |
| Paper target sentence F1 | 53.56% | 92.89% |
| Paper target human-token FPR | 0.00% | 2.27% |
| Reconstruction token F1 | 58.57% | 76.43% |

**Update:** retain LoRA as a strong candidate; do not conclude that it dominates full tuning at the same FPR. Both broad-domain document detectors remain weak at their frozen paper-calibrated thresholds. Full per-group results: [Qwen3.5 comparison](../benchmarks/otd/training/lora-comparison/continuation/qwen35-final-comparison.md).

### Score audit: calibration and learned separation both matter

For the 50% human ModernBERT model, the document cutoff is approximately 0.97. Using saved scores, an optimistic cutoff fitted directly to each test dataset would raise recall from 0% to about 30% on EPOCH, and from 0.12% to about 8.2% on DetectRL, with at most 1% observed human FPR within each dataset.

**Update:** threshold mismatch contributes, but changing the threshold alone does not deliver strong strict-FPR detection on these larger broad datasets. This supports a diversity pilot, not a conclusion that new model families are specifically required. These test-fitted numbers are diagnostics, not validated deployment performance; small human samples are especially unreliable at 1% FPR.

The token/sentence rerun initially differed from saved document means by up to 0.0292 after changing batch composition. Repeating the exact original batches reproduced saved means exactly (maximum difference 0.0). The discrepancy is batch-composition-sensitive inference, not evidence of checkpoint drift; preserve batching in reproducibility checks and assess sensitivity before deployment. For the 50% human ModernBERT model, reconstructed-paper token recall is 49.55% at 0.61% observed human FPR; a test-fitted diagnostic cutoff reaches 57.10% at at most 1% FPR. On paired paper targets, current token recall is 79.03% at 1.03% FPR and the diagnostic at-1% recall is 78.82%, so lowering that cutoff is not a free improvement. Sentence FPR on paired targets is 1.67%; enforcing 1% diagnostically reduces recall from 75.69% to 63.65%. These remain optimistic test diagnostics, not production calibration. No production thresholds were changed. Detailed records: `benchmarks/otd/training/score-audit/`.

### Effective-batch sweep: still running; limited speed conclusions

Qwen3-0.6B LoRA uses rank 128, alpha 32, attention and MLP adapters. Effective batches 32/64/128 are being tested with five batch/LR combinations. The larger LR candidates use a square-root scaling heuristic, not a proven optimum. The search is narrow and single-seed.

Microbatch remains four, so this mostly tests learning dynamics from changing optimizer-update frequency. It does not test each batch size's maximum throughput. Selection is currently by validation loss; that is a known limitation for our recall/FPR goal.

### Throughput profiling: useful enabling work, not a quality result

Qwen3-0.6B profiling found approximately 3.6× stage-1 speedup with microbatch 16 and 1.9× stage-2 speedup with microbatch 8, both with checkpointing off, versus microbatch four/checkpointing on. Effective batch stayed 32. BF16, finite-gradient, and frozen-parameter checks passed; sampled gradient-signature differences were about 0.14% and 0.12% relative L2.

These are short warmed measurements on selected training examples, not end-to-end speedups or proof of equal final quality. Validate longest-example memory headroom and learning behavior before applying them. Preserve completed runs as controls.

## What to avoid next

- Training on the new Arena training split while still reporting the old full Arena benchmark as independent evidence; all final profiles must respect the new group split.
- Attributing broad misses to generator identity without separating prompt, scope, domain, and calibration effects.
- Mixing AI chat answers with mismatched human prose and mistaking source/style shortcuts for detection.
- More arbitrary human-ratio or backbone sweeps before resolving the largest failure modes.
- Treating lower training loss, higher GPU utilization, or faster updates as the research objective.

## Handoff for a separate data-design discussion

Resolve these questions before implementation: the fixed Arena prompt-group split and benchmark version; which additional Arena/Hugging Face examples pass the overlap audit; where matched human prose comes from; which labels support document versus span supervision; which paper failure slice the addition targets; and the smallest controlled comparison that can change our decision. Keep expensive new-family generation last.

## Evidence locations

- Arena inventory: `benchmarks/otd/runs/arena100-combined/dataset.jsonl`; source manifest: `benchmarks/otd/runs/arena100/source_manifest.json`.
- Frozen evaluation profiles: `benchmarks/otd/eval_suite/bundle/`.
- Current training composition: `benchmarks/otd/training/backbone-comparison/prepared/manifest.json`.
- Space records: `/data/workspace/paper-mix-sweep-v1/`, `/data/workspace/paper-batch-sweep-v1/`, `/data/workspace/score-audit-v1/`.
- The repeat-inference audit and initial diversity pilots are now complete; their findings appear below. Follow the ranked backlog for current decisions.


## Low-effort generation coverage audit — 5/5 diagnostic, partially addressed

[Audit report](low-effort-audit/REPORT.md): the title-only RAID protocol already adds a lower-context drafting mode. A reproducible repetition flag fires on515/4211external RAID AI training draws versus0/15764AI-containing paper-control draws. These are differently composed windows and correlated draws, not validated quality prevalence. Assistant aesthetic judgments are provisional; no quality filtering or label changes were made. Wait for the running RAID comparison before commissioning a redundant generation batch. A later matched failure-slice experiment needs blinded human review and rough-human controls.


### Initial diversity results — October1

Control/RAID/MAGE completed. Paper-target tokenF1:89.56/80.68/81.32%; recall82.44/68.11/69.32%; humanFPR1.56/0.70/1.10%. Workflow reconstruction tokenF1:69.17/63.88/65.58%. RAID Arena document recall49.75% versus0%control; MAGE DetectRL document recall37.88% versus0.06%control. These separately calibrated models are not matched on observed FPR; one seed. Neither is a clear paper-quality promotion. RAID also raises untouched-paper and ELLIPSE false positives. Wait for boundary results and inspect validation-based recall/FPR separation before choosing a confirmation or curriculum followup. [Full metrics](../benchmarks/otd/training/diversity-sweep/RESULTS.md).
