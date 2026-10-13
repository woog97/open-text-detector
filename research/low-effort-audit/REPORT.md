# Low-context and surface-failure coverage audit

October 1, 2026. **This is not a validated quality-label dataset.** The assistant is an LLM and may favor familiar model-generated prose. No LLM judge scores, detector scores, or new generation were used. No training examples were filtered or relabeled from this audit.

## What was inspected

- RAID: deterministic seed1012026, two examples per model/source from its previously defined training source groups (24 records including humans). Read the full first sample per group:11 AI abstracts plus one human abstract. The second sample per group is retained but not qualitatively reviewed. Sampling was across raw training groups, not restricted to the final overlap-filtered donor pool; this is a coverage illustration, not a training-selection audit.
- Arena: inventory of100 prompts in the existing32-model collection; six purposively selected research/technical/proposal prompts, with one eligible response randomly selected per prompt using seed1012026. Inspected the first1600characters of each, not every full response. This is not a representative prevalence estimate.
- Separately measured observable signals on **all actual prepared training draws** of the current control and the external portions of RAID/MAGE, and all3141mechanically eligible responses in this Arena file. This does not cover every newer Arena expansion.

## Findings that do not require trusting an aesthetic judgment

1. The11 reviewed RAID AI examples were requested from **a paper title alone**. They did not receive an original paragraph, surrounding context, content bullets, or experimental evidence. This is a directly observable contrast to our fidelity-focused reconstruction protocol.
2. RAID GPT-2 example `3d957cd2-a5c2-464c-b493-f4c4caabd816` repeats the same long sentence and method introduction. RAID MPT example `57ab09c8-530c-4f33-9dc0-977af41a87c3` degenerates into a long chain of synonyms. These are visible output failures, not a model's taste about polished writing.
3. Some title-only outputs assert dataset counts, new methods and experimental success. Those details are **not supported by the supplied prompt**. They may reflect model knowledge or errors; this audit did not compare them against original papers, so it does not label them hallucinations.
4. Arena includes a one-sentence request for extensive Japanese M&A research. The sampled answer supplies a long report and numerical claims despite receiving no sources. Those claims were not independently fact-checked. Another sampled model explicitly labels its proposal a template with placeholders: missing context can produce an appropriately qualified response, not necessarily bad writing.
5. A sampled human abstract also contains broad benefit language. Wording such as 'insights' or formulaic introductions cannot establish authorship or low quality.

## Reproducible surface measurements

A word is a lowercase Unicode regex word. For every8-word sequence, count occurrences beyond its first occurrence; divide their sum by all8-word sequence positions. The10% threshold below is a descriptive, arbitrary flag, not a validated quality boundary. Repeated sentences require at least8normalized words. Code and assistant self-reference are format indicators, not errors.

| Collection | AI-containing training draws / eligible responses | At least10% repeated8-grams | Any repeated long sentence |
|---|---:|---:|---:|
| Paper-only control, actual windows |15,764|0 (0.00%)|7|
| RAID external AI, actual windows |4,211|515 (12.23%)|384|
| MAGE external AI, actual windows |4,210|33 (0.78%)|30|
| Arena eligible responses, full outputs |3,141|6 (0.19%)|11|

Draws are repeated exposures, not independent documents. Control windows can include human context; external AI windows are pure outputs. Arena full outputs differ in length. Do not interpret this table as matched 'slop prevalence,' proof of human/AI separability, or an estimate for real submissions. Nevertheless, it verifies that substantial repetition is present **inside the RAID windows actually going into training**, not only in tails removed by cropping.

## Decision

The concern is partially supported as a **coverage gap**, not as a claim about most real AI papers: our paper-only recipe lacks the substantial repetition seen in the prepared RAID addition, and its generation protocol provides much more guidance than title-only drafting. The already-running RAID experiment therefore tests part of this hypothesis without another generation batch.

Do not discard polished examples, label all title-only writing low quality, or add another redundant experiment yet. The RAID result cannot isolate quality effects from generator/domain/prompt changes. A causal follow-up would compare training pools matched on generator, domain, length and budget, differing in an independently defined failure slice, with untouched paper evaluation.

For quality ground truth, use two human reviewers blinded to generator/detector result and assess separate dimensions: readability, repetition, specificity, factual support checked against the source paper, and fitness for a paper section. Include human writing and allow 'uncertain'; record disagreements. My annotations can identify passages for review but should not determine inclusion/exclusion. At deployment, both rough-human FPR and careful-AI recall must remain explicit checks.

## Files

- `signals.py`: reproducible feature definitions.
- `training-signals.json`, `arena-signals.json`: aggregate measurements.
- `raid-sample.json`, `arena-sample.json`: sampled inputs/outputs and IDs (removed from the repo on Oct 13, 2026: third-party text).
- `observations.json`: provisional observations with explicit review scope.
