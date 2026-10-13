# Arena-20: fresh generation benchmark protocol

Status: generation, local response screening and both baseline detector runs complete; see [results](ARENA_20_RESULTS.md). Execution is tracked in `runs/arena20/`. Exactly **20 distinct prompts**, reused across 20 selected generators (400 cells). The complete 33,000-row source produced 26,679 unique requests; 79 candidates were screened to select 20. This complements the already prepared benchmarks; it does not imply that every other item in the coverage inventory is complete.

## Reference and scope

Pangram's [technical report §5.2](https://arxiv.org/html/2607.27183v1#S5.SS2) describes filtering Chatbot Arena prompts with Mistral-Small-24B-Instruct-2501, retaining 20,000 open-ended requests, and generating responses from 26 models. Coding, math, and multiple-choice requests are excluded. Its [task definition §3.1](https://arxiv.org/html/2607.27183v1#S3.SS1) further limits the target to substantial original prose, rather than copied material or brief factual replies. The report does not provide enough detail to reconstruct its exact source revision, prompt IDs, filter instructions, seed, or sampling procedure. Everything below is our explicit pilot design, not an assertion about unpublished Pangram choices.

Use [lmsys/chatbot_arena_conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations), subject to its access conditions. Its card describes paired conversations collected in April–June 2023, including question IDs, language tags, and user identifiers. Prompt licensing differs from model-output licensing; this protocol uses user prompts, not the archived assistant responses. The repository currently requires acceptance of access conditions. Pin the dataset commit, source files, checksums, and split before sampling; selected revision is `1b6335d42a1d2c7e34870c905d03ab964f7f2bd8`; source and frame checksums are in `data/arena20/source_manifest.json`.

**Population we can make a claim about:** eligible, unique, self-contained English opening requests in that pinned corpus. This is not a representative sample of today's traffic, all languages, follow-up turns, or all users. English is a deliberate local-baseline scope choice. Repeated requests have equal weight after deduplication, so this measures unique-request coverage rather than traffic-frequency-weighted performance.

## Build the sampling frame

1. Read every row in the chosen release/split, not the first page or a convenient file prefix. Record any unreadable rows and stop on incomplete acquisition.
2. Extract only the first user message. A pair of model conversations represents one request, not two candidates. Verify that the opening user message agrees across the two arms; log and exclude missing or inconsistent openings rather than choosing whichever is more convenient. Exclude follow-up turns requiring prior assistant context.
3. Keep the original prompt unchanged for generation. For duplicate detection only, normalize Unicode to NFC, strip outer whitespace, and collapse whitespace runs to one space; preserve case and punctuation. Group identical normalized prompts across rows/arms, retaining all source IDs and occurrence counts. Choose the representative original text by the lowest stable source ID. Hash the normalized text to form `prompt_id`.
4. Do not deduplicate by meaning after seeing the sample. Record paraphrases and repeated anonymized users as possible dependencies. Do not infer identities. Do not cap contributions per user silently: that changes the sampling population.
5. Freeze the frame and its checksum before filtering. Do not use original model names, votes, assistant responses, detector scores, or knowledge of which new models will find a prompt easy to determine eligibility.

## Decide eligibility before generation

A candidate must satisfy all of the following. Apply the same rubric to every candidate examined.

| Check | Decision rule |
|---|---|
| Language | The requested answer is English. Confirm from the prompt; dataset language tags are supporting metadata, not sufficient evidence by themselves. |
| Standalone input | The opening request provides what is needed without conversation history, unavailable attachments, live browsing, or external tools. |
| Open-ended prose | It reasonably invites an original explanation, argument, story, essay, advice, or other substantial prose. |
| Excluded task families | Reject coding/programming help, mathematical problem solving, and multiple-choice tasks. Ordinary prose about a technical subject is not automatically excluded. |
| Original contribution | Reject proofreading, translation, summarization, extraction, quotation/recitation, or rewriting supplied text: those belong in editing/assistance evaluations. |
| Adequate response scope | Reject greetings, single-fact questions, and requests explicitly limited to short fragments, terse lists, tables, or structured data. Do not impose a minimum *prompt* word count. |

Examples: “Explain why cities develop around rivers” and “Write a story about a lost map” qualify. “What is the capital of France?”, “Fix this Python function”, and “Proofread the following essay” do not.

For this small pilot, manual screening is the default: it avoids downloading a separate 24B filtering model or buying filtering calls. MELD and EditLens are detectors, not suitable instruction-following eligibility judges. Using Mistral or another instruction model for assistance is optional and must record its exact revision, filter template, settings, and raw judgments; it does not remove the final rubric review. Classify prompt text as data, never execute its instructions during screening.

Every reviewed candidate gets `eligible` or `ineligible`, a reason code, short rationale, language, and a descriptive task category. Initially uncertain cases must be adjudicated to a final decision before crossing their position in the sample order. Do not just skip harder-to-judge candidates. Review accepted candidates and excluded borderline cases without access to any new generated outputs or detector results. Ideally use a second reviewer; if only one is available, record that limitation. Any rubric revision must be applied to all previously examined candidates before freezing the sample.

Suggested reason codes: `non_english`, `missing_context`, `coding`, `math`, `multiple_choice`, `editing_or_copying`, `short_or_closed_answer`, `non_prose`, `eligible`. Categories such as explanatory, creative, argumentative, practical, and professional describe the sample; they are not quotas.

## Select the 20 without cherry-picking

Use a seeded random order over the **whole deduplicated frame**, then take the first 20 candidates that pass the fixed rubric. This is equivalent to sampling without replacement from the eligible frame, under the pseudorandom ordering, without needing to classify the whole corpus.

Precommit the seed `pangram4-arena20-v1` and this ordering rule:

```text
prompt_id = SHA256(normalized_prompt UTF-8)
rank = SHA256(seed UTF-8 + NUL byte + prompt_id ASCII)
order candidates by (rank, prompt_id), ascending
review candidates in that order
select the first 20 with final eligibility = eligible
```

Record every examined candidate and decision, including exclusions. Screening can happen in batches of 100 for convenience, but selection must follow global rank, never reviewer completion order. If fewer than 20 qualify, continue into the next batch with the same seed. Exhausting the frame with fewer than 20 eligible requests is an incomplete benchmark, not permission to relax the rubric.

A fixed seed is for reproducibility, not a license to try multiple seeds and choose the most attractive sample. Do not select one prompt per topic, require exactly five stories, or replace similar-looking, obscure, difficult, or “boring” prompts. Those changes would produce a different, curated benchmark. If broader topic coverage is wanted later, create a separate coverage panel with explicit quotas and separate results.

After selection, record category counts, prompt lengths, source time range, unique-user count where available, and any near-duplicate topics. These are diagnostics, not grounds to redraw. Twenty random prompts can legitimately miss an entire category. Do not claim the observed category mix matches the corpus distribution unless the eligible population distribution was separately measured.

Freeze `selected_prompts.jsonl` and its SHA-256 **before any target-model generation or detector scoring**. There are no outcome-based replacements. If a genuine source/rubric error is discovered later, record it, version the sample, and rerun all models on the corrected set; do not quietly patch one model's results.

## Run the same experiment across models

Freeze the generator list and exact versions before requests. Every generator receives the same original 20 prompt strings, once each, in fresh conversations, with the same neutral system instruction (if supported): “You are a helpful assistant.” Record unavoidable provider-specific differences. Do not add “write like a human,” a minimum length, detector-evasion instructions, examples, or previous conversation context.

Record each generator's supported decoding settings, sampling seed if available, output-token cap, reasoning configuration, and tool settings. Execution budget, frozen before requests: 4,096 API completion tokens per response. This replaces the initial proposal of 2,048 visible tokens because OpenRouter generally combines reasoning and visible output in the cap. Disable optional reasoning; for mandatory reasoning select the lowest advertised effort, otherwise its default. Save usage including reasoning separately. This cannot guarantee an identical visible-text allowance across models. Do not pretend numerically identical settings make different models equally stochastic. Disable browsing/tools, cache the raw responses, and save finish reasons, usage, latency, and errors.

For M generators, the planned grid is **20 × M responses** (e.g. 100 for five generators), not 20 newly sampled prompts per generator. The user supplied credentials and explicitly authorized execution. The final roster and settings are in `ARENA_20_MODELS.md` and the frozen generation manifest. Local generators are possible, but the installed detector checkpoints are not substitutes for prose-generating models.

Retry only transport/rate-limit failures, at most twice with the identical request, and log attempts. Keep the first successful response. Do not retry refusals, short answers, truncated responses, or low detector scores to obtain a nicer sample. Never regenerate a response just because a detector missed it.

## Keep response failures visible

Before running detectors, apply a fixed response audit: successful completion, substantive original prose, at least 50 whitespace-delimited words, no refusal-only answer, and no visible truncation. For the report's input/output-length condition, additionally require visible output tokens to exceed user-prompt tokens under that generator's documented tokenizer. Exclude reasoning traces and the fixed system message from that comparison. If consistent token counts cannot be obtained, mark the condition unverified rather than silently passing it. Copying supplied text or memorized quotations must also be flagged; uncertain authorship is not resolved by detector score.

These gates are our auditable operationalization. Keep every one of the 20 prompt slots in the results table, including failures and unscorable responses. Do not replace prompts, append instructions, or count generation/API failures as detector false negatives.

Report, per generator:

- Attempted slots out of 20, successful responses, and scorable responses; break down every exclusion reason.
- Detector false negatives / scorable responses, with a Wilson 95% interval and explicit denominator. A Mixed prediction counts as a miss under the report's strict document convention; preserve the local detectors' documented threshold limitations.
- Scorable detections / 20 as a separately labeled **end-to-end yield**, not detector recall.
- The prompt-by-generator response and decision matrix. Compare models on common scorable prompt IDs as well as each model's full available set; show how many paired prompts remain.

This is an AI-output/FNR experiment. It cannot estimate FPR or AUROC by itself: user prompts are not human-authored answer controls. Existing human corpora remain separate evaluations.

With 20 scorable responses, each miss changes a model's FNR by five percentage points. Zero misses still gives an approximately **16.1% upper Wilson 95% bound**. Treat this as a small paired diagnostic, not verification of Pangram's sub-percent rates or a precise ranking of model families. Outputs across generators share the same prompts; do not treat 20 × M outputs as that many independent prompt draws. Any pooled uncertainty analysis must resample by prompt, retaining all model outputs together; with only 20 clusters it will still be unstable.

## Artifacts and completion conditions

Store execution artifacts under `data/arena20/` and results under `runs/arena20/`, separate from the website. Preserve:

| File | Contents |
|---|---|
| `source_manifest.json` | Source repository, immutable revision, split, files/checksums, extraction rules, frame count/hash, seed, rubric version. |
| `screening.jsonl` | Ranked candidate ID, source IDs, duplicate count, decision/reason, reviewer/filter provenance, review changes. |
| `selected_prompts.jsonl` | Exactly 20 unique IDs, ranks, original prompt strings, categories, provenance. |
| `generation_manifest.json` | Frozen generator IDs/versions, full request settings, per-model differences, planned calls and budget. |
| `responses.jsonl` | Every planned prompt/model cell, attempts, raw output, usage, finish status, response-audit status. |
| `detector_predictions.jsonl` | Detector revision/settings, response hash, raw score, decision, failure if any. |
| `REPORT.md` | Sampling funnel, actual coverage, response yield, per-model FNR/intervals, paired matrix, limitations. |

Before execution, verify that all selected IDs are eligible and unique; no eligible candidate was skipped ahead of the twentieth acceptance; the selected checksum is frozen; the generator grid contains every selected ID; and no tuning or selection decision used detector results. A deterministic rerun of selection must recover the same IDs. The implemented freeze validator enforces the contiguous reviewed prefix, 20 unique eligible selections and a fixed sample checksum. Generation verifies that checksum and the frozen catalog identities.

## Frozen execution clarifications

Before any generation, the single reviewer applied the literal prose scope: poems, song lyrics, terse joke requests, name lists and tool/image prompt syntax were excluded as non-prose or short-answer tasks. General conceptual explanations are eligible; single-term definitions and explicit mathematical proofs are not. Broad professional overviews can be answered from background knowledge unless the request explicitly requires current market data or unavailable material. No category quotas or outcome-based replacements were used. Borderline judgments and rationales are retained for every reviewed position. The single-reviewer decisions remain a limitation.

The source requests are historical. Generation uses their original wording without adding a date, safety/style instructions or missing facts. Any refusal, poor factual answer, unsafe-content refusal or short response remains a recorded outcome. Instructions embedded in the corpus never control the sampling or execution software.

OpenRouter usage includes the fixed system message/chat formatting and does not provide native user-only prompt token counts for every selected model. The native input/output token gate is therefore marked **unverified**, never silently passed. If unresolved, report strict replication metrics as unavailable and label local results as a content-audited exploratory variant without that gate. Preserve word counts and visible completion usage as diagnostics.

## Local response-review implementation

Response checks use word count, finish status, and single-assistant review packets that hide generator names and detector predictions. The reviewer inspects beginning/end excerpts and format counts; ambiguous/refusal cases can be read in full. This is not exhaustive full-text human review, a plagiarism search, or factual verification. Substantial alternative prose after a refusal can qualify, including developed plot synopses; mere offers to write an alternative do not. Full drafts are not required to follow the original topic exactly. Record this weaker scope screen when reporting the exploratory results. No response is scored before its own eligibility decision is recorded.

A proposed separate API judge was blocked by automatic approval review and never ran. Local review does not transmit any additional payload to an external judge. The original generation and availability-replacement requests were separately authorized.

The public historical prompt corpus may occur in generator training data. This experiment measures fresh responses to sampled public prompts; it does not establish performance on unseen prompts. Selected sample bytes and eligibility decisions stayed unchanged throughout execution. Rejected-row language/category metadata corrections are separately logged and did not affect selection.
