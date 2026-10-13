# OpenAI writer handoff: sentence and paragraph edits (Oct 6, 2026)

**For a new Codex conversation with no prior context. Read this whole file before starting.**

> **Current task (Oct 7): Round 3, GPT-6.1 Sol small edits only. Start with [`ROUND3-GPT61-SOL.md`](ROUND3-GPT61-SOL.md).** Rounds 1 and 2 below are reference. Round 2 small-edit batches and GPT-6 Sol round-2 paragraph batches were superseded and moved to `batches/superseded-20261007/`; do not run them.

## Purpose

This project trains a detector for AI-written text inside human research writing. The training mix needs AI edits from several writer models, and this handoff covers the OpenAI share. Each input item is a paragraph from a human scientific text (PubMed Central, peS2o or arXiv, published 2022 or earlier) with its neighbouring paragraphs. A writer makes one requested edit. Job D is a small sentence-level edit. Job E writes or rewrites one whole paragraph. Ingest code then aligns each output against the original and labels exactly the AI-written characters. Your job: run the writer subagents on the prepared batches, self-check every output, and run ingest.

## What to produce

| Job | Model | Batch files | Items | Success threshold (≥85% accepted) |
|---|---|---|---|---|
| D: sentence edits | `gpt-6.1-sol` | `batches/d/gpt-6.1-sol-001` … `-010` | 500 | ≥ 425 |
| D: sentence edits | `gpt-6-sol` | `batches/d/gpt-6-sol-001` … `-010` | 500 | ≥ 425 |
| E: whole paragraphs | `gpt-6.1-sol` | `batches/e/gpt-6.1-sol-001` … `-005` | 210 | ≥ 179 |
| E: whole paragraphs | `gpt-6-sol` | `batches/e/gpt-6-sol-001` … `-005` | 210 | ≥ 179 |

- Every batch file has 50 items, except the last E file per model, which has 10.
- D edit mix per model: rewrite one sentence 30%, rewrite two adjacent sentences 30%, insert one sentence 30%, split one sentence 10%.
- E per model: 50% `draft` (write the paragraph from its neighbours without seeing the original) and 50% `rewrite` (polish the given paragraph).

## Model rule (strict)

- **The batch file prefix is the model.** Run `gpt-6.1-sol-*.jsonl` only with a `gpt-6.1-sol` subagent and `gpt-6-sol-*.jsonl` only with a `gpt-6-sol` subagent.
- **Never substitute another model.** If a model is unavailable or refused, **stop and report**. Do not swap models and do not write the edits yourself.
- One subagent handles one batch file.

## Writer instructions

Give each subagent its instruction file, verbatim, plus its one batch file path:

- **D:** [`D-INSTRUCTIONS.md`](D-INSTRUCTIONS.md)
- **E:** [`E-INSTRUCTIONS.md`](E-INSTRUCTIONS.md)

They were copied verbatim from the Claude writer run. The note at the top of each file overrides its scratchpad paths: use the paths below.

## Files and output format

All paths are relative to `research/openai-writer-handoff-20261006/`.

- **Input:** `batches/{d,e}/<model>-NNN.jsonl`. **Never edit input files.**
- **Output:** `outputs/{d,e}/<model>/<model>-NNN.jsonl`, with the same file name as the batch. One JSON line per input item, in input order:
  - D: `{"id": "<id>", "edited_paragraph": "<the full paragraph after the edit>"}`
  - E: `{"id": "<id>", "paragraph": "<the one paragraph>"}`
- Exactly these keys. No commentary lines, no Markdown, no code fences.

## Acceptance criteria (what ingest checks)

**D: sentence edits**

- **Only the requested edit, on the named sentence(s).** The edit must land on the sentence number(s) given in `instruction`, using the numbering in `numbered_sentences`. For `insert_one`, the new sentence goes directly after the named sentence.
- **Everything else is byte-identical** to the original paragraph, including spacing, citations, symbols and extraction quirks. Only the single space next to the edit may differ. Copy untouched text exactly; do not re-type it.
- **Sentence counts:**

  | Edit type | Original sentences changed | Sentences in the result |
  |---|---|---|
  | `rewrite_one` | 1 | 1 |
  | `rewrite_two` | 2 adjacent | 2 |
  | `insert_one` | none | 1 new sentence after the named one |
  | `split_one` | 1 | 2 |

  Sentence numbers refer to the item's `numbered_sentences`.
- A rewrite must be noticeably reworded. If its word overlap with the original exceeds 0.9, it is rejected.
- Output the full paragraph, not just the changed sentence.

**E: whole paragraphs**

- Exactly one paragraph: no blank lines inside it.
- Length within 0.5–1.8× `target_words`; aim for about ±25%.
- Word overlap with the original must not exceed 0.9 for either kind. A `rewrite` must also keep the content: overlap below 0.12 is rejected.
- No Markdown, headings or commentary, and never reproduce `prev_context` or `next_context`; only the target paragraph is written.

## Self-check (each subagent runs this before finishing)

`selfcheck.py` calls the exact validation functions ingest uses: named-sentence placement, byte-identical untouched text, sentence counts, overlap and length limits, the context-copy check and the never-train exclusion. It also requires one output line per input id, in input order. **Passing the self-check means the batch passes ingest.** It needs the local item files in `research/claude-hosted-edits-20261006/`, so run it in this checkout.

From the repository root:

    uv run -q --with requests python research/openai-writer-handoff-20261006/selfcheck.py d gpt-6-sol-007
    uv run -q --with requests python research/openai-writer-handoff-20261006/selfcheck.py e gpt-6.1-sol-003

It prints one line per failing item.

**Retry rule:**
- **Before handing a batch back,** the subagent may rewrite any item the self-check rejects and re-run the check, ideally until it reports `0 problems`.
- **Once a batch is handed back,** it is final. Ingest simply drops rejected items; there are no later retries. The item counts include margin above the targets.

## Concurrency and resume

- **Concurrency:** run as many writer subagents in parallel as the session allows (3–6), mixing both models. Each subagent writes only its own output file.
- **Resume:** each writer first writes `<batch>.jsonl.tmp`, runs the self-check on it, then renames it to `<batch>.jsonl` as its very last step. A batch is finished only if `<batch>.jsonl` exists; a leftover `.tmp` means the write was interrupted, so delete that `.tmp` and rerun the batch. Never delete or rewrite a finished `<batch>.jsonl`.
- **Partial progress:** ingest can run at any time on partial outputs and can be re-run.

## Ingest (after all batches, or at any checkpoint)

From `research/claude-hosted-edits-20261006`:

    uv run -q --with requests python build_hosted.py ingest-d --writer gpt-6.1-sol
    uv run -q --with requests python build_hosted.py ingest-d --writer gpt-6-sol
    uv run -q --with requests python build_hosted.py ingest-e --writer gpt-6.1-sol
    uv run -q --with requests python build_hosted.py ingest-e --writer gpt-6-sol

- Each command prints a JSON summary for that writer only: `assigned`, `returned`, `accepted`, `acceptance_of_returned`, `acceptance_of_assigned` and `rejects` by reason.
- Rows are labeled with generator `openai/<model>` and `written_by: codex-subagent`.
- Outputs go to:
  - `llm-edits-claude-hsm-v1.jsonl.gz` (D);
  - `paragraph-edits-claude-hsm-v1-train.jsonl.gz` (E, AI rows plus matched human rows).
- Re-running for a writer replaces only that writer's rows.

**Success:** at least 85% accepted per model and job, i.e. at least 425 of 500 for D and at least 179 of 210 for E, matching the table above. If a model falls short, report its reject reasons; do not re-run handed-back batches.

## Rules

- No web search. Do not look up or read the original papers or passages.
- Never edit input batches, `build_*.py` or any `*never-train*.json` file.
- Do not commit anything; the batches and outputs are gitignored because they contain third-party text.
- Report all times in Pacific time (PDT).
- **Final report:** per model and job, report batches done, accepted/returned, top reject reasons, the start and end times (PDT), and any model-availability problems.

## Round 2 (T2 scale-up, Oct 6 afternoon)

Same rules, instructions, output format, self-check, retry rule and ingest commands as above. Only the batch names and counts are new. Round-2 batch names contain `-r2-`, so they cannot collide with round 1.

| Job | Model | Batch files | Items | Success threshold (≥96% accepted) |
|---|---|---|---|---|
| D: sentence edits | `gpt-6.1-sol` | `batches/d/gpt-6.1-sol-r2-001` … `-016` | 780 | ≥ 750 |
| D: sentence edits | `gpt-6-sol` | `batches/d/gpt-6-sol-r2-001` … `-016` | 780 | ≥ 750 |
| E: whole paragraphs | `gpt-6.1-sol` | `batches/e/gpt-6.1-sol-r2-001` … `-005` | 210 | ≥ 200 |
| E: whole paragraphs | `gpt-6-sol` | `batches/e/gpt-6-sol-r2-001` … `-005` | 210 | ≥ 200 |

- **Targets:** 1,500 accepted small edits and 400 accepted paragraph edits in total. Round 1 accepted 100% and 99.5%, so the item counts carry only a small margin.
- **Last files:** the last D file per model has 30 items, and the last E file per model has 10.
- **Outputs:** `outputs/{d,e}/<model>/<model>-r2-NNN.jsonl`, the same name as the batch.
- **Self-check:** `selfcheck.py d gpt-6-sol-r2-004`. It handles `-r2-` names.
- **Ingest:** run the same four commands as round 1 after round 2. Re-running for a writer rebuilds that writer's rows from round-1 and round-2 outputs together, so round-1 rows are kept.
- **Model rule:** the batch prefix is the model, and you never substitute. If a model is unavailable, stop and report.
