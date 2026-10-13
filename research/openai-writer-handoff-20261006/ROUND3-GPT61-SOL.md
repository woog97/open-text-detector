# Round 3: GPT-6.1 Sol small edits (Oct 7, 2026)

**For a new Codex conversation with no prior context.** Read this file, then `README.md` in this folder. `README.md` has the writer instructions, output format, acceptance criteria, self-check, retry rule, concurrency and resume rules. All of them apply here unchanged. This file lists only what is new.

## Why this round exists

The detector's training mix needs 1,800 GPT small edits for 3 full-length epochs without repeats. Only 397 usable ones exist. GPT-6 Sol is reserved as an unseen writer for evaluation, so none of its rows may be used for training. GPT-6.1 Sol has to supply the rest: **1,403 more accepted rows.**

The round-2 batches were never run. Their host paragraphs were chosen before the held-out split existed, and many sit on held-out papers. They were moved to `batches/superseded-20261007/`. **Do not run anything in that folder.** The round-3 hosts were screened against every held-out list, including `research/t21-heldout-20261006/`, before assignment.

## What to produce

| Job | Model | Batch files | Items | Success threshold |
|---|---|---|---|---|
| D: sentence edits | `gpt-6.1-sol` | `batches/d/gpt-6.1-sol-r3-001` … `-030` | 1,500 | ≥ 1,403 accepted (93.5%) |

- 30 batch files of 50 items each.
- Edit mix: rewrite one sentence 30%, rewrite two adjacent sentences 30%, insert one sentence 30%, split one sentence 10%.
- Writer instructions: [`D-INSTRUCTIONS.md`](D-INSTRUCTIONS.md), given verbatim to each subagent with its one batch file path.
- Round 1 accepted 100% of GPT-6.1 Sol small edits, so the margin is small.

## Model rule (strict)

- Run every `gpt-6.1-sol-r3-*` batch only with a `gpt-6.1-sol` subagent.
- **Never run GPT-6 Sol in this round.** There are no GPT-6 Sol batches to run.
- Never substitute another model, and never write edits yourself. If `gpt-6.1-sol` is unavailable or refuses, stop and report.

## Paths

All relative to `research/openai-writer-handoff-20261006/`:

- **Input:** `batches/d/gpt-6.1-sol-r3-NNN.jsonl`. Never edit input files.
- **Output:** `outputs/d/gpt-6.1-sol/gpt-6.1-sol-r3-NNN.jsonl`. Write `.jsonl.tmp` first, self-check it, then rename it as the very last step.
- **Self-check** (from the repository root):

      uv run -q --with requests python research/openai-writer-handoff-20261006/selfcheck.py d gpt-6.1-sol-r3-007

  It prints one line per failing item and exits 0 only when there are no problems. It now also rejects any row on a held-out paper; none should occur, because the hosts were pre-screened.

## Ingest

After all batches, or at any checkpoint, from `research/claude-hosted-edits-20261006`:

    uv run -q --with requests python build_hosted.py ingest-d --writer gpt-6.1-sol

This rebuilds all GPT-6.1 Sol rows (rounds 1 and 3) in `llm-edits-claude-hsm-v1.jsonl.gz` and prints `assigned`, `returned`, `accepted` and `rejects`. Round 1 rows on papers that are now held out are dropped at this step. That is expected: those rows were never usable for training.

**Success:** at least 1,403 round-3 rows accepted. To get the round-3 count, subtract the round-1 rows kept, or count `host_id`s that appear in `items-d-scale3.jsonl`.

## Rules

- No web search. Do not look up the original papers.
- Never edit input batches, `build_*.py`, `selfcheck.py` or any `*never-train*.json` file.
- Do not commit anything. Batches and outputs are gitignored because they contain third-party text.
- Report times in Pacific time (PDT).
- **Final report:** batches done, accepted/returned for round 3, top reject reasons, start and end times (PDT), and any model-availability problems.

## Background (for the person running this)

- Items: `research/claude-hosted-edits-20261006/items-d-scale3.jsonl` (writer `gpt-6.1-sol`, round `scale3`), made by `build_hosted.py prepare-scale3`. Stats are in `prepare-scale3-stats.json`.
- The same step assigned 1,450 Opus 5.5 and 1,450 Sonnet 5.5 items to Claude subagents (run separately, not part of this handoff).
- GPT-6.1 Sol round-2 paragraph edits (`batches/e/gpt-6.1-sol-r2-*`, 210 items, 41 of them on held-out papers) are not part of this handoff. Leave them unrun.
