> **Codex handoff note (overrides the paths below):** this file is copied verbatim from the Claude writer run. For this handoff, read batches from `batches/e/<model>-NNN.jsonl` and write each output to `outputs/e/<model>/<model>-NNN.jsonl` (same file name as the batch), both under `research/openai-writer-handoff-20261006/`. Ignore the scratchpad paths and the ingest section below; see README.md. Write the output first as `<model>-NNN.jsonl.tmp`, run `selfcheck.py` on it, and rename it to `<model>-NNN.jsonl` only as your very last step.

# Instructions for Claude paragraph writers (Job E: whole-paragraph edits in scientific passages)

You produce ONE whole paragraph inside a human-written scientific text (PubMed Central, peS2o or arXiv), the way an author using an AI writing assistant would. Your paragraph replaces the original one in place, so it must read as a natural part of the text.

## Input

Your batch file `e-batches/<writer>-NNN.jsonl` has 50 items, one JSON object per line:

- `id`, `kind` (`draft` or `rewrite`).
- `prev_context`: the paragraph before. `next_context`: the paragraph after. Either may be empty.
- `target_words`, `target_sentences`: the original paragraph's length.
- `paragraph`: **rewrite items only.** The original paragraph.

## The two kinds

- **draft:** you have NOT seen the original paragraph. Write the paragraph that belongs between `prev_context` and `next_context`. It should continue the line of argument, set up what the next paragraph relies on, keep the same terminology and notation, and invent nothing that contradicts the context. If one side is empty, the paragraph opens or closes the passage.
- **rewrite:** rewrite and polish `paragraph` as an AI writing assistant would when asked to improve it. Keep all the content, claims, numbers, citations and notation. Improve the wording and flow noticeably; restructure sentences as needed. Add nothing new and drop nothing.

## Form

- Exactly one paragraph: no blank lines inside it, about `target_words` words (±25%), about `target_sentences` sentences.
- Match the text's conventions as they appear in the context: citation markers such as `[12]`, `(Smith et al., 2019)` or `\\cite{...}`, and math written the way the context writes it (LaTeX `$...$` if the context uses it, otherwise plain characters).
- No Markdown, headings, bold, lists, commentary or notes.

## Output

For each item, output one JSONL line, in input order:

    {"id": "<id>", "paragraph": "<your paragraph>"}

Write one output file per batch, with the same file name, into `e-outputs/<writer>/` under this directory.

## Ingest (main session)

From `research/claude-hosted-edits-20261006`. It works on partial outputs and can be re-run:

    uv run -q --with requests python build_hosted.py ingest-e $SCRATCH/claude-hosted [--writer sonnet|opus|haiku]

The paragraph is labeled as one AI span inside its context. Each accepted item also gets a matched human row: the same context with the original paragraph.

Output files:

- `paragraph-edits-claude-hsm-v1-train.jsonl.gz`.
- `paragraph-edits-claude-hsm-v1-heldout.jsonl.gz`. Held-out paragraphs are also added to `heldout-never-train.json`.

Rejection rules:

- not exactly one paragraph;
- Markdown;
- length outside 0.5–1.8× the original;
- word overlap with the original above 0.9 (rejected for both kinds);
- a rewrite whose word overlap is below 0.12 (content lost).
