> **Codex handoff note (overrides the paths below):** this file is copied verbatim from the Claude writer run. For this handoff, read batches from `batches/d/<model>-NNN.jsonl` and write each output to `outputs/d/<model>/<model>-NNN.jsonl` (same file name as the batch), both under `research/openai-writer-handoff-20261006/`. Ignore the scratchpad paths and the ingest section below; see README.md. Write the output first as `<model>-NNN.jsonl.tmp`, run `selfcheck.py` on it, and rename it to `<model>-NNN.jsonl` only as your very last step.

# Instructions for Claude edit writers (Job D: sentence edits in scientific passages, claude-hsm-v1)

You are helping build a research dataset on AI-assisted editing of scientific papers. You edit a paragraph from a real scientific text the way an author using an AI writing assistant would. Keep the paper's voice, terminology, notation and citation style. Change only what the instruction asks; copy every other sentence exactly, character for character.

## Input

Each line of your batch file is one JSON object:

- `id`: copy it unchanged into your output.
- `edit_type`: one of `rewrite_one`, `rewrite_two`, `insert_one`, `split_one`.
- `instruction`: the exact edit to make. It names sentences by number.
- `numbered_sentences`: the paragraph split into sentences, numbered `[1]`, `[2]` and so on. Sentence numbers in `instruction` refer to this list.
- `paragraph`: the paragraph to edit, as plain running text.
- `prev_context`, `next_context`: the surrounding paragraphs. They are for reading only; never edit or output them.

The edit types mean:

- `rewrite_one`: rewrite the named sentence so it reads more polished and fluent, keeping its meaning. Reword it noticeably, not just one word.
- `rewrite_two`: rewrite the two named adjacent sentences so they read more polished and fluent, keeping their meaning. Reword them noticeably. You may restructure them, but the result must still be exactly two sentences.
- `insert_one`: insert one new sentence directly after the named sentence. It should add a natural, plausible clarification or transition consistent with the paragraph. Do not change any existing sentence.
- `split_one`: split the named sentence into two well-written sentences that keep its meaning, rewording as needed.

## Output

For each input line, output exactly one JSON line, in input order:

{"id": "<the input id>", "edited_paragraph": "<the full paragraph after the edit>"}

Rules:

- `edited_paragraph` is the whole paragraph, as plain running text with no sentence numbers and no line breaks.
- Make only the requested edit. Every other sentence stays byte-for-byte identical, including its spacing, citations, symbols and any odd extraction artifacts.
- Output no commentary, Markdown, code fences or extra keys.
- Write one output file per batch file, with the same name (for example `sonnet-003.jsonl`), into `$SCRATCH/claude-hosted/d-outputs/<writer>/`.

Outputs that change anything beyond the requested sentences, change nothing, or change the wrong number of sentences are rejected automatically during ingest.

## Ingest

From `research/claude-hosted-edits-20261006`. It works on partial outputs and can be re-run:

    uv run -q --with requests python build_hosted.py ingest-d $SCRATCH/claude-hosted [--writer sonnet|opus|haiku]

Accepted rows go to `llm-edits-claude-hsm-v1.jsonl.gz` (dataset `hsm_llm_edit`, training only), after the same alignment filters as claude-v1 and a never-train check.
