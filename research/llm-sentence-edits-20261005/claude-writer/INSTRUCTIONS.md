# Instructions for Claude edit writers (held-out small-edit evaluation set)

You are helping build a research dataset on AI-assisted editing of scientific papers. You edit a paragraph from a real paper the way an author using an AI writing assistant would. Keep the paper's voice, terminology, notation and citation style. Change only what the instruction asks; copy every other sentence exactly, character for character.

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
- Write one output file per batch file, with the same name (for example `sonnet-03.jsonl`), into the writer's outputs directory.

Outputs that change anything beyond the requested sentences, change nothing, or change the wrong number of sentences are rejected automatically during ingest.

## Ingest

After the outputs exist:

    cd research/llm-sentence-edits-20261005
    uv run -q --with requests python build_llm_edits.py ingest-claude --writer sonnet --in <outputs dir>

Use `--writer haiku`, `sonnet` or `opus`. Accepted rows are written to `heldout-eval-v1.jsonl.gz` with generator `claude-<model>`. Re-running for a writer replaces that writer's rows.
