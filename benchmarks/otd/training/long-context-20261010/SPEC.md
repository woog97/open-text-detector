# Long-context (1,020-token) copy of the Oct 7 training mix: extension modules

Goal: a row-for-row copy of the Oct 7 final mix (72,000 stage-2 rows) where each row's text is extended with its real surrounding
context to about 1,000 tokens, so that a 510-token run and a 1,020-token run differ only in context length.

## Inputs
- Mix rows (local): `$SCRATCH/longctx/input/stage2-epoch{0,1,2}.jsonl.gz` (72,000 rows total). Fields: dataset (the mix key), id, paper_id,
  group, host_id, kind, text_format, generator, text, regions [{start,end,label}] (label 1 AI, 0 human, 2 = mixed/unknown),
  target_start/target_end, source_start/source_end (offsets of this window in its pool text, where set).
  `SCRATCH` is the Claude Code scratch folder of the session that ran this (`/private/tmp/claude-501/<project>/<session>/scratchpad`).
- Tokenizer: Qwen/Qwen3.6-35B-A3B at revision 995ad96eacd98c81ed38be0c5b274b04031597b0 (download only the tokenizer files with
  huggingface_hub into `$SCRATCH/longctx/tokenizer/` if not already there; it is public). Count tokens with add_special_tokens=False.
- Bucket (read-only for you): `open-text-detector/training-storage`. Local bucket access needs `uv run --with 'huggingface_hub>=1.0'`
  (`hf buckets cp hf://buckets/open-text-detector/training-storage/<path> <local>`). Download only what you need into `$SCRATCH/longctx/<family>/`.

## Contract for each module (`<family>.py` in this folder)
`extend(row, tok, max_tokens=1000) -> dict | None`
- Returns a NEW row dict (copy all fields) whose `text` = added_before + row['text'] + added_after, where the ORIGINAL row text appears
  verbatim and unchanged at offset len(added_before). Assert this. Return None when no longer context is available for that row.
- Total length <= max_tokens tokens. Add context on both sides as evenly as available (token-balanced), snapped to whitespace or paragraph
  boundaries, using the paragraph separator the source uses (usually '\n\n').
- Shift every region and target_start/target_end by len(added_before). Add regions covering the added text with the correct label:
  human context in a human paper = 0; context inside an AI-generated document = that document's label (1). Never invent labels:
  if the added text's provenance is unknown, label it 2 (ignored) or do not add it.
- The added context must come from the SAME text extraction / format as the row text (same text_format), so formatting cannot reveal
  the label. If the source paragraphs around the row cannot be matched exactly to the row's own text, return None (do not guess).
- Set fields: `context_extended` (bool), `ctx_before_chars`, `ctx_after_chars`, `ctx_source` (short string naming the source used).
- Pure function apart from cached source loading; deterministic.

## Validation each module must report
- Per mix key and text_format it handles: rows attempted, extended, returned None (with reasons), median/p90 token length before and after.
- For extended rows: 100% pass of the verbatim-containment assert; regions cover only valid offsets; AI-labelled characters unchanged.
- 5 printed examples (truncated) showing the join points.
- Run on ALL rows of the families it handles (not a sample) and write `$SCRATCH/longctx/<family>/extended-ids.jsonl.gz` with the
  extended rows, plus `report.json`.

## Rules
- Do not modify anything outside `benchmarks/otd/training/long-context-20261010/` and `$SCRATCH/longctx/`. Read any repo file.
- No GPU, no ssh to the H200, no uploads, no deletions outside $SCRATCH/longctx/<family>/.
- Never use or add Fable / GPT-6 Sol / GLM / MiMo text. Do not call any LLM API.
- Cite the source functions you reuse (file:line). If the original extraction code cannot be reproduced exactly, say so and return None
  for those rows instead of approximating.
