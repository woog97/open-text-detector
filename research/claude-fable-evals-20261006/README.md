# Fable 5.1 evaluation-only sets (Oct 6, 2026)

These sets are for evaluation only. All generated text is registered in `heldout-never-train.json` in this folder, which `build_llm_edits.never_train_hit()` loads. Inputs come only from held-out sources:

- **Light polish and co-writing:** the Job E held-out host passages (human-source-mix pmc/pes2o/arxiv, 2022 or earlier). They are already on the hosted never-train list, and no job wrote them.
- **Humanizer:** AI paragraphs from the held-out Claude sections.

Builder: `build_fable_evals.py` (`prepare`, `ingest {polish,humanizer,cowrite}`). Writer: Claude Fable 5.1 (`claude-fable-5-1`). Every AI row has a matched human control. Rows use the held-out scoring format (`slot`, `label`, `condition`, `writer`, `text`, `regions`).

| Set | Items | Output file |
|---|---|---|
| Light polish | 300 | `polish-eval-v1.jsonl.gz` |
| Humanizer | 200 | `humanizer-eval-v1.jsonl.gz` |
| Co-writing | 100 | `cowrite-eval-v1.jsonl.gz` |

## Labeling rules

**Light polish: character labels by diff alignment.** The polished paragraph is aligned to the original with a word-level diff. A token is a word plus its trailing whitespace, compared without that whitespace.

- Tokens inside unchanged (`equal`) blocks are human (0).
- Tokens that are new or replaced are AI (1). Every character of a token gets the token's label, so a corrected word counts entirely as AI.
- Deleted original words leave no characters in the polished text, so they get no label.
- The context paragraphs are human (0).

Word-level alignment is used instead of a raw character diff because a character diff would mark scattered single letters of rewritten words as human. `ai_token_fraction` records how much was changed. The matched negative is the unedited original window.

**Humanizer:** the whole AI section is labeled AI (1), including the humanized paragraph (`humanized_start`/`humanized_end`). Each item also has two comparison rows: the same section before humanizing (`ai_before_humanizing`) and the matched human version of the section (`human_controls`).

**Co-writing: three provenance classes.**

| Class | Meaning | Label |
|---|---|---|
| `human` | original text, verbatim | 0 |
| `ai` | assistant text accepted verbatim | 1 |
| `ai_edited` | assistant text the author then edited | 1 |

`ai_edited` counts as AI because the wording and structure come from the assistant, and the human edits are small. That matches how the detector would be used: text that starts as AI output is AI-assisted. The full three-class provenance is kept in each row's `provenance` field, so other labelings, for example treating `ai_edited` as a separate class, can be scored later. The matched negative is the original three-paragraph document.
