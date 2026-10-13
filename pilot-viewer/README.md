# Paper revisions local viewer

Browse all completed paper-generation runs at http://127.0.0.1:3020. Reconstruction v3 contains 250 fresh paragraphs from the same five targets in each of 50 papers. V2 remains available as the matched baseline. Both use Luna; v3 revises the notes and writing prompts to separate private constraints from paragraph content and preserve voice, claim strength and logical conditions.

From the project root:

```sh
python3 pilot-viewer/build_data.py
python3 pilot-viewer/serve.py
```

The builder reads the existing run artifacts in `research/data`, including Sol 10, Luna 50, quality-first Luna 50, and both paragraph reconstruction runs. Select a run in the sidebar; shared passage selections carry across runs. It makes no model calls and does not read credentials. The server binds only to loopback and serves only `pilot-viewer/dist`.

Each run includes its saved examples, exact prompts and responses, attempts, costs, sentence labels, and editing regions. Paragraph reconstruction runs also expose content notes and full fidelity evidence. Use the verdict filter to browse fully faithful, minor-difference, material-difference or uncertain judgments. Expand a paper to browse its five paragraphs. The latest run appears first in the selector. URL fragments preserve the run, passage, operation, detail view and attempt.

The expanded collection reports new spending separately from the historical costs of reused examples. Its fidelity table separates the 200 new judgments from the 50 retained judgments. Earlier runs remain available for comparison.

The layout uses compact neutral controls and side-by-side word diffs. In narrow panels, “Browse papers” opens the paper list and filters; choosing a paper closes it again.

AI-token counts use the existing MELD tokenizer and provenance labels; API token usage includes reasoning within output tokens. Neither figure should be substituted for the other.

The V2 / V3 evaluation view reports unchanged strict fidelity judgments, balanced blinded pairwise writing quality, and lexical overlap separately. It shows the old and new generated paragraphs side by side, exact quality requests and responses, and all new-run costs. Detailed machine-readable results are saved in `research/data/paper-gap250-luna-v3-20260930/comparison-report.json`. The source-quality strata use flags from the baseline audit, so membership is fixed across versions. These are development comparisons and same-model judgments, not expert validation or an untouched test benchmark.
