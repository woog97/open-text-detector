# Wider evaluation and remaining human paragraphs

The original ModernBERT checkpoint and token/sentence thresholds are frozen.
No training, threshold fitting, or new generation calls were performed here.

## Main findings at the frozen thresholds

| Check | Result |
|---|---:|
| Arena, 50 generators: AI token recall | 0.91% |
| Arena: AI sentence recall | 0.42% |
| OpAI native spans: AI token recall | 8.25% |
| Clean novel human paragraphs, isolated: token FPR | 5.63% |
| Same human paragraphs with neighbors: focal-token FPR | 3.58% |
| Same contextual human paragraphs: focal-sentence FPR | 2.36% |

The human comparison covers 7,713 clean novel paragraphs from 390 held-out
papers. The contextual token-FPR 95% paper-bootstrap interval is
3.13–4.10%.
Scorer parity was exact on 16 identical inputs checked against the original
evaluator. The transfer failure is therefore not explained by a scorer mismatch.

The prior 73.9% token recall on the narrow reconstruction test does not establish
broad AI detection. These results mix domain, generator, task and calibration
shifts; they do not isolate which cause dominates. Some public-benchmark ranking
scores remain useful even though recall at the frozen operating point is poor.
Broaden training examples and recalibrate on validation data; never mine these
test failures into training. Neither the checkpoint nor thresholds was changed.

Primary scoring took 300 seconds; the paired context check took 100 seconds.
There were no new generation calls. GPU billing is not inferred from runtime.

## Coverage

**65,348 examples / 63,035 distinct texts / 17 benchmark groups.**

- 18,222 remaining human paragraphs from 390 held-out historical papers.
- 4,898 eligible Arena responses across 50 generators and 100 shared prompts.
- Expanded MELD-eval, DetectRL, GEDE and PELIC samples; all locally available
  OpAI test trajectories, Epoch examples, Sem-Detect test reviews, Liang, Saha,
  Perkins and VUB examples.
- Explicit local length/interleaving proxies and 435 development-exposed paper
  pilot variants, reported separately as exploratory checks.

Large released datasets use deterministic samples of 32 per existing stratum;
PELIC uses 2,000 answers. Smaller collections use all available examples.
See `manifest.json` for counts and selection, and the workbench's
[coverage inventory](../../../COVERAGE.md) for source provenance and licenses.

## Human pools

Extracted **95,334 non-target body paragraphs** from the same 2,000 historical
papers, preserving their existing paper/author split assignments:

| Split | All retained paragraphs | Clean, fresh, novel paragraphs* |
|---|---:|---:|
| Train | 57,369 | 23,586 |
| Validation | 19,296 | 7,895 |
| Test | 18,669 | 7,713 |

*`clean_prose=true`, `development_exposed=false`, `cohort=novel_body`, and no
normalized exact cross-split duplicate. “Novel” excludes the neighbors used in
prior generation passages. These are additional human negatives, not yet mined
hard negatives. The wider test suite further applies overlap screening.

Files `human_remaining_{train,validation,test}.parquet` contain text, `paper_id`,
nullable verified `forum_id`, split, character `regions` labeled human `0`,
PDF URL/hash, page, section, bounding box and extraction flags. The row-level
`label` is the string `human`; token labels should be rebuilt from `regions`
with the chosen tokenizer, masking special/padding tokens.

Use the training pool for future training or false-positive mining, and the
validation pool for calibration. Preserve the test pool for evaluation. Sample
by paper and balance against AI examples; do not let the larger negative pool
silently dominate training. Deduplicate training text within its own split.

PDF paragraph recovery is approximate. The extractor filters recognized
headings, captions and small-font material, and stops at reference/appendix
boundaries. It retains imperfect prose with flags rather than claiming every
physical paragraph was recovered. Pre-2022 provenance supports human labels;
it is not an authorship log. Modern OpenReview papers were not assumed human.

## Evaluation safeguards

- No evaluation paper is a training or validation paper. External texts are
  screened for normalized exact sentence/paragraph matches and contiguous
  32-word overlaps with the actual training/validation inputs. External author
  identities are generally unavailable, so this is not proof of author separation
  for every public benchmark.
- Exact texts are scored once and reused; each text has one vote per reported
  condition. Do not add overlapping dataset/condition counts as independent data.
- Known spans and historical-human paragraphs support token/sentence metrics.
  Document-native labels support document metrics. Polish and ambiguous/mixed
  cases without spans remain separate; document labels are not invented spans.
- Document flagging means at least half of tokens exceed the frozen token
  threshold. This is a diagnostic rule, not a separately calibrated document
  classifier. AI-only collections cannot measure precision or human FPR.
- The annotation audit found 140 identical-text groups with conflicting spans;
  their 337 rows are excluded from primary span metrics in every condition.
  Raw predictions and original annotations remain available for inspection.
- Intervals resample papers or the source's group IDs. Sentence boundaries are
  heuristic. Per-generator results share prompts and are not independent.

`results.json` contains per-source, generator, domain/category, attack, length,
conference/year, and quality breakdowns. `verification.json` records hashes and
coverage. A separate paired context check lives in `../wide-eval-context-v1/`;
it supplies original neighbors and scores only the same focal human paragraphs.
That check was added after observing paragraph-only false positives and does
not constitute an independent holdout or change the model/thresholds.

The suite, human Parquet pools, predictions and reports are saved in the private
training Space at `/data/workspace/paper-v3-modernbert-20260930/`. Local copies
of inputs, human pools and reports live alongside this README. The public
Hugging Face dataset has not been modified by this evaluation task.
