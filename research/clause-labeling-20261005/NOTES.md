# Soft n-gram labeling (Pangram 4 report §3.5): test and overnight relabel

Started 2026-10-05 evening, run overnight into 2026-10-06. The user pre-approved Luna (flex) runs on as much
data as could be found and asked for no questions overnight. The decisions made under that approval are
listed below.

## What the report does

Haiku 4.5 only **splits** the edited text into clauses. Each clause is then labeled with lexical n-gram
similarity L and embedding similarity E against the source: Human if L is high, AI-Assisted if it was
reworded, AI-Generated if nothing matches. The report's "seemed better" claim compares Haiku with classical
clause splitters, and it gives no metrics, thresholds or embedding model.

## Test (synthetic edits with exact per-character truth)

`gen_synthetic.py` edits 300 human paragraphs (pre-2022 papers) with Luna. Edit types: polish, paraphrase,
inserted sentence, a reworded single clause, an appended clause. 257 of the 300 paragraphs passed
validation. Thresholds are tuned on the dev half (split by paper) and scored on the 114 test pairs
(`label_eval.py`, `bootstrap_eval.py`).

| Splitter | macro-F1, report rule (polish = human) | macro-F1, our rule (polish = assisted) |
|---|---|---|
| spaCy en_core_web_trf clauses (`split_spacy.py`) | **0.895** | **0.884** |
| Luna clauses | 0.890 | 0.880 |
| sentences | 0.785 | 0.796 |
| whole paragraph | 0.39 | 0.35 |

- Luna minus spaCy: −0.005 (95% CI −0.03 to +0.02). There is no difference.
- Clauses minus sentences: about +0.10 (95% CI +0.05 to +0.16).
- The gain comes from edits inside sentences.
- No splitter flagged anything in the 45 untouched control paragraphs.
- Embedding model: Qwen3-Embedding-4B was no better than 0.6B on sentence-level edits.

**Decision: spaCy clauses + Qwen3-Embedding-0.6B.** It's free and deterministic, and it never alters the
text (Luna altered it in 6% of splits). Thresholds (spaCy, report rule): Human if L ≥ 0.72;
AI-Assisted if L ≥ 0.35 or E ≥ 0.70; otherwise AI-Generated.

The 300-sentence hand gold set (`annotator.template.html`, artifact YVG8QfNMXZBKEU6rdZsQna) was dropped
after 14 annotations: the user's labels used a coarser but consistent convention.

## Decisions made overnight

- **Binary training labels:** the trainer takes 0/1, so Human → 0 and Assisted/Generated → 1.
  The three-way labels are kept in `soft_regions` (0/1/2), with per-unit `soft_units` (L, E, label).
- **Light polish counts as Human** (report rule). Only the L ≥ 0.72 cut affects the binary labels.
- **No Luna clause-splitting at scale:** spaCy tied with it.
- **New Luna edits only on clean text already in use:**
  - gap10000 human originals with 4–12 sentences, source splits kept, never-train dropped
    (5,119 paragraphs).
  - Unused paragraphs from the sentence-edit builder's screened pool (`candidates-v1.jsonl`,
    `paired_train_nontarget` only; 3,238 after never-train).
  - The 14,561-paper archive, the Paper Atlas PDFs and the pool's 303 `archive_pre2023` rows were **not**
    used, because extraction improvements come first.
  - The edit mix is weighted to small edits: one sentence 45%, two 35%, half the paragraph 20%; no controls.
- **Spending cap:** $3 in total, flex only, no fallback to standard.

## Relabel of existing pairs (pilot, 1,000 rows)

Share of the old AI-region characters that soft labeling marks as **human** (near-verbatim copies):

| Pair type | Share now human |
|---|---|
| gap10000 paragraph rewrites | 24% (354/400 rows contain a copied clause) |
| llm-edits split_one | 55% |
| splices, two sentences | 12% |
| llm-edits rewrite_two | 6% |
| llm-edits rewrite_one, splices one sentence | 1–2% |
| llm-edits insert_one | 0% |

## Relabel of all existing pairs (done 2026-10-06 ~11:15 UTC)

All 25,283 pairs: 10,000 gap10000 rewrites, 5,000 LLM edits, 10,283 splices.
Rows: `research/data/soft-relabel-20261006/soft-rows-existing.jsonl.gz`, also in the bucket at
`workspace/clause-labeling-20261006/soft-rows/`.

| Pair type | AI chars now human | Rows with a copied clause |
|---|---|---|
| gap10000 rewrites | 25.4% | 8,939 / 10,000 |
| llm split_one | 30% | 407 / 449 |
| splices, two sentences | 14.8% | 2,123 / 2,894 |
| llm rewrite_two | 8.7% | 740 / 1,431 |
| splices, one sentence | 4.5% | 3,778 / 7,389 |
| llm rewrite_one | 2.7% | 467 / 1,533 |
| llm insert_one | 0% | 1 / 1,587 |

**Soft A/B arm for the 4B recipe:** `prepared/qwen35-4b-soft/prepared-v2` in the bucket (local copy in
`research/data/soft-relabel-20261006/prepared-4b-soft/`).
- It's a copy of `backbone-launch-20261003/runs/qwen35-4b/prepared-v2` with soft regions on every
  paragraph-rewrite row in stage 1 and stage 2. About 3,000 rows per epoch are relabeled, and AI-labeled
  characters fall 33% (for example 1.85M → 1.23M in stage-2 epoch 0).
- Every window mapped; none were missed. Selection and calibration windows are unchanged, so checkpoint
  selection is the same in both arms.
- The baseline arm is the original directory.
- For wave-2-style variants that add LLM edits or splices, run `soften_prepared.py` on the variant's
  prepared dir with the same soft-rows file.

**Never-train leak in existing training data (not caused by this work):** 9, 5 and 8 rows in base 4B
stage-2 epochs 0, 1 and 2 (and 9 in stage 1) now match the expanded never-train lists. `build_llm_edits.py`
now also loads the Claude full-paper and section held-out lists. Ids are in
`research/data/soft-relabel-20261006/never-train-in-base-epoch0.txt`; `never_train` is set on those soft rows.

**No GPU training was launched.** pangram-54 reported the user had said no new GPU work overnight.

## New Luna small edits (done 2026-10-06 ~12:40 UTC)

`research/data/soft-edits-v1-20261006/soft-edits-v1.jsonl.gz`, also in the bucket at
`workspace/clause-labeling-20261006/soft-rows/`. 13,767 edited paragraphs:

| Batch | Rows |
|---|---|
| gap10000 originals, two random edit plans (`gen_gap10000_a`, `gen_gap10000_a2`) | 4,195 + 4,193 |
| unused screened pool, two edit plans (`gen_pool_b`, `gen_pool_b2`) | 2,685 + 2,694 |

- Splits: train 11,080 / validation 1,792 / test 895, inherited from the source papers.
- 40 rows were dropped on the never-train recheck.
- Each row has:
  - `regions`: soft binary labels.
  - `truth_binary_regions`: **exact** construction labels, with polish counted as human. Prefer these for
    training.
  - `soft_regions`, `soft_units` and `truth_regions` (three-way, with the edit op).
- The validation and test rows give a new small-edit evaluation set with exact truth (Luna-only, so
  same-generator).

**Soft labels against exact truth, 13.8k generated rows, report rule:**
- Binary character agreement 97.7%.
- AI recall 96%, AI precision 89%.
- 2.0% of human characters flagged as AI, mostly the untouched part of sentences where only one clause
  was edited.
- Per edit type, the share labeled AI: insert 100%, paraphrase 98%, appended clause 95%,
  reworded clause 78%, polish 11% (as intended).

**Cost:** $1.87 of OpenRouter credit for the whole session (key usage 45.87 → 47.74), flex only.

## Morning to-do (needs the user)

1. **A/B on 4B, ≥3 seeds per arm:** original `prepared-v2` vs `prepared/qwen35-4b-soft/prepared-v2`.
   Metric: small-edit recall at 1% FPR on the held-out set, plus retained-human FPR.
2. **If soft labels help, a third arm:** add `soft-edits-v1` train rows (with `truth_binary_regions`)
   to the mix, for example in place of GRADTEX, as the LLE arm did.
3. **Remove the 22 never-train rows** from the existing 4B prepared epochs (see above).
4. **Rerun wave 2** if wanted: its `/tmp` state was lost in the 07:19 UTC Space restart.

## Files

- Code: this directory.
  - `build_relabel_inputs.py`: existing pairs.
  - `gen_synthetic.py --production` / `--candidates`: new edits.
  - `gen_to_pairs.py`: generated edits → pairs.
  - `space_relabel.py` + `launch_relabel.py`: Space CPU job.
  - `make_soft_rows.py`: training rows.
- Data (git-ignored):
  - `research/data/soft-relabel-20261006/`: existing pairs, units, rows.
  - `research/data/soft-edits-v1-20261006/`: new edits.
- Space: `/data/workspace/clause-labeling-20261006/` (inputs, units, logs).

## Incidents

- **Model download rule broken (fixed):** two spaCy models were downloaded on the Mac, against AGENTS.md.
  They were deleted (492 MB) and everything since runs on the Space.
- **Space restart:** the training Space restarted at about 07:19 UTC on 2026-10-06. `/tmp` was wiped and all
  GPUs went idle, so wave-2 training in `/tmp/pangram-splice-20261006` appears to be lost.
  - My two CPU relabel jobs were killed and resumed from `/data`.
  - Cause unknown. My jobs used a few GB against a 1.16 TB limit, and the OOM counters reset with the
    restart.
  - Session pangram-54 was told.

## T2.1 soft-label arm (2026-10-06 evening)

- Relabelled the paired rows T2.1 trains on that had no soft labels: Claude/GPT small edits, Claude/GPT paragraph
  edits and Claude sections (4,940 pairs; `benchmarks/otd/training/t21-20261006/build_soft_t21.py`).
  Output: bucket `workspace/clause-labeling-20261006/soft-rows/soft-rows-t21new.jsonl.gz`.
- Share of AI characters now labelled human: section rewrites 57%, paragraph rewrites 38%, section drafts 3%,
  paragraph drafts 0.2%, Claude rewrite_two 16%, rewrite_one 5%, inserts 0%.
- **Speed:** don't run the relabel on CPU when sections are included. Four CPU shards (32 threads each) wrote no
  rows in 82 minutes; sections have many clause units, so the embedding step dominates. Six shards with the
  embedding model on a GPU (`launch_relabel.py ... GPU_INDEX`, `RELABEL_THREADS=12`) finished in 7 minutes.
  spaCy stays on CPU (no cupy on the Space; `space_relabel.py` now uses `spacy.prefer_gpu()`).
