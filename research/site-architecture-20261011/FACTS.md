# How the detector works: verified facts (Oct 11 2026)

Source material for the Architecture page (`app/components/site-architecture.tsx`). Every number below was checked against code or configs; file paths are the sources. **Naming (Alice, Oct 11 2026):** `final-moe` now means the Oct 10 loss-ablation winner (run `abl-noseg-nomix`). The Oct 7 model is renamed `final-moe-oct7`. The site's scores and cutoffs still come from final-moe-oct7 until the Atlas is re-scored.

## 1. Text extraction (PDF → text)
- Extractor: Poppler `pdftotext -enc UTF-8 -bbox-layout` (word boxes). Pages with no text layer fall back to Tesseract OCR at 300 dpi. (`backend/open_text_detector_backend/pdf_extraction.py:90,113-115`)
- Clean-up `positioned-clean-v2` (`backend/open_text_detector_backend/text_cleanup.py`):
  - ligatures expanded; invisible/control characters dropped (:19-25)
  - removed only when clearly identified: running headers (ICLR/COLM/TMLR, top 8.5% of the page, on ≥2 pages), "Anonymous authors" placeholder, margin line-number rails, page numbers (:183-248)
  - NeurIPS checklist template lines removed; Answer/Justification lines kept (:105-115)
  - line-end hyphen joined only if the joined word appears ≥2 times in the paper and the prefix isn't a compound prefix like "non-"/"self-" (:425-437)
  - math kept as extracted; broken math fonts / LaTeX source only flagged (:36-37,476-478)
  - references are not removed from the text
- Atlas input: datasets `iclr2027-clean-text-v2-20261005` and `iclr2027-backfill-clean-text-v2-20261005`.
- Training text: new human paper text is clean v2; some human windows are older PMC JATS, peS2o and arXiv text, mixed so text format can't predict the label (`benchmarks/otd/training/final-20261007/mix_README.md:51-57`).
- Sentences: regex `\S.*?(?:[.!?](?=\s|$)|$)` (same in training `data.py:9`).
- Sections (after scoring, for counting only; `research/atlas-bib-exclusion-20261008/bib.py`): bibliography starts at a References/Bibliography heading with ≥3 citation-like lines in the next 20; ends at an appendix-style line or a run of ≥5 lines / ≥400 chars with no citation signal. Appendix = everything after the bibliography if >200 chars remain. Counting modes: all / nobib (default) / noapp / body.

## 2. Input format and scoring (`research/atlas-rescore-moe-20261007/score_moe.py`)
- Qwen tokenizer, no special tokens, character offsets kept.
- Windows: 510 tokens, stride 510 (no overlap); the last window is shifted back to end at the last token, so only the tail overlaps. (Research evals use stride 256.)
- Repeat2: `ids + ids`, window fed twice, no separator/prompt/BOS/EOS; only the second copy is read (`backbone-launch-20261003/data.py:43-53`). Padded to 1,024.
- Token p(AI) = softmax class 1, averaged over the windows covering the token. BF16, LoRA merged.
- Sentence score = mean token p(AI) over the sentence's non-whitespace tokens.
- Document score: document head on the token-weighted mean over windows; stored as `document_prob`; shown only on the Calibration page.
- Cutoffs (nobib, `app/lib/paper-discovery.ts`): 0.1% FPR = 0.18322 (default), 1% = 0.008258; red at ≥ 0.80 (≈0.015% of human sentences). Flagged at ≥ cutoff.
- Calibration: 2,400 human papers (2019–2022), 823,565 nobib sentences; held-out FPR 0.1008% at the 0.1% cutoff.

## 3. Architecture
- Backbone Qwen/Qwen3.6-35B-A3B, revision 995ad96e; frozen, BF16.
  - 40 layers: 30 linear-attention + 10 full-attention (every 4th layer).
  - MoE feed-forward: 256 experts, 8 routed + 1 shared per token; ~3B of 35B parameters active per token.
  - Hidden size 2,048; 16 attention heads, 2 KV heads; vocabulary 248,320.
- LoRA: rank 128 on every linear layer, rank 16 on each expert's gate_up/down projections, alpha 32, no dropout (`final-20261007/models.json`, `train_sweep.py:241`, `adapters_short.py`).
- Heads (`overnight-sweep-20261004/modeling_sweep.py:11-31`):
  - token head Linear(2048→2) on every second-copy token → p(AI) per token (all site scores)
  - document head Linear(2048→2) on the mean of the window's token states
  - segment head Linear(2048→15) and mixed head Linear(2048→2) on the last token (final-moe-oct7 only). Segment label = AI share of the window's characters in 15 equal bins; mixed label = 1 if the minority label covers >15% of labelled tokens (`backbone-launch-20261003/data.py:62-64`)

## 4. Training loss (final-moe = abl-noseg-nomix)
- No stages: 72,000 distinct windows, one pass, 2,250 steps of 32 rows, Repeat2. Micro-batch 32 (no accumulation), `--fast-moe --pack`, 1 H200; 0 rollbacks (`sweeps/abl-noseg-nomix/train.log`).
- Per step: L = (1/32) Σ_rows [ℓ_token + 0.2 ℓ_sentence + 0.1 ℓ_document]; document-only rows have ℓ_document only. Exact for 71,998 of 72,000 rows (2 paragraph-edit rows have no single-label sentence). Code: per micro-batch mean × share of labelled rows, then × micro-batch size/32 (`modeling_sweep.py:124-142`, `train_sweep.py:552-559`).
  - token: per row ½ mean CE over target + ½ over context (one part if only one exists)
  - sentence: CE on the mean token logit of each single-label sentence, mean per row
  - document: CE on the document head; label AI if any AI region
- AdamW 8-bit, weight decay 0.01, clip 1.0; LoRA lr 1e-4, heads 2e-5, 6% warmup + cosine; seed 1; checkpoint = end of training.

## 5. final-moe-oct7 (previous; still serves the site's scores)
- Stage 1: 12,000 windows (375 steps), segment loss only, single copy. Stage 2: same 72,000 windows as three ~24k files, `--stage2-epochs 3`.
- Loss adds 0.2·segment + 0.1·mixed (heads on the last token). Micro-batch 8 × 4 accumulation, 1 H200 GPU 0 (`final-20261007/placements/placement-2026-10-07T110523Z.json`). Atlas used checkpoint stage2-epoch2.
