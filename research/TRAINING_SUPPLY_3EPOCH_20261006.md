# Training supply for 3 full-length epochs on one pass (counted 2026-10-06, 11:30 PM PDT)

**Target:** the T2.1 mix (24,000 stage-2 rows per full-length epoch) for 3 epochs without repeating a window: 72,000 rows, so each key needs 3× its per-epoch rows.

**Why rows fail `never_train`.** The T2.1 held-out split (`research/t21-heldout-20261006/build_t21_split.py`, Oct 6 evening) holds out every paper hosting a GPT-6 Sol or Fable row, plus 15% of papers hosting Opus/Sonnet small edits. Every row on a held-out paper, from any writer, is then dropped from training. Tonight's items were picked with `build_llm_edits.never_train_hit`, whose file list does not include `t21-heldout-never-train.json`, so about 17% of tonight's Claude edits landed on held-out papers. Add that file to host selection before the next batch.

**How it was counted.** Every number is distinct usable 510-token windows after the same screening `build_t21.py` applies: 510-token crop, `never_train_hit` with the T2.1 v2 lists (`nt21.py`), evaluation paper ids, and shared 10+-word sentences. Fable is excluded everywhere (it is a held-out writer). Sections from any paper with a Fable section are also excluded, human side included.
- **Generated pools** (Claude/GPT edits, sections, paragraph edits, human matches and hosts) were rebuilt from tonight's ingested files and screened on the Space (`/data/workspace/recount-20261006/screen.json`).
- **Prepared sources** (mirrors, GRADTEX, generic human text, Luna paragraph edits, original human paper text, GPT/Luna manuscripts) were counted from the original source pools (`backbone-launch-20261003/source-pools`), not from the three already-drawn epoch files (`recount.json`). Long documents count as `ceil(tokens / 510)` non-overlapping windows. Paired paper rows count one window each, because the AI span must be inside it. GRADTEX counts one row per document (document labels only).

**Correction to the earlier figures.** The T2.1 build report counted prepared sources by document id, among windows already drawn. That undercounted them: GPT/Luna manuscripts are 5,537 windows, not 377; original human paper text is 4,786, not 3,819; Luna paragraph edits are 4,786, not 3,793.

## AI-generated keys

| Key | Per epoch | Need (×3) | Have | Left to make | Notes |
|---|---|---|---|---|---|
| Claude small edits (`edit_claude`) | 4,440 | 13,320 | **10,622** (Opus 5,360, Sonnet 4,308, Haiku 954) | **2,698** | 17% of rows fail `never_train`; screen hosts before assigning |
| Luna small edits (`edit_luna`) | 3,360 | 10,080 | 12,989 | 0 | |
| GPT small edits (`edit_gpt`) | 600 | 1,800 | **397** | **1,403** | All GPT-6 Sol rows are held out by design (unseen-writer eval), so this must come from GPT-6.1 Sol |
| Luna paragraph edits (`para_luna`) | 1,740 | 5,220 | **4,786** | **434** | Codex/OpenAI pipeline, or repeat about 9% |
| Claude/GPT paragraph edits (`para_e`) | 1,160 | 3,480 | **2,278** (Sonnet 1,157, Opus 720, Haiku 221, GPT-6.1 Sol 180) | **1,202** | All 209 GPT-6 Sol rows fail `never_train` |
| AI sections (`sections_ai`) | 1,400 | 4,200 | **1,998** (Opus 846, Sonnet 839, Haiku 313) | **2,202** | Limited by clean human sections: the last prepare found only 699 of 2,470 wanted |
| Luna mirrors | 1,400 | 4,200 | 34,126 | 0 | |
| GPT/Luna full manuscripts (`fullpapers`) | 500 | 1,500 | 5,537 (366 papers) | 0 | about 4 windows per paper used |
| Claude full papers (`claude_fp`) | 500 | 1,500 | 5,644 (210 papers) | 0 | about 7 windows per paper used |
| GRADTEX (document labels) | 1,200 | 3,600 | 77,142 | 0 | |

## Human keys

| Key | Per epoch | Need (×3) | Have | Left | Notes |
|---|---|---|---|---|---|
| Original human paper text (`human_paper_orig`) | 3,000 | 9,000 | **4,786** | **4,214** | Needs new papers (the 14,561-paper archive after extraction work) |
| Matched human sections (`human_section`) | 1,000 | 3,000 | **1,983** | **1,017** | One per accepted AI section; grows with sections |
| Matched human paragraphs (`human_para_e`) | 1,000 | 3,000 | **2,272** | **728** | One per accepted paragraph edit; grows with `para_e` |
| Human host passages (`human_host`) | 1,000 | 3,000 | 5,167 | 0 (+2,167 spare) | Same host pool as new edits: each host used for an edit leaves this pool |
| Generic human text (`human`) | 1,700 | 5,100 | 121,322 (72,967 docs) | 0 | |

**Human paper text as a group:** need 18,000, have 14,208, short 3,792. The spare host passages cover part of this if shares move within the group.

## What is left, by route

1. **Claude small edits (2,698 rows).** Screen the 9,711 unassigned hosts with `nt21` first; about 5,167 pass. Then about 2,800 Opus/Sonnet items (about 56 batches) is enough, at about 1.5k subagent tokens per item (about 4M tokens). This uses about 2,800 of the 5,167 clean hosts and leaves about 2,370 for `human_host`, below its 3,000 need. The 1,600 unrun Fable `scale2` items (hosts already assigned, in `items-d-scale2.jsonl`) can be reassigned to Opus/Sonnet instead.
2. **GPT small edits (1,403) and Luna paragraph edits (434):** Codex, so no Claude usage. Screen hosts against `nt21` first; T2.1 lost 60% of GPT rows to `never_train`.
3. **Paragraph edits (1,202 AI, 728 human matches).** About 1,640 more Claude items (79% screening survival, 93% acceptance). Each accepted row also adds a matched human row. Needs hosts, so it competes with route 1.
4. **Sections (2,202 AI, 1,017 human matches).** Limited by full papers dated 2022 or earlier, not by extraction. The 14,561-paper archive already has extracted text, but only 343 of its papers are 2022 or earlier (NeurIPS 2021 excluded); 295 of those are already segmented. Other sources are the 1,482 paired papers (already segmented) or a new source of pre-2023 full texts (arXiv, PMC, peS2o).
5. **Original human paper text (4,214).** Same limit: needs more pre-2023 papers.

**If nothing more is generated,** the shortfalls repeat within 3 full-length epochs. Roughly: human paper original 1.9 passes, sections 2.1, `para_e` 1.5, GPT small edits 4.5, Claude small edits 1.25. Repeating human text is the low-risk part. GPT small edits at 4.5 passes is the one most likely to overfit.

## Tonight's generation (Oct 6, 10:55–11:25 PM PDT)

Workflows `wf_95a695bf-b07`, `wf_fa56f70a-caa`, `wf_483d5d81-a05` (Opus), `wf_87110867-2e1` (Sonnet), `wf_d7313c8b-b5f` (Haiku), and `wf_e86b58a2-b3c` and `wf_f80d97a3-27a` (Opus `scale2`). That's 324 batches, no errors, about 20M subagent tokens (plus a few minutes of runs stopped and relaunched). Ingested 11:25 PM. Small edits: 10,080 accepted in `llm-edits-claude-hsm-v1` (all writers, 92% of returned). Paragraph edits: 2,877. Sections: 2,733 (152 held out).

## Update, Oct 7 12:35 AM PDT

- **Never-train check:** `build_llm_edits.NEVER_TRAIN_FILES` now includes the T2.1 held-out lists, v1 and v2. Host selection, the writer self-check and ingest all apply them. v2 adds the later Fable papers, so it drops slightly more of the older rows than the Space screen above, which used v1 only.
- **Scale-up 3** (`build_hosted.py prepare-scale3`, stats in `prepare-scale3-stats.json`): hosts were screened before assignment. Never-run items with pre-split hosts were released: 1,600 Fable, 1,560 GPT round-2 small edits, 210 GPT-6 Sol round-2 paragraph edits. 4,505 clean hosts; assigned Opus 1,450, Sonnet 1,450, GPT-6.1 Sol 1,500 (Codex handoff `openai-writer-handoff-20261006/ROUND3-GPT61-SOL.md`). 3,106 clean hosts remain for `human_host` (need 3,000).
- **Claude small edits after scale-up 3** (workflows `wf_94228a82-a05`, `wf_9900cf2b-88c`, 4.7M subagent tokens): 12,754 pass the never-train check (Opus 6,333, Sonnet 5,511, Haiku 910), against 13,320 needed. Short 566, about 4%. Not yet re-screened on the Space for evaluation-sentence overlap, which removed almost nothing last time.

## Audit after the corpus extraction (Oct 7, 1:20 AM PDT)

**Human paper supply is no longer a bottleneck.** R2 holds 18,493 PDFs dated 2022 or earlier (ICLR 2013–2022, NeurIPS 2021–2022, CoRL 2021–2022, TMLR 2022). Excluding the calibration set and the Atlas catalogue leaves 15,830, or 13,511 without NeurIPS 2021. The corpus job (`research/corpus-extraction-20261007/`) gives every one of them direct, clean-v2 and omissions-v2 text. At about 18 windows per paper that is roughly 280k windows, against these needs:

| Need | Papers | Status |
|---|---|---|
| Original human paper text, 4,214 windows short | about 850 at 5 windows per paper | covered |
| Sections, about 2,500 new items | about 1,300 (2.4 sections per paper, 15% held out) | covered |
| Hosts for small edits and paragraph edits | each paper has about 40 paragraphs | covered |

**What is left is generation, not data:**

| Slice | Accepted rows short | Items to write | Writer / cost |
|---|---|---|---|
| Claude small edits | 566 | about 600 | Opus/Sonnet, about 1M subagent tokens |
| GPT small edits | 1,403 | 1,500 (prepared) | GPT-6.1 Sol via Codex (`ROUND3-GPT61-SOL.md`) |
| Claude/GPT paragraph edits | 1,202 (plus 728 human matches, which come with them) | about 1,400 | Claude, about 3M tokens |
| AI sections | 2,202 (plus 1,017 human matches, which come with them) | about 2,500 | Claude, about 23M tokens (about 9k per section, as in Job C) |
| Luna paragraph edits | 434 | about 450 | Luna pipeline, or repeat about 9% |

**Decisions still open (not data):**
1. **Which text stage training uses.** Paired, section and edit rows were built on the older Poppler extraction. Using clean v2 for the new human text matches them. Using omissions v2 only on the human side would make `⟦…⟧` markers a human-only signal (the symmetric-omissions problem in the omission notes). Recommendation: train on clean v2 until omissions are applied symmetrically.
2. **NeurIPS 2021.** Excluded by earlier builders for an unrecorded reason (2,319 papers).
3. **Leakage screening** of the new papers against the evaluation suites and held-out lists, done at sampling time (`never_train_hit` plus the eval-paper checks).

## Wide-pool sampling (Oct 7, 1:30 AM PDT)

Decisions: train on clean v2; NeurIPS 2021 included (exclusions removed from build_llm_edits / build_sections / build_fullpapers); leakage screening at sampling time.

`research/corpus-sampling-20261007/` (runs on the Space; push with `push.py`):
- `build_pool.py`: papers dated 2022 or earlier from `corpus-clean-text-v2`, minus the calibration set and papers whose id is on a held-out or eval list. Paragraphs come from the clean v2 line structure (one paragraph per line; a lowercase-start line continues the previous one). Hosts pass `host_ok`; sections come from `segment_archive` plus `eligible_sections`. `never_train_hit` and the shared-eval-sentence check run per host (with context) and per section, not per paper. Outputs `hosts.parquet`, `segments.jsonl.gz`, `papers.parquet`, `stats.json`.
- `sample.py`: round-robin over papers (seeded shuffle within venue/year strata, strata interleaved by size). Every eligible paper gives one item before any gives a second; `--max-per-paper` caps it, and `--exclude` skips used hosts/papers.
- Test on 300 papers: all kept, about 18 hosts and 2.7 eligible sections per paper. 200 sampled sections came from 200 papers. Extrapolated to about 15.8k papers: about 280k hosts and 43k sections.
- Open: eligible sections skew toward introductions and related work; add role quotas (`ROLE_MIX`) if the section mix matters. Clean v2 keeps some line-break hyphenation ("re- sources").
