# Open Text Detector project ledger

As of Wed Oct 7, 1:55 AM PDT. Built from all 20 Claude Code session transcripts on this project (15 with substantial work, Oct 3–7, 2026). Times are PDT. Items marked **[today]** are from Oct 6, 6:00 AM PDT onward.

New to the jargon? See the [Glossary](#glossary) at the end. Exact mix compositions are on the [run board](https://claude.ai/artifact/Cnh2kuWCVC7UDbVFVsp3pj) (Runs & mixes tab).

## What you are trying to solve

- **Core research.** Detect AI-written text in research papers at document, sentence and token level with very low false positives on human writing. Hardest case: one or two AI sentences inside an otherwise human paper.
- **Training data.** Give the model enough short AI edits, from enough writers, inside enough real human paper text. The old mix had 5 small-edit windows out of 24,000 per epoch.
- **Evaluation.** Decide between models on clean held-out data from several writers, at a calibrated 1% human false-positive rate, with intervals that resolve the differences being tested.
- **Human calibration.** Set thresholds on thousands of pre-2023 human papers (rule: 2022 or earlier is human) with paper-level intervals, at sentence and document level.
- **Model and recipe.** Pick a stable LoRA recipe and a backbone (Qwen3.5-4B or Qwen3.6-35B-A3B MoE) that is cheap enough to iterate on and to score the Atlas with.
- **Text extraction.** Clean text from PDFs (the 'baseline': text-layer extraction plus cleanup) for every paper in R2, without teaching a formatting shortcut.
- **Paper Atlas.** Publish calibrated classifications with PDF overlays on the public Atlas site to collect feedback before the announcement.
- **Infrastructure.** Run across the HF Space (8×A100), the shared H200 and Cloudflare R2 without losing work to restarts, quotas, rate limits or usage limits.

## What changed since last night

Last night = the state at about midnight Oct 5→6 (pick-up note).

| Topic | Last night | Now |
|---|---|---|
| Best data mix | SPG (splices replace GRADTEX): small-edit recall 0.63 on Luna edits; best available model. | T2/T2.1 (multi-writer small edits, more human paper text, sections). Held-out writers 0.60 vs 0.47, cross-model 0.83 vs 0.41, standalone rewrites 0.78 vs 0.26. SPG's Luna lead is home advantage: on clean v2 T2.1 0.79 vs SPG 0.38. |
| Evaluation sets | Headline = 150 Luna small-edit sentences (±7 pts). Held-out writer set: 1,185 edits, 4 writers. | Held-out v2: 4,000 AI rows + 4,000 human, 6 writers (Fable 5.1 and GPT-6 Sol never trained on), sources tagged (PMC, peS2o, arXiv, paired ML). Plus Fable polish 299, humanizer 200, co-writing 100. CI on v2 about ±1.7 pts. |
| Backbone | MoE best on edits at 20% length; full-length MoE overfit. | MoE T2 (20% length) leads 4B T2 on held-out writers 0.63 vs 0.60 and cross-model 0.91 vs 0.83; equal on Luna small edits. One seed. |
| Soft (clause) labels | Built and validated on synthetic data (+0.10 macro-F1); A/B not run. | A/B on T2.1: worse. Clean v2 0.72 vs 0.84, humanizer 0.61 vs 0.83. Not adopted. |
| Human calibration | 215 pre-2023 papers; document threshold rested on 2 papers; sentence intervals treated sentences as independent. | 2,403 papers with paper-clustered intervals. Live Atlas threshold 0.0283 gives 0.71% FPR; proposed 0.0248 (held-out FPR 0.93%). Document threshold 0.211 on 25 papers. Each model needs its own threshold (T2 0.165, SPG 0.0070). |
| Threshold stability | Assumed ~1 pt seed noise. | T2.1 seed 3 set its 1% threshold at 0.667 vs ~0.41 for seeds 1–2, with the same AUROC. Recall at a dev-set threshold can swing 30 pts from calibration alone. |
| Boundary bleed | 12–19% of human sentences beside AI sections flagged. | That figure came from old-mix MoE runs. T2-family models flag 2.6–5.0% of human sentences in mixed documents vs 0.6% in human documents. 76% of T2 section training windows are cropped to pure AI. |
| AI training data | Claude sentence edits 4,919; sections 1,164; full papers in progress; no GPT data. | Claude small edits 12,754 (Opus 6,333, Sonnet 5,511, Haiku 910); paragraph edits 5,754; sections 2,733 (+2,733 matched human); full papers 254; GPT-6.1/6 Sol 1,000 small + 418 paragraphs; RAID chat slice 6,000 windows. Fable data is evaluation-only. |
| Writers in training | Luna only in the trained models; Claude and GPT data generated but untrained. | Claude (Opus, Sonnet, Haiku), Luna and GPT-6.1 Sol in T2.1. Queued: DeepSeek V4.1 Flash as the first open-weight writer (4,000 small + 1,200 paragraph edits, 15% held out). Evaluation-only writers: Fable 5.1, GPT-6 Sol, and queued GLM 5.3 Flash and MiMo V2.6 Flash. |
| Host sampling | Hosts and sections came from ~1,130 paired papers plus a few thousand PMC/peS2o/arXiv passages; heavy reuse of the same papers. | Queued wide pool: every pre-2023 R2 paper split into ~280,000 host paragraphs and ~43,000 sections across ~15,800 papers, leakage-screened per passage, sampled one item per paper before any paper gets a second. NeurIPS 2021 back in. |
| Human paper text | ~1,130 training papers; thought to be the main gap, blocked on extraction. | R2 already holds 18,493 PDFs from 2022 or earlier (~280k windows). Extraction of all of them (and parity for all 80,226 R2 PDFs) is running on the Space. |
| Seeds | 3 seeds per arm by default. | Default is 1 seed per arm with a paired bootstrap; more seeds only for close calls, equivalence checks and the final model. In every past arm seed 1 gave the same decision. |
| Space persistence | Work in /tmp; two restarts had already lost runs. | persist_daemon mirrors to /data every 5 min, restart watcher, outputs written to /data. Two more unexplained restarts (Oct 6 12:19 AM, 2:36 PM) caused no loss after the fix. |
| Text-extraction omissions | Active hill-climb (v2: noise 126→22 per 1k words). | Deprioritized. Omission markers on human text only would be a 'this is human' cue, so train on clean v2 until omissions apply to both sides. |

## Priorities

### Suggested order

1. **Check the corpus extraction finished and is safe** — Two Space jobs started ~1 AM: 12,134 pre-2023 papers, then all 80,226 R2 PDFs. Your time: ~5 min. Compute: 2–3 h of Space CPU. Depends on: Jobs from session adff.
   - Steps: Confirm both jobs finished and counts match R2; Confirm datasets are in the bucket; Delete the export Worker and its token.
   - Risks: A Space restart mid-job (outputs go to /data in 500-paper chunks).
   - Done when: Every R2 PDF has direct, clean v2 and omission v2 text in the bucket.
2. **Decide the ICLR 2027 68% question** — Open 31 h. The Atlas shows per-paper flags on named papers that nobody has validated. Your time: ~45 min reading flagged sentences. Compute: ~20 OpenReview batch requests for ICLR 2026 submissions. Depends on: none.
   - Steps: Read top-flagged sentences in ~20 papers; Score ICLR 2026 submission versions; Say whether 2027 reflects AI use or format drift.
   - Risks: The fast10 model is weak; rescoring with the chosen model may change the picture.
   - Done when: A written verdict, and the Atlas label updated to match.
3. **Approve the threshold switch and Calibration page deploy** — 0.0283 → 0.0248 (human FPR 0.71% → 0.93%). Deploy would also ship other sessions' uncommitted app changes. Your time: ~10 min. Compute: Rebuild and re-upload all detail files (bundled). Depends on: Choose deploy-all or calibration-only.
   - Steps: Choose deploy scope; Edit build_atlas_details.py:25; Rebuild and upload bundles; update catalogue calibration metadata.
   - Risks: Shipping unreviewed app changes.
   - Done when: Atlas flags at a 1% FPR measured on 2,403 human papers.
4. **Build the next mix and train one full-length run** — One 72,000-row epoch: Claude, Luna, GPT-6.1 Sol and DeepSeek small edits, wide-pool sections and paragraphs, pre-2023 human text, RAID slice. One seed. Your time: ~15 min to approve. Compute: 4B ~1.5 h on 2 A100s; MoE ~2.6 h on 2 H200s. Depends on: Corpus job; queued generation (session adff); GPT-6.1 Sol round 3.
   - Steps: Fix section crops to keep a boundary; Build with every never-train list, train on clean v2; Score on clean v2, Fable sets, cross-model, humancal per-model threshold.
   - Risks: Overfitting if any pool repeats >2×; Threshold noise: calibrate per model on humancal.
   - Done when: A model that beats T2.1 on clean v2 at a calibrated 1% FPR, ready to rescore the Atlas.
5. **Run GPT-6.1 Sol round 3 in Codex** — GPT small edits: 397 usable vs 1,800 needed. Your time: ~10 min to start Codex, then wait. Compute: Codex subagents. Depends on: none.
   - Steps: Open Codex in the repo; Run research/openai-writer-handoff-20261006/ROUND3-GPT61-SOL.md; Ingest with the strict check.
   - Risks: Codex output format drift.
   - Done when: ≥1,403 accepted GPT-6.1 Sol small edits.
6. **Commit everything that is safe** — Most of the last 24 h of code lives in scratchpads or is uncommitted. Your time: ~5 min. Compute: none. Depends on: none.
   - Steps: Bring the T2/T2.1 builders and calibration/corpus code into the repo; Commit on overnight-20261006; Push to both remotes.
   - Risks: Other sessions' half-finished changes in the same tree.
   - Done when: A fresh session can rebuild every mix from the repo.

### Where your time went today

Today = Oct 6, 6:00 AM – Oct 7, 1:55 AM PDT. Your active time is measured from your message timestamps: gaps under 20 minutes count as active. Across all sessions that is about 6.2 h of wall-clock time (370 min); parallel sessions overlap, so topic minutes below add up to more (~470 session-minutes). Topic minutes are approximate (±25%); grades judge payoff against your time, not compute.

| Topic | Your minutes | Grade | Payoff |
|---|---:|---|---|
| Data generation (Claude resume, Fable, Codex GPT, scale-ups, supply counting) | 195 | B− | mixed |
| Mix decision and evaluation (T2, T2.1, MoE T2, soft A/B, eval coverage, held-out v2) | 100 | A | high |
| Infrastructure (restarts, persistence, backups, commits, pushes, tokens, permissions) | 70 | B | overhead |
| Dashboards and ledgers (run board, bleed plan, ledgers, priorities) | 45 | C+ | mixed |
| Human calibration (2,403 papers) | 20 | A | high |
| Public dataset audit (RAID, MAGE, SeqXGPT) | 15 | B+ | high |
| Omissions wrap-up | 13 | C | low |
| Break-reminder hook fixes | 12 | C | overhead |

### Usefulness grades for today's work

| Grade | Work | Why |
|---|---|---|
| A | T2 → T2.1 mix and its clean evaluation | Largest generalization gain of the project; decisions now rest on 6 writers and 3,366 clean rows instead of 150 Luna sentences. |
| A | Human calibration on 2,403 papers | Replaced a document threshold resting on 2 papers; showed the live threshold is conservative. |
| A | Held-out v2 and Fable evaluation sets | First unseen-writer tests; exposed the co-writing blind spot (0.14–0.17). |
| B+ | RAID/MAGE/SeqXGPT audit | Found that GRADTEX (20% of the old mix) inherits MAGE's mislabeled continuations. |
| B | MoE T2 runs | Backbone evidence at low cost; one seed. |
| B | Claude generation resume and Codex GPT edits | Most of the AI data needed for full-length training; ~17% wasted on held-out papers. |
| B | Persistence, restart watcher, backups | Overhead, but no work lost after noon despite two restarts. |
| B | Corpus extraction parity job | Removes the human-text bottleneck if it finishes; still running. |
| C+ | Soft-label A/B | Clear negative result for ~1 h of GPU; the CPU relabel stall cost 82 min of waiting. |
| C | Dashboards and ledgers | Useful orientation, but rebuilt several times as facts changed. |
| D | 72k scale-up launched then stopped | 7% of the weekly budget for 11–15% of the work; most items later redone differently. |
| D | Fable Opus-fallback debugging | ~45 min of your time; the data is evaluation-only anyway. |
| D | 3-epoch supply recount | ~45 min of confusion from a manuscripts-vs-windows miscount and jargon. |

### Least payoff today

| Work | Your time | Compute | Outcome | Assessment |
|---|---|---|---|---|
| Fable generation and Opus-fallback debugging | ~45 min | Opus usage on 3 quarantined batches | 2,186 Fable items; all now evaluation-only | The debugging was the waste; a Fable-only rule up front would have avoided it. |
| 3-epoch supply estimate and recount | ~45 min | none | Correct supply table after a recount | Counted manuscripts instead of windows (377 vs ~5,500) and explained in jargon; two rounds of "I don't understand". |
| 72k scale-up and Fable jobs (stopped) | ~25 min | 7% of the weekly Claude budget | Stopped at 11–15%; partial rows ingested | Cost estimate ignored orchestration overhead and fork workers. |
| Claude small edits on held-out papers | ~10 min | ~1–2M subagent tokens (est.) | ~17% of the batch unusable for training | Host screening missed the T2.1 list; fixed for scale3. |
| "Only 343 pre-2023 papers" claim | ~15 min | none | You planned a browser download session | R2 held 18,493 such PDFs; the calibration run had used them hours earlier. |
| CPU clause relabel stall | ~5 min | 82 min of waiting on Space CPU | Redone on GPUs in 7 min | Thread oversubscription; no progress check until a peer session looked. |

### Probably not worth continuing

- **Omission hill-climb.** Deprioritized by you; markers would leak the label if applied to one side only.
- **Wave 2 rerun (LLE, MIX).** Questions answered better by T2/T2.1.
- **Soft clause labels.** A/B lost by 12 pts on clean v2.
- **Multi-seed screening.** Seed 1 never changed a decision.
- **FSDP MoE on A100s.** Only useful if the H200 becomes the bottleneck.

## Open threads

Age is hours since the thread was last touched. Over 12 h counts as stale.

| Thread | State | Priority | Owner | Last touched | Age (h) | Next step |
|---|---|---|---|---|---:|---|
| Publication decisions: per-paper scores, PDF licences, project name vs Pangram Labs, brand assets | open | high | Alice | Oct 3 6:41 PM | 79.2 | Decide before the announcement. Per-paper ICLR 2027 scores are public as 'Experimental'. |
| Is ICLR 2027's 68% real? | open | high | Alice | Oct 5 6:56 PM | 31.0 | Read top-flagged sentences in ~20 papers; score ICLR 2026 submission versions. Open 31 h; blocks any honest Atlas claim. |
| Per-model thresholds and threshold noise **[today]** | open | high | Claude | Oct 6 9:46 PM | 4.2 | Calibrate each candidate model on the 2,403 human papers instead of 7,321 dev sentences; report AUROC beside recall. |
| Pick the model to ship (mix and backbone) **[today]** | waiting on you | high | Alice | Oct 6 11:00 PM | 2.9 | T2.1 hard labels chosen over soft. Decide 4B vs MoE; the final model should train on the next mix (new Claude data + pre-2023 human text) with the v2 never-train list. |
| Switch Atlas threshold 0.0283 → 0.0248 and deploy the Calibration page **[today]** | waiting on you | high | Alice | Oct 7 12:39 AM | 1.3 | Deploy would also ship ~177 lines of other sessions' uncommitted app changes: choose deploy-all or calibration-only. Threshold is hard-coded in build_atlas_details.py:25; every detail file must be rebuilt. |
| Build the next mix (one 72,000-row epoch) and run the format check **[today]** | open | high | Claude (session adff) | Oct 7 1:27 AM | 0.5 | T2.1-style builder with all leakage screens and no repeated windows; check the human vs AI share per extraction format before training. |
| Corpus extraction: all R2 PDFs → direct, clean v2, omission v2 **[today]** | running | high | Claude (session adff) | Oct 7 1:45 AM | 0.2 | Pre-2023 job: all 12,134 PDFs fetched 1:43 AM, extracting. Full R2 job (80,226 PDFs) starts after it, ~1–2 h. Verify counts and bucket persistence; delete the export Worker and token after. |
| Queued data generation for a 72,000-row epoch **[today]** | waiting | high | Claude (session adff) | Oct 7 1:54 AM | 0.0 | Space orchestrator waits for the corpus job, builds the wide pool, samples sections and hosts, then runs Luna (470 paragraph edits), DeepSeek (4,000 small + 1,200 paragraph, 15% held out) and eval-only GLM 5.3 / MiMo V2.6 (300 + 100 each). Then local Claude workflows: ~2,600 sections, ~1,400 paragraph edits, ~650 small edits (~27M subagent tokens). |
| Boundary-bleed falloff metric and builder fixes **[today]** | open | medium | Claude | Oct 6 8:04 PM | 5.8 | Compute bleed by sentence and token distance from saved scores; boundary-centred section crops and longer host context in the next build. Plan: artifact Lhibge5QEN8JfnoZLZApBN. |
| Paper-embedded evaluation **[today]** | open | medium | Claude | Oct 6 8:04 PM | 5.8 | Put held-out edits back into full human papers and score with sliding windows, as deployment does. |
| Commit and push all code **[today]** | waiting on you | medium | Alice | Oct 6 11:00 PM | 2.9 | f479f32, 4fdcbdd and this ledger are on overnight-20261006; T2/T2.1 builders, strict check, calibration and corpus code are uncommitted or only in scratchpads. The T2 builder exists only in a scratchpad and the bucket. |
| Eval gaps: GPT-6.1 Sol 130 rows, 66 archive controls, no Luna in v2, polish only at document level **[today]** | open | medium | Claude | Oct 6 11:28 PM | 2.5 | Add the ~17% held-out-paper Claude edits as eval rows; token-level metric for polish. |
| GPT-6.1 Sol round-3 small edits via Codex **[today]** | waiting on you | medium | Alice | Oct 7 12:23 AM | 1.5 | Run research/openai-writer-handoff-20261006/ROUND3-GPT61-SOL.md in Codex (1,500 items, ≥1,403 needed), then ingest. |
| Atlas search latency (2.5–5 s) | stale | low | Claude | Oct 3 5:17 PM | 80.6 | 80 MB index exceeds Worker memory; stream it or move search out of the Worker. |
| PyMuPDF AGPL licence for the omission code | open | low | Alice | Oct 5 5:03 PM | 32.9 | Matters only if the omission pipeline ships; pypdfium2 is the alternative. |
| Calibrated colour bands in the reader | stale | low | Alice | Oct 5 8:49 PM | 29.1 | Raw-score bands colour flagged sentences red and some green; use per-model calibration metadata. |
| FSDP training of the MoE on A100s | stale | low | Alice | Oct 5 10:32 PM | 27.4 | 2 A100s ≈ 1 H200 with FSDP2; ~1–2 h to build. Only worth it if the H200 is the bottleneck. |
| Third-party paper text in shared artifacts | waiting on you | low | Alice | Oct 5 11:45 PM | 26.2 | Omission Review, Text-Layer vs MinerU and Clause Cuts pages contain paper text; check share settings. |
| Wave 2 (LLE, MIX, Arep) lost in the 12:19 AM restart | abandoned | low | Alice | Oct 6 12:23 AM | 25.5 | Superseded by T2/T2.1. Rerun only if 'LLM edits alone' is needed. |
| 22 held-out rows leaked into the old 4B prepared-v2 | stale | low | Alice | Oct 6 5:37 AM | 20.3 | Affects BASE-mix runs only; T2.1 builder screens them. |
| Unexplained Space restarts **[today]** | open | low | Alice | Oct 6 3:29 PM | 10.4 | Four restarts in four days (two unexplained on Oct 6). Optional HF support note; keep the restart watcher and persist daemon. |
| Old HF token revoked; Jupyter token unchanged **[today]** | accepted | low | Alice | Oct 6 5:57 PM | 8.0 | HF token rotated and the old one is invalid. Jupyter token left as is by choice. |
| Omission thread (both-sides application, regressions, manual review) **[today]** | deferred | low | Alice | Oct 6 6:05 PM | 7.8 | Deprioritized by you. Resume note in memory omission-eval-state.md. |
| Selected checkpoints pruned on 7 runs **[today]** | open | low | Claude | Oct 6 9:46 PM | 4.2 | SPG/SPH and moe-A-s1 lost their selected epochs (e2 kept). Make pruning keep stage2-selection.json epochs. |
| autoMode allow rule for R2/Worker/token work **[today]** | waiting on you | low | Alice | Oct 7 12:56 AM | 1.0 | Add the rule text to ~/.claude/settings.json yourself; Claude cannot edit its own permissions. |

## Findings

- **Wide-pool sampler works** (Oct 7 1:26 AM) **[today]**. 300-paper test: every paper kept, ~18 hosts and 2.7 sections per paper; 200 sampled sections came from 200 different papers. Cleaned text has one paragraph per line, so the pool builder has its own splitter. Eligible sections lean toward introductions and related work.
- **No data bottleneck after the corpus job** (Oct 7 1:15 AM) **[today]**. R2 has 18,493 PDFs from 2022 or earlier (~280k windows). Remaining generation for 3 full epochs: AI sections 2,202, paragraph edits 1,202, GPT small edits 1,403 (Codex), Claude small edits 566.
- **Generation landed on held-out papers** (Oct 6 11:36 PM) **[today]**. Host selection ignored the T2.1 never-train list, so ~17% of tonight's Claude small edits are unusable for training (usable as extra eval rows).
- **Extra seeds never changed a decision** (Oct 6 11:02 PM) **[today]**. Effects were 10–40 pts against 1–3 pts of seed noise. Only the T2.1-vs-T2 equivalence check needed more than one seed.
- **Soft labels hurt** (Oct 6 11:00 PM) **[today]**. T2.1 with clause soft labels: clean v2 0.72 vs 0.84, humanizer 0.61 vs 0.83, test small −4.9 pts. Part of the loss is the labeling rule (copied clauses called human, eval labels the whole span AI).
- **Live Atlas threshold is conservative** (Oct 6 9:46 PM) **[today]**. On 2,188 new pre-2023 papers 0.0283 gives 0.71% human FPR. Pooled 1% threshold 0.0248 (held-out 0.93%, CI ±0.07 pts with paper resampling). Human flag rate rises 0.64% (ICLR 2019) → 0.96% (ICLR 2022).
- **Realistic co-writing is nearly invisible** (Oct 6 9:45 PM) **[today]**. Fable 3-round co-writing: recall 0.14–0.17 (T2 family), AUROC 0.78. Humanized sections stay detectable (T2 0.90). Light polish has only 2 fully AI sentences, so sentence metrics cannot score it.
- **SPG's lead was home advantage** (Oct 6 9:35 PM) **[today]**. On clean v2 (no Luna, SPG overlaps removed): T2.1 0.79 vs SPG 0.38 at similar human FPR, ahead on every writer and source. Hardest: Fable 5.1 (0.69), arXiv hosts (0.67).
- **T2.1 matches T2** (Oct 6 8:18 PM) **[today]**. Test small edits −2.7 pts [−8.2, +2.4]; held-out AUROC 0.953 vs 0.954. Seed 3 differs only in calibration (threshold 0.667).
- **MoE T2 leads off-domain** (Oct 6 7:18 PM) **[today]**. moe-T2-p20: held-out writers 0.63, cross-model 0.91, Luna small edits 0.47 (4B T2 0.60 / 0.83 / 0.48). Old-mix MoE: held-out 0.25.
- **Fable workers fall back to Opus** (Oct 6 7:06 PM) **[today]**. After a safety-classifier stop, Fable subagents continued on claude-opus-5; three batches were quarantined.
- **GRADTEX inherits MAGE's label problems** (Oct 6 6:20 PM) **[today]**. MAGE: 61% of continuations labeled whole-document machine, labels inverted. GRADTEX (20% of the old mix) derives from it. RAID usable only as a filtered chat-model slice.
- **T2 generalizes far better than SPG** (Oct 6 5:20 PM) **[today]**. 4B, 3 seeds: held-out writers 0.60 vs 0.47; cross-model 0.83 vs 0.41 (Opus 3 0.88 vs 0.06); standalone 0.78 vs 0.26; public AUROC 0.979 vs 0.911. Luna small edits 0.48 vs 0.63.
- **GPT writers paraphrase more than Claude** (Oct 6 3:56 PM) **[today]**. GPT-6.1/6 Sol rewrites are 44–52% similar to the original vs 60–77% for Claude; 1,000/1,000 small edits accepted.
- **Loose validation accepted wrong-sentence edits** (Oct 6 12:46 PM) **[today]**. Strict named-sentence check rejected 93 Claude training edits and 27 held-out edits (4,919 → 4,905; 885 → 871). Haiku passes only 59%.
- **Full-length MoE overfits a small repeated pool** (Oct 6 12:19 PM) **[today]**. 1e-4 run: val loss 0.773 → 0.921 → 0.818, test small-edit 0.233 → 0.207 over epochs. 4B and 9B did not degrade. Paired windows came from 1,133 papers.
- **Clause labels beat sentence labels on synthetic truth** (Oct 5 11:37 PM). spaCy clauses 0.895 vs sentences 0.785 macro-F1; 25% of AI-labeled text in paragraph rewrites is near-verbatim human.
- **Short-edit data is the main lever** (Oct 5 10:52 PM). Splices lifted small-edit recall 0.25 → 0.58 (SPH) / 0.63 (SPG). Costs: standalone rewrites 0.51 → 0.26 (SPG), public AUROC 0.975 → 0.911 without GRADTEX.
- **Seed noise is small with a stable recipe** (Oct 5 9:32 PM). Arm A seeds agree within ~1 pt; the 8–12 pt swings came from diverging arms. Eval-set size was the larger noise source.
- **Extraction polishing is low leverage** (Oct 5 9:02 PM). About 25% of training tokens came from PDF extraction. Cleaning human text changed FPR by +0.000; 10× injected artifacts cut AI small-edit recall 4–5 pts.
- **High learning rates diverge near step 950** (Oct 5 7:55 PM). All 5 diverged runs (mostly 5e-4) blew up near step 950 on seed-1 data order. Divergence guard catches all 5 within 20–60 steps, 0 false alarms on 30 healthy runs.
- **ICLR flag rates rise by year** (Oct 5 6:56 PM). Median sentences flagged: 2022 or earlier 0.5%, 2023–24 0.6–0.9%, ICLR 2025 1.5–1.8%, ICLR 2026 14.5%, ICLR 2027 68%. Not validated.
- **Thresholds set on ordinary human text do not transfer** (Oct 5 11:13 AM). Cutoffs flag 0.2–1.6% of ordinary human papers but 4.5–7% of human sentences next to edits and 16–29% of human paper_v3 sentences.
- **Recipe changes cannot fix small edits** (Oct 5 7:26 AM). Small-edit recall at 1% FPR stayed 0.21–0.27 across every stable recipe, 9B, sentence-loss weight, oversampling and full length.
- **Constant LR without warmup caused the collapses** (Oct 5 6:22 AM). 5e-4 constant (LoRA Without Regret) collapsed near step 1,000; 5e-4 with cosine and warmup matched baseline. Arm A (2e-4 cosine) became the recipe.
- **Training almost never showed small edits** (Oct 4 10:05 PM). 2 of 24,000 stage-2 windows had an AI span under 200 characters. Later count: small edits were 0.02% of an epoch.
- **Decoders beat encoders on public text** (Oct 4 8:34 PM). Pooled public AUROC: Qwen3.5 4B/9B and MoE 0.90–0.92 vs Ettin/ModernBERT 0.73–0.74. Paper AUROC leaders tied (MoE 0.970, Ettin 0.964, 9B 0.962).

## Experiments

| When | Experiment | Setup | Result | Status |
|---|---|---|---|---|
| Oct 7 1:53 AM | Generation orchestrator (Luna, DeepSeek, GLM, MiMo) **[today]** | Space; waits for the corpus job | Queued. | planned |
| Oct 7 1:26 AM | Wide-pool sampler test **[today]** | 300 pre-2023 papers on the Space | All papers kept; 200 sections from 200 papers. | done |
| Oct 7 1:16 AM | Corpus extraction (pre-2023 + all R2) **[today]** | Space CPU, export Worker bundles | Running. | running |
| Oct 7 12:29 AM | Claude generation resume (Opus, Sonnet, Haiku) **[today]** | ~10 workflows, ~13M subagent tokens | Claude small edits 12,754; paragraph edits 5,754; sections 2,733. | done |
| Oct 6 11:00 PM | Soft-label A/B (T2.1S) **[today]** | 3 seeds, GPU relabel | Worse: clean v2 0.72 vs 0.84. | done |
| Oct 6 9:46 PM | Human calibration (2,403 papers) **[today]** | Atlas 4B, T2-s1, SPG-s1 on H200 | Atlas threshold 0.0248; document threshold on 25 papers. | done |
| Oct 6 9:15 PM | Held-out v2 + Fable set scoring **[today]** | T2.1 (clean file), T2, SPG | T2.1 0.83 all writers; co-writing 0.14; humanizer 0.86. | done |
| Oct 6 8:18 PM | T2.1 mix (4B) **[today]** | T2 minus held-out split; 3 seeds | Matches T2; clean v2 0.83; seed 3 calibration outlier. | done |
| Oct 6 7:39 PM | Fable 5.1 writer sets **[today]** | Manual Fable subagents | 1,251 small edits, 210 held-out, 299 polish, 200 humanizer, 100 co-writing, 115 sections. All evaluation-only. | done |
| Oct 6 7:18 PM | MoE T2 (p20 and pass1) **[today]** | H200, 1e-4, guard with rollback | p20 held-out 0.63, cross-model 0.91; pass1 0.57 / 0.87. | done |
| Oct 6 6:37 PM | 72k scale-up + Fable jobs **[today]** | 314 batches | Stopped at 11–15% after 7% of the weekly budget. | stopped |
| Oct 6 6:31 PM | RAID chat slice + public dataset audit **[today]** | RAID, MAGE, SeqXGPT | 6,000 filtered RAID windows; MAGE and SeqXGPT rejected. | done |
| Oct 6 5:20 PM | T2 mix (4B) **[today]** | 20%, arm A, 3 seeds | Held-out 0.60, cross-model 0.83, rewrites 0.78; Luna small 0.48. | done |
| Oct 6 3:56 PM | GPT-Sol edits via Codex **[today]** | GPT-6.1 Sol + GPT-6 Sol | 1,000 small edits, 418 paragraphs accepted. | done |
| Oct 6 1:17 PM | Claude gap-fill **[today]** | 156 batches, ~11.7M tokens | 248 small edits, 1,228 paragraph edits, ~914 sections. | done |
| Oct 6 7:53 AM | Overnight Claude generation **[today]** | Sentence edits, sections, full papers; ~120M subagent tokens | 4,905 edits, 1,164 sections, 254 papers; usage limit hit 4:50 AM. | done |
| Oct 6 12:23 AM | Wave 2 (LLE, MIX, Arep) | 4B, 20%, 1 seed | Trained, then lost in the 12:19 AM restart. | lost |
| Oct 5 11:54 PM | Held-out writer set v1 | 1,185 edits, Claude subagents + Luna | SPG: Haiku 63%, Luna 53%, Sonnet/Opus 37–41%. | done |
| Oct 5 11:37 PM | Clause-labeling synthetic test | spaCy vs Luna vs sentence splits, 114 test pairs | Clauses +0.10 macro-F1 over sentences. | done |
| Oct 5 10:52 PM | Splice wave 1 (SPH, SPG) | 4B, 20%, 3 seeds | Small-edit 0.25 → 0.58 / 0.63; costs on rewrites and public AUROC. | done |
| Oct 5 10:31 PM | moe-A-full-lr1e4 | 2 H200s, guard on | Stable but overfit; best epoch 0; small 0.23 → 0.21. | done |
| Oct 5 9:45 PM | Artifact-shortcut test | 4B + 9B on cleaned/dirtied eval variants | No FPR change from cleaning human text; 10× artifacts cost 4–5 pts. | done |
| Oct 5 9:29 PM | Omission hill-climb (v4 → v6 → math policy → references) | Sonnet judges, 176 pages | Noise left 175 → 22 per 1k words; clean pages 16% → 45%. | done |
| Oct 5 7:44 PM | moe-A-full (MoE full length, 2e-4) | 2 H200s | Diverged near step 950; ~1.5 h unnoticed. | failed |
| Oct 5 6:56 PM | Year diagnostic | 7,102 papers 2023–2026 + OpenReview downloads | Flat through 2024, rising from 2025; ICLR 2027 68%. | done |
| Oct 5 4:32 PM | ICLR 2027 scoring (fast10 4B) | 42,419 papers; 8-GPU queue | Done; published as 'Experimental'. | done |
| Oct 5 2:37 PM | moe-A-s1 (MoE 20%) | 1 H200, 62 min | All-edit 0.84, small 0.30. | done |
| Oct 5 1:21 PM | MinerU2.5-Pro | A100, vLLM | First run lost ~10 h to OOM; too slow; kept 300 papers × 9 pages. | stopped |
| Oct 5 12:47 PM | Full-length learning curves (4B, 9B) | Arm A and J, H200 | Longer training helps paper rewrites (0.89), not small edits (0.21–0.26). | done |
| Oct 5 8:28 AM | Overnight LoRA sweep (arms A–L, 4B/9B) | 20% length, up to 3 seeds, A100 + H200 | Arm A 0.770 all-edit / 0.249 small. Constant 5e-4 collapses; no recipe beats ~0.25 on small edits. | done |
| Oct 5 12:03 AM | positioned-clean-v2 + ICLR backfill | 41,004 + 1,414 papers | ~28 min; 59.5M line-number words removed. | done |
| Oct 3 7:46 PM | Six-backbone comparison (fast10) | ModernBERT, Ettin-1B, Qwen3.5-4B/9B, Gemma-4-12B, Qwen3.6-35B-A3B at 10% length | Decoders +0.16 public AUROC; paper AUROC tied; MoE best on small edits (32/46% at 1% FPR). | done |

## Decisions

### Flags raised and not answered

| Severity | Flag | Why it matters | Raised |
|---|---|---|---|
| high | Per-paper AI scores on named papers, PDF licences, unverified passage reuse | Per-paper ICLR 2027 scores are public. | Oct 3 6:29 PM |
| high | Project name close to Pangram Labs; workbench imitates their dashboard and brand | Matters before the announcement. | Oct 3 6:41 PM |
| high | Read top-flagged ICLR 2027 sentences and score ICLR 2026 submission versions | 68% is still unvalidated. | Oct 5 6:56 PM |
| medium | Real-world FPR on the year-diagnostic pre-2022 classifications | Partly answered by the 2,403-paper calibration. | Oct 5 10:05 PM |
| medium | Par3 human translations and human-only sets as false-positive checks | Non-native writing was demoted after discussion; translations untested. | Oct 5 10:18 PM |
| medium | HF support note about unexplained Space restarts | Two unexplained restarts on Oct 6. | Oct 6 3:29 PM |
| medium | Thresholds may understate FPR on recent human writing | Human flag rate rises 0.64% → 0.96% from 2019 to 2022. | Oct 6 9:46 PM |
| medium | Omission markers on human-only text are a giveaway; train on clean v2 | Unanswered; corpus job writes all three stages. | Oct 7 1:15 AM |
| low | PyMuPDF AGPL licence | Raised twice. | Oct 5 5:03 PM |
| low | 4 Fable held-out insert items land inside citations | Kept; all 210 passed ingest. | Oct 6 7:39 PM |
| low | Break and bedtime nudges (8 PM, 9:46 PM, 10:52 PM, 11 PM, 12:35 AM, 1:15 AM) | Respected; reminders continue. | Oct 6 9:46 PM |

### Where you went a different way

| When | Claude's position | What you decided |
|---|---|---|
| Oct 5 12:23 AM | Re-add the 9B curve on the baseline recipe | "i dont trust u, i trust john schulman": stayed on LoRA Without Regret (its constant schedule later collapsed) |
| Oct 5 1:48 PM | Pause the Space to save cost | GPUs are a community grant; only storage costs money |
| Oct 5 6:56 PM | Validate ICLR 2027's 68% before publishing | Published publicly as 'Experimental' to iterate on the UX |
| Oct 5 9:41 PM | Hand-label a 300-sentence gold set | Abandoned after 14 labels; synthetic known-truth test instead |
| Oct 5 10:52 PM | ~$20 API budget for multi-writer evaluation edits | Claude subagents on your account |
| Oct 5 11:15 PM | Pushing to rain-1/pangram-at-home publishes before the announcement | Pushed your branch woog97/workbench (and again on Oct 7) |
| Oct 5 11:56 PM | Not yet time for a mega generation run | Pre-approved Luna and Claude generation at scale overnight |
| Oct 6 12:29 PM | GPT data through the API under a $50 cap | Codex handoff instead |
| Oct 6 10:52 PM | Drop Haiku as a writer (~50% reject rate) | Ran Haiku too, prioritizing Opus to use the weekly quota |
| Oct 6 10:52 PM | Resume would hit the 5-hour limit | "The goal here is to hit the week limit" |

### Recommendations and decisions you accepted

- Oct 3 6:31 PM: Repo public-ready from the first commit, private until the announcement
- Oct 4 10:10 PM: Papers from 2022 or earlier are human-written (no pushback)
- Oct 5 4:39 PM: Flag at 1% human FPR with ROC curves as standard practice
- Oct 5 7:45 PM: Divergence guard, snapshots and auto-rollback after the MoE divergence
- Oct 5 8:18 PM: Bundle bulk transfers; run downloads on the Space
- Oct 5 9:07 PM: Data priorities: splices, artifact-shortcut test, LLM sentence edits
- Oct 6 12:00 PM: Restart watcher and /data as the default output for Space jobs **[today]**
- Oct 6 12:13 PM: Anything Alice's sessions didn't start is River's **[today]**
- Oct 6 12:32 PM: GPT data via a Codex handoff, never the OpenAI API **[today]**
- Oct 6 3:58 PM: Run T2 on the 4B (3 seeds vs SPG) with all three speed-ups **[today]**
- Oct 6 5:47 PM: Run T2 on the MoE **[today]**
- Oct 6 5:52 PM: Deprioritize the omission work **[today]**
- Oct 6 7:03 PM: Fable workers must stay on Fable **[today]**
- Oct 6 8:00 PM: Expand human calibration from R2 PDFs **[today]**
- Oct 6 8:06 PM: Run the soft-label A/B on T2.1 **[today]**
- Oct 6 8:20 PM: T2.1 held-out split: 15% of Opus/Sonnet papers, all GPT-6 Sol and Fable **[today]**
- Oct 6 11:02 PM: Default to 1 seed per arm **[today]**
- Oct 6 11:14 PM: All Fable data is a held-out writer evaluation; no more Fable generation **[today]**
- Oct 7 12:19 AM: Fix host screening; GPT-6.1 Sol via Codex; keep GPT-6 Sol unseen **[today]**
- Oct 7 12:59 AM: R2 ↔ Space parity: direct, clean v2 and omission v2 text for every R2 PDF, not through the Mac **[today]**
- Oct 7 1:20 AM: Train on clean v2; NeurIPS 2021 back in; sample sections and paragraphs uniformly from the wide pool **[today]**
- Oct 7 1:35 AM: Generate Claude sections, paragraph edits and small edits; Luna paragraph edits via OpenRouter flex instead of repeating **[today]**
- Oct 7 1:39 AM: Add DeepSeek V4.1 Flash as an open-weight training writer (Baseten first, DeepInfra fallback), 15% held out **[today]**
- Oct 7 1:50 AM: Evaluation-only open-weight writers: GLM 5.3 Flash and MiMo V2.6 Flash; skip space-bunny **[today]**

## Errors and time wastes

| Cause | Problem | Times | Last | Cost | Fix |
|---|---|---:|---|---|---|
| Analysis | Wrong claims later corrected **[today]** | 12 | Oct 7 12:43 AM | Hours of confusion or misdirected work | Corrected: H200 free space, MinerU text loss, seed noise 8–12 pts, upload latency, '4fdcbdd on main', 45-min estimate, 377 'manuscripts' vs ~5,500 windows, 'only 343 pre-2023 papers' (R2 had 18,493), '12–19% bleed' (old MoE only) |
| Tooling | Blocked actions and permission prompts **[today]** | 9 | Oct 7 12:56 AM | Blocked deletes, pushes, publishes, Worker work; push to rain-1 delayed ~3 h | Manual mode for approvals; you ran pushes yourself; autoMode rule pending |
| Environment | Missing tools after restarts **[today]** | 9 | Oct 6 8:32 PM | poppler, tesseract, pypdf, Pillow, micromamba, pyarrow, W&B key, HF token | Pinned envs on /data, keys via env at launch, dependency guards |
| Communication | Jargon and over-hedged answers **[today]** | 6 | Oct 7 12:18 AM | Several "I don't understand" rounds (hosts, windows, never-train) | Literal prose rule; explain plainly |
| Training | Loss divergence noticed late | 5 | Oct 5 7:44 PM | ~2.6 h of 2 H200s on an unusable run | Divergence guard, rolling snapshots, loss-aware health checks, auto-rollback |
| Coordination | Parallel sessions interfered **[today]** | 5 | Oct 6 9:33 PM | Duplicate GPU runs, a crashed trial under another session's job name, other sessions' uncommitted changes in one tree | Session messages, per-thread scope, GPU reservations |
| Storage | Space restarts wiped /tmp | 4 | Oct 6 2:42 PM | q4b-A checkpoints, wave 2 (3 trained runs + scores), tool envs; ~75 min of reruns | persist_daemon to /data every 5 min, restart watcher, outputs written to /data. No loss after Oct 6 noon. |
| Monitoring | Jobs launched without progress checks **[today]** | 4 | Oct 6 9:31 PM | Silent MoE crash 15 min; CPU relabel stalled 82 min before anyone looked | 1/2/4/…/60-min backoff checks; check progress, not liveness |
| Transfer | Per-file transfers and fast polling | 3 | Oct 5 8:16 PM | 30-min Space 429 lockout; ~20 min slow R2 uploads | 500 requests / 5 min budget; bundle everything; run on the Space |
| Usage | Generation cost estimates far too low **[today]** | 3 | Oct 6 6:38 PM | 7% of the weekly budget for 15% of the 72k scale-up; Opus fallback in Fable workers | Stopped workflows; trimmed CLAUDE.md; fresh sessions, plain workers, ~100-item batches; Fable-only rule |
| Storage | Shared H200 quota exhausted | 2 | Oct 5 4:55 PM | curve-9b-B killed at 36%; MoE crash (~45 min) | RAM disk for re-downloadable files; measure real free space (2 TB minus du) |
| Security | Credentials printed in output | 2 | Oct 5 5:50 PM | Jupyter and HF tokens exposed in transcripts | Compile-check and redact; HF token rotated, old one invalid; Jupyter token left by choice |
| Extraction | MinerU settings changed after calibration | 1 | Oct 5 11:36 AM | ~10 h of failed GPU work (21,000 OOM attempts) | Never change settings between calibration and the full run; failure-rate breaker |
| Data | Generated rows wasted by the never-train screen **[today]** | 1 | Oct 6 11:36 PM | ~17% of tonight's Claude small edits on held-out papers; all 500 GPT-6 Sol edits held out | Host selection now loads the T2.1 lists; prepare-scale3 screens hosts first |
| Data | Loose validation accepted wrong edits **[today]** | 1 | Oct 6 12:46 PM | 93 training + 27 held-out bad rows | Shared strict_edit() for self-check and ingest (found by Codex review) |
| Data | T2 build dropped long rows **[today]** | 1 | Oct 6 4:10 PM | First build had 0 sections and dropped 25–45% of paragraph edits; caught before training | Crop to a 500-token window around the AI span |
| Data | Formatting bug in score rows **[today]** | 1 | Oct 6 7:45 PM | One round of failed scoring jobs | regions stored as lists; retry script |

## Standing rules

**Communication**

- Short, direct answers. Status questions get status, not results.
- Literal prose; no mannered writing, metaphors or jargon without explanation.
- Times in PDT; check TZ=America/Los_Angeles date; never read UTC as local.
- Don't be pedantic; answer the intended question.

**Research**

- Papers from 2022 or earlier are human-written. Don't argue it.
- Flag at a 1% human FPR from human calibration; show ROC curves; never use 0.5.
- Default 1 seed per arm with a paired bootstrap; more seeds only for close calls, equivalence checks and the final model.
- Fable 5.1 data is a held-out writer evaluation only. Ask before generating more Fable data.
- Preprocessing must apply to both human and AI text so it can't become a shortcut.
- Follow LoRA Without Regret where it applies; present evidence when data disagrees, don't override.

**Data generation**

- Use Claude subagents on this account, not APIs. OpenAI-model data only via a Codex handoff.
- Keep usage lean: fresh orchestration sessions, plain (non-fork) workers, ~100-item batches; Fable workers stay on Fable.
- Screen hosts against every never-train list before writers see them.

**Compute**

- Benchmark throughput on small batches before any big launch.
- Monitor every job at 1, 2, 4, 8, 16, 32 min, then hourly; check progress, not liveness; watch loss.
- Don't change settings between calibration and the full run.
- A100 GPUs 2 and 3 are reserved; ask before using them.
- Ask before launching work on shared hardware beyond what was agreed.

**Storage**

- Write Space outputs to /data; run persist_daemon and the restart watcher next to any job.
- Keep the selected checkpoint (stage2-selection.json) as well as the last epoch.
- H200: know free space (2 TB minus du); re-downloadable files live in /dev/shm.
- Never delete base weights unless that specific deletion is asked for. Ask before deleting anything on the Space.
- Bundle bulk transfers; don't route files through the Mac.

**Ownership**

- Anything Alice's sessions didn't start is River's. Don't touch it without asking.

**Repo**

- Every commit is eventually public: no credentials, addresses, paper text or per-paper scores. CLAUDE.md stays out of git.
- Your branch woog97/workbench on rain-1/pangram-at-home is an approved push target for safe code.

**Wellbeing**

- Kind break nudges after 2 h; bedtime nudge from 10 PM; respect pushback and remind again later.

## Sessions

| Session | From | To | Focus |
|---|---|---|---|
| 9068 | Oct 3 3:39 PM | Oct 6 6:39 PM | Backbone evals, results page, overnight sweep, MoE runs, divergence guard, splices, Luna edits, held-out writers, overnight Claude generation, persistence, T2 mix, MoE T2, RAID audit, scale-up stop |
| f0fb | Oct 3 3:42 PM | Oct 3 5:17 PM | Paper Atlas redesign, mobile fixes, search speed |
| 2f2a | Oct 3 5:36 PM | Oct 3 6:42 PM | State survey, git repo, publication rules |
| d22e | Oct 4 8:44 PM | Oct 4 8:55 PM | ROC/FPR explainer |
| 67fb | Oct 4 9:17 PM | Oct 4 9:35 PM | H200 downloads, Space helpers out of /tmp |
| 25a9 | Oct 4 9:33 PM | Oct 6 6:05 PM | Extraction artifacts, cleaner v2, MinerU, omissions hill-climb; deprioritized |
| 3b26 | Oct 5 1:06 AM | Oct 6 4:02 PM | ICLR scoring, calibration, year diagnostic, Atlas publish, reader fixes, bucket cleanup, first ledger |
| a79b | Oct 5 5:10 PM | Oct 6 4:09 PM | H200 storage, MoE divergence, splices, cross-model eval, overnight generation, Codex handoff, River's sweep |
| 33a3 | Oct 5 7:01 PM | Oct 6 12:02 PM | Clause soft labels: test, relabel, Luna edits |
| 6962 | Oct 6 3:58 PM | Oct 6 6:16 PM | Ledger priorities, HF token rotation |
| 09d2 | Oct 6 6:25 PM | Oct 7 1:30 AM | Eval coverage review, T2.1 split and runs, boundary-bleed plan, run board, backups, this ledger |
| 55ac | Oct 6 6:40 PM | Oct 6 7:56 PM | MoE T2 scoring, Fable generation and Opus fallback |
| 0369 | Oct 6 7:56 PM | Oct 7 12:39 AM | Human calibration (2,403 papers), push to rain-1 |
| 506e | Oct 6 8:04 PM | Oct 6 11:28 PM | T2.1 decision, soft-label A/B, multi-seed retrospective |
| adff | Oct 6 10:51 PM | Oct 7 1:16 AM | Claude generation resume, 3-epoch supply, R2 pre-2023 discovery, corpus extraction |

## Glossary

Exact row counts for each mix are on the [run board](https://claude.ai/artifact/Cnh2kuWCVC7UDbVFVsp3pj), Runs & mixes tab. Dates (PDT) say when each term was coined or the thing was made; 'before Oct 3' predates these sessions.

Status: 🟢 core (main things we rely on now) · 🔵 in use (secondary) · 🟡 iffy (in use but questionable or under review) · ⚪ retired (no longer used).

**Data mixes**

- 🟢 **Mix** (Oct 3 (BASE)): The recipe of training rows the model sees per epoch: which kinds of text, in what proportions. Labels below name the mix a run was trained on. Exact row counts per mix are on the run board (Runs & mixes tab).
- 🔵 **BASE (prepared-v2)** (Oct 3 (backbone launch)): The original mix from the Oct 3 backbone launch. Per epoch: 25% generic human text, 25% Luna mirrors (fully AI), 25% paired paper passages (half human originals, half Luna paragraph rewrites placed in human context), 20% GRADTEX, 5% AI manuscripts. Almost no short AI edits. *Status: Only as stage 1 and as components inside T2/T2.1; River's sweep and the Atlas model use it whole.*
- ⚪ **SPH** (Oct 5, 9:19 PM): Splices replace half (expansion inferred from the run notes): BASE with half of the paired-paper rows swapped for splices. Oct 5. *Status: Weaker than SPG.*
- 🔵 **SPG** (Oct 5, 9:19 PM): Splices replace GRADTEX (expansion inferred): BASE with every GRADTEX row swapped for splices. Oct 5. Best on Luna edits, weak on other writers. *Status: Kept as the baseline to beat.*
- ⚪ **LLE** (Oct 5, 10:59 PM): LLM edits: BASE with GRADTEX swapped for Luna sentence edits. Lost in a Space restart before it was scored. *Status: Lost; superseded by T2.*
- ⚪ **MIX** (Oct 5, 10:59 PM): BASE with half the Luna mirrors swapped for splices plus Luna sentence edits. Also lost. *Status: Lost; superseded by T2.*
- ⚪ **Arep** (Oct 5, 10:59 PM): A rerun of arm A on BASE, because the original checkpoints were lost. Also lost. *Status: Lost.*
- 🔵 **T2** (Oct 6, 12:28 PM (designed); built 4:20 PM): The rebalanced mix built Oct 6: about 35% small edits (Claude 50%, Luna 40%, GPT 10%), 25% human paper text, plus paragraph edits, AI sections, fewer mirrors and less GRADTEX. No recorded meaning for the name 'T2'. *Status: Superseded by T2.1 for new runs; the MoE T2 runs still inform the backbone choice.*
- 🟢 **T2.1** (Oct 6, 6:50 PM): T2 with a held-out split removed: 15% of the papers carrying Opus/Sonnet edits, and every paper with GPT-6 Sol or Fable text. Lets us evaluate cleanly on writers the model never saw. *Status: Current candidate mix.*
- ⚪ **T2.1S** (Oct 6, 8:06 PM (approved); results 11:00 PM): T2.1 trained with soft labels instead of hard labels. Tested Oct 6; worse. *Status: Soft labels lost the A/B.*
- 🔵 **Arms A–L** (Oct 4, 10:45 PM (A–G); H–L Oct 5 early morning): Hyperparameter variants from the Oct 4–5 sweep, all on BASE. Arm A = learning rate 2e-4, cosine schedule, 6% warmup: the recipe used since. Arm B = 5e-4 constant ("LoRA Without Regret"). *Status: Arm A is the recipe; the other arms are finished experiments.*

**Kinds of training text**

- 🟢 **Small edit** (Oct 4, 10:05 PM (gap found)): One or two AI sentences inside an otherwise human paragraph: a rewritten sentence, two rewritten sentences, an inserted sentence, or a split sentence. The hardest and most important case.
- 🔵 **Splice** (Oct 5, 4:34 PM): A small edit made without calling any model: take a human paragraph and an existing Luna rewrite of the same paragraph, and swap in one or two aligned rewritten sentences. *Status: Part of the Luna small-edit pool; Luna-only.*
- 🟢 **LLM sentence edit** (Oct 5, 9:28 PM (Luna pilot)): A small edit made by asking a model (Luna, Claude or GPT) to rewrite, insert or split one named sentence in a human paragraph. Checked so that only the named sentence changed. *Status: Most new small edits.*
- 🟢 **Paragraph edit** (before Oct 3 (Luna); Claude/GPT Oct 6, 12:32 PM): A whole paragraph drafted or rewritten by AI, with the human paragraphs around it left as they were.
- 🟢 **AI section** (Oct 6, 1:17 AM): A whole section of a human paper (for example the introduction) drafted or rewritten by Claude, inside the rest of the human paper. *Status: Needed; but 76% of section windows are cropped to pure AI, fix pending.*
- 🔵 **Luna mirrors** (before Oct 3): Whole passages rewritten end to end by GPT-6 Luna from human sources. Fully AI, every token labeled AI. *Status: Cut from 25% to ~6% of the mix.*
- 🔵 **AI manuscripts / full papers** (before Oct 3 (GPT/Luna); Claude Oct 6, 7:53 AM): Complete papers written by a model from a real paper's title and abstract (the title and abstract are excluded from training).
- 🔵 **Paired paper passages** (before Oct 3): Human paragraphs from real conference papers together with Luna rewrites of them, placed back into the human context.
- 🔵 **Generic human text** (before Oct 3): Human writing from 34 sources (Wikipedia, Project Gutenberg, PMC, arXiv, reviews, essays and others). Every token labeled human. *Status: Cut from 25% to ~7% of the mix.*
- 🟡 **GRADTEX** (before Oct 3; MAGE label problems found Oct 6, 6:20 PM): Public documents derived from the MAGE dataset that carry only a whole-document human/AI label, no per-token labels. Used to train the document-level output. MAGE's labels have known problems. *Status: Derived from MAGE, whose labels are partly wrong; still 5% of T2/T2.1.*
- 🟡 **RAID chat slice** (Oct 6, 6:31 PM): 6,000 windows from the public RAID dataset, restricted to chat models and filtered so they end at sentence boundaries. *Status: Built, not yet in any training mix.*
- 🟢 **Wide pool** (Oct 7, 1:26 AM): Every pre-2023 paper in R2 split into candidate host paragraphs and sections (~280,000 and ~43,000 across ~15,800 papers), leakage-screened per passage and sampled one item per paper before any paper gets a second. *Status: Queued; built after the corpus job.*
- 🟢 **Host** (Oct 6, 12:38 AM): The human paragraph (with its neighbours) that a small edit or paragraph edit is written into.
- 🟢 **Window** (before Oct 3): A chunk of up to 510 tokens: the unit the model reads. One training row is one window. Long documents are scored as overlapping windows.
- 🟢 **2022-or-earlier rule** (Oct 4, 10:10 PM): Papers published in 2022 or earlier are treated as human-written. They are the human side of training and calibration.

**Labels**

- 🟢 **Hard labels** (before Oct 3 (named Oct 5, 7:01 PM)): Every character inside an AI-edited span is labeled AI, even words the model copied unchanged from the human original. *Status: Current labeling.*
- ⚪ **Soft labels** (Oct 5, 7:01 PM): A finer scheme from the Pangram 4 technical report (section 3.5). The edited span is split into clauses; clauses copied almost verbatim from the human original are labeled human, reworded clauses AI-assisted, new clauses AI-generated. We trained with copied = human, everything else = AI. It made the detector worse (T2.1S). *Status: Made the detector worse.*
- 🟢 **Never-train list** (Oct 5, 11:08 PM): Files listing held-out papers and fingerprints of held-out paragraphs. Every training builder must drop rows that match, so evaluation text never leaks into training.

**Writers**

- 🔵 **Luna, Sol** (Luna before Oct 3; Sol Oct 6, 12:32 PM): OpenAI models: GPT-6 Luna (wrote most early AI text), GPT-6 Sol and GPT-6.1 Sol (GPT edits, generated through Codex). *Status: Luna is still most of the small-edit pool; GPT-6.1 Sol edits being generated; GPT-6 Sol kept unseen.*
- 🟢 **Opus, Sonnet, Haiku, Fable** (Oct 5, 11:18 PM (Claude writers); Fable Oct 6, 6:18 PM): Anthropic models: Claude Opus 5.5, Sonnet 5.5, Haiku 4.5 and Fable 5.1. Fable's data is used only for evaluation. *Status: Haiku is iffy as a writer (~50% of its edits rejected); Fable is evaluation-only.*
- 🟢 **DeepSeek, GLM, MiMo** (Oct 7, 1:39 AM (DeepSeek); GLM and MiMo 1:50 AM): Open-weight models added Oct 7. DeepSeek V4.1 Flash writes training edits (15% of its papers held out). GLM 5.3 Flash and MiMo V2.6 Flash write evaluation-only edits, so we can test open-weight writers the model never saw. *Status: Queued; generation starts after the corpus job.*
- 🟢 **Unseen writer** (Oct 6, 6:50 PM): A model none of whose text is in the training mix. For T2.1: Fable 5.1 and GPT-6 Sol; next, also GLM 5.3 Flash and MiMo V2.6 Flash. Tests whether detection carries over to new writers.

**Evaluation sets**

- 🟡 **Sweep eval (in-domain small edits)** (Oct 4, 10:58 PM): The original 3,511-row test set: Luna edits inside ML conference papers, split into dev and test halves by paper. Its test half has only 150 small-edit sentences, so numbers move by about ±7 points. *Status: Luna-only and small (±7 pts); favours models trained on Luna.*
- 🔵 **Held-out writers v1** (Oct 5, 11:26 PM): 1,185 small edits by Claude (Opus, Sonnet, Haiku) and Luna on papers never used in training, plus the unedited originals. *Status: Superseded as the main test by held-out v2.*
- 🟢 **Held-out v2 (clean)** (Oct 6, 7:42 PM): 4,000 AI rows and 4,000 human rows from 6 writers, including two unseen writers, on ML, PMC, peS2o and arXiv papers. The 'clean' file drops papers T2.1 might have seen. *Status: Main decision set.*
- 🟢 **Cross-model set** (Oct 5, 10:18 PM): The heterogeneous-ai-spans dataset: long AI sections by six writers inside human documents, mostly outside ML papers. Usually scored on a fixed 500-document sample.
- 🟢 **Fable polish / humanizer / co-writing** (Oct 6, 7:39 PM): Fable-written tests of realistic use: light polishing of a human paragraph, an AI section rewritten to sound human, and three rounds of a human and AI writing together. *Status: Polish needs a token-level metric; sentence scores can't measure it.*
- 🔵 **Standalone rewrites, public docs** (Oct 3 (frozen evaluation suite)): Standalone: AI rewrites shown without their human context. Public docs: documents from public detection benchmarks, scored by AUROC. *Status: Secondary checks.*
- 🟢 **Human calibration (humancal)** (Oct 5, 4:50 PM (215 papers); 2,403 papers Oct 6, 9:46 PM): 2,403 human papers from 2019–2022 (ICLR, NeurIPS, CoRL) scored to set and check thresholds, with intervals that resample whole papers.

**Metrics**

- 🟢 **Recall at 1% FPR** (Oct 4, 8:34 PM; standard practice Oct 5, 4:39 PM): The headline number. Pick the score threshold at which 1% of human sentences are flagged, then report the share of AI sentences above it. *Status: Headline metric.*
- 🟡 **Paper threshold** (Oct 5, 11:54 PM): The threshold set at 1% FPR on human sentences from the sweep-eval dev half. All the recall numbers here use it unless stated. *Status: Set on only 7,321 dev sentences; moved from 0.41 to 0.67 between seeds. To be replaced by per-model thresholds from human calibration.*
- 🟢 **AUROC** (Oct 3): Probability that a random AI sentence scores higher than a random human sentence. Needs no threshold, so it is not affected by calibration.
- 🟢 **Human FPR** (Oct 3): Share of human sentences flagged as AI at a given threshold.
- 🔵 **Boundary bleed** (Oct 5, 11:05 PM): Human sentences in documents that also contain AI text get flagged more often than sentences in fully human documents. *Status: Measured; the distance-based metric isn't built yet.*
- 🔵 **Seeds, paired bootstrap** (Oct 4, 8:34 PM): A seed is one training run with a given random start. Paired bootstrap compares two models on the same evaluation rows to get a confidence interval on their difference. *Status: Default is now 1 seed per arm.*

**Models and training**

- 🟢 **4B, MoE** (Oct 3): 4B = Qwen3.5-4B. MoE = Qwen3.6-35B-A3B, a mixture-of-experts model with ~3B active parameters. Both are fine-tuned with LoRA adapters. *Status: Backbone choice pending.*
- 🔵 **20% length, full length, p20, pass1** (Oct 4, 10:45 PM; p20/pass1 Oct 6, 5:49 PM): 20% length = training on the first 20% of each epoch (525 steps), our cheap screening setting. Full length = all of it. p20 / pass1 = the two MoE T2 runs: 20% length × 3 epochs, and one full-length pass. *Status: 20% is the screening length; next run is full length.*
- 🟢 **Stage 1, stage 2** (before Oct 3): Training runs in two stages: stage 1 trains only a coarse 'is this window AI?' output; stage 2 trains the per-token and per-sentence outputs.
- 🟡 **fast10 / atlas-fast10** (Oct 3): The Oct 3 Qwen3.5-4B trained on 10% of BASE. It produces the scores currently on the Paper Atlas. *Status: Still scores the Atlas; weak and due to be replaced.*
- 🔵 **Divergence guard** (Oct 5, 7:49 PM): A check in the trainer that stops (or rolls back) a run when the loss rises sharply above its running minimum.

**Text and infrastructure**

- 🟢 **Baseline extraction, clean v2, omission v2** (clean v2 Oct 4, 11:03 PM; 'baseline' Oct 5, 2:34 PM; omission v2 Oct 5, 9:29 PM): Baseline = text taken from the PDF's text layer. Clean v2 = after our cleanup (line numbers, headers, broken characters). Omission v2 = clean v2 with figures, garbled math and references cut and marked. *Status: Clean v2 is what we train on; omission v2 is iffy (markers on human text only would leak the label).*
- 🟢 **Paper Atlas** (before Oct 3): The public website that shows papers with their classifications overlaid on the PDF.
- 🟢 **Space, H200, R2, bucket** (before Oct 3): Space = Hugging Face Space with 8 A100 GPUs, where most runs happen. H200 = a shared 4-GPU server. R2 = Cloudflare storage holding the PDFs and the Atlas. Bucket = the Space's persistent storage (/data).
- 🔵 **Codex** (Oct 6, 12:32 PM (GPT data handoff)): OpenAI's coding agent, used to generate the GPT-written edits (we don't call the OpenAI API directly). *Status: For GPT-written data only.*

## Links

- [Paper Atlas (live)](https://open-text-detector.woog09.workers.dev)
- [This ledger (page)](https://claude.ai/artifact/BWtCZZrM6FHWQZpwB1qzV4)
- [Run board (runs, mixes, decision tree)](https://claude.ai/artifact/Cnh2kuWCVC7UDbVFVsp3pj)
- [Model comparison](https://claude.ai/artifact/1Wey5VmYi8yN2gYgZJ7Zjy)
- [Training map and eval coverage](https://claude.ai/artifact/4M3V9un6gzqXCrbaSbzbzU)
- [Boundary-bleed plan](https://claude.ai/artifact/Lhibge5QEN8JfnoZLZApBN)
- [Backbone and sweep results](https://claude.ai/artifact/QJJUjcHwHNmXuqFNKX28QQ)
- [Previous ledger (Oct 6, 6 PM)](https://claude.ai/artifact/VDHHWuQhGUPNcxFvwJRSRn)
- [Shared branch](https://github.com/rain-1/pangram-at-home/tree/woog97/workbench)
