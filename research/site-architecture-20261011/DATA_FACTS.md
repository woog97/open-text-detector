# Data page: verified facts (Oct 11 2026)

Sources for `app/components/site-data.tsx`. Counts of the shipped 72,000-row mix were taken by encoding every row of `final-20261007/mix/prepared-v2/stage2-epoch0.jsonl.gz` with the trainer's `data.py`.

## Training mix (72,000 windows, one pass)
| Group | Rows | Keys (rows) | AI token share |
|---|---|---|---|
| Small edits | 25,200 | edit_claude 13,334 · edit_luna 8,132 · edit_deepseek 3,356 · edit_gpt 378 | 0.10–0.13 |
| Human paper text | 18,000 | human_paper_orig 9,000 · human_section 3,000 · human_para_e 3,000 · human_host 3,000 | 0 |
| Paragraph edits | 8,700 | para_luna 4,385 · para_e 3,505 · para_deepseek 810 | 0.35–0.56 |
| Fully AI | 7,200 | mirrors 4,200 · fullpapers 1,500 · claude_fp 1,500 | 1 |
| Other human writing | 5,100 | human (34 sources) | 0 |
| AI sections | 4,200 | sections_ai | 0.92 |
| Document-only | 3,600 | gradtex | no token labels |
Plan and builder: `benchmarks/otd/training/final-20261007/mix_build.py:25-35`, `mix_README.md`. Shortfalls: edit_gpt 378/1,800, para_deepseek 810/999 (docs/pickup-notes.md:135), filled from other keys in the group (mix_build.py:369-399).

## Human text
- Rule: dated 2022 or earlier = human (build_pool.py:4,110; build_hosted.py:1).
- Clean v2 pool: ICLR/NeurIPS 2021–2022, 10,413 papers with hosts (pickup-notes.md:123). Older paired PDFs: ACL, ICML, NeurIPS 2013–2021. Hosted passages: PMC, peS2o, arXiv, ACL ≤ 2022.
- 34 general sources (TRAINING_DATA_INVENTORY.md:84-122): gutenberg, writingprompts, scp, wikisource, pmc, arxiv, pes2o, acl, wikipedia, pressbooks, libretexts, wikibooks, amazon2018, imdb, opinrank, stack_nontech, stack_tech, ubuntu_irc, wiki_talk, cccc, eff, foodista, wikivoyage, globalvoices, voa, wikinews, scidev, oanc_slate, asap2, persuade, govreport, fed, worldbank, oanc_icic; ≤ 2 windows per document.
- Format check: clean-v2 AI share 0.51 vs 0.69–0.78 older formats (pickup-notes.md:135).

## Other human writing in the mix (5,100 windows, by source; joined to pool-human.jsonl.gz)
creative 1,104 (gutenberg 536, writingprompts 388, scp 110, wikisource 70) · scientific 946 (arxiv 305, pmc 286, pes2o 191, acl 164) · reference 744 (wikipedia 496, wikibooks 105, pressbooks 83, libretexts 60) · social 684 (stack_nontech 367, stack_tech 152, ubuntu_irc 110, wiki_talk 55) · general_web 549 (cccc 321, foodista 115, wikivoyage 95, eff 18) · reviews 461 (amazon2018 338, imdb 63, opinrank 60) · news 428 (voa 119, globalvoices 103, oanc_slate 76, wikinews 68, scidev 62) · professional 184 (govreport 61, worldbank 51, fed 43, oanc_icic 29). Essays (asap2, persuade): 0. Categories follow the Pangram 4 technical report (arXiv 2607.27183).

## AI text
- Who made the AI-labelled training rows (counted in the shipped mix): generated for this project 45,300 (small edits 25,200 + paragraph edits 8,700 + fully AI 7,200 + sections 4,200); from an outside source 1,796 GRADTEX rows with document_label 1 (AI-involved), document label only, no token labels. GRADTEX also gives 1,804 human (document_label 0) rows. So 96.2% of AI-labelled rows and all character-level AI labels are ours. GRADTEX: public, derived from MAGE (TRAINING_DATA_INVENTORY.md:41-43).
- Small edits: rewrite_one / rewrite_two / insert_one / split_one at 30/30/30/10; strict alignment check; similarity ≤ 0.9 (build_llm_edits.py:131,551-639; build_hosted.py:246-268).
- Paragraph edits: draft/rewrite 50/50; checks length 0.5–1.8×, context similarity ≤ 0.6, original similarity ≤ 0.9, rewrites ≥ 0.12 change (build_hosted.py:271-305).
- Sections: Claude draft/rewrite; roles intro 25, related work 25, discussion 20, method 15, experiments 15 (build_sections.py:1-14).
- Mirrors (GPT-6 Luna, from topics of human passages), manuscripts (GPT-6.1 Sol 196, GPT-6 Luna 204; GPT-6 Sol excluded), Claude full papers (draft → Opus review → revision; 210 used).
- Writers: Claude Opus 5.5 / Sonnet 5.5 / Haiku 4.5 (Claude Code), GPT-6 Luna (OpenRouter), GPT-6.1 Sol (Codex), DeepSeek V4.1 Flash (OpenRouter). Never trained: Fable 5.1, GPT-6 Sol, GLM 5.3 Flash, MiMo V2.6 Flash (mix_build.py:38).
- GRADTEX: public dataset derived from MAGE; document labels only (data.py:72-111).

## Leakage screens and windows
- never_train_hit: paper ids, paragraph SHA-256, 60-char shingles (build_llm_edits.py:250-292); eval/selection/calibration ids and 10+-word sentences (mix_build.py:85-104, 286-294).
- crop510: rows > 510 tokens cropped to ≤ 500 tokens containing the AI span, random offset, word boundary (mix_build.py:65-82); long documents cut into back-to-back ≤ 500-token windows (mix_build.py:128-145).

## Evaluation and calibration
See the evaluations artifact facts (held-out sets, long-context sets, Fable sets, cross-model, sweep) and FACTS.md section 2 for the calibration set.
