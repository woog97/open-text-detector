# Evaluation coverage and feasibility

Sources: [technical report, §§4–5 and Appendix C](https://arxiv.org/html/2607.27183v1), [model card](https://www.pangram.com/research/model-card/pangram-4), and [release technical overview](https://www.pangram.com/blog/pangram-4-technical). Inventory checked 22 September 2026. “Ready” means the importer and local scoring path exist; it does not mean a full benchmark run has completed.

The report's evaluation inventory is below. Status and implementation choices are our assessment, not claims made by Pangram.

| Evaluation | Local status / dependency |
|---|---|
| Overall human FPR | Proxy: historical PG-19 passages. Exact private holdout unavailable. |
| Overall AI FNR | Released MELD-eval and MDTA responses ready. [Arena-20](ARENA_20_PROTOCOL.md) specifies a new 20-prompt paired generation pilot; sampled with the fixed seed and implemented; see [execution results](ARENA_20_RESULTS.md). Exact Pangram holdout unavailable. |
| Generator/family and release-date generalization | Generator breakdown ready; matching the original generator panel needs its outputs or fresh generation. No release-date correlation claimed from our small panel. |
| Domain performance | Dataset/domain breakdown ready. |
| Length performance | DetectRL native lengths plus explicit local truncation proxies ready. |
| Light AI polish | Liang and Saha released polish; GEDE improvement cohort separately identified. |
| Editing-prompt Mixed recall | Sem-Detect refined reviews and OpAI trajectories available as substitutes. |
| WildChat/soft-n-gram Mixed recall | Exact selected edits and clause labels unavailable; recreating requires generation and clause/semantic labeling. |
| Interleaved authorship, N=1,4,8,12,16,20 | Local labeled sentence-block proxies and OpAI spans ready. |
| Multilingual FPR | Language-aware reporting ready; no multilingual corpus prepared. Exact pre-2022 FineWeb2 selection unavailable. |
| Multilingual FNR | No multilingual AI corpus prepared; generation or released outputs needed. |
| Non-native English: ELLIPSE | Prepared 30 September 2026: all 2,571 nonempty, unique publisher-test essays, no detector-dependent filtering. Exact Pangram subset and third-party training overlap unknown. |
| Non-native English: ICNALE | Not imported; corpus access and transcript selection needed. |
| Non-native English: PELIC | Ready: historical original answers, with an explicit local >=50-word eligibility rule. Exact report subset unavailable. |
| Non-native English: Liang TOEFL | Ready, including released polished versions and native-English controls. |
| UChicago | Original text download not located; needs released texts and split mapping. |
| VUB | Ready: released fully AI-generated Word papers. |
| GEDE | Ready: released database, missing/empty human source text skipped explicitly. |
| Perkins | Ready: released original/manipulated text and human controls. |
| Epoch style imitation | Ready: human, vanilla, style-transfer passages. |
| DetectRL | Ready: published test files for Tasks 1, 3, 4; Task 2 uses the same test texts but its training/generalization protocol is not reproduced. |
| MELD-eval | Ready: original released JSONL, all generators/domains/attack types available. |
| Sem-Detect | Ready: ICLR 2022 official test reviews; other conference shards not downloaded. |
| Saha peer review | Ready: bounded easy/hard/human path-list sample. |
| OpAI-Bench | Ready: six test files, all nine version stages; coverage varies by generator/domain. |
| Commercial humanizer AI detection | Released attack datasets are substitutes. Exact 13-system private paired corpus unavailable. |
| Auxiliary humanizer classification | Unsupported by our local model interfaces; requires an appropriate head and paired labels. |
| BLADER | No released outputs imported. Reproduction needs generation/rewriting; not executed. |
| Manual red-team | Human evaluation process; no fixed dataset to replay. |
| Agent red-team | Requires an agent, detector access, generation budget, and an explicit run budget; not executed. |
| Backbone/LoRA ablations | Private architectures, training data, and checkpoints unavailable. |

The release sources differ in some counts and rounded results. For example, the [model card](https://www.pangram.com/research/model-card/pangram-4) reports an English FineWeb human cohort separately from multilingual FineWeb2, while the [blog](https://www.pangram.com/blog/pangram-4-technical) describes the combined human evaluation more broadly. We do not turn these into a single target dataset or invent matching example IDs. The report is the protocol reference; the pinned dataset revision defines what we actually score.

## Released sources and exact local choices

| Adapter | Publisher source | Selection and provenance |
|---|---|---|
| `liang` | [ChatGPT-Detector-Bias](https://github.com/Weixin-Liang/ChatGPT-Detector-Bias) | TOEFL original 91, polished 91, Hewlett original 88. Retain original document strings. The report/card's TOEFL counts differ; we expose actual imported counts. |
| `ellipse` | [Author repository](https://github.com/scrosseye/ELLIPSE-Corpus) | Official final test CSV at revision `dc3b8f0b3b4332fc9f64302c4ccfc4ed582f4b43`; 2,571 essays, including short texts; Git blob hashes verified. No existing prepared-text duplicates. CC-BY-NC-SA-4.0. Author IDs unavailable; group by essay. Prepared data omit demographic fields; proficiency scores retained. |
| `pelic` | [PELIC dataset](https://github.com/ELI-Data-Mining-Group/PELIC-dataset) | Original answers dated before 2022 and at least 50 words; sampled by answer ID, author grouping retained. Publisher LFS SHA-256 verified. This local subset differs from the report. |
| `meld_eval` | [MELD-eval](https://huggingface.co/datasets/anon-review-meld-2026/meld-eval) | `meld_eval.jsonl`, not the later `v1.1` file. Full source downloaded; default sample by label/generator/domain/attack. Preserve prompt IDs. Metadata CC-BY-4.0; underlying text retains source terms. |
| `epoch` | [Author repository](https://github.com/jaeholee-brown/ai-text-detectors) | Read `corpus/*/snippet_*.txt`, `stages/vanilla`, and `stages/style_transfer`. Exclude prompts and detector prediction files. Group by author. Source texts retain underlying rights. |
| `detectrl` | [NLP2CT/DetectRL](https://github.com/NLP2CT/DetectRL) | Test files only. Native `human`/`llm` labels retained. Transformations of human text are not silently relabeled AI. Test files overlap: pooled counts are not independent unique documents. |
| `gede` | [GEDE repository](https://github.com/lukasgehring/Assessing-LLM-Text-Detection-in-Educational-Contexts) | Read-only SQLite queries. Use publisher binary `is_human`, retaining `prompt_mode`; improved human essays remain native positives but also have an explicit polish false-alarm metric. Missing AAE originals require upstream separate acquisition. Underlying corpus terms apply. |
| `sem_detect` | [ML Conference Peer Reviews](https://huggingface.co/datasets/Sem-Detect/ML_Conferences-Peer-Reviews) | Only `split=test` in ICLR 2022 reviews; never download paper bodies. `rewrite` is Mixed and excluded from pure binary ROC. This does not reproduce the Sem-Detect classifier, whose reference-review pipeline requires additional models/APIs. |
| `saha` | [FLAIR-IISc/ai-in-peer-review](https://github.com/FLAIR-IISc/ai-in-peer-review) | Eight lowest seeded path hashes per easy/hard level, plus eight human reviews. Skip keypoint input files. Level 4 is human-base polish; levels 1–3 are AI positives. No source prompts are executed. |
| `opai` | [OpAI-Bench](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench) | Test Qwen3-8B in four domains; GPT-5.4 and Gemini-2.5-Flash abstracts. Native character spans/ratios; all intermediate fractions labeled Mixed. Apache-2.0 dataset release; original source provenance still applies. |
| `vub` | [Public OSF project](https://osf.io/4p5ab/overview?view_only=f74d55edeaf441f39bbb6d6483259ead) | 40 AI-named `.docx` stimuli currently available from Results → AI-generated paper; the report scores 39, so our full download is not assumed to be its identical subset. Read body text only. Mutable source files are frozen locally by checksum receipts. No actual student theses or detector-report documents downloaded. |
| `perkins` | [Mendeley version 3](https://data.mendeley.com/datasets/xv6fk2mmh9/3) | Word stimuli from named control/generator/attack folders, excluding instruction documents. Validate publisher SHA-256. CC-BY-4.0. |
| `local` | [Existing baseline collection](../../research/README.md) | One centered 500-word excerpt per selected PG-19 book, MDTA responses, length cutoffs, and two-block human/AI sentence concatenations. Blocks may have unmatched topics; these are implementation/proxy tests, not a natural coauthoring distribution. |

Data acquisition needs internet access, but **none of these local scoring paths needs a paid API**. Reproducing Pangram's actual detector results requires Pangram 4 access; local model scores cannot stand in for its predictions. Producing fresh proprietary frontier outputs or commercial-humanizer outputs requires the relevant service access. Downloaded published outputs avoid that generation cost.

Public-dataset detector scoring stays local. The separately authorized Arena experiments transmit selected prompts to OpenRouter for generation; generated text is scored by local detectors. Licensing/provenance pointers are recorded here and in cached publisher cards; no claim is made that every underlying source text can be redistributed under the repository's code license.

## Arena-20 execution

The [20-model roster](ARENA_20_MODELS.md), [frozen prompt sample](ARENA_20_SAMPLE.md) and [sampling protocol](ARENA_20_PROTOCOL.md) are separate from the website. `arena20.py` builds/validates the sample; `arena_generate.py` runs the frozen 400-cell OpenRouter grid. Credentials belong only in ignored `.env.secrets`. Execution artifacts are in `data/arena20/` and `runs/arena20/`; completed response audits and detector results are in [Arena-20 results](ARENA_20_RESULTS.md), explicitly separated from exact Pangram replication. The original 400 cells and 40 replacement cells are preserved.

## Arena-100 generation

The [completed generation report](ARENA_100_GENERATION_RESULTS.md) covers all 1,200 successful cells: the same 100 frozen prompts across 12 retained models from the Pangram report. Failed cells were retried under explicit user authorization; successful responses were preserved. Reported generation charges total $1.66934653. The full dataset includes 13 responses below 50 words and seven non-stop completions; those remain in the dataset but are excluded from the separate mechanical detector screen. Full substantive-prose/refusal adjudication and the native user-only token gate remain unverified.

## Earlier cheap-roster expansion

[Generation is complete](ARENA_100_CHEAP_RESULTS.md) for all 20 earlier cheap variants on the same 100 prompts: 2,000 successful responses, with 365 matching earlier responses reused. New charges were $0.64111519. Together with the 12-model report roster, the combined dataset has 3,200 unique prompt/model pairs and generation IDs. The 1,961 mechanically eligible cheap-roster responses are prepared for local detector scoring; scoring this added roster is not included in the completed 12-model detector results. Qwen free-route failures used the same canonical model through DeepInfra BF16 at a quoted $1.875/M output, protected by the unchanged below-$2/M cap.

The [skipped-model expansion](ARENA_100_SKIPPED_RESULTS.md) completes generation for all 26 Pangram Table 3 models on the same 100 prompts, plus earlier cheap models and requested available GPT releases: 49 unique models and 4,900 successful responses overall. GPT-6 Terra is unavailable in the frozen OpenRouter catalog. This completes generation coverage, not local detector scoring for the newly added models. Two Claude Fable responses have missing individual usage metadata, recorded explicitly in the report.

## Fresh paper workflow expansion — 30 September 2026

Completed [protocol and review passes](EVAL_EXPANSION_PLAN.md): 189 fresh historical papers, split 27 pilot / 54 calibration / 108 test, with paper and author-name separation from the prior 2,000-paper collection. Seven Luna Flex workflows produced 756 provenance-labeled reconstructions and 567 assistance outputs. Assistance has no invented binary token gold. The untouched body pool contains 9,301 paragraphs; the locked-test clean/novel slice contains 923.

The [generation results](../../research/data/paper-eval-workflows-luna-20260930/RESULTS.md) and [scoring manifest](../../research/data/paper-eval-workflows-luna-20260930/scoring/manifest.json) preserve original thresholds and specify BF16 scoring. Prepared test suite: 8,552 rows, including the new ELLIPSE supplement; correlated views and extraction strata are explicit. Generation is complete; detector scores on this new suite have not yet been calculated. This is Luna workflow transfer, not unseen-generator validation or an exact reproduction of Pangram's private interleaving/polishing tests.
