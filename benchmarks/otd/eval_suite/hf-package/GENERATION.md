# Paper-authorship evaluation expansion — 2026-09-30

## Decision and scope

Generate a new, evaluation-only Luna collection, using OpenRouter's OpenAI Flex route with no standard fallback. Keep existing training, evaluations and public dataset unchanged. This tests new papers and writing workflows; it cannot establish unseen-generator performance. Future non-Luna runs should reuse the same sources and specifications.

Target 189 new papers: 27 pilot, 54 calibration, 108 locked test. Sample historical NeurIPS, ICML and ACL, 2013–2021, initially seven papers per venue/year: one pilot, two calibration, four test. Exclude all 2,000 prior papers by ID, normalized title and PDF hash, and exclude their normalized author names. Isolate authors between new splits. If a stratum runs out, report the shortfall rather than silently relaxing independence. Name matching is conservative and is not author identity resolution. This selects eligible historical prose, not a representative sample of every paper or today's writing.

## Conditions on one source paragraph per paper

| Condition | Writer sees | Purpose / labels |
|---|---|---|
| Sentence reconstruction | Abstract, surrounding prose, notes for one hidden sentence | Sparse AI insertion; exact replacement span |
| Two-sentence reconstruction | Same, with a hidden adjacent pair | Short AI block; exact replacement span |
| Paragraph v3 reconstruction | Abstract, adjacent paragraphs, frozen v3 notes | Matched control for our existing protocol |
| Concise authoring from notes | Same information, shorter independent writer instructions | Prompt/workflow transfer within Luna |
| Proofreading | Original paragraph and context | Human-origin, AI-assisted diagnostic; not binary AI gold |
| Light polish | Original paragraph and context | Same diagnostic, with measured edit extent |
| Substantial rewrite | Original paragraph and context | Same diagnostic, not automatically wholly AI gold |

Use one target with at least four plausible sentences so sparse edits leave both human and AI content. Deterministic assembly preserves all non-target text exactly. Export paragraph-alone and three-paragraph-context views with shared family IDs: correlated views are not independent samples. Preserve complete requests, outlines, responses, generation IDs, model, tier, paper_id, nullable forum_id, source location, original text, offsets, and transformation metadata. Do not fabricate OpenReview IDs for proceedings sources.

Produce untouched controls and all non-target body paragraphs from every selected paper. Keep extraction-quality flags and target/context-overlap flags; report clean novel body text separately from all eligible body text. Normalization is deterministic and identical before either human control or AI generation. No model 'cleanup' of human controls. Similarity and edit distance describe changes; neither determines authorship.

## Labels and scoring contract

Reconstruction labels record provenance of the replacement operation, not proof that every copied word was newly invented. Preserve exact character intervals for human context and generated replacement. Also export matching source/candidate blocks to support an unchanged-copy sensitivity analysis. Sentence boundaries are heuristic and explicitly marked as such. Sentence targets must pass abbreviation/boundary checks in the pilot.

For proofreading/polishing/rewrite, preserve the human-origin parent, process label, exact textual diff operations, and unchanged spans. Changed characters are not automatically AI-authorship gold; there is no honest binary token label for all assisted writing. Score assistance cohorts separately: unchanged-human false alarms, flagged proportion versus edit extent, and explicit assistance detection only if a compatible classifier output exists. Do not force our binary detector into Pangram's three-way task.

Later scoring: use BF16 inference; freeze existing thresholds for the primary comparison. If recalibrating, use only the new calibration split and publish that as a separate result. Report token and sentence precision/recall/F1, human FPR, recall at a validation-calibrated 1% human FPR, span overlap/boundary error, and per-paper macro averages. Bootstrap by paper, with venue/year and condition slices. Report accuracy only as secondary; balanced-set precision is not deployment precision. Include illustrative precision at 1%, 5% and 20% AI prevalence. Test-set thresholds must never be tuned.

The new human sample cannot validate 0.01% paper-level FPR. Report numerator, denominator, uncertainty and paper clustering; zero observed errors is not zero risk. Match Pangram's metric and unit only when actually comparable; otherwise label the result a local analogue. Public ELLIPSE non-native writing is a useful separate acquisition task, not something to synthesize with Luna. Defer unrelated multilingual, dialogue and humanizer benchmarks. Keep unseen generators as an explicitly unfilled coverage gap.

## Quality assurance and freezing

A separate Luna call judges each output against the hidden original and context for fidelity, clarity, naturalness, context fit, voice/hedging, added and omitted claims. It sees no detector predictions and cannot revise the output. Evidence quotes must be literal substrings. This is same-model screening, not independent expert ground truth. Preserve every first mechanically valid generation, including poor ones. Report all valid outputs as primary, plus a predeclared faithful/non-degraded stratum as secondary; never select examples using detector results.

Pilot only: inspect exact assembly, source boundaries, hidden-original exclusion, edit labels and representative semantic judgments. Run mechanical tests before paid requests. Require at least one successfully judged example for every condition, verified Flex billing, and no source/label/prompt-leakage errors before main launch. Quality failures are measured, not a reason to silently reroll. Record any prompt revision, pilot exposure and freeze hashes before calibration/test generation. Only pilot outputs may inform revisions; main results remain unused for model or prompt selection.

## Review 1 — construct validity

Initial idea: assign every edited paragraph a binary AI label and call the new batch a generalization benchmark.

Criticism: that would teach and reward an invalid label definition; polishing leaves extensive human-authored material. Luna-only samples also cannot establish generator transfer, and a single reconstruction recipe repeats the training distribution.

Revision: separate provenance-labeled reconstruction from assistance diagnostics; include one-sentence, two-sentence and alternate-prompt conditions; retain unchanged-copy sensitivity metadata. Explicitly leave unseen-generator evaluation open. Keep quality ratings separate from authorship labels.

## Review 2 — leakage, selection and statistical validity

Initial idea: reuse all available paper paragraphs, keep only Luna-approved outputs, and report extremely low false-positive operating points.

Criticism: those papers are already exposed; model-based filtering may select style shortcuts; extraction artifacts could dominate; thousands of paragraphs from a small number of papers do not establish tiny document-level FPR. A same-model judge can miss systematic errors.

Revision: select fresh paper/author-separated sources and distinct pilot/calibration/test partitions, retain bad generations and extraction strata, compare paired context views, freeze before main generation, and use paper-clustered uncertainty. Historical provenance and automated judging remain qualified evidence. Record exclusions and cell shortfalls so convenience sampling is visible.

## Source pilot correction before paid generation

The inherited five-target requirement excluded almost all ACL candidates unnecessarily. Restarted selection with one valid target per paper (four sentence spans, including an internal sentence of at least 15 words). Archived the initial selection, retained the download cache, and preserved all author/paper independence rules. No paid generation preceded this correction.

## Pilot revision 2 — extraction and task scope

The first broad pilot exposed embedded numerical tables, broken hyphenation and flattened mathematics in a few apparently eligible targets. Added deterministic source flags for these cases; reselect a valid target from the same paper where possible, otherwise replace the paper within its original venue/year/split slot. Preserve all earlier pilot-author exposure when replacing sources. Raw flagged human paragraphs remain available as an extraction diagnostic.

Some substantial rewrites expanded into adjacent context. Editing prompts now explicitly restrict the output and its claims to original_paragraph; context is read-only. Re-run the pilot under protocol version 2 and retain version 1 outputs and charges in pilot-v1. This is a documented protocol revision, not quality-based rerolling within a frozen run. The v3 reconstruction writer stays unchanged. Historical source texts may have appeared in generator pretraining; hidden here means absent from the writer request, not proven absent from model training.

## Final boundary review — protocol version 3

Recognize sentence starts after parentheses, quotations and numbered citations while retaining abbreviation exceptions. The earlier heuristic merged some such sentences. Revalidate every selected target. Reuse version-2 pilot responses only when the complete initial request hash matches, and re-parse them against the current target. Changed requests get fresh responses. Preserve both earlier pilots and count their charges once. Formatting repairs may request shorter note fragments to enforce the unchanged ten-word copying limit; they never request a higher quality writing outcome.

## Execution and accounting

Budget ceiling: $10 in reserved worst-case request cost at the verified Flex rate; no automatic increase. Expected generation is 1,323 writer calls, three outlines per paper and 1,323 quality calls, about 3,213 successful calls plus bounded format retries. Actual tokens and charges, including failed/invalid billable attempts, are authoritative. Maximum 12 concurrent requests, 900-second timeout. Capacity errors retry with bounded backoff; ambiguous transport outcomes are quarantined instead of automatically spending again. Persist requests before dispatch and responses before parsing; resume completed requests without rerunning. Price/tier/model mismatches stop generation. Do not publish sealed evaluation data automatically.

Deliver source manifest, frozen protocol, provenance-labeled reconstruction and assistance files, untouched-human pool, quality results, failure/coverage report, full token/cost ledger, and resumable worker/status files. Generation does not by itself produce new classifier benchmark scores; those require the later common-suite BF16 scoring pass.
