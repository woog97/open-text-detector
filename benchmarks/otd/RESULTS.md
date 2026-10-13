# Local benchmark implementation and validation

Prepared 4,607 examples across 12 input files. Raw-download checksum audit passed for 278 files (1,037,155,761 bytes).

Nine named public benchmark families have working importers: MELD-eval, GEDE, DetectRL, Epoch, Sem-Detect, Saha, OpAI-Bench, VUB, and Perkins. Additional inputs cover Liang TOEFL, PELIC, and local binary/length/mixing proxies.

All four initial public-dataset runs completed: **350 scored examples, zero inference failures** (not 350 unique texts). These runs validate the pipeline. Sample sizes and exploratory thresholds are insufficient for a reliable model leaderboard, and the runs do not use identical selections.

| Run | Model | Completed predictions | Failures | Report |
|---|---|---:|---:|---|
| meld-pilot | meld | 224 | 0 | [Open](runs/meld-pilot/REPORT.md) |
| meld-additional | meld | 32 | 0 | [Open](runs/meld-additional/REPORT.md) |
| editlens-smoke | editlens | 88 | 0 | [Open](runs/editlens-smoke/REPORT.md) |
| editlens-additional | editlens | 6 | 0 | [Open](runs/editlens-additional/REPORT.md) |

Validation: nine unit tests passed; 100 independent pairwise-AUROC and ROC-monotonicity checks passed; current source lint passed. The source audit verifies row IDs, labels, text checksums, character spans, and raw download receipts.

The initial two runs preceded later dataset additions and the final label-balanced pilot selector. Their exact source/data hashes and selected IDs remain in their manifests. Additional runs archive source code as well.

The initial public-dataset runs involved no website changes, detector API calls, frontier generation charges, threshold training, or adaptive red-team runs.

## Arena-20 fresh generation pilot

The separate [Arena-20 results](ARENA_20_RESULTS.md) now cover 20 reproducibly sampled prompts and an intended roster of 20 model variants across 13 named families. Nineteen models returned responses; one service was unavailable. Two account-blocked routes were replaced without changing account settings. All 440 original/replacement cells remain recorded: 365 successful responses and 351 eligible responses scored by both MELD and EditLens, with zero detector errors. Reported successful-request usage cost was $0.15302697.

This adds 702 predictions to the initial 350, and 351 prepared responses to the prior 4,607. The pinned Arena source adds 41,572,998 bytes; raw dataset payloads total 1,078,728,759 bytes (about 1.08 GB), excluding model weights and small metadata/results files. No new detector weights were downloaded.

The native user-only token-length gate remains unverified, so these are exploratory results. The report preserves exclusions, per-generator uncertainty and comparisons on shared eligible prompts; it does not claim an exact Pangram replication or reliable leaderboard. Arena sampling/metrics validation: 13 tests passed.

See [the run guide](README.md) and [complete evaluation coverage](COVERAGE.md).

## Arena-100 Pangram-report roster

The [Arena-100 report](ARENA_100_RESULTS.md) contains all 1,200 successful responses (100 frozen prompts × 12 retained models), after user-authorized failed-cell cleanup. Reported generation charges total **$1.66934653**. Thirteen short responses, five length-limited responses and two responses without a finish reason remain in the complete dataset but are excluded from mechanical scoring.

MELD and EditLens each scored all **1,180 mechanically eligible responses**, with **zero inference errors**. MELD produced 20 non-AI decisions; EditLens produced 258 non-AI decisions, counting Mixed as a miss. These are descriptive AI-only results under the saved thresholds, not a human false-positive evaluation or an exact Pangram replication. Full substantive-prose/refusal adjudication and the native output-longer-than-input token gate remain unverified.

An EditLens queue-snapshot completion bug was detected by final coverage validation. The run resumed only the missing 355 responses, preserved all 825 earlier predictions, and verified matching response hashes for all 1,180 outputs. Original and resumed code/manifests are archived. Nineteen current sampling, routing/reuse and scoring-completion tests pass.

## Earlier cheap models: 100-prompt generation expansion

All 2,000 cells succeeded after cleanup, adding 1,635 new successful responses and reusing 365 exact matching earlier responses. The [cost table and generation report](ARENA_100_CHEAP_RESULTS.md) separate $0.64111519 in new charges from $0.15302697 attached to reused responses. The combined 32-model dataset contains 3,200 unique pairs. Saved successful responses account for $2.46348869; the API key reports $2.46648383 total usage, leaving $0.00299513 unattributed. All cheap-roster requests and observed usage obeyed the below-$2/M output routing cap.

There are 1,961 mechanically eligible added-roster inputs prepared for the detectors. No added-roster detector score is claimed here. The original 12-model detector run remains complete and separately reported. Full grid, response hashes, cost coverage, routing caps, request reuse, and a secret-value scan passed; see `runs/arena100-combined/VALIDATION.json`.

## Completed skipped-model generation expansion

[Skipped-model generation results](ARENA_100_SKIPPED_RESULTS.md): all 14 previously excluded Table 3 generators now have 100 successful responses each. GPT-5.6 Luna, GPT-6 Sol and GPT-6 Astra add another 300; the existing 100 GPT-6 Luna responses are reused. All 1,700 new requests succeeded using 32 concurrent calls with low/disabled reasoning. No failed-request cleanup was needed. GPT-6 Terra is absent from the saved OpenRouter catalog and remains uncovered.

The deduplicated cumulative corpus is now 49 models × 100 prompts = **4,900 successful responses**. The extension increased API-key usage by **$20.31784505**, matching the known response-cost sum. Two Claude Fable responses lack individual usage records, so per-response cost/token coverage is incomplete for those two cells; their successful texts are preserved. Total account usage is $22.784328878, including prior runs. New local-detector inputs are prepared (1,750 mechanically eligible rows in this extension including reused Luna); detector scoring has not been run for these added models.
