# Stage 2: synthetic mirrors

This is data-generation stage 2 in the user's flowchart, not Pangram's second optimization stage.

Pangram 4 §2.2 describes extracting a topic from a human source, generating independent prose from that topic (optionally grounding with titles/keywords, or a Q&A question), then rejecting excessive verbatim copying. The generated example is wholly AI-written. In this project, the planned unit is a standalone passage matched to a selected human passage, not the whole parent book or paper. Source: https://arxiv.org/html/2607.27183v1#S2.SS2

## Implemented

- `generator.json`: user-selected main generator **`openai/gpt-6-luna` via OpenRouter**, for both topic extraction and independent writing. OpenAI Flex only, low reasoning, no fallback to another model or a more expensive route. Secondary generators are not selected.
- `run_openrouter.py`: paid-call runner using the project's existing OpenRouter credential transport. Explicit execution and a positive run budget are required. Durable request/response files preserve actual model, usage, billed cost, lineage, and failures. Uncertain dispatches halt instead of risking duplicate charges. Visible-token checks subtract reasoning tokens and fail closed when their count is unavailable. No model assets are downloaded.
- `mirror_core.py`: isolated topic and writer prompts; the writer receives only a short topic, genre, language, and target length. Human source prose, surrounding paragraphs and detailed content notes are excluded from the writer request.
- `protocol.json`: versioned local choices for length and copy checks. Numerical thresholds are proposals, not disclosed Pangram settings.
- `run_local.py`: cached local-model inference in BF16 only; deterministic per-call seeds; immutable request/response files; resumable records; retained rejection reasons; no paid API calls or model downloads.
- `test_mirrors.py`: eight tests covering parent admission gates, writer isolation, source hashing, verbatim copying, truncation, and source-family requirements.
- `stage3-audit.json`: direct counts from the existing edit-generation corpus.

Production inputs must be admitted human records with a passed protected-overlap audit and frozen family/split. A pilot can explicitly use quarantined inputs, but all resulting mirrors remain quarantined, development-exposed, and ineligible for training or locked test. Mechanical QC alone does not admit a generation. Topic/genre, factual interpretation and downstream release reviews remain separate.

## First pilot

Space directory: `/data/workspace/synthetic-mirrors-v1`.

Scheduler job: `/data/workspace/paper-diversity-v1/auto-dispatch/jobs/synthetic-mirrors-pilot-v1.json`.

Generator: cached `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, BF16, no thinking output. The same model handles topic extraction and independent writing in separate calls. This is a single-model implementation test, not the intended final generator mixture.

Ten inputs cover creative, scientific, reference, general-web and news categories. Inputs use seven source slices. The requested nontechnical Q&A pilot slots remained empty because the bounded candidate collection had no records passing the specified cooking-site selection; no replacement source was silently inserted. The selected records have 140–300 words and distinct parent documents. Basic intake exclusions remove obvious discussion boilerplate, heavy equation markup and list-like fragments. They do not certify genre or historical authorship.

Recovery locations:

- `pilot/status.json`: completed/planned counts, mechanically passing counts, rejection reasons.
- `pilot/manifest.json`: input hash, model revision, protocol hash and inference dtype.
- `pilot/calls/`: exact prompts, outputs, token usage and settings.
- `pilot/records/` and `pilot/mirrors.jsonl`: output-source lineage and QC.
- Dispatcher `states/synthetic-mirrors-pilot-v1.json` and `logs/synthetic-mirrors-pilot-v1.log`: ownership and process diagnostics.

Inspect live PID/start ticks and completion status before resuming. Use the existing dispatcher; do not launch a duplicate worker. A changed protocol, input list or model requires a separately versioned run. Resumption reuses completed calls and records; it does not regenerate rejected content to get a preferred quality result.

## Source eligibility and scale

The uploaded human pool currently contains 34,206 quarantined candidates and zero admitted passages. It cannot yet support a training-ready mirror release. The first collection also exposed genre errors, such as Wikipedia discussion boilerplate appearing in the provisional reference slice. Complete parent admission, genre routing, exclusions and family partitioning before scaling.

Luna is the selected main hosted generator. The user authorized a cumulative $1 generation run, now running on the Space under `/data/workspace/synthetic-mirrors-luna-dollar-v1`. See `LUNA_DOLLAR_RUN.txt` for recovery and scope. Secondary generators are not selected. Preparing stage 2 does not start detector training or change existing evaluation splits.

## Running Luna

Default invocation validates inputs and prints the plan without API calls:

```sh
python research/synthetic-mirrors/run_openrouter.py --input INPUT.jsonl --out RUN_DIRECTORY --pilot --limit 10
```

To execute, add `--execute --budget-usd APPROVED_CAP`. The cap covers both extraction and writing calls in this run, including calls retained from a previous invocation. Remove `--pilot` only for admitted, family-partitioned parents. Use a new output directory for each changed input selection or protocol. Keep the Qwen pilot directory separate. The runner uses four concurrent workers by default. A shared budget includes completed charges and conservative reservations for in-flight calls; it stops gracefully when another request cannot fit.

Local use reads the API key through the existing `benchmarks/otd/arena100_generate.py` transport. The Space worker receives it in process memory and uses `openrouter_transport.py` to send requests only to OpenRouter. It is not saved in run records. No new credential file is required.

Verified October 2, 2026 UTC: [OpenRouter's live endpoint metadata](https://openrouter.ai/api/v1/models/openai/gpt-6-luna/endpoints) lists Flex at $0.05 per million input tokens and $0.25 per million output tokens (cache write ceiling $0.0625/M). The runner checks prices and freezes the canonical model revision before dispatch. For illustration, 100 million billed input tokens plus 100 million billed output tokens would cost $30 at those ordinary rates; this is not an estimate of the final corpus, whose length and reasoning usage remain unmeasured.

Hosted inference precision is provider-managed and unreported. The hosted manifest records that explicitly rather than inheriting the local pilot's BF16 claim. All locally controlled inference remains BF16. The original `protocol.json` is preserved for the already-launched local pilot; the hosted run snapshots the same content rules with the hosted precision field corrected.

## Existing stage 3

The local `paper-eval-workflows-luna-20260930/dataset.jsonl` contains 2,646 correlated views of 1,323 unique family/condition pairs: 189 each for seven workflows. Three are genuine editing workflows: proofread, light polish, and substantial rewrite, giving 567 distinct edited targets. Each appears in two exported views.

These editing rows preserve `human_origin_ai_assisted_no_binary_gold` and mark edited target regions `assisted_unknown`. Their source families are assigned to pilot, calibration and test, so they are evaluation-only. Reuse their implementation and prompt/provenance conventions when useful; do not import their text into training. This confirms stage-3 generation exists but does not establish an implementation of Pangram 4 §3.5 soft n-gram clause labeling.
