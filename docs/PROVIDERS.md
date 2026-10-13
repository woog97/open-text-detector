# Classification provider contract

## HTTP model adapter

Create a registry record:

```json
{
  "name":"My classifier",
  "provider":"http",
  "task":"text",
  "model_id":"my-model-version",
  "endpoint":"https://my-inference-host.example/classify",
  "api_key":"optional-upstream-bearer-secret",
  "enabled":true,
  "lower_threshold":0.2,
  "upper_threshold":0.8
}
```

The backend sends `POST` JSON, `Content-Type: application/json`, and an optional `Authorization: Bearer ...` header. It sends exactly the model ID and submitted content; it does not forward the owner's workspace API key.

Text request:

```json
{"model":"my-model-version","text":"Original document text"}
```

Image request (`task: "image"`):

```json
{"model":"my-image-model","image_base64":"BASE64_ENCODED_IMAGE_BYTES"}
```

Required response:

```json
{
  "score":0.68,
  "segments":[
    {"start":0,"end":22,"score":0.68}
  ]
}
```

`score` is a finite number between 0 and 1, increasing with AI intervention. Segments are optional. Offsets refer to **Python Unicode code-point indices in the original submitted text**, not UTF-8 bytes or JavaScript UTF-16 units. Frontends should convert as needed for astral characters. Each segment has a positive range; segments must be ordered, nonoverlapping, within the document, and have finite [0,1] scores. Image providers should omit segments.

Provider responses are capped at 5 MB. Redirects are rejected to avoid leaking credentials. The transport validates public DNS destinations and pins the selected IP while retaining TLS host verification. Private/link-local addresses are rejected. Local inference is available only via loopback addresses when `OTD_ALLOW_LOCAL_MODEL_ENDPOINTS=true`; that flag never permits arbitrary LAN hosts or metadata endpoints.

An upstream failure, timeout, malformed output, or invalid score becomes a failed scan. There is no automatic fallback to another model.

The built-in HTTP contract is intentionally explicit. An arbitrary commercial API, including Pangram's commercial API, is **not automatically compatible**; use a small mapping service if its payloads differ.

## Local Qwen EditLens v3 (active)

The selected backend is [DarrenJiaImbue/editlens-qwen3-4b-merged-v3](https://huggingface.co/DarrenJiaImbue/editlens-qwen3-4b-merged-v3), pinned to `33716e3667e3514a92cd1e9a6f1511655f12ef86`. It is a full merged checkpoint; no separate base model or gated adapter is required.

From the workspace root:

```sh
cd backend
uv sync --frozen --extra models --extra research
cd ..
backend/.venv/bin/python scripts/download_model.py
# Start the backend, then select this model:
backend/.venv/bin/python scripts/activate_model.py
```

The downloader writes to `models/editlens-qwen3-4b-merged-v3/`, checks the weights against the published LFS SHA-256, and records every file's hash in `download-manifest.json`. `OTD_MODEL_DIR` overrides the model folder's parent directory. Serving uses local files only and refuses a missing/wrong revision manifest. Downloading is an explicit setup operation, not part of request handling.

The loader installs EditLens's LayerNorm + bias-free Linear head before loading the safetensors weights, and rejects missing, unexpected or mismatched tensors. This prevents the ordinary Transformers classifier from silently using a random head. `trust_remote_code` is disabled. The Transformers version matches the model's v5 configuration, including rotary-position settings.

The service uses the original EditLens text cleaning and expected-bucket calculation (`softmax(logits) @ [0,1,2,3] / 3`). Qwen runs BF16 on Apple MPS/CUDA or FP32 on CPU. Inputs longer than 1,024 tokens are split into nonoverlapping chunks and token-weighted; this extension differs from upstream's single truncated window. Each live result records model revision, device, precision and aggregation method. The model's training label distance thresholds (`0.03`, `0.15`) are **not** thresholds for this expected bucket score. The UI retains explicit, uncalibrated workspace thresholds of `0.2` and `0.8`.

The active model is CC-BY-NC-SA-4.0. Four live setup scans, including a multi-chunk document, completed successfully. These check serving, not detection accuracy. The two ICLR year cohorts must not be treated as ground-truth human/AI labels.

## Open Pangram / EditLens activation (deferred)

No activation is required to use or test the rest of the backend. Once access is approved:

1. Confirm Hugging Face access to both [Pangram's adapter](https://huggingface.co/pangram/editlens_Llama-3.2-3B) and [Meta's base model](https://huggingface.co/meta-llama/Llama-3.2-3B).
2. Install optional model libraries with `uv sync --frozen --extra models` from `backend/`. This downloads dependencies; model weights are fetched on the first inference request.
3. Configure `HF_TOKEN` securely in the inference process environment, the private `backend/.env` file, or the standard Hugging Face login. Do not place it in the model's HTTP `api_key` field: that field is for external HTTP providers.
4. Update the existing registry model through `PUT /v1/models/open-pangram-llama`:

```json
{
  "name":"Open Pangram · Llama 3.2 3B",
  "provider":"editlens",
  "task":"text",
  "model_id":"pangram/editlens_Llama-3.2-3B",
  "base_model_id":"meta-llama/Llama-3.2-3B",
  "enabled":true,
  "lower_threshold":0.2,
  "upper_threshold":0.8
}
```

5. Submit a validation scan and check memory, latency, and the result. Live inference has not yet been tested because access is pending.

The adapter follows the official EditLens sequence-classification implementation, including the LayerNorm-plus-linear head and expected bucket score. It applies Pangram's published emoji, preamble, case, and whitespace preprocessing with a mapping back to original text offsets. Long documents are evaluated in token-bounded chunks and aggregated by token count. This long-document aggregation and highlighting are workspace behavior, not a claim to reproduce Pangram's proprietary passage algorithm.

The adapter supports CUDA, Apple MPS, and CPU without requiring CUDA-only bitsandbytes. It loads one model at a time and runs sequentially to limit memory use. CPU uses float32; GPU/MPS uses float16. It does not execute remote model code or load pickle weight files. The Qwen release is pinned and checksum-verified as described above; the legacy Llama/RoBERTa adapters still resolve their upstream default revisions.

Thresholds require calibration for the intended dataset. Open Pangram is English-focused, unlike broader language coverage advertised for Pangram's commercial product. The published release is noncommercial; see the model card and third-party notice.


## Local MELD v5 localization

Install with `backend/.venv/bin/python scripts/download_meld.py`, restart the backend,
then select it using `backend/.venv/bin/python scripts/activate_meld.py`.
The provider name is `meld`; the accepted model ID is `anon-review-meld-2026/meld`.
Revision: `453acf594d48f8c55c3a38bde396f9178516d817` (v5, released 2026-07-31).
All weights load locally with strict tensor matching; no remote model code executes.

`score_type=ai_evidence`. `raw_score` is the mean of the top 25% token margins;
`score` is its sigmoid, not a calibrated probability or AI-authorship fraction.
The document label is `ai_evidence`, `below_threshold`, or `uncertain` (under 100 words).
MELD uses its shipped overall document threshold, not the registry's EditLens thresholds.

Long documents use 2048-token inputs including special tokens, with 256 source tokens
of overlap. Each token keeps the observation farthest from its window edge, then
pooling is performed over the whole document. This extension is not separately calibrated.
Original text and line breaks are preserved. `tokens` contains raw and scaled evidence
with half-open Unicode code-point offsets; byte tokens sharing a character are coalesced.
`segments` partitions the exact source into sentence-like units with token-weighted mean
evidence. Sentence splitting is a punctuation/blank-line heuristic and may split abbreviations.
The UI supports sentence/token heatmaps, score tooltips and disabling highlights.

Sentence/token labels use the document cutoff only as an exploratory reference.
They are **not** calibrated Human/AI-Assisted/AI-Generated predictions. The API explicitly
sets `localization.span_threshold_calibrated=false`. The model's 1% reference false-positive
rate does not transfer automatically to our corpus, individual spans or long-paper aggregation.

Source and reference scorer: https://huggingface.co/anon-review-meld-2026/meld

### Local MELD performance controls

`OTD_MELD_PRECISION` selects the backbone precision (`float32`, `float16`, `bfloat16`; portable default `float32`). The evidence head and rotary-frequency buffers stay float32. `OTD_MELD_BATCH_SIZE` controls how many full-length sliding windows are evaluated together (1–8; portable default 1). Batching preserves the same 2,048-token contexts, 256-token overlap, source offsets and aggregation; it does not combine attention across documents or truncate text. Settings are local inference controls, and cached reports retain the precision used for their original run.

Checkpoint loading skips random initialization of backbone weights that are immediately replaced, while retaining strict checkpoint key/shape validation. The backend keeps the loaded model between jobs. Reports record precision, batch size, phase timings and GPU tensor/driver allocations after the run (these are not peak GPU memory measurements).

The device benchmark scripts and results are in `scripts/tune_meld.py`, `scripts/sweep_meld_final.py`, and `research/benchmarks/meld-device`. Timing comparisons are device- and workload-specific; reductions in precision must be checked against the original outputs before selecting a device profile.

## Laya contextual phrase baseline

The `laya` provider supports the pinned local English `convaiinnovations/laya` checkpoint. It returns experimental phrase scores with original offsets and a length-weighted document summary. See [setup, context policy, and Apple GPU benchmarks](LAYA.md). Register it with `scripts/activate_laya.py`; selecting Laya does not require replacing the MELD installation.
