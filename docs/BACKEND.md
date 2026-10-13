# Backend operation and API

## Configuration and storage

Copy `backend/.env.example` to `backend/.env` only if you need to change defaults. Run all commands from `backend/`. Settings named `PANGRAM_*` (the project's old name) are still read when the `OTD_*` name is not set.

| Setting | Default | Purpose |
|---|---|---|
| `OTD_DATA_DIR` | `.data` | SQLite database, encryption key, owner key, image uploads |
| `OTD_ALLOWED_ORIGINS` | JSON list of localhost:3000 origins | Browser CORS allowlist; not an authentication mechanism |
| `OTD_WORKER_ENABLED` | `true` | Run the persisted scan worker |
| `OTD_ALLOW_LOCAL_MODEL_ENDPOINTS` | `false` | Allow explicit loopback HTTP model endpoints; never applies to article URLs |
| `OTD_MODEL_DIR` | workspace `models/` | Parent folder containing the pinned Qwen checkpoint |
| `OTD_MODEL_DEVICE` | `auto` | Choose CUDA, Apple MPS, or CPU for EditLens when activated |
| `OTD_PROVIDER_TIMEOUT` | `120` | HTTP provider timeout in seconds |
| `OTD_REQUESTS_PER_MINUTE` | `120` | Per-key request limit within the running process |

Secrets for HTTP providers are encrypted with Fernet. The key is generated as `.data/encryption.key`. Files are restricted to the owner. **Scan text and notes are not encrypted in SQLite**; protect the host and disk. Back up the complete data directory, including its encryption key. An encrypted-secret database cannot be restored without that key.

The service records audit action, actor key ID, target ID, and timestamp. It does not store raw request payloads in its audit log. Access logging is off in the supplied launcher to avoid recording share tokens.

Use one application process for a local workspace. Claims and leases prevent concurrent scan execution across worker instances, but rate limits are per process and each model process allocates its own memory. Scale-out deployment, hardened authentication, and tenant isolation are separate work.

`backend/Dockerfile` installs the locked base service without model weights or ML extras. To containerize, build with `backend` as the build context and mount a private writable `/data` volume owned by UID 10001. Container port 8000 must remain behind an authenticated TLS reverse proxy when accessed remotely. The Docker image has not been built in this session.

## SQLite runtime

The backend uses the bundled pysqlite3 driver with SQLite 3.51.3, pinned through uv.
Run `uv sync --extra models --extra research` after dependency changes. Building the driver
requires Xcode command line tools on macOS or a C compiler on Linux; the Dockerfile installs
build tools temporarily. Source provenance is in `backend/vendor/README.md`.

Do not switch this service back to the system SQLite 3.51.1: it contains a Unix connection
lifecycle deadlock fixed in 3.51.2. The bundled 3.51.3 also includes the subsequent WAL reset
fix. Database format and existing compressed findings are unchanged.

Verified on September 22, 2026: 65 backend tests passed, including 1,280 concurrent connection
lifecycles. After recovery, five-request median local API latencies were 12.4 ms for checks,
12.2 ms for cached findings, and 63.4 ms for the 51,529-paper library. A 120-request mixed
concurrency check completed without hangs (33.5 ms median, 500.9 ms maximum). These are API
measurements, not full browser rendering times.

## API key use

Use the owner key through the `Authorization: Bearer ...` header. In the interactive API docs, click **Authorize** and enter the token. Owner-only operations include model configuration, default selection, corpus management, key management, report sharing, and audit access.

Create a restricted integration key with `POST /v1/keys`, for example:

```json
{"name":"My integration","scopes":["read","scan"]}
```

The secret is returned once. The server stores only its SHA-256 hash. `DELETE /v1/keys/{id}` revokes it immediately. Keys represent access to the same workspace, not separate user accounts.

## Select a model

1. `GET /v1/models` lists models, enabled status, and defaults.
2. `POST /v1/models` creates another model. `PUT /v1/models/{id}` fully replaces its public configuration; omitting `api_key` preserves the encrypted credential. `clear_api_key: true` removes it.
3. `PUT /v1/settings/default-model` with `{"model_id":"registry-id"}` selects the default for that model's task.
4. Supply `model_id` in a scan request to override the default for that scan.

The active local default is `EditLens · Qwen3 4B v3`. A fresh database still seeds the disabled `open-pangram-llama` record; run `scripts/activate_model.py` after downloading Qwen to register and select it. The original Llama record is retained for later access approval.

Lower and upper thresholds are workspace classification policy. The `.2` and `.8` defaults are **not Pangram's calibrated thresholds**. The stored score is expected AI intervention on a continuous scale, not a calibrated probability of AI authorship and not a fraction of words written by AI. Passage scores are chunk-level estimates, not sentence-level certainty.

## Scan lifecycle

`POST /v1/scans` accepts:

```json
{
  "text":"At least fifty words of text to analyze...",
  "title":"Optional report title",
  "model_id":"optional-registry-id",
  "check_plagiarism":false
}
```

It returns HTTP 202 with the saved scan and a `queued` state. Poll `GET /v1/scans/{id}` for `running`, `completed`, `failed`, or `cancelled`. Disabled or missing models return HTTP 409 before queueing. Provider failures create a saved failed scan with an error code; they never produce a fabricated score.

An `Idempotency-Key` header makes text submission safe to repeat. The key is scoped to the caller's API key. A changed request with the same key receives 409. Repeating the original request returns its original scan, even if the default model has changed.

A scan snapshots the registry configuration, thresholds, and encrypted provider credential at submission. The server recovers expired running jobs after interruption, with a maximum of three claims. Recovery can repeat a remote inference request if the process died after the remote service completed but before persistence. Provider-side billing idempotency is not guaranteed.

Cancel queued/running work with `POST /v1/scans/{id}/cancel`. An in-flight model computation may finish, but its result will not overwrite the cancellation. Retry a failed or cancelled scan with `/retry`; this creates a **new** scan using that registry model's current settings while preserving the failed record.

Soft-delete with `DELETE /v1/scans/{id}` and restore with `POST /v1/scans/{id}/restore`. Deletion cancels pending work and revokes share links. Restoring does not restore share access or automatically restart inference. There is no automatic permanent purge or retention scheduler yet.

## Inputs and limits

- Text: 50–100,000 whitespace-delimited words and 500,000 characters maximum.
- JSON batches: `POST /v1/batches`, up to 100 documents and 5 million characters total, atomically validated before queueing.
- Files: multipart `POST /v1/uploads`, repeated `files` fields plus optional `model_id` and `check_plagiarism`; 20 MiB per file and 100 MB batch total. Unsupported files are listed in `errors`; successfully extracted documents are queued together. Originals are not retained for text documents.
- URLs: `POST /v1/scans/url`. Only HTTP(S), public addresses, port 80/443; redirects are revalidated and DNS addresses pinned before connecting. Up to 2 MB is downloaded. It does not execute JavaScript, sign into sites, or bypass paywalls.
- Images: multipart `POST /v1/images`; JPEG/PNG/WebP, 512×512 minimum, 40 megapixels maximum, 20 MiB upload limit. Image bytes remain in the private data directory. No image model is bundled.
- Reference corpus: at most 1,000 documents and 10 million characters total.

Password-protected PDFs and scanned PDFs without extractable text return explicit errors. OCR is not currently available. Original uploads may contain untrusted content; the backend extracts text only and does not execute scripts or macros.

## Results, reports, and sharing

- `GET /v1/scans?q=...&status=completed&kind=text&limit=25&offset=0` searches/paginates history. Use `trash=true` for recoverable items.
- `PATCH /v1/scans/{id}` updates `title`, `notes`, and `feedback` (`helpful`, `unhelpful`, or null).
- `GET /v1/batches/{id}` returns per-state counts, progress, and items.
- `GET /v1/reports/{id}?format=json|csv|pdf` downloads a report.
- `GET /v1/export` exports the most recent 10,000 non-deleted scans as CSV. CSV formula-like values are escaped. PDF export uses standard fonts; some non-Latin scripts may require an additional font configuration. JSON preserves Unicode exactly.
- `POST /v1/scans/{id}/shares` creates a bearer link for a completed scan; `expires_in_hours` defaults to 24 and is limited to 720. Only the owner can create links.
- Anyone holding the returned `/shared/{token}` path can read the document/report. Notes, provider endpoint/credentials, and private corpus details are omitted. The path is relative to the service's origin.
- `DELETE /v1/shares/{id}` revokes it. Shared responses use `no-store` and `no-referrer`.

Reference corpus matching reports overlap of exact eight-word phrases. Matches are not evidence of misconduct; quoted and common passages can match. The service does not search the internet or Pangram's proprietary plagiarism index.

## Pending integrations

`GET /v1/capabilities` explicitly reports unconfigured capabilities. Gmail, LMS, billing, OCR, and web plagiarism adapters are not implemented. External integrations can use the authenticated scan API, but this does not constitute native OAuth or an LMS integration.

## Cached findings

The `/cached-findings` page lists completed reports already persisted in the workspace SQLite database. Cards include the model snapshot, classification, text preview and date. Opening a card only retrieves the existing report, including its original sentence/token evidence; it never invokes a classifier.

The dataset browser reads the four collected JSONL sources in `research/data` (override with `OTD_DATASET_DIR`). Papers and AI responses use full collected text; PG-19 books use explicitly labelled first-2,000-word excerpts. Size limits still apply. Browsing does not run inference. Use **Compute / open** to reuse an existing report or enqueue one analysis with the selected model. Failed/cancelled reports open for explicit retry; trashed cached analyses must be restored through All Checks.

- `GET /v1/findings`: completed, non-trashed reports; paginated title search and model filter.
- `GET /v1/finding-datasets`: source catalogue and counts.
- `GET /v1/finding-datasets/{dataset}`: paginated example previews.
- `POST /v1/finding-datasets/{dataset}/{example}/compute`: `{ "model_id": "…" }`; returns `{ "scan": …, "reused": … }`.

Reuse compares exact submitted text and the stored model configuration (registry ID, provider, checkpoint IDs, endpoint and thresholds). A stable idempotency key prevents duplicate concurrent submissions. Renaming a model does not invalidate reports. Replacing weights in place or changing a remote endpoint's implementation is not automatically detectable: register a new model/checkpoint ID for a new version. Prior reports retain their original snapshot. The dataset catalogue uses file offsets and refreshes when the source file changes; source texts are copied into SQLite only when submitted for analysis.

## ReviewBench paper library

`/papers` is a dedicated read-only archive with title/author/ID search, conference and year filters, source text, provenance links and per-model report links. `/papers?paper=iclr%3AN92hjSf5NNh` opens a particular paper. Classification uses the usual persistent queue and report dashboard, with editing hidden for source-backed scans; text updates are not accepted by the API.

Run `backend/.venv/bin/python scripts/download_reviewbench.py` from the project root to reproduce the full pinned download and catalogue. All original Parquet shards (including peer reviews and metadata) are retained under `research/data/reviewbench/original`. The separate `catalogue.sqlite3` contains searchable paper metadata and losslessly compressed OCR text, opened read-only by the backend. `manifest.json` records revision, file sizes, SHA-256 checksums, counts and missing text. The original files are never changed by classification. Classification links and reports live in the existing workspace database.

`GET /v1/papers` accepts `q`, `conference`, `year`, `offset`, `limit`; `GET /v1/papers/{paper_id}` returns immutable source text and saved classifications; `POST /v1/papers/{paper_id}/compute` accepts `model_id`. The current classifier submission limits remain 50–100,000 words and 500,000 characters. Missing/oversized texts remain readable and are not silently truncated. The source's conference/decision metadata is not an AI-authorship label.

Paper-library model filtering: `GET /v1/papers?classified_by=<model registry ID>` returns papers with a completed, non-trashed classification by that model. `classified_by=any` includes any completed model result; omit the parameter for all papers. Filters apply before pagination and combine with title, conference and year filters.

New successful scans store `result.performance`: provider-call runtime (including loading/preprocessing), processing time, original start/completion times, first-attempt queue wait, attempt number and words/second. Detail responses also report input UTF-8 bytes, stored result bytes, and submission-to-completion elapsed time. Existing reports retain their original completion time; separate classification runtime is shown as unavailable if it was never measured. Provider metadata supplies device, precision, token/window counts and pinned revision where available. Cached reads never reset these measurements.

## Compressed findings and cloud object storage

New findings use verified, lossless compressed objects with small database summaries.
Local storage is the default; set `OTD_RESULT_STORAGE=s3` to use a private R2 or
S3-compatible bucket. See [RESULT_STORAGE.md](RESULT_STORAGE.md) for configuration,
existing-result migration, deployment limits, recovery and the 50,000-paper cost model.

## Filtered bulk paper processing

The classification filter accepts `not:<model-id>` and `not:any` as complements of
completed, non-trashed classifications. Failed, cancelled and in-flight scans do not
count as completed classifications.

`POST /v1/paper-runs` accepts `model_id`, `q`, `conference`, `year`, and `classified_by`.
It snapshots all matching paper IDs across all pages in the same order as the library
(year descending, conference, title, ID), excluding completed results for the chosen
model. A single durable run can be active at a time. The worker creates only one scan
at a time rather than copying 50,000 paper texts into the queue. It rechecks completed
and in-flight classifications before each paper and skips them. Invalid/missing paper
text is recorded as a failure and the run continues. Model configuration is pinned at
start; later disabling/deleting the model prevents new scans using it.

`GET /v1/paper-runs` returns the latest run, progress counts, current paper and first
10 errors. Start/stop requires `scan` scope; viewing requires `read`. Closing the page
or restarting the server does not cancel the persistent run. Changing library filters
after starting does not change its snapshot.

`POST /v1/paper-runs/{id}/stop` accepts `mode: finish|discard`. Both cancel pending
papers and any not-yet-running scan owned by this run. Finish allows its running scan
to persist normally before stopping. Discard marks it cancelled so the result is not
published or retained; an object written during the cancellation race is removed.
The underlying inference may finish its current call before releasing the device.
Previously completed results and unrelated manual scans remain untouched. Cancelled
papers are eligible for a later run. All completed results use compressed storage.
