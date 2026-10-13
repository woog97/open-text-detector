# Open Text Detector

Open Text Detector detects AI-written text in research papers, at document, sentence and token level, with thresholds calibrated to a fixed false-positive rate on human writing. This repository holds the training and evaluation code, the paper ingestion tools, the Paper Atlas website (`app/`) and a private workbench (`backend/`).

The project is not affiliated with Pangram Labs. It was called Pangram during development, and some paths and internal names still use that word.

## Released

| What | Where |
|---|---|
| Website: every ICLR 2027 submission scored sentence by sentence, with calibration, evaluation and the full project history | [opentextdetector.com](https://opentextdetector.com) |
| The live model (LoRA adapter on Qwen3.6-35B-A3B, trained Oct 10, 2026) | [woog/open-text-detector-qwen36-35b-a3b-oct10](https://huggingface.co/woog/open-text-detector-qwen36-35b-a3b-oct10) |
| Its training data (72,000 windows) | [woog/open-text-detector-mix-oct7](https://huggingface.co/datasets/woog/open-text-detector-mix-oct7) |
| The held-out evaluation sets behind the published numbers | [woog/open-text-detector-heldout](https://huggingface.co/datasets/woog/open-text-detector-heldout) |
| All of the above | [Hugging Face collection](https://huggingface.co/collections/woog/open-text-detector-6ac947a6a58766dba41f131b) |

## What is in this repository

| Path | Contents |
|---|---|
| `benchmarks/otd/training/` | Training code by experiment. `overnight-sweep-20261004/` holds the trainer (`train_sweep.py`, `modeling_sweep.py`) and the fast evaluation path; `final-20261007/` the mix builder and the evaluation runner; `loss-ablation-20261010/` the ablations that chose the live recipe. |
| `benchmarks/otd/training/final-moe-oct10-snapshot/` | The byte-exact code that trained the live model, with each file's SHA-256 (matching the `run.json` published with the adapter) and the exact training command. |
| `benchmarks/otd/eval_suite/` | The frozen evaluation suite (hash-pinned). |
| `research/` | Data generation, evaluation, calibration and site-data pipelines, one folder per experiment and date. `research/atlas-rescore-moe-20261007/score_atlas_fast.py` scores papers for the site; `research/atlas-bib-exclusion-20261008/` finds reference lists and appendices; `research/atlas-oct10-20261012/` builds and publishes the site data for the live model. |
| `app/` | The website (Cloudflare Worker + React). |
| `backend/`, `scripts/` | A private workbench service and paper ingestion tools. |

## Reproducing the live model

1. Get the base model `Qwen/Qwen3.6-35B-A3B` (revision `995ad96eacd98c81ed38be0c5b274b04031597b0`) and the training data `woog/open-text-detector-mix-oct7`.
2. Follow `benchmarks/otd/training/final-moe-oct10-snapshot/README.md`: it gives the exact command and the code as it ran (one H200, about 2 hours).
3. Score text with `research/atlas-rescore-moe-20261007/score_atlas_fast.py` (510-token windows, Repeat2 layout) and flag sentences with the cutoffs on the model card or the site's Calibration page.

## License

Code and the published adapters: Apache License 2.0 (`LICENSE`). Datasets, quoted paper text and model outputs keep the terms of their sources; see `NOTICE`.

## Workbench

A private, model-selectable classification service with a dashboard, report views and an API for running detectors on submitted text, URLs and uploads.

The local interface connects to the backend for text, URL and upload submission, history, reports, model settings and API keys. The default detector is **MELD v5 · Local evidence**, with sentence and token heatmaps. **EditLens · Qwen3 4B v3** remains selectable for passage-level editing scores. **Laya · Experimental contextual phrases** is available as a local zero-shot alternative, with surrounding paragraph context and Apple GPU acceleration; see [Laya setup and design](docs/LAYA.md). The original Llama adapter remains disabled pending access.

## Run the interface

Start the backend below, then in another terminal:

```sh
cd app
npm ci
npm run dev -- --host 127.0.0.1 --port 3000
```

Open http://localhost:3000 for Paper Atlas, the searchable paper library and PDF reading interface. The existing private workbench is at http://localhost:3000/workbench. “Try a sample text” previews an explicitly illustrative report without running a model. The Models & settings screen configures detectors.

The development server forwards `/backend` to the loopback backend and reads the owner key on the server; the key is never embedded in the browser bundle. Keep both servers bound to loopback. This automatic owner authorization is for a trusted local workspace only. Production deployment needs authenticated server-side routing; the static/Worker build alone does not include this local proxy.

## Run the backend

```sh
cd backend
uv sync --frozen --extra models
uv run python -m open_text_detector_backend serve
```

Open http://127.0.0.1:8000/docs. The **Authorize** button accepts the owner key stored in `backend/.data/admin.key`; the key is generated on first start and is never printed by the service. Do not put keys in URLs or paste them into chat. The root page is a short service overview.

The installed environment can also run directly:

```sh
cd backend
.venv/bin/python -m open_text_detector_backend serve
```

Default binding is loopback only. Requests to `/v1` require a scoped Bearer API key. This is a **single-workspace backend**: authorized keys share its scan history. It is not a multi-tenant SaaS or a replacement for Pangram's proprietary production service.

## Backend features

- Persistent model registry: select a default or select a model for each scan.
- Pinned MELD v5 baseline (1.58 GB) with full-document token evidence, sentence highlights and Unicode character offsets.
- Selectable, pinned EditLens Qwen3 4B v3 model (8.06 GB), running locally. Original Llama and RoBERTa EditLens adapters are also supported.
- Generic HTTP adapter for hosted text or image classifiers, with encrypted upstream credentials.
- Durable SQLite scan queue with atomic claims, worker leases, restart recovery, cancellation, explicit retries, and idempotent text submission.
- Immutable model/configuration snapshots for every scan. Changing a default doesn't change past reports.
- Text, public article URL, and TXT, Markdown, CSV, RTF, DOCX, and text-based PDF inputs.
- Batch text submissions and up to 100 document uploads, including per-file error reporting.
- Image uploads (PNG/JPEG/WebP) to a configured image classifier.
- Searchable/paginated history, notes, feedback, recoverable trash, usage counts, and audit events.
- JSON, CSV, and PDF report export; expiring, revocable report links.
- Read/scan API keys, revocation, rate limits, request limits, and SSRF-resistant URL fetching with pinned DNS resolution.
- Exact phrase matching against a private reference corpus, clearly labeled as corpus-only matching.
- Interactive OpenAPI documentation and automated integration tests.

## What is not active

- **Original Open Pangram Llama inference:** awaiting gated access. The requested Qwen EditLens model is active instead.
- **Image detection:** needs an external image-classification endpoint.
- **Internet-wide plagiarism search:** needs a separately licensed search/index provider; local corpus matching is not a substitute.
- **OCR, Gmail, Google Docs, browser extensions, and LMS integrations:** not connected or implemented. The HTTP API provides a base for future integrations.
- **Billing, enterprise SSO, organization roles and multi-tenancy:** not implemented.
- **Complete product parity:** the interface follows the inspected dashboard and report layout, but paid-only services and the integrations listed above are unavailable. Promotional controls explain their availability rather than simulating purchases.

See [backend setup and API guide](docs/BACKEND.md), [provider contract](docs/PROVIDERS.md), and [third-party attribution](docs/THIRD_PARTY.md).

## Verification

```sh
cd backend
uv run pytest -q
uv run ruff check open_text_detector_backend tests
```

The tests use synthetic fixtures and a local HTTP test classifier. They do **not** establish detection quality or prove gated model inference works. No test-only classifier is shipped as a production provider.

## Local model and baseline data

See [model serving](docs/PROVIDERS.md#local-qwen-editlens-v3-active) and [baseline inventory](research/README.md). The four collections include 100 ICLR 2023 papers, 100 ICLR 2026 papers, 100 historical books and 6,397 modern AI responses. These are research inputs, not a calibrated accuracy claim.

## Local PDF reading prototype

Open **PDF reader** in the sidebar, or http://localhost:3000/pdf-reader after starting both local servers. The reader lists downloaded PDFs under `research/data` (restart the backend to discover newly added files). Nothing is uploaded or classified by opening a PDF.

- Original PDF pages rendered locally with PDF.js, zoom, page overview and navigation.
- Saved ReviewBench reports linked by exact paper ID; papers with reports are marked with a dot and listed first. Choose a saved model result in the right panel.
- Click page highlights or passage entries to inspect the original classified text and score. Toggle highlights or switch to the full OCR source text.
- Unique normalized text matches map existing OCR segments onto the native PDF text layer. Short, changed, and ambiguous passages are **not** overlaid and remain in the “Not located” filter. The original report's PDF revision is not known, so this prototype does not certify version correspondence.
- Pages scroll continuously; only nearby pages are rendered, with page-sized placeholders preserving the document layout. Local byte-range responses allow partial PDF loading. Papers with saved reports currently read text from every page to build the alignment index; this may fetch most of that PDF. Production ingestion should precompute verified segment coordinates and bind them to a PDF hash.
- The local worker, fonts, character maps, and WASM assets are copied from the pinned PDF.js dependency by `predev`/`prebuild`; there is no external PDF rendering service or CDN.

Verification: `node --experimental-strip-types --test app/tests/pdf-alignment.test.mjs`, plus `backend/.venv/bin/python -m pytest backend/tests/test_pdf_reader.py` (run Python tests from `backend` instead if the package is not on the import path). The API retains the workspace's read authorization and restricts files to its PDF catalogue; the existing development proxy supplies local authorization.

## Paper Atlas website preview

The homepage is a responsive library backed by the existing authenticated local PDF catalogue. Search titles/IDs, filter by collection or result availability, sort, and paginate. Opening a paper uses a deep link (`/?paper=<catalogue-id>`) and displays the original PDF with available saved classifications. Browser Back and the collection button return to the library. About explains the source collection and model limitations.

The public research preview is deployed at https://open-text-detector.woog09.workers.dev using a separate, read-only Cloudflare build. The library automatically discovers PDFs under the bucket's public `papers/<sha256>.pdf` prefix, including uploads without extraction or classification results. Use **PDF only** to browse those papers, or select **ICLR 2027** from the collection menu. Inventory is refreshed on demand every minute per Worker instance and the browser refreshes each minute while visible. Unknown metadata falls back to the filename under Other uploads.

## Cloudflare deployment

The site runs in one Cloudflare account:

- Worker `open-text-detector` (`app/wrangler.atlas.json`): static assets and the read-only catalogue/PDF API, served at https://opentextdetector.com, https://www.opentextdetector.com and https://open-text-detector.woog09.workers.dev. Deploy it only with `npm run deploy:atlas` from `app/` on a clean `main` (see `scripts/deploy_atlas.sh`).
- R2 Standard bucket `pangram-paper-atlas` (the bucket keeps its original name; R2 buckets cannot be renamed): deduplicated PDFs under `papers/<sha256>.pdf`, public report JSON under `results/<paper-id>.json`, catalogue and index files.
- Worker `open-text-detector-upload` (`app/wrangler.atlas-upload.json`): token-gated uploads into that bucket, used by the Atlas publishing scripts; they read its URL and token from a local access file, never from the repository.
- Worker `pangram-paper-atlas` (`app/wrangler.atlas-redirect.json`): the site's old name; it only redirects the old workers.dev address to the new one (`npm run deploy:atlas-redirect`).

No wrangler configuration binds a D1 database: the catalogue is read from R2. An older D1 database named `pangram-paper-atlas` may still exist in the account; nothing in this repository uses it.

The standalone entry is `app/atlas-cloud/main.tsx`. It reuses the public library and reader without deploying the local workbench, Python service, owner key, or model configuration. PDFs are delivered through the Worker binding with byte-range support; the bucket does not need a public R2 URL or CORS configuration. Hosting uses the custom domains and the included workers.dev address.

To update the published dataset from the saved local results (the backend need not be running):

```sh
backend/.venv/bin/python scripts/prepare_cloud_atlas.py
backend/.venv/bin/python scripts/upload_cloud_atlas.py
cd app
npm run deploy:atlas
```

The exporter reads the local database without starting inference, considers all classification selection manifests, and verifies PDF/input hashes. It retains previously published papers and adds new papers only once results are complete. The upload script uses the existing Wrangler login and checkpoints uploads locally. Run `cd app && npx wrangler whoami` first if the login token needs refreshing. `prepare_cloud_atlas.py` still writes a `catalogue.sql` from the old D1 setup; it is not imported anywhere. Removing papers requires an explicit reviewed deletion. Frontend-only updates need only `cd app && npm run deploy:atlas`, after the initial export has prepared PDF.js assets.

To upload all downloaded PDFs independently of classification/publication:

```sh
cd app && npx wrangler whoami && cd ..
backend/.venv/bin/python scripts/upload_paper_pdfs.py
```

The archive index now contains 14,561 distinct PDFs, including the latest 6,834-paper batch. All PDFs in the latest batch were verified against R2 sizes and checksums. The uploader preserves older archive entries when local files have been removed and does not delete local copies.

This deduplicates PDFs under `papers/<sha256>.pdf`, skips objects whose remote size and MD5 already match, verifies every object after upload, and stores a conference/year index at `indexes/downloaded-papers.json`. The website now discovers these PDFs automatically, using this index for conference/year and filenames. Rerunning resumes from the verified R2 objects; local summaries are saved under `app/.sites-runtime/atlas/`.

This is a snapshot, not an automatic ICLR ingestion pipeline. Before the planned 60,000-paper release, implement server-side pagination/search in the UI, verified version-bound annotation coordinates, and load testing against the account's Workers quotas. Current cache headers allow browser caching; they do not implement a shared Worker API cache. PDF range requests each invoke the Worker and can issue R2 metadata/read operations. R2 free egress does not eliminate operation or compute usage charges. No paid Workers plan was activated by this deployment.

When retiring the site, remove the `open-text-detector`, `open-text-detector-upload` and `pangram-paper-atlas` (redirect) Workers, the custom domains, and the R2 objects/bucket `pangram-paper-atlas` in Cloudflare. Export anything to retain first. Removing the Worker alone leaves stored data in R2. Review account subscriptions separately; no automatic shutdown has been scheduled.


### Automatic PDF discovery (October 3, 2026)

The Cloudflare reader merges the verified results manifest with a paginated R2 listing of the public `papers/` prefix. Published classifications take precedence and are never invented for new uploads. An unclassified PDF needs no report or extracted-text object: the API supplies an empty report list and the existing PDF.js reader opens the original file. Other bucket prefixes remain inaccessible. ETags cover the merged public inventory, so new uploads invalidate the browser catalogue even if results have not changed.

ICLR submission titles come from `indexes/iclr2027-pdfs.json`. Refresh that metadata independently with `backend/.venv/bin/python scripts/index_iclr2027_pdfs.py`; the regular PDF uploader and ICLR publisher also refresh it automatically. This command uploads only title/conference metadata, and the site includes an entry only once the corresponding PDF exists. It does not run inference, alter results, or remove files. Bucket custom metadata `title`, `filename`, and `collection` can describe other direct uploads.

### Research discovery browser (October 3, 2026)

The standalone Cloudflare homepage now defaults to an ICLR 2027 discovery browser. It offers research-area and keyword filters; search across titles, keywords, TL;DRs, abstract previews and submission numbers; title/update/submission sorting; a browser-local reading list; full abstracts and BibTeX; OpenReview links; related papers ranked by shared area/keywords; and next/previous navigation within filtered results. Filters and pagination survive shared URLs and browser history. The PDF reader opens in reading-only mode; the prior classification explorer is available at `/?view=classifications`.

The indexer reads locally archived public OpenReview notes, choosing latest mdate per ID and excluding restricted fields. `indexes/iclr2027-pdfs.json` remains lean for bucket discovery. Rich preview metadata is streamed through `/backend/v1/discovery` from `indexes/iclr2027-discovery.json` and merged in the browser, avoiding the Worker memory limit. Full abstracts/citations are in 256 metadata shards, served only for papers present in the public PDF catalogue through `/{id}/metadata`. Existing uploader/indexer hooks refresh these artifacts; no model calls are required.

Search currently covers metadata and abstract previews, not full PDF text. Anonymous submissions stay anonymous; no review ratings or acceptance decisions are inferred. Reading lists are local to each browser, not account-synced. The extracted text/position recovery archives are preserved separately and were not modified by this UI change.
