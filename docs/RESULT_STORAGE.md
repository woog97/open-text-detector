# Deployable compressed findings

Classification results are immutable compressed objects. SQLite stores the scan metadata,
source text, small verdict summary, and an internal object reference. Heavy token and
sentence data is fetched only when a detail report/export is requested. Existing JSON
reports remain readable until migrated. The browser uses the same authenticated API.

## Format and access

`PGF1` is a versioned lossless format: JSON metadata plus typed, byte-shuffled numeric
columns compressed with Zstandard level 19. Repeated labels use dictionary indices;
offsets use reversible deltas/lengths. Float32 is used only when every saved value
round-trips exactly; otherwise float64 is retained. Saved transformed scores are never
recomputed. Unknown provider fields/schemas stay intact in compressed JSON metadata.
Each write validates canonical JSON equality, uploads/writes the object, reads it back,
checks SHA-256 and size, then commits the summary/reference atomically in SQLite.
A failed write never marks a scan completed. Failed reads return 503 without rerunning
inference. Objects are content addressed and cannot be confused across model runs.

History, findings and paper preview lists use the database summary and do not fetch
objects. Full reports use a byte-bounded, per-process LRU cache of compressed objects
(default 32 MiB); decoding returns independent objects so sharing/redaction cannot alter
later responses. The backend negotiates gzip for HTTP delivery of reconstructed JSON.
Browser payloads are gzip JSON, not PGF1; they are larger than the stored columnar object.
A direct binary browser decoder/lazy token endpoint is a possible later optimization,
not required for cloud storage. Keep the bucket private; only the backend needs access.

The local backend stores objects under `.data/findings/<hash-prefix>/<hash>.pgf`.
Selecting `s3` uses the same format through the S3 API (R2 or AWS-compatible storage).
No cloud account/bucket is created automatically, and no report is uploaded until S3
storage is configured and a write/migration is explicitly run.

## R2 setup

Create a private R2 Standard bucket. Give the backend a bucket-scoped read/write API
credential. Set these backend environment variables through deployment configuration
and its secret manager (never public frontend variables):

```dotenv
OTD_RESULT_STORAGE=s3
OTD_RESULT_S3_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com
OTD_RESULT_S3_BUCKET=open-text-detector-findings
OTD_RESULT_S3_REGION=auto
OTD_RESULT_S3_ACCESS_KEY=<secret>
OTD_RESULT_S3_SECRET_KEY=<secret>
OTD_RESULT_S3_PREFIX=findings/v1
OTD_RESULT_CACHE_BYTES=33554432
```

For AWS S3, omit the endpoint, set the real AWS region, and preferably use an instance
role/standard AWS credential chain instead of explicit keys. Backend SDK calls use
bounded timeouts/retries. Cloud credentials and internal references never enter public
scan responses. The browser does not need bucket CORS or credentials since reads go
through the authenticated backend. R2 integration is tested with the official SDK's
stubbed S3 contract; a live R2 roundtrip requires an actual configured bucket.

## Existing reports: resumable migration

From `backend/`, with the target storage configuration in `.env` or the environment:

```sh
.venv/bin/python ../scripts/migrate_result_storage.py
.venv/bin/python ../scripts/migrate_result_storage.py --apply
```

The first command reports eligible records without converting them. Apply creates a
SQLite recovery backup, processes one result at a time, verifies exact reconstruction
and persisted object integrity, and compare-and-swaps the database payload. Rerunning
skips already migrated results. It migrates legacy JSON or local objects into the chosen
store. Local objects remain readable while new writes target S3, allowing migration
without losing access. Changing from one remote bucket/configuration to another is not
an automatic migration: retain the old settings and deliberately copy/reconcile objects
first. References include a store fingerprint to reject accidental bucket mismatches.

Recovery backups retain the old data intentionally and consume extra local space.
Old local objects are also retained after upload. Neither is deleted automatically.
Only after verification and recovery planning should an operator remove these copies.
SQLite can reuse freed pages; its file does not immediately shrink. An offline `VACUUM`
can reclaim them after a database backup. Objects left unreferenced by cancelled jobs
or interrupted writes are harmless; garbage collection must respect retained backups
and trash. No automatic bucket lifecycle expiration should target active findings.

Deploy the metadata database together with its configured encryption key and object
store configuration; back them up coherently. Never deploy only the frontend and expect
it to read a local database. For a single API instance, SQLite on a persistent volume
is sufficient for this result catalogue; multiple API writers/replicas require a shared
database design. Source texts remain in the existing scan rows/catalogue, outside these
result-only size estimates. Deduplicating source text is a separate optimization.

Classification can run on a separate machine from serving. The current queue is still
SQLite based: setting `OTD_WORKER_ENABLED=false` makes a serving instance read/write
API-only, but does not implement distributed worker synchronization. Copy/sync a coherent
metadata snapshot for a read-only published corpus, or implement a shared job database
before concurrent remote workers. Production frontend authentication and the local
owner-key development proxy still need deployment configuration; see README.md.

## 50,000-paper budget

The local compression sample covers 11 reports from 10 ICLR 2023 papers. Its mean is
about 247 KB per result, predicting 12.35 GB for 50,000 papers **per model/version**.
Paper lengths and providers vary; budget 15–20 GB for findings until the full corpus is
measured. This is not a guarantee that every ReviewBench paper matches this sample.
Keep source files, database, backup copies and model weights separate in the budget.

Cloudflare R2 Standard prices checked September 22, 2026:

- Storage: $0.015/GB-month; 10 GB-month monthly free allowance.
- Writes/list operations: $4.50/million; 1 million monthly free allowance.
- Reads: $0.36/million; 10 million monthly free allowance.
- Direct R2 egress is free; Standard has no retrieval fee.
- Billing rounds storage up to GB and paid request usage to million-operation units.

At 13 GB sustained storage, gross storage is $0.195/month; if the full 10 GB allowance
is available, billable storage is roughly $0.045/month. At the user's 150 GB uncompressed
estimate, gross storage is $2.25/month. Fifty thousand verified writes involve roughly
50,000 PUTs and 50,000 read-back GETs, within the free request allowances if otherwise
unused. Traffic above these allowances can dominate storage cost, so retain the cache.
Hosting, database, inference, backups and any application-host bandwidth are additional.
Do not use the Infrequent Access class for interactive findings solely to save fractions
of a cent: it has retrieval fees and different free-tier/minimum-duration rules.

Sources:
- https://developers.cloudflare.com/r2/pricing/
- https://developers.cloudflare.com/r2/api/s3/api/
- https://github.com/facebook/zstd

Benchmarks: `research/benchmarks/result-compression/REPORT.md`.
