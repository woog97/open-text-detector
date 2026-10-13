import asyncio
import json
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4
from fastapi import FastAPI, Depends, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from .result_store import ResultStorageError
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
from .config import Settings
from .db import Database, digest, now, write_private
from .documents import extract_document, fetch_article, validate_image, DocumentError
from .network import resolve_target, NetworkError
from .schemas import (ModelConfig, ScanRequest, URLScanRequest, BatchRequest, ScanPatch, KeyRequest,
                      DefaultRequest, ShareRequest, CorpusRequest)
from .security import require, RateLimiter, BodyLimitMiddleware
from .service import ScanService, public_model, public_scan as serialize_scan
from .papers import Papers
from .pdf_reader import PDFReader
from .paper_bulk import PaperBulk
from .schemas import PaperBulkRequest, PaperBulkStop
from .findings import Findings, DATASETS
from .reports import csv_report, pdf_report, json_report

Admin = Annotated[dict, Depends(require("admin"))]
Reader = Annotated[dict, Depends(require("read"))]
Scanner = Annotated[dict, Depends(require("scan"))]


def create_app(settings=None, providers=None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        app.state.settings = settings
        app.state.db = Database(settings.data_dir)
        app.state.limiter = RateLimiter()
        app.state.service = ScanService(app.state.db, settings, providers)
        app.state.findings = Findings(app.state.service, settings.dataset_dir)
        app.state.papers = Papers(app.state.findings)
        app.state.pdf_reader = PDFReader(app.state.service, settings.dataset_dir)
        app.state.bulk = PaperBulk(app.state.service, app.state.papers)
        app.state.service.bulk = app.state.bulk
        runner = asyncio.create_task(app.state.service.run()) if settings.worker_enabled else None
        yield
        app.state.service.stopping.set()
        if runner:
            # Finish the current operation before shutdown. A hard interruption is
            # recovered by the persistent lease, including after a process crash.
            await runner

    app = FastAPI(title="Open Text Detector Workbench API", version="0.1.0", lifespan=lifespan,
                  description="Private, model-selectable classification. All /v1 routes require a scoped Bearer API key.")
    app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=4)
    app.add_middleware(BodyLimitMiddleware, limit=settings.max_request_bytes)
    app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins,
                       allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
                       allow_headers=["Authorization", "Content-Type", "Idempotency-Key"])

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        # Pydantic's default error payload can echo submitted provider credentials.
        return JSONResponse(status_code=422, content={"detail": [
            {"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in error.errors()]})

    @app.exception_handler(ResultStorageError)
    async def storage_error(request, error):
        return JSONResponse(status_code=503, content={"detail": "Saved findings are temporarily unavailable. Please retry."})

    @app.exception_handler(DocumentError)
    @app.exception_handler(NetworkError)
    async def input_error(request, error):
        return JSONResponse(status_code=422, content={"detail": str(error)})

    def db():
        return app.state.db

    def service():
        return app.state.service

    def public_scan(scan, detail=True):
        return serialize_scan(scan, detail, service().results)

    def get_scan(scan_id, include_deleted=False):
        row = db().one("SELECT * FROM scans WHERE id=?" + ("" if include_deleted else " AND deleted_at IS NULL"), (scan_id,))
        if not row:
            raise HTTPException(404, "Scan not found")
        return row

    def get_model(model_id):
        row = db().one("SELECT * FROM models WHERE id=?", (model_id,))
        if not row:
            raise HTTPException(404, "Model not found")
        return row

    def validate_model(body):
        if body.provider == "http":
            resolve_target(body.endpoint, model_endpoint=True, allow_local=settings.allow_local_model_endpoints)

    def model_values(body, current=None):
        return (body.name, body.provider, body.task, body.model_id, body.base_model_id, body.endpoint,
                db().encrypt(body.api_key) if body.api_key else None if body.clear_api_key else (current or {}).get("secret"),
                int(body.enabled), body.lower_threshold, body.upper_threshold)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def root():
        return """<!doctype html><html><head><title>Open Text Detector backend</title><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{font:17px/1.6 system-ui;max-width:720px;margin:10vh auto;padding:24px;color:#234b3c}a{color:#d9550b}li{margin:9px 0}code{background:#f1f4f1;padding:4px}</style></head><body><h1>Open Text Detector backend</h1><p>Your private classification service is running.</p><ul><li><a href="/docs">Open the interactive API documentation</a></li><li><a href="/health">Check service health</a></li></ul><p>Choose the active detector in the local interface under Models &amp; settings.</p><p>Use the workspace owner's key from <code>.data/admin.key</code> with the Authorize button in the API documentation. Keep this key private.</p></body></html>"""

    @app.get("/health", tags=["System"])
    def health():
        db().one("SELECT 1")
        return {"status": "ok", "version": "0.1.0", "worker_enabled": settings.worker_enabled}

    @app.get("/v1/capabilities", tags=["System"])
    def capabilities(actor: Reader):
        return {"text_detection": "requires_enabled_model", "image_detection": "requires_http_image_model",
                "models": ["editlens", "meld", "laya", "http"], "default_text_model": db().one("SELECT value FROM settings WHERE key='default_text_model'")["value"],
                "uploads": ["txt", "md", "csv", "rtf", "docx", "pdf"], "batch_limit": 100,
                "plagiarism": "workspace_corpus_exact_phrase_matching", "web_plagiarism": "not_configured",
                "ocr": "not_configured", "gmail": "not_connected", "lms": "not_connected",
                "model_access": "Managed on the inference host; no model is downloaded at startup"}

    @app.get("/v1/pdf-reader", tags=["PDF reader"])
    def reader_files(actor: Reader):
        return app.state.pdf_reader.list()

    @app.get("/v1/pdf-reader/{file_id}", tags=["PDF reader"])
    def reader_detail(file_id: str, actor: Reader):
        return app.state.pdf_reader.detail(file_id)

    @app.get("/v1/pdf-reader/{file_id}/file", tags=["PDF reader"])
    def reader_pdf(file_id: str, actor: Reader):
        path = app.state.pdf_reader.path(file_id)
        return FileResponse(path, media_type="application/pdf", filename=path.name,
                            content_disposition_type="inline", headers={"Cache-Control": "private, no-cache", "Content-Encoding": "identity"})

    @app.get("/v1/papers", tags=["Paper library"])
    def papers(actor: Reader, q: str = "", conference: str = "", year: int | None = None, classified_by: str = "",
               offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100)):
        return app.state.papers.list(q, conference, year, offset, limit, classified_by)

    @app.get("/v1/paper-runs", tags=["Paper library"])
    def latest_paper_run(actor: Reader):
        return {"run": app.state.bulk.get()}

    @app.post("/v1/paper-runs", tags=["Paper library"], status_code=202)
    def start_paper_run(body: PaperBulkRequest, actor: Scanner):
        return app.state.bulk.start(body)

    @app.post("/v1/paper-runs/{run_id}/stop", tags=["Paper library"])
    def stop_paper_run(run_id: str, body: PaperBulkStop, actor: Scanner):
        return app.state.bulk.stop(run_id, body.mode)

    @app.get("/v1/papers/{paper_id}", tags=["Paper library"])
    def paper(paper_id: str, actor: Reader):
        return app.state.papers.get(paper_id)

    @app.post("/v1/papers/{paper_id}/compute", tags=["Paper library"], status_code=202)
    def compute_paper(paper_id: str, body: DefaultRequest, actor: Scanner):
        return app.state.papers.compute(paper_id, body.model_id)

    @app.get("/v1/finding-datasets", tags=["Findings"])
    def finding_datasets(actor: Reader):
        return {"items": [{"id": key, "name": value[0], "description": value[2],
                            "count": len(app.state.findings.entries(key))} for key, value in DATASETS.items()]}

    @app.get("/v1/finding-datasets/{dataset}", tags=["Findings"])
    def finding_examples(dataset: str, actor: Reader, q: str = "", offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100)):
        rows = app.state.findings.entries(dataset)
        if q:
            rows = [r for r in rows if q.casefold() in (r["title"] + " " + r["source_id"]).casefold()]
        return {"items": [{k: v for k, v in r.items() if k != "offset"} for r in rows[offset:offset+limit]], "total": len(rows)}

    @app.post("/v1/finding-datasets/{dataset}/{example}/compute", tags=["Findings"], status_code=202)
    def compute_finding(dataset: str, example: str, body: DefaultRequest, actor: Scanner):
        return app.state.findings.compute(dataset, example, body.model_id)

    @app.get("/v1/findings", tags=["Findings"])
    def saved_findings(actor: Reader, q: str = "", model_id: str | None = None,
                       offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100)):
        clauses = ["deleted_at IS NULL", "status='completed'"]
        args = []
        if q:
            clauses.append("instr(lower(title), lower(?)) > 0")
            args.append(q)
        if model_id:
            clauses.append("json_extract(model_snapshot, '$.model.id')=?")
            args.append(model_id)
        where = " AND ".join(clauses)
        total = db().one("SELECT count(*) AS n FROM scans WHERE " + where, args)["n"]
        rows = db().all("SELECT * FROM scans WHERE " + where + " ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?", [*args, limit, offset])
        items = [public_scan(row, False) | {"preview": row["text"][:260]} for row in rows]
        return {"items": items, "total": total}

    @app.get("/v1/models", tags=["Models"])
    def list_models(actor: Reader):
        return {"items": [public_model(m) for m in db().all("SELECT * FROM models ORDER BY created_at")],
                "defaults": {r["key"]: r["value"] for r in db().all("SELECT * FROM settings WHERE key LIKE 'default_%_model'")}}

    @app.post("/v1/models", status_code=201, tags=["Models"])
    def add_model(body: ModelConfig, actor: Admin):
        validate_model(body)
        model_id, stamp = str(uuid4()), now()
        db().execute("INSERT INTO models VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (model_id, *model_values(body), stamp, stamp))
        db().audit("model.created", actor["id"], model_id)
        return public_model(get_model(model_id))

    @app.put("/v1/models/{model_id}", tags=["Models"])
    def update_model(model_id: str, body: ModelConfig, actor: Admin):
        current = get_model(model_id)
        validate_model(body)
        if body.task != current["task"]:
            raise HTTPException(409, "A model's task cannot change; create a separate model")
        db().execute("UPDATE models SET name=?,provider=?,task=?,model_id=?,base_model_id=?,endpoint=?,secret=?,enabled=?,lower_threshold=?,upper_threshold=?,updated_at=? WHERE id=?",
                     (*model_values(body, current), now(), model_id))
        db().audit("model.updated", actor["id"], model_id)
        return public_model(get_model(model_id))

    @app.put("/v1/settings/default-model", tags=["Models"])
    def default_model(body: DefaultRequest, actor: Admin):
        model = get_model(body.model_id)
        db().execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                     (f"default_{model['task']}_model", body.model_id))
        db().audit("model.default_changed", actor["id"], body.model_id)
        return {"task": model["task"], "model_id": body.model_id}

    @app.delete("/v1/models/{model_id}", tags=["Models"])
    def delete_model(model_id: str, actor: Admin):
        get_model(model_id)
        if db().one("SELECT 1 FROM settings WHERE key LIKE 'default_%_model' AND value=?", (model_id,)):
            raise HTTPException(409, "Choose another default before removing this model")
        db().execute("DELETE FROM models WHERE id=?", (model_id,))
        db().audit("model.deleted", actor["id"], model_id)
        return {"deleted": True}

    @app.post("/v1/scans", status_code=202, tags=["Scans"])
    def create_scan(body: ScanRequest, actor: Scanner, idempotency_key: Annotated[str | None, Header(max_length=200)] = None):
        key = actor["id"] + ":" + idempotency_key if idempotency_key else None
        row = service().prepare(body, idempotency=key)
        result = service().insert([row])[0]
        db().audit("scan.submitted", actor["id"], result["id"])
        return result

    @app.post("/v1/scans/url", status_code=202, tags=["Scans"])
    async def scan_url(body: URLScanRequest, actor: Scanner):
        # Validate provider before fetching any third-party content.
        service().model(body.model_id)
        text, final_url = await asyncio.to_thread(fetch_article, body.url)
        if len(text) > settings.max_text_chars:
            raise HTTPException(413, "Article text exceeds the character limit")
        doc = ScanRequest(text=text, title=body.title, model_id=body.model_id, check_plagiarism=body.check_plagiarism)
        result = service().insert([service().prepare(doc, source=final_url)])[0]
        db().audit("scan.url_submitted", actor["id"], result["id"])
        return result

    @app.post("/v1/batches", status_code=202, tags=["Batches"])
    def create_batch(body: BatchRequest, actor: Scanner):
        batch_id = str(uuid4())
        if sum(len(doc.text) for doc in body.documents) > 5_000_000:
            raise HTTPException(413, "Batch exceeds 5 million characters")
        rows = [service().prepare(doc, batch_id=batch_id) for doc in body.documents]
        results = service().insert(rows)
        db().audit("batch.submitted", actor["id"], batch_id)
        return {"id": batch_id, "items": results}

    @app.post("/v1/uploads", status_code=202, tags=["Uploads"])
    async def upload_documents(actor: Scanner, files: Annotated[list[UploadFile], File()],
                               model_id: Annotated[str | None, Form()] = None,
                               check_plagiarism: Annotated[bool, Form()] = False):
        if not 1 <= len(files) <= 100:
            raise HTTPException(422, "Upload between 1 and 100 documents")
        service().model(model_id)
        batch_id, rows, errors, total = str(uuid4()), [], [], 0
        for index, file in enumerate(files):
            name = Path(file.filename or "document.txt").name[:200]
            try:
                data = await file.read(settings.max_upload_bytes + 1)
                if len(data) > settings.max_upload_bytes:
                    raise DocumentError("File exceeds the upload limit")
                total += len(data)
                if total > 100_000_000:
                    raise HTTPException(413, "Batch upload exceeds 100 MB")
                text = await asyncio.to_thread(extract_document, name, data, settings.max_text_chars)
                body = ScanRequest(text=text, title=name, model_id=model_id, check_plagiarism=check_plagiarism)
                rows.append(service().prepare(body, batch_id=batch_id, source="file:" + name))
            except (DocumentError, HTTPException) as e:
                if isinstance(e, HTTPException) and e.status_code == 413:
                    raise
                errors.append({"index": index, "filename": name, "error": str(e)})
            finally:
                await file.close()
        if not rows:
            raise HTTPException(422, {"message": "No documents could be scanned", "errors": errors})
        results = service().insert(rows)
        db().audit("batch.uploaded", actor["id"], batch_id)
        return {"id": batch_id, "items": results, "errors": errors}

    @app.post("/v1/images", status_code=202, tags=["Uploads"])
    async def upload_image(actor: Scanner, file: Annotated[UploadFile, File()], model_id: Annotated[str | None, Form()] = None):
        service().model(model_id, "image")
        data = await file.read(settings.max_upload_bytes + 1)
        await file.close()
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, "Image exceeds the upload limit")
        metadata = await asyncio.to_thread(validate_image, data)
        filename = Path(file.filename or "Image").name[:200]
        body = ScanRequest(text="Image", title=filename, model_id=model_id)
        body.text = ""
        path = db().root / "uploads" / (str(uuid4()) + ".bin")
        path.parent.mkdir(exist_ok=True, mode=0o700)
        row = service().prepare(body, kind="image", source=json.dumps(metadata), upload_path=str(path))
        write_private(path, data)
        try:
            result = service().insert([row])[0]
        except Exception:
            path.unlink(missing_ok=True)
            raise
        db().audit("image.submitted", actor["id"], result["id"])
        return result

    @app.get("/v1/scans", tags=["Scans"])
    def scans(actor: Reader, q: str = Query(default="", max_length=200),
              status: Literal["queued", "running", "completed", "failed", "cancelled"] | None = None,
              kind: Literal["text", "image"] | None = None, trash: bool = False,
              limit: int = Query(default=25, ge=1, le=100), offset: int = Query(default=0, ge=0)):
        clauses = ["deleted_at IS NOT NULL" if trash else "deleted_at IS NULL"]
        args = []
        if q:
            clauses.append("(title LIKE ? ESCAPE '\\' OR text LIKE ? ESCAPE '\\')")
            escaped = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            args += [f"%{escaped}%", f"%{escaped}%"]
        if status:
            clauses.append("status=?")
            args.append(status)
        if kind:
            clauses.append("kind=?")
            args.append(kind)
        where = " AND ".join(clauses)
        total = db().one("SELECT count(*) AS n FROM scans WHERE " + where, args)["n"]
        rows = db().all("SELECT * FROM scans WHERE " + where + " ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?", [*args, limit, offset])
        return {"items": [public_scan(r, False) for r in rows], "total": total, "limit": limit, "offset": offset}

    @app.get("/v1/extractions/{pdf_hash}/{text_hash}", tags=["Reports"])
    def extraction_map(pdf_hash: str, text_hash: str, actor: Reader):
        import re
        from .result_codec import decode
        if not re.fullmatch(r"[0-9a-f]{64}", pdf_hash) or not re.fullmatch(r"[0-9a-f]{64}", text_hash):
            raise HTTPException(404, "Extraction not found")
        path = settings.dataset_dir.parent / "extractions/positioned/objects" / pdf_hash / (text_hash + ".pgf")
        if not path.is_file():
            raise HTTPException(404, "Extraction not found")
        return decode(path.read_bytes())

    @app.get("/v1/scans/{scan_id}", tags=["Scans"])
    def scan(scan_id: str, actor: Reader):
        return public_scan(get_scan(scan_id))

    @app.patch("/v1/scans/{scan_id}", tags=["Scans"])
    def patch_scan(scan_id: str, body: ScanPatch, actor: Scanner):
        get_scan(scan_id)
        changes = body.model_dump(exclude_unset=True)
        if any(value is None for key, value in changes.items() if key != "feedback"):
            raise HTTPException(422, "Title and notes cannot be null")
        if changes:
            db().execute("UPDATE scans SET " + ",".join(f"{key}=?" for key in changes) + ",updated_at=? WHERE id=?",
                         [*changes.values(), now(), scan_id])
            db().audit("scan.updated", actor["id"], scan_id)
        return public_scan(get_scan(scan_id))

    @app.post("/v1/scans/{scan_id}/cancel", tags=["Scans"])
    def cancel_scan(scan_id: str, actor: Scanner):
        get_scan(scan_id)
        changed = db().execute("UPDATE scans SET status='cancelled',updated_at=? WHERE id=? AND status IN ('queued','running')", (now(), scan_id))
        if not changed:
            raise HTTPException(409, "Only queued or running scans can be cancelled")
        db().audit("scan.cancelled", actor["id"], scan_id)
        return public_scan(get_scan(scan_id))

    @app.post("/v1/scans/{scan_id}/retry", status_code=202, tags=["Scans"])
    def retry_scan(scan_id: str, actor: Scanner):
        row = get_scan(scan_id)
        if row["status"] not in {"failed", "cancelled"}:
            raise HTTPException(409, "Only failed or cancelled scans can be retried")
        # Create a separate attempt while retaining the historical failed record.
        snapshot = json.loads(row["model_snapshot"])
        model = service().model(snapshot["model"]["id"], row["kind"])
        snapshot["model"] = model
        stamp = now()
        retry = {"id": str(uuid4()), "title": row["title"], "kind": row["kind"], "text": row["text"],
                 "source": row["source"], "model_snapshot": json.dumps(snapshot), "status": "queued",
                 "created_at": stamp, "updated_at": stamp, "upload_path": row["upload_path"]}
        result = service().insert([retry])[0]
        db().audit("scan.retried", actor["id"], result["id"])
        return result

    @app.delete("/v1/scans/{scan_id}", tags=["Scans"])
    def trash_scan(scan_id: str, actor: Scanner):
        get_scan(scan_id)
        with db().connect() as c:
            c.execute("UPDATE scans SET deleted_at=?,updated_at=?,status=CASE WHEN status IN ('queued','running') THEN 'cancelled' ELSE status END WHERE id=?", (now(), now(), scan_id))
            c.execute("UPDATE shares SET revoked_at=? WHERE scan_id=?", (now(), scan_id))
        db().audit("scan.trashed", actor["id"], scan_id)
        return {"deleted": True, "recoverable": True}

    @app.post("/v1/scans/{scan_id}/restore", tags=["Scans"])
    def restore_scan(scan_id: str, actor: Scanner):
        get_scan(scan_id, True)
        db().execute("UPDATE scans SET deleted_at=NULL,updated_at=? WHERE id=?", (now(), scan_id))
        db().audit("scan.restored", actor["id"], scan_id)
        return public_scan(get_scan(scan_id))

    @app.get("/v1/batches/{batch_id}", tags=["Batches"])
    def batch(batch_id: str, actor: Reader):
        rows = db().all("SELECT * FROM scans WHERE batch_id=? AND deleted_at IS NULL ORDER BY created_at,id", (batch_id,))
        if not rows:
            raise HTTPException(404, "Batch not found")
        counts = {status: sum(r["status"] == status for r in rows) for status in ["queued", "running", "completed", "failed", "cancelled"]}
        return {"id": batch_id, "counts": counts, "finished": not (counts["queued"] + counts["running"]),
                "items": [public_scan(r, False) for r in rows]}

    @app.get("/v1/reports/{scan_id}", tags=["Reports"])
    def report(scan_id: str, actor: Reader, format: Literal["json", "csv", "pdf"] = "json"):
        row = public_scan(get_scan(scan_id))
        payload, content_type = (json_report(row), "application/json") if format == "json" else (
            (csv_report([row]), "text/csv") if format == "csv" else (pdf_report(row), "application/pdf"))
        return Response(payload, media_type=content_type, headers={"Content-Disposition": f'attachment; filename="scan-{scan_id}.{format}"'})

    @app.get("/v1/export", tags=["Reports"])
    def export(actor: Reader):
        rows = db().all("SELECT * FROM scans WHERE deleted_at IS NULL ORDER BY created_at DESC LIMIT 10000")
        return Response(csv_report([public_scan(r, False) for r in rows]), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="scan-history.csv"'})

    @app.post("/v1/scans/{scan_id}/shares", status_code=201, tags=["Sharing"])
    def share(scan_id: str, body: ShareRequest, actor: Admin):
        row = get_scan(scan_id)
        if row["status"] != "completed":
            raise HTTPException(409, "Only completed scans can be shared")
        share_id, token, expires = str(uuid4()), secrets.token_urlsafe(32), time.time() + body.expires_in_hours*3600
        db().execute("INSERT INTO shares VALUES(?,?,?,?,NULL)", (share_id, scan_id, digest(token), expires))
        db().audit("share.created", actor["id"], share_id)
        return {"id": share_id, "path": "/shared/" + token, "expires_at": expires,
                "notice": "Anyone holding this link can read the text and report until it expires or is revoked. Notes and provider connection details are excluded."}

    @app.delete("/v1/shares/{share_id}", tags=["Sharing"])
    def revoke_share(share_id: str, actor: Admin):
        if not db().execute("UPDATE shares SET revoked_at=? WHERE id=?", (now(), share_id)):
            raise HTTPException(404, "Share not found")
        db().audit("share.revoked", actor["id"], share_id)
        return {"revoked": True}

    @app.get("/shared/{token}", tags=["Sharing"])
    def public_report(token: str, request: Request):
        client = request.client.host if request.client else "anonymous"
        if not app.state.limiter.admit("shared:" + client, 60):
            raise HTTPException(429, "Rate limit exceeded")
        row = db().one("SELECT scan_id FROM shares WHERE token_hash=? AND expires_at>? AND revoked_at IS NULL", (digest(token), time.time()))
        if not row:
            raise HTTPException(404, "Share is invalid or expired")
        result = public_scan(get_scan(row["scan_id"]))
        result = {k: result[k] for k in ["id", "title", "kind", "text", "status", "result", "created_at", "word_count", "model"]}
        result["model"] = {k: result["model"][k] for k in ["name", "model_id", "provider", "lower_threshold", "upper_threshold"]}
        # Corpus documents are private even if a scan itself is shared.
        if result["result"]:
            result["result"].pop("plagiarism", None)
        return JSONResponse(result, headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})

    @app.get("/v1/keys", tags=["API Keys"])
    def keys(actor: Admin):
        rows = db().all("SELECT id,name,scopes,created_at,revoked_at FROM api_keys ORDER BY created_at")
        return {"items": [r | {"scopes": json.loads(r["scopes"])} for r in rows]}

    @app.post("/v1/keys", status_code=201, tags=["API Keys"])
    def create_key(body: KeyRequest, actor: Admin):
        key_id, token = str(uuid4()), "pgw_" + secrets.token_urlsafe(36)
        db().execute("INSERT INTO api_keys VALUES(?,?,?,?,?,NULL)", (key_id, body.name, digest(token), json.dumps(body.scopes), now()))
        db().audit("key.created", actor["id"], key_id)
        return {"id": key_id, "name": body.name, "key": token, "scopes": body.scopes,
                "notice": "This key is shown only once"}

    @app.delete("/v1/keys/{key_id}", tags=["API Keys"])
    def revoke_key(key_id: str, actor: Admin):
        if key_id == "bootstrap":
            raise HTTPException(409, "The workspace owner key is managed on the server")
        if not db().execute("UPDATE api_keys SET revoked_at=? WHERE id=?", (now(), key_id)):
            raise HTTPException(404, "API key not found")
        db().audit("key.revoked", actor["id"], key_id)
        return {"revoked": True}

    @app.post("/v1/corpus", status_code=201, tags=["Reference Corpus"])
    def add_corpus(body: CorpusRequest, actor: Admin):
        if db().one("SELECT COUNT(*) AS n FROM corpus")["n"] >= 1000:
            raise HTTPException(409, "Reference corpus limit reached (1,000 documents)")
        if db().one("SELECT COALESCE(SUM(LENGTH(text)),0) AS n FROM corpus")["n"] + len(body.text) > 10_000_000:
            raise HTTPException(409, "Reference corpus exceeds the 10 million character limit")
        item_id = str(uuid4())
        db().execute("INSERT INTO corpus VALUES(?,?,?,?,?)", (item_id, body.title, body.text, body.source_url, now()))
        db().audit("corpus.created", actor["id"], item_id)
        return {"id": item_id, **body.model_dump()}

    @app.get("/v1/corpus", tags=["Reference Corpus"])
    def corpus(actor: Admin, limit: int = Query(default=25, ge=1, le=100), offset: int = Query(default=0, ge=0)):
        return {"items": db().all("SELECT id,title,source_url,created_at,LENGTH(text) AS characters FROM corpus ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)),
                "total": db().one("SELECT count(*) AS n FROM corpus")["n"]}

    @app.delete("/v1/corpus/{item_id}", tags=["Reference Corpus"])
    def delete_corpus(item_id: str, actor: Admin):
        if not db().execute("DELETE FROM corpus WHERE id=?", (item_id,)):
            raise HTTPException(404, "Reference document not found")
        db().audit("corpus.deleted", actor["id"], item_id)
        return {"deleted": True}

    @app.get("/v1/usage", tags=["System"])
    def usage(actor: Reader):
        rows = db().all("SELECT status,COUNT(*) AS count FROM scans WHERE deleted_at IS NULL GROUP BY status")
        return {"scans": {r["status"]: r["count"] for r in rows},
                "models": db().one("SELECT count(*) AS n FROM models")["n"],
                "storage_bytes": db().path.stat().st_size,
                "billing": "No billing or subscription is connected"}

    @app.get("/v1/audit", tags=["System"])
    def audit(actor: Admin, limit: int = Query(default=50, ge=1, le=200)):
        return {"items": db().all("SELECT * FROM audit ORDER BY created_at DESC LIMIT ?", (limit,))}

    return app


app = create_app()
