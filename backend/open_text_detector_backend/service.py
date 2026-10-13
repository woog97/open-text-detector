import asyncio
import base64
import hashlib
import json
import logging
from .sqlite_runtime import sqlite3
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from fastapi import HTTPException
from .db import now
from .result_store import ResultStore, ResultStorageError
from .providers import Providers, ProviderError
from .plagiarism import compare_corpus

logger = logging.getLogger(__name__)


def public_model(model):
    return {k: v for k, v in model.items() if k != "secret"} | {"has_api_key": bool(model.get("secret")), "enabled": bool(model["enabled"])}


def public_scan(scan, detail=True, store=None):
    data = {k: v for k, v in scan.items() if k not in {"model_snapshot", "upload_path", "worker_id", "lease_until", "request_hash", "idempotency_key", "result_storage"}}
    snapshot = json.loads(scan["model_snapshot"])
    data["model"] = public_model(snapshot["model"])
    data["result"] = json.loads(scan["result"]) if scan["result"] else None
    reference = scan.get("result_storage")
    if detail and reference:
        if store is None:
            raise ResultStorageError("Result store is required for this report")
        data["result"] = store.get(reference)
    data["error"] = json.loads(scan["error"]) if scan["error"] else None
    data["word_count"] = len(scan["text"].split())
    if detail and data["result"]:
        if store is not None:
            from .pdf_extraction import find_maps
            maps = find_maps(scan["text"], store.settings.dataset_dir)
            if maps: data["result"]["source_maps"] = maps
        metrics = dict(data["result"].get("performance") or {})
        completed = data["result"].get("completed_at")
        if completed:
            metrics["elapsed_since_submission_seconds"] = max(0, (datetime.fromisoformat(completed) - datetime.fromisoformat(scan["created_at"])).total_seconds())
        metrics["stored_result_bytes"] = json.loads(reference)["bytes"] if reference else len(scan["result"].encode("utf-8"))
        if reference:
            metrics["uncompressed_result_bytes"] = json.loads(reference)["json_bytes"]
            metrics["storage_format"] = json.loads(reference)["format"]
        metrics["input_utf8_bytes"] = len(scan["text"].encode("utf-8"))
        data["result"]["performance"] = metrics
    if not detail:
        data.pop("text", None)
        data.pop("notes", None)
        if data["result"]:
            data["result"] = {k: v for k, v in data["result"].items() if k in {"score", "label", "score_type"}}
    return data


class ScanService:
    def __init__(self, db, settings, providers=None):
        self.db, self.settings = db, settings
        self.results = ResultStore(settings)
        self.bulk = None
        self.providers = providers or Providers(settings, db)
        self.worker_id = str(uuid4())
        self.stopping = asyncio.Event()

    def model(self, model_id=None, task="text"):
        if not model_id:
            default = self.db.one("SELECT value FROM settings WHERE key=?", (f"default_{task}_model",))
            model_id = default["value"] if default else None
        model = self.db.one("SELECT * FROM models WHERE id=?", (model_id,)) if model_id else None
        if not model or model["task"] != task:
            raise HTTPException(409, detail={"code": "model_not_configured", "message": f"Configure a {task} classification model first"})
        if not model["enabled"]:
            raise HTTPException(409, detail={"code": "model_disabled", "message": "This model is disabled. Enable it after its dependencies and access are ready"})
        return model

    def prepare(self, body, *, batch_id=None, source="text", kind="text", upload_path=None, idempotency=None):
        if kind == "text" and not 50 <= len(body.text.split()) <= 100_000:
            raise HTTPException(422, "Text scans require between 50 and 100,000 words")
        if len(body.text) > self.settings.max_text_chars:
            raise HTTPException(413, "Text exceeds the configured character limit")
        # Hash caller intent before resolving defaults; retries keep the original model.
        request_hash = hashlib.sha256(json.dumps({"body": body.model_dump(), "source": source, "kind": kind}, sort_keys=True).encode()).hexdigest()
        if idempotency:
            prior = self.db.one("SELECT * FROM scans WHERE idempotency_key=?", (idempotency,))
            if prior:
                if prior["request_hash"] != request_hash:
                    raise HTTPException(409, "Idempotency key was already used for another request")
                return prior
        model = self.model(body.model_id, kind)
        stamp = now()
        return {"id": str(uuid4()), "title": body.title or (body.text[:80].replace("\n", " ") or "Image scan"),
                "kind": kind, "text": body.text, "source": source,
                "model_snapshot": json.dumps({"model": model, "check_plagiarism": body.check_plagiarism}),
                "status": "queued", "created_at": stamp, "updated_at": stamp,
                "batch_id": batch_id, "upload_path": upload_path,
                "idempotency_key": idempotency, "request_hash": request_hash}

    def insert(self, scans):
        with self.db.connect() as conn:
            for row in scans:
                if conn.execute("SELECT 1 FROM scans WHERE id=?", (row["id"],)).fetchone():
                    continue
                columns = list(row.keys())
                try:
                    conn.execute(f"INSERT INTO scans({','.join(columns)}) VALUES({','.join('?' for _ in columns)})", tuple(row.values()))
                except sqlite3.IntegrityError:
                    # A simultaneous idempotent request may have won the insertion.
                    previous = conn.execute("SELECT * FROM scans WHERE idempotency_key=?", (row.get("idempotency_key"),)).fetchone()
                    if not previous or previous["request_hash"] != row["request_hash"]:
                        raise HTTPException(409, "Conflicting request")
                    row["id"] = previous["id"]
        return [public_scan(self.db.one("SELECT * FROM scans WHERE id=?", (row["id"],)), store=self.results) for row in scans]

    def claim(self):
        stamp = time.time()
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("UPDATE scans SET status='failed',error=?,updated_at=? WHERE status='running' AND lease_until<? AND attempts>=3",
                         (json.dumps({"code": "worker_interrupted", "message": "Processing was interrupted three times; retry manually"}), now(), stamp))
            row = conn.execute("SELECT * FROM scans WHERE deleted_at IS NULL AND (status='queued' OR (status='running' AND lease_until<? AND attempts<3)) ORDER BY created_at LIMIT 1", (stamp,)).fetchone()
            if not row:
                return None
            conn.execute("UPDATE scans SET status='running',worker_id=?,lease_until=?,attempts=attempts+1,updated_at=? WHERE id=?",
                         (self.worker_id, stamp + self.settings.lease_seconds, now(), row["id"]))
            return dict(row)

    async def heartbeat(self, scan_id):
        while True:
            await asyncio.sleep(max(1, self.settings.lease_seconds / 3))
            self.db.execute("UPDATE scans SET lease_until=? WHERE id=? AND worker_id=? AND status='running'",
                            (time.time() + self.settings.lease_seconds, scan_id, self.worker_id))

    async def process_one(self):
        row = self.claim()
        if row is None:
            return False
        started_at, started = now(), time.perf_counter()
        heartbeat = asyncio.create_task(self.heartbeat(row["id"]))
        try:
            snapshot = json.loads(row["model_snapshot"])
            image = base64.b64encode(Path(row["upload_path"]).read_bytes()).decode() if row["kind"] == "image" else None
            classification_started = time.perf_counter()
            result = await asyncio.to_thread(self.providers.classify, snapshot["model"], row["text"], image)
            classification_seconds = time.perf_counter() - classification_started
            if snapshot["check_plagiarism"]:
                result["plagiarism"] = await asyncio.to_thread(compare_corpus, row["text"], self.db.all("SELECT * FROM corpus"))
            from .pdf_extraction import find_maps
            maps = find_maps(row["text"], self.settings.dataset_dir)
            if maps: result["source_maps"] = maps
            result["completed_at"] = now()
            result["performance"] = {
                "started_at": started_at,
                "classification_seconds": round(classification_seconds, 6),
                "processing_seconds": round(time.perf_counter() - started, 6),
                "attempt": row["attempts"] + 1,
                "words_per_second": round(len(row["text"].split()) / classification_seconds, 2) if classification_seconds > 0 and row["text"] else None,
                "timing_scope": "Provider call, including model loading when needed, preprocessing and inference",
            }
            if row["attempts"] == 0:
                result["performance"]["queue_seconds"] = max(0, (datetime.fromisoformat(started_at) - datetime.fromisoformat(row["created_at"])).total_seconds())
            if not self.db.one("SELECT id FROM scans WHERE id=? AND status='running' AND worker_id=? AND deleted_at IS NULL", (row["id"], self.worker_id)):
                return True
            summary, reference = await asyncio.to_thread(self.results.put, result)
            saved = self.db.execute("UPDATE scans SET status='completed',result=?,result_storage=?,error=NULL,updated_at=?,lease_until=NULL WHERE id=? AND worker_id=? AND status='running' AND deleted_at IS NULL",
                            (summary, reference, now(), row["id"], self.worker_id))
            if not saved and not self.db.one("SELECT id FROM scans WHERE result_storage=? LIMIT 1", (reference,)):
                await asyncio.to_thread(self.results.discard, reference)
        except ResultStorageError:
            logger.exception("Could not persist findings: %s", row["id"])
            self.fail(row["id"], "result_storage_failed", "Findings could not be saved. Check storage configuration and retry.")
        except ProviderError as e:
            self.fail(row["id"], e.code, e.message)
        except Exception:
            logger.exception("Scan processing failed: %s", row["id"])
            self.fail(row["id"], "processing_failed", "The scan could not be completed. Check server diagnostics and retry")
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass
        return True

    def fail(self, scan_id, code, message):
        self.db.execute("UPDATE scans SET status='failed',error=?,updated_at=?,lease_until=NULL WHERE id=? AND worker_id=? AND status='running' AND deleted_at IS NULL",
                        (json.dumps({"code": code, "message": message}), now(), scan_id, self.worker_id))

    async def run(self):
        while not self.stopping.is_set():
            try:
                if self.bulk:
                    await asyncio.to_thread(self.bulk.advance)
                if await self.process_one():
                    continue
            except Exception:
                logger.exception("Queue polling failed")
            try:
                await asyncio.wait_for(self.stopping.wait(), timeout=self.settings.worker_poll_seconds)
            except TimeoutError:
                pass
