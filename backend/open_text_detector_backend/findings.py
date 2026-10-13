"""Local dataset catalogue and persistent, model-specific reuse of scan reports."""
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException
from .schemas import ScanRequest
from .service import public_scan

DATASETS = {
    "iclr_2023": ("ICLR 2023", "papers.jsonl", "Historical comparison; authorship is not verified."),
    "iclr_2026": ("ICLR 2026", "papers.jsonl", "Contemporary comparison; year is not an AI label."),
    "human_pg19": ("PG-19 · human writing", "books.jsonl", "First 2,000 words of each pre-1919 book."),
    "ai_mdta_2025": ("MDTA · AI writing", "responses.jsonl", "Responses from models released in 2025 or later."),
}


def fingerprint(model):
    # Display names, credentials and enabled state do not change the detector.
    return {k: model.get(k) for k in (
        "id", "provider", "task", "model_id", "base_model_id", "endpoint",
        "lower_threshold", "upper_threshold")}


def cache_key(text, model):
    return "finding:" + hashlib.sha256(json.dumps(
        [text, fingerprint(model)], sort_keys=True).encode()).hexdigest()


def excerpt(row, dataset):
    text = row["text"]
    if dataset == "human_pg19":
        words = list(re.finditer(r"\S+", text))
        if len(words) > 2000:
            return text[:words[1999].end()], True
    return text, False


@lru_cache(maxsize=8)
def catalogue(path, dataset, mtime, size):
    # Store byte offsets, not copies of the source corpus in memory or SQLite.
    rows = []
    with Path(path).open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            row = json.loads(line)
            text, shortened = excerpt(row, dataset)
            rows.append({"id": str(len(rows)), "source_id": str(row.get("id", len(rows))),
                         "title": row.get("title") or row.get("prompt") or row.get("id"),
                         "preview": text[:260], "word_count": len(text.split()),
                         "characters": len(text), "excerpt": shortened,
                         "source_url": row.get("source_url"),
                         "generator": row.get("generator"), "offset": offset})
    return rows


class Findings:
    def __init__(self, service, root):
        self.service, self.db, self.root = service, service.db, root

    def entries(self, dataset):
        if dataset not in DATASETS:
            raise HTTPException(404, "Dataset not found")
        path = self.root / dataset / DATASETS[dataset][1]
        if not path.is_file():
            return []
        stat = path.stat()
        return catalogue(str(path), dataset, stat.st_mtime_ns, stat.st_size)

    def document(self, dataset, example):
        rows = self.entries(dataset)
        if not example.isdigit() or int(example) >= len(rows):
            raise HTTPException(404, "Example not found")
        meta = rows[int(example)]
        with (self.root / dataset / DATASETS[dataset][1]).open("rb") as stream:
            stream.seek(meta["offset"])
            row = json.loads(stream.readline())
        return meta, excerpt(row, dataset)[0]

    def compute(self, dataset, example, model_id):
        meta, text = self.document(dataset, example)
        title = (meta["title"] + (" · first 2,000 words" if meta["excerpt"] else ""))[:200]
        return self.compute_text(text, title, "dataset:" + dataset, model_id)

    def compute_text(self, text, title, source, model_id):
        model = self.service.model(model_id)
        # Reuse reports made before the catalogue feature, including in-flight work.
        for scan in self.db.all("SELECT * FROM scans WHERE text=? AND kind='text' AND deleted_at IS NULL ORDER BY status='completed' DESC,created_at DESC", (text,)):
            if fingerprint(json.loads(scan["model_snapshot"])["model"]) == fingerprint(model):
                self.db.execute("UPDATE scans SET source_locked=1 WHERE id=?", (scan["id"],))
                scan["source_locked"] = 1
                return {"scan": public_scan(scan, store=self.service.results), "reused": True}
        key = cache_key(text, model)
        prior = self.db.one("SELECT * FROM scans WHERE idempotency_key=?", (key,))
        if prior and prior["deleted_at"]:
            raise HTTPException(409, "This saved finding is in Trash. Restore it in All Checks first.")
        if not 50 <= len(text.split()) <= 100_000:
            raise HTTPException(422, "This source must contain between 50 and 100,000 words to be classified")
        if len(text) > min(500_000, self.service.settings.max_text_chars):
            raise HTTPException(413, "This full source exceeds the current analysis character limit")
        body = ScanRequest(text=text, title=title[:200], model_id=model["id"])
        prepared = self.service.prepare(body, source=source, idempotency=key)
        prepared["source_locked"] = 1
        return {"scan": self.service.insert([prepared])[0], "reused": prior is not None}
