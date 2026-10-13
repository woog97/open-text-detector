"""Durable filtered paper runs; one scan at a time, in the catalogue's display order."""

import json
from .sqlite_runtime import sqlite3
import zlib
from uuid import uuid4
from fastapi import HTTPException
from .db import now
from .service import public_model


class PaperBulk:
    def __init__(self, service, papers):
        self.service, self.papers, self.db = service, papers, service.db

    def get(self, run_id=None):
        row = (
            self.db.one("SELECT * FROM paper_runs WHERE id=?", (run_id,))
            if run_id
            else self.db.one("SELECT * FROM paper_runs ORDER BY created_at DESC,id DESC LIMIT 1")
        )
        if not row:
            if run_id:
                raise HTTPException(404, "Bulk run not found")
            return None
        model = json.loads(row.pop("model_snapshot"))["model"]
        row["model"] = public_model(model)
        row["filters"] = json.loads(row["filters"])
        counts = self.db.all(
            "SELECT status,count(*) AS n FROM paper_run_items WHERE run_id=? GROUP BY status", (row["id"],)
        )
        row["counts"] = {r["status"]: r["n"] for r in counts}
        row["current"] = self.db.one(
            "SELECT i.paper_id,i.scan_id,s.title,s.status FROM paper_run_items i JOIN scans s ON s.id=i.scan_id WHERE i.run_id=? AND i.status='queued' ORDER BY i.ordinal LIMIT 1",
            (row["id"],),
        )
        # Fetch only queue metadata: never read/decompress paper bodies for polling.
        row["upcoming"] = self.db.all(
            "SELECT paper_id,ordinal FROM paper_run_items WHERE run_id=? AND status='pending' ORDER BY ordinal LIMIT 4",
            (row["id"],),
        )
        if row["upcoming"]:
            archive = self.papers.connect()
            try:
                ids = [item["paper_id"] for item in row["upcoming"]]
                titles = {r["id"]: r["title"] for r in archive.execute(
                    "SELECT id,title FROM papers WHERE id IN (" + ",".join("?" for _ in ids) + ")", ids
                )}
                for item in row["upcoming"]:
                    item["title"] = titles.get(item["paper_id"], item["paper_id"])
            finally:
                archive.close()
        row["errors"] = self.db.all(
            "SELECT paper_id,error FROM paper_run_items WHERE run_id=? AND error IS NOT NULL ORDER BY ordinal LIMIT 10",
            (row["id"],),
        )
        return row

    def start(self, body):
        model = self.service.model(body.model_id)
        if self.db.one("SELECT id FROM paper_runs WHERE status IN ('running','stopping')"):
            raise HTTPException(409, "A paper run is already active. Stop it first.")
        filters = body.model_dump(exclude={"model_id"})
        ordered = self.papers.list(**filters, ids_only=True)
        completed = {
            r["paper_id"]
            for r in self.db.all(
                "SELECT p.paper_id FROM paper_scans p JOIN scans s ON s.id=p.scan_id WHERE s.status='completed' AND s.deleted_at IS NULL AND json_extract(s.model_snapshot,'$.model.id')=?",
                (model["id"],),
            )
        }
        ordered = [p for p in ordered if p not in completed]
        rid, stamp = str(uuid4()), now()
        snapshot = json.dumps({"model": model, "check_plagiarism": False})
        try:
            with self.db.connect() as c:
                c.execute(
                    "INSERT INTO paper_runs VALUES(?,?,?,?,?,?,?)",
                    (
                        rid,
                        "running" if ordered else "completed",
                        snapshot,
                        json.dumps(filters),
                        len(ordered),
                        stamp,
                        stamp,
                    ),
                )
                c.executemany(
                    "INSERT INTO paper_run_items(run_id,ordinal,paper_id) VALUES(?,?,?)",
                    ((rid, i, p) for i, p in enumerate(ordered)),
                )
        except sqlite3.IntegrityError:
            raise HTTPException(409, "A paper run is already active. Stop it first.")
        return self.get(rid)

    def stop(self, rid, mode):
        with self.db.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            run = c.execute("SELECT status FROM paper_runs WHERE id=?", (rid,)).fetchone()
            if not run:
                raise HTTPException(404, "Bulk run not found")
            if run["status"] not in ("running", "stopping"):
                return self.get(rid)
            c.execute("UPDATE paper_runs SET status='stopping',updated_at=? WHERE id=?", (now(), rid))
            c.execute(
                "UPDATE paper_run_items SET status='cancelled' WHERE run_id=? AND status='pending'", (rid,)
            )
            # Only scans created by this run: never cancel unrelated manual jobs.
            states = ("queued", "running") if mode == "discard" else ("queued",)
            placeholders = ",".join("?" for _ in states)
            c.execute(
                f"UPDATE scans SET status='cancelled',updated_at=? WHERE id IN (SELECT scan_id FROM paper_run_items WHERE run_id=?) AND status IN ({placeholders})",
                (now(), rid, *states),
            )
            self._reconcile(c, rid)
        return self.get(rid)

    def _reconcile(self, c, rid):
        c.execute(
            "UPDATE paper_run_items SET status=(SELECT status FROM scans WHERE id=scan_id),error=(SELECT CASE WHEN error IS NOT NULL THEN json_extract(error,'$.message') END FROM scans WHERE id=scan_id) WHERE run_id=? AND status='queued' AND scan_id IN (SELECT id FROM scans WHERE status NOT IN ('queued','running'))",
            (rid,),
        )
        active = c.execute(
            "SELECT 1 FROM paper_run_items WHERE run_id=? AND status IN ('pending','queued') LIMIT 1", (rid,)
        ).fetchone()
        if not active:
            c.execute(
                "UPDATE paper_runs SET status=CASE WHEN status='stopping' THEN 'stopped' ELSE 'completed' END,updated_at=? WHERE id=?",
                (now(), rid),
            )

    def advance(self):
        with self.db.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            run = c.execute(
                "SELECT * FROM paper_runs WHERE status IN ('running','stopping') LIMIT 1"
            ).fetchone()
            if not run:
                return
            rid = run["id"]
            self._reconcile(c, rid)
            if (
                run["status"] != "running"
                or c.execute(
                    "SELECT 1 FROM paper_run_items WHERE run_id=? AND status='queued'", (rid,)
                ).fetchone()
            ):
                return
            snapshot = json.loads(run["model_snapshot"])
            # Bounded work per tick when records are missing text or exceed limits.
            for item in c.execute(
                "SELECT * FROM paper_run_items WHERE run_id=? AND status='pending' ORDER BY ordinal LIMIT 100",
                (rid,),
            ).fetchall():
                existing = c.execute(
                    "SELECT 1 FROM scans s JOIN paper_scans p ON p.scan_id=s.id WHERE p.paper_id=? AND s.deleted_at IS NULL AND s.status IN ('completed','queued','running') AND json_extract(s.model_snapshot,'$.model.id')=? LIMIT 1",
                    (item["paper_id"], snapshot["model"]["id"]),
                ).fetchone()
                if existing:
                    c.execute(
                        "UPDATE paper_run_items SET status='skipped' WHERE run_id=? AND ordinal=?",
                        (rid, item["ordinal"]),
                    )
                    continue
                try:
                    # Read archive directly, avoiding nested write connections while holding this transaction.
                    archive = self.papers.connect()
                    try:
                        paper = archive.execute(
                            "SELECT title,text FROM papers WHERE id=?", (item["paper_id"],)
                        ).fetchone()
                    finally:
                        archive.close()
                    if paper is None:
                        raise ValueError("Paper is no longer in the archive")
                    text = zlib.decompress(paper["text"]).decode()
                    if not 50 <= len(text.split()) <= 100_000 or len(text) > min(
                        500_000, self.service.settings.max_text_chars
                    ):
                        raise ValueError("Archived text is missing or outside the supported length limits")
                    # Retain the chosen model configuration for the entire run.
                    self.service.model(snapshot["model"]["id"])  # respect later disable/delete
                    scan = {
                        "id": str(uuid4()),
                        "title": paper["title"][:200],
                        "text": text,
                        "kind": "text",
                        "source": "reviewbench:" + item["paper_id"],
                        "source_locked": 1,
                        "model_snapshot": run["model_snapshot"],
                        "status": "queued",
                        "created_at": now(),
                        "updated_at": now(),
                        "batch_id": rid,
                    }
                    c.execute(
                        f"INSERT INTO scans({','.join(scan)}) VALUES({','.join('?' for _ in scan)})",
                        tuple(scan.values()),
                    )
                    c.execute("INSERT INTO paper_scans VALUES(?,?)", (item["paper_id"], scan["id"]))
                    c.execute(
                        "UPDATE paper_run_items SET status='queued',scan_id=? WHERE run_id=? AND ordinal=?",
                        (scan["id"], rid, item["ordinal"]),
                    )
                    break
                except (ValueError, HTTPException, zlib.error) as error:
                    message = str(error.detail) if isinstance(error, HTTPException) else str(error)
                    c.execute(
                        "UPDATE paper_run_items SET status='failed',error=? WHERE run_id=? AND ordinal=?",
                        (message, rid, item["ordinal"]),
                    )
            self._reconcile(c, rid)
