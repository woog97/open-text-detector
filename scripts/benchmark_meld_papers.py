"""Run a reproducible ten-paper MELD baseline via the normal persistent queue."""

import json
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/benchmarks/meld-device"
key = (ROOT / "backend/.data/admin.key").read_text().strip()


def api(path, body=None):
    req = urllib.request.Request(
        "http://127.0.0.1:8000" + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=60))


if "--resume" in sys.argv:
    selected = json.loads((OUT / "papers.json").read_text())
else:
    model = next(
        m
        for m in api("/v1/models")["items"]
        if m["provider"] == "meld" and m["enabled"]
    )
    selected = []
    for line in (ROOT / "research/data/iclr_2023/papers.jsonl").open():
        row = json.loads(line)
        pid = "iclr:" + row["id"]
        paper = api("/v1/papers/" + pid)
        if any(
            s["model"]["id"] == model["id"] and s["status"] == "completed"
            for s in paper["classifications"]
        ):
            continue
        result = api("/v1/papers/" + pid + "/compute", {"model_id": model["id"]})
        selected.append(
            {
                "paper_id": pid,
                "scan_id": result["scan"]["id"],
                "title": row["title"],
                "words": len(row["text"].split()),
            }
        )
        if len(selected) == 10:
            break
    (OUT / "papers.json").write_text(json.dumps(selected, indent=2))
print("Queued", len(selected), "papers", flush=True)
remaining = {r["scan_id"] for r in selected}
while remaining:
    for sid in list(remaining):
        conn = sqlite3.connect(ROOT / "backend/.data/workspace.sqlite3")
        conn.row_factory = sqlite3.Row
        row = dict(conn.execute("SELECT * FROM scans WHERE id=?", (sid,)).fetchone())
        conn.close()
        if row["status"] not in ("completed", "failed", "cancelled"):
            continue
        sys.path.insert(0, str(ROOT / "backend"))
        from open_text_detector_backend.service import public_scan

        scan = public_scan(row)
        if scan["status"] in ("completed", "failed", "cancelled"):
            (OUT / (sid + ".json")).write_text(json.dumps(scan))
            remaining.remove(sid)
            print(
                json.dumps(
                    {
                        "id": sid,
                        "status": scan["status"],
                        "performance": (scan.get("result") or {}).get("performance"),
                        "remaining": len(remaining),
                    }
                ),
                flush=True,
            )
    if remaining:
        time.sleep(5)
