import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / "research/benchmarks/meld-device"
selected = json.loads((out / "papers.json").read_text())
while True:
    with sqlite3.connect(root / "backend/.data/workspace.sqlite3") as conn:
        states = [
            conn.execute(
                "select status from scans where id=?", (p["scan_id"],)
            ).fetchone()[0]
            for p in selected
        ]
    if all(s == "completed" for s in states) and all(
        (out / (p["scan_id"] + ".json")).exists() for p in selected
    ):
        break
    if any(s in ("failed", "cancelled") for s in states):
        raise RuntimeError(states)
    time.sleep(5)
for precision, batch in [
    ("float32", 1),
    ("float16", 1),
    ("float16", 2),
    ("float16", 4),
    ("bfloat16", 1),
]:
    print("Testing", precision, batch, flush=True)
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/tune_meld.py"),
            "--precision",
            precision,
            "--batch",
            str(batch),
        ],
        check=True,
        cwd=root,
    )
