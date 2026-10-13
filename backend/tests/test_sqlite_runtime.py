import subprocess
import sys
from pathlib import Path

from open_text_detector_backend.sqlite_runtime import sqlite3


def test_fixed_sqlite_runtime():
    assert sqlite3.sqlite_version_info >= (3, 51, 3)


def test_concurrent_wal_connection_lifecycle(tmp_path):
    # Isolate native deadlocks so a regression fails within a bounded time.
    script = """
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from open_text_detector_backend.db import Database
db = Database(Path(sys.argv[1]))
def work(i):
    for j in range(80):
        with db.connect() as conn:
            conn.execute("SELECT count(*) FROM scans").fetchone()
            if j % 8 == 0:
                conn.execute("INSERT OR REPLACE INTO settings VALUES(?,?)", (f"stress-{i}", str(j)))
with ThreadPoolExecutor(max_workers=16) as pool:
    list(pool.map(work, range(16)))
assert db.one("PRAGMA quick_check")["quick_check"] == "ok"
assert db.one("SELECT count(*) AS n FROM settings WHERE key LIKE 'stress-%'")["n"] == 16
"""
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        timeout=30,
        capture_output=True,
        text=True,
    )
