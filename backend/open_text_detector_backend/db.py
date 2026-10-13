import hashlib
import json
import os
import secrets
from .sqlite_runtime import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from cryptography.fernet import Fernet


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write_private(path: Path, data: bytes):
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return
    with os.fdopen(fd, "wb") as f:
        f.write(data)


class Database:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.root / "workspace.sqlite3"
        write_private(self.root / "encryption.key", Fernet.generate_key())
        self.cipher = Fernet((self.root / "encryption.key").read_bytes())
        self.initialize()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self):
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations(version TEXT PRIMARY KEY, applied_at TEXT NOT NULL)")
            for path in sorted((Path(__file__).parent.parent / "migrations").glob("*.sql")):
                if not conn.execute("SELECT 1 FROM schema_migrations WHERE version=?", (path.name,)).fetchone():
                    conn.executescript("BEGIN IMMEDIATE;\n" + path.read_text())
                    conn.execute("INSERT INTO schema_migrations VALUES(?,?)", (path.name, now()))
                    conn.commit()
            write_private(self.root / "admin.key", ("pgw_" + secrets.token_urlsafe(36)).encode())
            token = (self.root / "admin.key").read_text().strip()
            conn.execute("INSERT OR IGNORE INTO api_keys VALUES(?,?,?,?,?,NULL)",
                         ("bootstrap", "Workspace owner", digest(token), json.dumps(["admin", "read", "scan"]), now()))
            conn.execute("INSERT OR IGNORE INTO models VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                "open-pangram-llama", "Open Pangram · Llama 3.2 3B", "editlens", "text",
                "pangram/editlens_Llama-3.2-3B", "meta-llama/Llama-3.2-3B", None, None, 0, .2, .8, now(), now()))
            conn.execute("INSERT OR IGNORE INTO settings VALUES('default_text_model','open-pangram-llama')")
        os.chmod(self.path, 0o600)

    def one(self, sql, args=()):
        with self.connect() as c:
            row = c.execute(sql, args).fetchone()
            return dict(row) if row else None

    def all(self, sql, args=()):
        with self.connect() as c:
            return [dict(r) for r in c.execute(sql, args).fetchall()]

    def execute(self, sql, args=()):
        with self.connect() as c:
            return c.execute(sql, args).rowcount

    def encrypt(self, value):
        return self.cipher.encrypt(value.encode()).decode() if value else None

    def decrypt(self, value):
        return self.cipher.decrypt(value.encode()).decode() if value else None

    def audit(self, action, actor, target=None):
        self.execute("INSERT INTO audit VALUES(?,?,?,?,?)", (str(uuid4()), action, actor, target, now()))
