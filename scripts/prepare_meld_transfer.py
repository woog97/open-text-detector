# The Worker was deleted Oct 13, 2026 (as pangram-meld-transfer); redeploy it from app/wrangler.meld-transfer.json before use.
"""Prepare a fresh scoped transfer token; checkpoints already in R2 stay there."""
import argparse
import hashlib
import json
import os
import secrets
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--hours", type=float, default=2)
p.add_argument("--access-file", type=Path, default=Path("/tmp/pangram-meld-transfer-access.json"))
args = p.parse_args()
if not 0 < args.hours <= 24:
    p.error("Transfer access must last between 0 and 24 hours; stored files do not expire")
atlas = json.loads((ROOT / "app/wrangler.atlas.json").read_text())
token = secrets.token_urlsafe(32)
args.access_file.write_text(json.dumps({"token": token,
    "url": "https://open-text-detector-meld-transfer.woog09.workers.dev"}))
os.chmod(args.access_file, 0o600)
config = {"name": "open-text-detector-meld-transfer", "account_id": atlas["account_id"],
    "main": "meld-transfer/worker.ts", "compatibility_date": "2026-09-01", "workers_dev": True,
    "r2_buckets": atlas["r2_buckets"], "observability": {"enabled": False},
    "vars": {"TOKEN_SHA256": hashlib.sha256(token.encode()).hexdigest(),
             "EXPIRES_AT": str(int(time.time() + args.hours * 3600))}}
(ROOT / "app/wrangler.meld-transfer.json").write_text(json.dumps(config, indent=2))
print("Prepared expiring download access; deploy app/wrangler.meld-transfer.json to activate it.")
