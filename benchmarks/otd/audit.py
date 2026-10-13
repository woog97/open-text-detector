"""Validate normalized corpora and checksum receipts without inference or network."""

import hashlib
import json
from collections import Counter

from prepare import B, readjsonl, validate


def main():
    manifest = json.loads((B / "data/prepared/manifest.json").read_text())
    report = {"datasets": {}, "raw_receipts": 0, "raw_bytes": 0}
    for receipt in (B / "data/raw").rglob("*.receipt.json"):
        m = json.loads(receipt.read_text())
        p = receipt.with_name(receipt.name.removesuffix(".receipt.json"))
        with p.open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        assert actual == m["sha256"] and p.stat().st_size == m["bytes"], p
        report["raw_receipts"] += 1
        report["raw_bytes"] += p.stat().st_size
    for p in sorted((B / "data/prepared").glob("*.jsonl")):
        rows = validate(list(readjsonl(p)))
        m = manifest["datasets"][p.stem]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == m["sha256"], p
        assert len(rows) == m["selected"], p
        for r in rows:
            assert hashlib.sha256(r["text"].encode()).hexdigest() == r["text_sha256"]
        report["datasets"][p.stem] = {
            "rows": len(rows),
            "unique_texts": len({r["text_sha256"] for r in rows}),
            "labels": dict(Counter(r["label"] for r in rows)),
            "cohorts": len({r.get("cohort") for r in rows}),
        }
    report["passed"] = True
    (B / "AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
