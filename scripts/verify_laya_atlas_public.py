"""Read-only end-to-end verification through the public website, without credentials."""

import concurrent.futures
import gzip
import json
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/classifications/laya-atlas-five-per-year"
SITE = "https://open-text-detector.woog09.workers.dev"


def read(path):
    req = urllib.request.Request(
        SITE + path,
        headers={
            "User-Agent": "open-text-detector-atlas-public-audit/1.0",
            "Cache-Control": "no-cache",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        raw = response.read()
    return json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)


def main():
    for attempt in range(10):
        catalogue = read("/backend/v1/pdf-reader?model=laya")
        if len(catalogue["items"]) == 125:
            break
        if attempt == 9:
            raise RuntimeError("Public catalogue has not exposed the full sample")
        time.sleep(10)
    assert set(Counter(i["collection"] for i in catalogue["items"]).values()) == {5}
    assert len({i["collection"] for i in catalogue["items"]}) == 25
    model = next(m for m in catalogue["models"] if m["id"] == "laya")
    assert model["available"] == 125 and model["sample"]
    records = {
        r["item"]["id"]: r
        for r in [json.loads(p.read_text()) for p in (OUT / "staged").glob("*.json")]
    }
    assert {i["id"] for i in catalogue["items"]} == set(records)

    def verify(item):
        r = records[item["id"]]
        expected = json.loads(
            gzip.decompress(
                (OUT / "objects" / Path(r["item"]["detail_key"]).name).read_bytes()
            )
        )
        for attempt in range(10):
            actual = read("/backend/v1/pdf-reader/" + item["id"])
            if actual == expected:
                break
            if attempt == 9:
                raise RuntimeError("Public paper detail differs from verified upload")
            time.sleep(10)
        assert item["score_summaries"] == r["item"]["score_summaries"]
        return len(
            next(x for x in actual["reports"] if x["model"].get("id") == "laya")[
                "result"
            ]["segments"]
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        phrases = sum(pool.map(verify, catalogue["items"]))
    original = json.loads((OUT / "live-catalogue-before.json").read_text())
    assert [m for m in catalogue["models"] if m["id"] != "laya"] == original["models"]
    receipt = {
        "public_papers_verified": 125,
        "conference_year_pairs": 25,
        "papers_per_pair": 5,
        "phrases": phrases,
        "existing_model_availability_preserved": True,
        "verified_at": time.time(),
        "url": SITE + "/?model=laya#collection",
    }
    (OUT / "public-verification.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
