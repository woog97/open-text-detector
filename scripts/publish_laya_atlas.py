"""Add Laya scores to already-public paper details, preserving existing reports."""

import argparse
import concurrent.futures
import copy
import fcntl
import gzip
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

from atlas_scores import summarize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/classifications/laya-atlas-five-per-year"
PUBLIC = ROOT / "research/exports/atlas-public"
SITE = "https://open-text-detector.woog09.workers.dev"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def atomic(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    tmp.replace(path)


def parse(raw):
    return json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)


def http(method, key, raw=None, etag=None, public=False):
    access = json.loads(Path("/tmp/pangram-atlas-publish-access.json").read_text())
    for attempt in range(5):
        try:
            headers = {"User-Agent": "open-text-detector-atlas-publisher/1.0"}
            if not public:
                headers["Authorization"] = "Bearer " + access["token"]
            if etag:
                headers["If-Match"] = etag
            url = (SITE if public else access["url"]) + "/" + key
            req = urllib.request.Request(url, data=raw, method=method, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as response:
                return response.read(), response.headers.get("ETag")
        except urllib.error.HTTPError as error:
            if error.code in {401, 403, 404, 412} or attempt == 4:
                raise RuntimeError(
                    f"Cloudflare {method} failed: HTTP {error.code}"
                ) from None
        except OSError:
            if attempt == 4:
                raise RuntimeError("Cloudflare request failed after retries") from None
        time.sleep(min(2**attempt, 10))


def put(key, raw, etag=None):
    ack = json.loads(http("PUT", key, raw, etag)[0])
    assert (
        ack["size"] == len(raw)
        and ack["etag"].strip('"') == hashlib.md5(raw).hexdigest()
    )
    assert http("GET", key)[0] == raw, "Read-back verification failed"


def stage_one(meta, item, manifest):
    result_path = OUT / "results" / (meta["pdf_sha256"] + ".json")
    if not result_path.exists():
        return None
    result_raw = result_path.read_bytes()
    result = json.loads(result_raw)
    fingerprint = digest(result_raw)
    assert (
        result["text_sha256"] == meta["text_sha256"]
        and result["recipe_sha256"] == manifest["recipe_sha256"]
    )
    record_path = OUT / "staged" / (meta["pdf_sha256"] + ".json")
    if record_path.exists():
        previous = json.loads(record_path.read_text())
        if (
            previous["source_detail_key"] == item["detail_key"]
            and previous["result_sha256"] == fingerprint
        ):
            return previous
    # Source all paper text and positions from the unauthenticated PUBLIC website.
    # Verify equality with the existing object before adding only the new scores.
    public_url = "backend/v1/pdf-reader/" + item["id"]
    detail = parse(http("GET", public_url, public=True)[0])
    existing = http("GET", item["detail_key"])[0]
    assert digest(existing) == Path(item["detail_key"]).name.split(".")[0]
    assert detail == parse(existing), (
        "Public content differs from current snapshot; retry later"
    )
    assert detail["version"] == meta["pdf_sha256"] and detail["version_verified"]
    matching = [
        r for r in detail["reports"] if r.get("text_sha256") == meta["text_sha256"]
    ]
    assert matching, "No matching public text"
    text = matching[0]["text"]
    assert digest(text.encode()) == meta["text_sha256"]
    positions = detail["position_maps"][meta["text_sha256"]]
    assert positions["pages"] and positions["rectangles"]
    segments = []
    end = 0
    for s in result["segments"]:
        assert (
            end <= s["start"] < s["end"] <= len(text)
            and math.isfinite(s["score"])
            and 0 <= s["score"] <= 1
        )
        end = s["end"]
        segments.append(
            {
                **s,
                "label": "low_evidence"
                if s["score"] <= 0.2
                else "ai_evidence"
                if s["score"] > 0.8
                else "uncertain",
            }
        )
    assert "".join(
        "".join(text[s["start"] : s["end"]] for s in segments).split()
    ) == "".join(text.split())
    detail["reports"] = [
        r for r in detail["reports"] if r.get("model", {}).get("id") != "laya"
    ]
    detail["reports"].append(
        {
            "id": "laya-"
            + manifest["recipe_sha256"][:12]
            + "-"
            + meta["pdf_sha256"][:24],
            "text": text,
            "text_sha256": meta["text_sha256"],
            "model": {"id": "laya", "name": manifest["model_name"]},
            "result": {
                "segments": segments,
                "score": result["score"],
                "score_type": result["score_type"],
                "notice": result["notice"],
                "inference": result["inference"],
                "recipe_sha256": manifest["recipe_sha256"],
            },
        }
    )
    raw = gzip.compress(
        json.dumps(detail, ensure_ascii=False, separators=(",", ":")).encode(),
        compresslevel=6,
        mtime=0,
    )
    key = "atlas-public/objects/" + digest(raw) + ".json.gz"
    put(key, raw)
    (OUT / "objects" / Path(key).name).write_bytes(raw)
    changed = copy.deepcopy(item)
    changed["detail_key"] = key
    changed["models"] = list(dict.fromkeys([*item["models"], "laya"]))
    changed.setdefault("score_summaries", {})["laya"] = summarize(segments, text)
    record = {
        "source_detail_key": item["detail_key"],
        "public_source": SITE + "/" + public_url,
        "underlying_content_verified_already_public": True,
        "result_sha256": fingerprint,
        "item": changed,
        "bytes": len(raw),
        "phrases": len(segments),
        "verified": True,
    }
    atomic(record_path, record)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()
    for name in ["staged", "objects"]:
        (OUT / name).mkdir(exist_ok=True)
    manifest = json.loads((OUT / "manifest.json").read_text())
    original_raw, etag = http("GET", "atlas-public/catalogue.json")
    catalogue = json.loads(original_raw)
    snapshot = OUT / "live-catalogue-before.json"
    if not snapshot.exists():
        snapshot.write_bytes(original_raw)
    by_pdf = {Path(i["pdf_key"]).stem: i for i in catalogue["items"]}
    assert len(by_pdf) == len(catalogue["items"]) and all(
        m["pdf_sha256"] in by_pdf for m in manifest["papers"]
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(
            pool.map(
                lambda m: stage_one(m, by_pdf[m["pdf_sha256"]], manifest),
                manifest["papers"],
            )
        )
    print(
        json.dumps(
            {
                "staged": sum(r is not None for r in records),
                "total": len(records),
                "publish": args.publish,
            }
        ),
        flush=True,
    )
    if not args.publish:
        return
    complete = json.loads((OUT / "complete.json").read_text())
    assert (
        complete["papers"] == 125
        and complete["recipe_sha256"] == manifest["recipe_sha256"]
        and all(records)
    )
    audit = json.loads((OUT / "browser-data-audit.json").read_text())
    assert audit["papers"] == 125 and audit["all_browser_reports_valid"]
    assert {p["id"]: p["detail_key"] for p in audit["paper_results"]} == {
        r["item"]["id"]: r["item"]["detail_key"] for r in records
    }, "Browser audit is stale; audit the newly staged objects before publishing"
    with (PUBLIC / "publication.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        live_raw, live_etag = http("GET", "atlas-public/catalogue.json")
        assert live_raw == original_raw and live_etag == etag, (
            "Live catalogue changed; restage before publishing"
        )
        changed = {r["item"]["id"]: r["item"] for r in records}
        catalogue["items"] = [changed.get(i["id"], i) for i in catalogue["items"]]
        catalogue["models"] = [m for m in catalogue["models"] if m["id"] != "laya"]
        catalogue["models"].append(
            {
                "id": "laya",
                "name": manifest["model_name"],
                "available": 125,
                "total": len(catalogue["items"]),
                "complete": False,
                "sample": True,
            }
        )
        catalogue["published_at"] = time.time()
        raw = json.dumps(catalogue, ensure_ascii=False, separators=(",", ":")).encode()
        atomic(OUT / "catalogue-to-publish.json", catalogue)
        put("atlas-public/catalogue.json", raw, etag)
        atomic(PUBLIC / "catalogue.json", catalogue)
        atomic(PUBLIC / "catalogue-with-scores.json", catalogue)
        ledger_path = PUBLIC / "ledger.json"
        if ledger_path.exists():
            ledger = json.loads(ledger_path.read_text())
            for meta, record in zip(manifest["papers"], records, strict=True):
                if meta["pdf_sha256"] in ledger:
                    ledger[meta["pdf_sha256"]]["item"] = record["item"]
            atomic(ledger_path, ledger)
        receipt = {
            "published_at": catalogue["published_at"],
            "papers": 125,
            "tuples": 25,
            "total_catalogue_papers": len(catalogue["items"]),
            "phrases": sum(r["phrases"] for r in records),
            "recipe_sha256": manifest["recipe_sha256"],
            "catalogue_sha256": digest(raw),
            "verified": True,
            "site": SITE + "/?model=laya",
        }
        atomic(OUT / "publication-receipt.json", receipt)
        print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
