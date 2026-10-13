"""Offline detector scoring of already reviewed Arena responses; no network or credentials."""

import argparse
import json
import time
from datetime import datetime, timezone

from arena20 import B, read, sha, write
from arena_review import R, collect
from metrics import report
from prepare import ROOT, record, validate
from run import build_model, convert


def main(a):
    dest = R / a.model
    dest.mkdir(exist_ok=True)
    predictions = dest / "predictions.jsonl"
    if predictions.exists():
        raise ValueError("Output already exists; refuse duplicate run")
    sources = list(B.glob("*.py")) + list(
        (ROOT / "backend/open_text_detector_backend/providers").glob("*.py")
    )
    model_dir = (
        ROOT
        / "models"
        / ("meld-v5" if a.model == "meld" else "editlens-qwen3-4b-merged-v3")
    )
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "model": a.model,
        "device": a.device,
        "precision": a.precision,
        "human_threshold": a.human_threshold,
        "ai_threshold": a.ai_threshold,
        "model_manifest": json.loads(
            (model_dir / "download-manifest.json").read_text()
        ),
        "code": {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in sources},
        "scope": "Local excerpt-reviewed prose variant, native input/output token gate unverified. Not Pangram API or an exact replication.",
    }
    for p in sources:
        target = dest / "code" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    model, cfg = build_model(a)
    done = []
    seen = set()
    start = time.monotonic()
    while time.monotonic() - start < 7200:
        rs = collect()
        reviews = (
            read(R / "content_reviews.jsonl")
            if (R / "content_reviews.jsonl").exists()
            else []
        )
        decisions = {(r["model_id"], r["prompt_id"]): r for r in reviews}
        eligible = []
        pending = 0
        for r in rs:
            if (
                not r["success"]
                or not r["audit"]["length_ok"]
                or not r["audit"]["not_truncated"]
            ):
                continue
            k = (r["model_id"], r["prompt_id"])
            review = decisions.get(k)
            if not review:
                pending += 1
                continue
            if not review["eligible"]:
                continue
            text = r["audit"]["text"]
            assert review["response_sha256"] == sha(text.encode())
            row = record(
                "arena20",
                sha((k[0] + "\0" + k[1]).encode()),
                text,
                "ai",
                generator=k[0],
                cohort=k[0],
                group_id=k[1],
                prompt_id=k[1],
                response_sha256=sha(text.encode()),
                task="ai_detection",
            )
            validate([row])
            eligible.append(row)
        for row in eligible:
            if row["id"] in seen:
                continue
            t = time.monotonic()
            try:
                out = convert(a, row, model.predict(cfg, row["text"]))
            except Exception as exc:
                out = {k: v for k, v in row.items() if k != "text"}
                out["error"] = type(exc).__name__ + ": " + str(exc)
            out["seconds"] = time.monotonic() - t
            with predictions.open("a") as f:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                f.flush()
            done.append(out)
            seen.add(row["id"])
            print(
                f"{len(done)} scored, {out.get('prediction', out.get('error'))}, {out['seconds']:.2f}s",
                flush=True,
            )
            if len(done) >= 3 and all("error" in r for r in done[-3:]):
                raise RuntimeError("Three consecutive detector failures")
        if len(rs) == 440 and pending == 0:
            break
        time.sleep(5)
    else:
        raise RuntimeError("Timed out waiting for generation/review completion")
    write(dest / "scored_inputs.jsonl", eligible)
    manifest.update(
        finished_utc=datetime.now(timezone.utc).isoformat(),
        scored=len(done),
        inputs_sha256=sha((dest / "scored_inputs.jsonl").read_bytes()),
        reviews_sha256=sha((R / "content_reviews.jsonl").read_bytes()),
    )
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (dest / "metrics.json").write_text(json.dumps(report(done), indent=2) + "\n")
    print("Local scoring complete", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["meld", "editlens"], required=True)
    a = p.parse_args()
    a.device = "mps"
    a.precision = "float32"
    a.batch_size = 1
    a.human_threshold = 0.1
    a.ai_threshold = 0.8
    main(a)
