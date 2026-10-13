"""Offline detector scoring of mechanically screened Arena-100 responses; no network or credentials."""

import argparse
import json
import time
from datetime import datetime, timezone

from arena20 import B, read, sha, write
from arena100_results import R, audit, collect
from metrics import report
from prepare import ROOT, record, validate
from run import build_model, convert


def eligible_ids(rows):
    return {
        "arena100:" + sha((r["model_id"] + "\0" + r["prompt_id"]).encode())
        for r in rows
        if audit(r)[0]
    }


def fully_scored(latest_rows, seen):
    return (
        len(latest_rows) == 1200
        and all(r["success"] for r in latest_rows)
        and eligible_ids(latest_rows) == seen
    )


def main(a):
    dest = R / a.model
    dest.mkdir(exist_ok=True)
    predictions = dest / "predictions.jsonl"
    if predictions.exists() and not a.resume:
        raise ValueError("Output already exists; refuse duplicate run")
    if a.resume and not predictions.exists():
        raise ValueError("No existing predictions to resume")
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
        "scope": "Mechanical screen only: successful stop finish and >=50 words. No full substantive-prose or refusal-only adjudication. Native token gate unverified; not Pangram replication.",
    }
    done = read(predictions) if a.resume else []
    seen = {r["id"] for r in done}
    assert len(seen) == len(done), "Duplicate existing predictions"
    assert not any("error" in r for r in done), (
        "Existing errors require explicit reconciliation"
    )
    code_dir = dest / "code"
    if a.resume:
        old = json.loads((dest / "manifest.json").read_text())
        for key in [
            "model",
            "device",
            "precision",
            "human_threshold",
            "ai_threshold",
            "model_manifest",
        ]:
            assert old[key] == manifest[key], f"Resume configuration changed: {key}"
        number = len(old.get("resumptions", [])) + 1
        receipt = dest / f"resume_{number:03d}"
        receipt.mkdir(exist_ok=False)
        for name in ["manifest.json", "scored_inputs.jsonl", "metrics.json"]:
            if (dest / name).exists():
                (receipt / name).write_bytes((dest / name).read_bytes())
        code_dir = receipt / "code"
        old.setdefault("resumptions", []).append(
            {
                "at": manifest["started_utc"],
                "code": manifest["code"],
                "existing_predictions": len(done),
                "existing_predictions_sha256": sha(predictions.read_bytes()),
                "reason": "Complete the remaining eligible responses after refreshing the generation snapshot",
            }
        )
        old.pop("finished_utc", None)
        manifest = old
    for p in sources:
        target = code_dir / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    model, cfg = build_model(a)
    existing_by_id = {r["id"]: r for r in done}
    start = time.monotonic()
    while time.monotonic() - start < 86400:
        rs = collect()
        eligible = []
        for r in rs:
            if not audit(r)[0]:
                continue
            k = (r["model_id"], r["prompt_id"])
            text = r["audit"]["text"]
            row = record(
                "arena100",
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
                assert (
                    existing_by_id[row["id"]]["response_sha256"]
                    == row["response_sha256"]
                ), "Previously scored response changed"
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
            existing_by_id[row["id"]] = out
            print(
                f"{len(done)} scored, {out.get('prediction', out.get('error'))}, {out['seconds']:.2f}s",
                flush=True,
            )
            if len(done) >= 3 and all("error" in r for r in done[-3:]):
                raise RuntimeError("Three consecutive detector failures")
        if (R / "sync_complete.json").exists():
            # Generation may have added successful responses while this older
            # snapshot was being scored. Refresh before deciding to terminate.
            if fully_scored(collect(), seen):
                break
            continue
        time.sleep(5)
    else:
        raise RuntimeError("Timed out waiting for generation/review completion")
    write(dest / "scored_inputs.jsonl", eligible)
    manifest.update(
        finished_utc=datetime.now(timezone.utc).isoformat(),
        scored=len(done),
        inputs_sha256=sha((dest / "scored_inputs.jsonl").read_bytes()),
        screen="Mechanical only; substantive prose/refusal-only and native-token gates unverified",
    )
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    metrics = report(done)
    metrics["limitations"] = [
        manifest["scope"],
        "100 repeated prompt clusters; pooled independent intervals suppressed. No FPR or AUROC without human controls.",
    ]
    for group in [metrics["overall"]] + [
        v
        for k, v in metrics["groups"].items()
        if " / generator / " not in k and " / cohort / " not in k
    ]:
        for value in group.values():
            if isinstance(value, dict) and "wilson95" in value:
                value["wilson95"] = None
    (dest / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("Local scoring complete", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["meld", "editlens"], required=True)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    a.device = "mps"
    a.precision = "float32"
    a.batch_size = 1
    a.human_threshold = 0.1
    a.ai_threshold = 0.8
    main(a)
