"""Offline, resumable benchmark runner. Never reads the website's DB or credentials."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from metrics import report
from prepare import ROOT, B, digest, readjsonl, validate


def localization(text, truth, predicted):
    """Model-independent word units: midpoint in half-open Unicode character spans."""
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    target = pred = 0
    for m in re.finditer(r"\S+", text):
        mid = (m.start() + m.end()) / 2
        y = any(s <= mid < e for s, e in truth)
        p = any(s <= mid < e for s, e in predicted)
        counts["tp" if y and p else "fn" if y else "fp" if p else "tn"] += 1
        target += y
        pred += p
    n = sum(counts.values())
    return {**counts, "fraction_error": abs(target - pred) / n if n else 0}


def build_model(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(ROOT / "backend"))
    if args.model == "meld":
        from open_text_detector_backend.providers.meld import Meld

        model = Meld(
            args.device,
            ROOT / "models",
            precision=args.precision,
            batch_size=args.batch_size,
        )
        config = {}
    else:
        from open_text_detector_backend.providers.checkpoints import QWEN_ID
        from open_text_detector_backend.providers.editlens import EditLens

        model = EditLens(args.device, model_dir=ROOT / "models")
        config = {"model_id": QWEN_ID, "base_model_id": QWEN_ID}
    return model, config


def convert(args, row, result):
    score = result["score"]
    if not math.isfinite(score):
        raise ValueError("Nonfinite score")
    if args.model == "meld":
        score = result["raw_score"]
        threshold = result["thresholds"]["raw_ai_above"]
        # Force an experimental binary decision for short-text benchmarks; report scope flag separately.
        prediction = "ai" if score > threshold else "human"
        spans = [
            [t["start"], t["end"]]
            for t in result["segments"]
            if t["raw_score"] > threshold
        ]
        fraction = sum(e - s for s, e in spans) / len(row["text"])
        kind = "fraction of characters in above-threshold sentences; exploratory localization proxy"
    else:
        prediction = (
            "human"
            if score < args.human_threshold
            else "ai"
            if score >= args.ai_threshold
            else "mixed"
        )
        spans = [
            [t["start"], t["end"]]
            for t in result["segments"]
            if t["score"] >= args.ai_threshold
        ]
        fraction = score
        kind = "EditLens expected editing bucket; not literal provenance coverage"
    out = {k: v for k, v in row.items() if k not in ["text", "ai_spans"]}
    out.update(
        score=score,
        prediction=prediction,
        predicted_fraction=fraction,
        fraction_kind=kind,
        inference=result["inference"],
        below_recommended_length=args.model == "meld" and row["words"] < 100,
    )
    if "ai_spans" in row:
        out["localization"] = localization(row["text"], row["ai_spans"], spans)
    return out


def select(rows, limit):
    if not limit:
        return rows
    # Cycle labels first, then cohorts, so many AI-only cohorts cannot hide human controls.
    pools = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        pools[row["dataset"]][row["label"]][str(row.get("cohort"))].append(row)
    for labels in pools.values():
        for cohorts in labels.values():
            for items in cohorts.values():
                items.sort(key=lambda row: digest(row["id"]))
    out = []
    for dataset, labels in sorted(pools.items()):
        count = 0
        positions = defaultdict(int)
        while labels and count < limit:
            for label in sorted(labels):
                cohorts = labels[label]
                names = sorted(cohorts)
                cohort = names[positions[label] % len(names)]
                out.append(cohorts[cohort].pop(0))
                count += 1
                positions[label] += 1
                if not cohorts[cohort]:
                    del cohorts[cohort]
                if not cohorts:
                    del labels[label]
                if count == limit:
                    break
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["meld", "editlens"], required=True)
    p.add_argument(
        "--datasets",
        default="all",
        help="comma-separated prepared file stems, e.g. liang,epoch,local",
    )
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--precision", default="float32", choices=["float32", "float16", "bfloat16"]
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--limit-per-dataset",
        type=int,
        default=0,
        help="0 uses all prepared rows; a positive limit makes a pilot",
    )
    p.add_argument("--human-threshold", type=float, default=0.1)
    p.add_argument("--ai-threshold", type=float, default=0.8)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if not 0 <= a.human_threshold < a.ai_threshold <= 1:
        p.error("Require 0 <= human < AI threshold <= 1")
    if a.limit_per_dataset < 0 or a.batch_size < 1:
        p.error("Invalid limit or batch size")
    files = sorted((B / "data/prepared").glob("*.jsonl"))
    if a.datasets != "all":
        requested = set(a.datasets.split(","))
        files = [f for f in files if f.stem in requested]
        if {f.stem for f in files} != requested:
            p.error("Unknown or missing dataset")
    rows = validate([r for f in files for r in readjsonl(f)])
    rows = select(rows, a.limit_per_dataset)
    if not rows:
        p.error("No examples prepared")
    model_dir = (
        ROOT
        / "models"
        / ("meld-v5" if a.model == "meld" else "editlens-qwen3-4b-merged-v3")
    )
    code_files = list(B.glob("*.py")) + list(
        (ROOT / "backend/open_text_detector_backend/providers").glob("*.py")
    )
    config = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(a).items()
        if k not in ["output", "resume"]
    }
    config.update(
        data={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        code={
            str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in code_files
        },
        model_manifest=json.loads((model_dir / "download-manifest.json").read_text()),
        selection_ids=[r["id"] for r in rows],
    )
    fingerprint = digest(json.dumps(config, sort_keys=True))
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = a.output / "manifest.json"
    predictions = a.output / "predictions.jsonl"
    if manifest.exists():
        if not a.resume:
            p.error("Output exists; use --resume or a new output directory")
        if json.loads(manifest.read_text())["fingerprint"] != fingerprint:
            p.error("Data, code, or configuration changed: use a new output directory")
    else:
        if predictions.exists():
            p.error("Predictions exist without manifest")
        manifest.write_text(
            json.dumps(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "fingerprint": fingerprint,
                    "config": config,
                    "platform": platform.platform(),
                    "python": sys.version,
                    "packages": {
                        k: importlib.metadata.version(k)
                        for k in ["torch", "transformers", "safetensors"]
                    },
                    "notice": "Local baseline evaluation, not Pangram 4 reproduction. No paid APIs.",
                },
                indent=2,
            )
            + "\n"
        )
    if not a.resume:
        for source in code_files:
            archived = a.output / "code" / source.relative_to(ROOT)
            archived.parent.mkdir(parents=True, exist_ok=True)
            archived.write_bytes(source.read_bytes())
    done = list(readjsonl(predictions)) if predictions.exists() else []
    seen = {r["id"] for r in done}
    model, cfg = build_model(a)
    with predictions.open("a") as f:
        for i, row in enumerate(rows):
            if row["id"] in seen:
                continue
            start = time.perf_counter()
            try:
                out = convert(a, row, model.predict(cfg, row["text"]))
            except Exception as e:  # noqa: BLE001 -- persist per-example failures for the audit
                out = {k: v for k, v in row.items() if k not in ["text", "ai_spans"]}
                out["error"] = type(e).__name__ + ": " + str(e)
            out["seconds"] = time.perf_counter() - start
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            done.append(out)
            print(
                f"{i + 1}/{len(rows)} {row['dataset']} {out.get('prediction', out.get('error'))} {out['seconds']:.2f}s",
                flush=True,
            )
            if (
                "error" in out
                and len(done) >= 3
                and all("error" in r for r in done[-3:])
            ):
                print(
                    "Stopping after three consecutive failures; inspect errors.",
                    flush=True,
                )
                break
    summary = report(done)
    summary["complete"] = len(done) == len(rows)
    summary["selected"] = len(rows)
    (a.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Local detector benchmark results",
        "",
        f"Model: {a.model}. Completed {len(done)}/{len(rows)} selected examples.",
        "",
        "These are stratified pilot/subset results, not reproductions of Pangram headline numbers.",
        "",
        "| Dataset | Scored | Failed | Human N | Strict FPR | AI N | Strict FNR | AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(v):
        return "—" if v is None else f"{v:.4f}"

    for key, v in summary["groups"].items():
        if " / dataset / " not in key:
            continue
        lines.append(
            f"| {key.split(' / ')[0]} | {v['scored']} | {v['failures']} | {v['strict_fpr']['n']} | {fmt(v['strict_fpr']['rate'])} | {v['strict_fnr']['n']} | {fmt(v['strict_fnr']['rate'])} | {fmt(v['ranking']['auroc'])} |"
        )
    lines += [
        "",
        "Rates are fractions (0.01 = 1%). Mixed counts as error on pure binary data.",
        "See metrics.json for confidence intervals, cohort breakdowns, polish, editing, and localization.",
        "MELD uses its shipped binary threshold, including a forced research-only decision below 100 words.",
        "EditLens uses exploratory cutoffs 0.1/0.8 by default; its editing score is not a provenance percentage.",
        "Low-FPR ROC estimates on small human samples are descriptive and poorly resolved.",
        "MELD v5 postdates the Pangram report; its MELD-eval scores are not an independent model-generalization result.",
    ]
    (a.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    if not summary["complete"] or summary["overall"]["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
