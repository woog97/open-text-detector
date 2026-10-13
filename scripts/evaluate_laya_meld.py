"""Freeze a development-selected profile, evaluate untouched papers, and score the complete batch."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from tune_laya_meld import (
    OUT,
    VARIANTS,
    Laya,
    calibrate,
    logit,
    metrics,
    predict,
    sigmoid,
)


def select():
    report = json.loads((OUT / "development-results.json").read_text())
    variants = report["variants"]
    # Fixed selection objective balances absolute color agreement and within-paper localization.
    name = min(
        variants,
        key=lambda n: (
            variants[n]["paper_disjoint_cv"]["mae"]
            + 0.05
            * (1 - variants[n]["paper_disjoint_cv"]["mean_within_paper_spearman"])
        ),
    )
    result = variants[name]
    config = {
        "id": "meld-v8-match-study-v1",
        "variant": name,
        "input": VARIANTS.get(name),
        "calibration": result["calibration"],
        "selection_objective": "development paper-disjoint CV MAE + 0.05*(1 - mean within-paper Spearman)",
        "development_cv": result["paper_disjoint_cv"],
        "teacher_revision": "8990324abd92e1fa17072f6887ea1e5c1cef5abc",
        "student_revision": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
    }
    path = OUT / "frozen-profile.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Profile already frozen; do not silently retune on holdout")
    path.write_text(json.dumps(config, indent=2))
    return config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    config = select()
    print(json.dumps({"frozen_profile": config}), flush=True)
    manifest = json.loads((OUT / "manifest.json").read_text())
    p = Laya(
        model_dir=ROOT / "models", runtime="mlx", precision="float16", batch_size=4
    )
    p._load()
    output = OUT / "evaluation"
    output.mkdir(exist_ok=True)

    def run(paper, indices, name):
        if name == "order_ensemble":
            a = np.array(predict(p, paper, indices, VARIANTS["baseline"]))
            b = np.array(predict(p, paper, indices, VARIANTS["reversed_choices"]))
            return sigmoid((logit(a) + logit(b)) / 2).tolist()
        return predict(p, paper, indices, VARIANTS[name])

    target = []
    baseline = []
    winner = []
    ids = []
    for j, meta in enumerate(manifest["papers"]):
        if meta["split"] != "holdout":
            continue
        paper = json.loads((OUT / (meta["pdf_sha256"] + ".json")).read_text())
        dest = output / ("holdout-" + meta["pdf_sha256"] + ".json")
        indices = meta["screen_indices"]
        if dest.exists():
            r = json.loads(dest.read_text())
        else:
            a = run(paper, indices, "baseline")
            b = (
                a
                if config["variant"] == "baseline"
                else run(paper, indices, config["variant"])
            )
            r = {
                "pdf_sha256": meta["pdf_sha256"],
                "indices": indices,
                "baseline": a,
                "selected_raw": b,
                "selected_calibrated": calibrate(
                    logit(np.array(b)), config["calibration"]
                ).tolist(),
            }
            dest.write_text(json.dumps(r))
        target.extend(paper["rows"][i]["teacher_score"] for i in indices)
        baseline.extend(r["baseline"])
        winner.extend(r["selected_calibrated"])
        ids.extend([j] * len(indices))
    development = json.loads((OUT / "development-results.json").read_text())
    baseline_fit = development["variants"]["baseline"]["calibration"]
    dev_targets = []
    for meta in manifest["papers"]:
        if meta["split"] == "development":
            paper = json.loads((OUT / (meta["pdf_sha256"] + ".json")).read_text())
            dev_targets.extend(
                paper["rows"][i]["teacher_score"] for i in meta["screen_indices"]
            )
    summary = {
        "profile": config,
        "holdout_screen": {
            "baseline": metrics(baseline, target, np.array(ids)),
            "baseline_calibrated": metrics(
                calibrate(logit(np.array(baseline)), baseline_fit),
                target,
                np.array(ids),
            ),
            "constant_development_mean": metrics(
                np.full(len(target), np.mean(dev_targets)), target, np.array(ids)
            ),
            "selected": metrics(winner, target, np.array(ids)),
        },
    }
    (OUT / "evaluation-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["holdout_screen"]), flush=True)
    if not args.full:
        return
    for n, meta in enumerate(manifest["papers"]):
        paper = json.loads((OUT / (meta["pdf_sha256"] + ".json")).read_text())
        dest = output / ("full-" + meta["pdf_sha256"] + ".json")
        if dest.exists():
            continue
        begin = time.perf_counter()
        raw = run(paper, list(range(len(paper["rows"]))), config["variant"])
        scores = calibrate(logit(np.array(raw)), config["calibration"]).tolist()
        result = {
            "pdf_sha256": meta["pdf_sha256"],
            "text_sha256": meta["text_sha256"],
            "split": meta["split"],
            "profile": config["id"],
            "variant": config["variant"],
            "seconds": time.perf_counter() - begin,
            "segments": [
                {
                    "start": r["start"],
                    "end": r["end"],
                    "teacher_score": r["teacher_score"],
                    "raw_laya_score": a,
                    "score": b,
                    "teacher_token_weight": r["teacher_token_weight"],
                }
                for r, a, b in zip(paper["rows"], raw, scores, strict=True)
            ],
        }
        dest.write_text(json.dumps(result))
        print(
            json.dumps(
                {
                    "full_paper": n + 1,
                    "of": len(manifest["papers"]),
                    "conference": meta["conference"],
                    "year": meta["year"],
                    "phrases": len(raw),
                    "seconds": result["seconds"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
