"""Exploratory follow-up: fit color boundaries on development papers only.

Designed after observing first-pass holdout collapse, so its holdout check is
explicitly exploratory and would need a fresh independent set before promotion.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from tune_laya_meld import OUT, calibrate, logit, metrics


def fit(x, y):
    # Select two ordered score cutoffs for balanced three-color agreement.
    candidates = np.unique(np.quantile(x, np.linspace(0, 1, 61)))
    labels = (y > 0.2).astype(int) + (y > 0.8).astype(int)
    best = None
    for low in candidates[:-1]:
        for high in candidates[candidates > low]:
            predicted = (x > low).astype(int) + (x > high).astype(int)
            balanced = float(
                np.mean(
                    [np.mean(predicted[labels == c] == c) for c in np.unique(labels)]
                )
            )
            # Prefer narrower raw-score interval only as deterministic tie-break, never use holdout.
            key = (balanced, -float(np.mean(abs(predicted - labels))))
            if best is None or key > best[0]:
                best = (key, float(low), float(high))
    _, low, high = best
    a = float(
        (logit(np.array([0.8]))[0] - logit(np.array([0.2]))[0])
        / (logit(np.array([high]))[0] - logit(np.array([low]))[0])
    )
    b = float(logit(np.array([0.2]))[0] - a * logit(np.array([low]))[0])
    return {"low_cutoff": low, "high_cutoff": high, "slope": a, "intercept": b}


def main():
    manifest = json.loads((OUT / "manifest.json").read_text())
    profile = json.loads((OUT / "frozen-profile.json").read_text())
    x = []
    y = []
    ids = []
    for j, paper in enumerate(manifest["papers"]):
        if paper["split"] != "development":
            continue
        source = json.loads((OUT / (paper["pdf_sha256"] + ".json")).read_text())
        pred = json.loads(
            (
                OUT
                / "screen"
                / (profile["variant"] + "-" + paper["pdf_sha256"] + ".json")
            ).read_text()
        )
        x.extend(pred["scores"])
        y.extend(source["rows"][i]["teacher_score"] for i in pred["indices"])
        ids.extend([j] * len(pred["scores"]))
    x = np.array(x)
    y = np.array(y)
    ids = np.array(ids)
    # Fixed paper folds; derive dense paper numbering as in the initial development study.
    mapping = {p: i for i, p in enumerate(np.unique(ids))}
    folds = np.array([mapping[p] % 4 for p in ids])
    cv = np.zeros(len(x))
    for fold in range(4):
        train = folds != fold
        test = ~train
        params = fit(x[train], y[train])
        cv[test] = calibrate(logit(x[test]), params)
    params = fit(x, y)
    report = {
        "status": "exploratory_followup_after_initial_holdout_review",
        "variant": profile["variant"],
        "calibration": params,
        "development_cv": metrics(cv, y, ids),
        "inference_weights_unchanged": True,
        "holdout_used_for_parameter_fitting": False,
    }
    testx = []
    testy = []
    testids = []
    for j, paper in enumerate(manifest["papers"]):
        if paper["split"] != "holdout":
            continue
        source = json.loads((OUT / (paper["pdf_sha256"] + ".json")).read_text())
        pred = json.loads(
            (
                OUT / "evaluation" / ("holdout-" + paper["pdf_sha256"] + ".json")
            ).read_text()
        )
        testx.extend(pred["selected_raw"])
        testy.extend(source["rows"][i]["teacher_score"] for i in pred["indices"])
        testids.extend([j] * len(pred["indices"]))
    report["exploratory_holdout"] = metrics(
        calibrate(logit(np.array(testx)), params), testy, np.array(testids)
    )
    (OUT / "color-followup.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
