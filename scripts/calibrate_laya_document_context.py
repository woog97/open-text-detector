"""Exploratory context calibration using only Laya predictions at inference.

The document feature is the mean log-odds over 24 uniformly spaced target phrases,
not a teacher score, venue, date, filename or training-paper identifier.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from tune_laya_meld import OUT, logit, metrics, sigmoid


def fit(x, y, ridge):
    mean = x.mean(0)
    scale = np.maximum(x.std(0), 1e-6)
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    beta = np.zeros(design.shape[1])
    penalty = np.diag([0] + [ridge] * x.shape[1])

    def objective(b):
        z = design @ b
        return float(np.sum(np.logaddexp(0, z) - y * z) + 0.5 * b @ penalty @ b)

    for _ in range(60):
        prob = sigmoid(design @ beta)
        grad = design.T @ (prob - y) + penalty @ beta
        hessian = (
            design.T @ ((prob * (1 - prob))[:, None] * design)
            + penalty
            + np.eye(len(beta)) * 1e-9
        )
        step = np.linalg.solve(hessian, grad)
        rate = 1.0
        while rate > 1e-7 and objective(beta - rate * step) > objective(beta):
            rate *= 0.5
        beta -= rate * step
        if np.max(abs(rate * step)) < 1e-8:
            break
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": beta.tolist(),
        "ridge": ridge,
    }


def apply(x, model):
    a = (np.asarray(x) - model["mean"]) / model["scale"]
    return sigmoid(
        np.column_stack([np.ones(len(a)), a]) @ np.array(model["coefficients"])
    )


def main():
    manifest = json.loads((OUT / "manifest.json").read_text())
    profile = json.loads((OUT / "frozen-profile.json").read_text())
    x = []
    y = []
    ids = []
    testx = []
    testy = []
    testids = []
    document_features = {}
    for j, paper in enumerate(manifest["papers"]):
        source = json.loads((OUT / (paper["pdf_sha256"] + ".json")).read_text())
        if paper["split"] == "development":
            pred = json.loads(
                (
                    OUT
                    / "screen"
                    / (profile["variant"] + "-" + paper["pdf_sha256"] + ".json")
                ).read_text()
            )
            raw = np.array(pred["scores"])
        else:
            pred = json.loads(
                (
                    OUT / "evaluation" / ("holdout-" + paper["pdf_sha256"] + ".json")
                ).read_text()
            )
            raw = np.array(pred["selected_raw"])
        local = logit(raw)
        prior = float(local.mean())
        document_features[paper["pdf_sha256"]] = prior
        features = np.column_stack([local, np.full(len(local), prior)])
        targets = [source["rows"][i]["teacher_score"] for i in pred["indices"]]
        if paper["split"] == "development":
            x.extend(features)
            y.extend(targets)
            ids.extend([j] * len(raw))
        else:
            testx.extend(features)
            testy.extend(targets)
            testids.extend([j] * len(raw))
    x = np.array(x)
    y = np.array(y)
    ids = np.array(ids)
    testx = np.array(testx)
    mapping = {p: i for i, p in enumerate(np.unique(ids))}
    folds = np.array([mapping[p] % 4 for p in ids])
    candidates = {}
    for name, columns in [("document_prior", [1]), ("local_and_document", [0, 1])]:
        for ridge in [0.1, 1.0, 10.0, 100.0]:
            pred = np.zeros(len(y))
            for fold in range(4):
                train = folds != fold
                test = ~train
                model = fit(x[train][:, columns], y[train], ridge)
                pred[test] = apply(x[test][:, columns], model)
            candidates[f"{name}_{ridge}"] = {
                "columns": columns,
                "ridge": ridge,
                "cv": metrics(pred, y, ids),
            }
    selected = min(
        candidates,
        key=lambda key: (
            candidates[key]["cv"]["mae"]
            + 0.05 * (1 - candidates[key]["cv"]["mean_within_paper_spearman"])
        ),
    )
    candidate = candidates[selected]
    columns = candidate["columns"]
    model = fit(x[:, columns], y, candidate["ridge"])
    report = {
        "status": "exploratory_followup_after_initial_holdout_review",
        "variant": profile["variant"],
        "feature_policy": "local logit plus document mean logit from 24 fixed uniformly spaced Laya target scores; no teacher inference features",
        "selection": "development-only paper-disjoint CV MAE + 0.05*(1-within-paper Spearman)",
        "selected": selected,
        "columns": columns,
        "model": model,
        "candidates": candidates,
        "exploratory_holdout": metrics(
            apply(testx[:, columns], model), testy, np.array(testids)
        ),
        "document_features": document_features,
    }
    (OUT / "document-context-followup.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps({k: report[k] for k in ["selected", "model", "exploratory_holdout"]})
    )
    print(json.dumps({k: v["cv"]["mae"] for k, v in candidates.items()}))


if __name__ == "__main__":
    main()
