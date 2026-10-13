"""Dependency-free evaluation; null means unavailable, never a fabricated zero."""

import math
from collections import Counter, defaultdict


def rate(k, n):
    if not n:
        return {"count": k, "n": n, "rate": None, "wilson95": None}
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {
        "count": k,
        "n": n,
        "rate": p,
        "wilson95": [max(0, center - half), min(1, center + half)],
    }


def ranking(pairs):
    """Exact empirical ROC with tied scores moved together; no interpolation."""
    pos = sum(y for y, s in pairs)
    neg = len(pairs) - pos
    if not pos or not neg:
        return {"auroc": None, "tpr_at_fpr": None, "human_n": neg, "ai_n": pos}
    ties = defaultdict(lambda: [0, 0])
    for y, s in pairs:
        if not math.isfinite(s):
            raise ValueError("Nonfinite score")
        ties[s][y] += 1
    tp = fp = 0
    points = [(0, 0)]
    auc = 0
    for s, (n, p) in sorted(ties.items(), reverse=True):
        newfp = fp + n
        newtp = tp + p
        auc += (newfp - fp) * (tp + newtp) / 2
        fp, tp = newfp, newtp
        points.append((fp / neg, tp / pos))
    return {
        "auroc": auc / (pos * neg),
        "tpr_at_fpr": {
            str(a): max(t for f, t in points if f <= a) for a in [0.001, 0.01, 0.05]
        },
        "human_n": neg,
        "ai_n": pos,
        "fpr_resolution": 1 / neg,
        "note": "Descriptive empirical test-set ROC; not a calibrated deployment threshold. Ties are indivisible.",
    }


def summarize(rows):
    ok = [r for r in rows if "error" not in r]
    binary = [r for r in ok if r["label"] in ("human", "ai")]
    human = [r for r in binary if r["label"] == "human"]
    ai = [r for r in binary if r["label"] == "ai"]
    tp = sum(r["prediction"] == "ai" for r in ai)
    fp = sum(r["prediction"] == "ai" for r in human)
    mixed = [r for r in ok if r["label"] == "mixed"]
    polish = [r for r in ok if r.get("task") == "polish"]
    out = {
        "attempted": len(rows),
        "scored": len(ok),
        "failures": len(rows) - len(ok),
        "prediction_counts": dict(Counter(r["prediction"] for r in ok)),
        "strict_fpr": rate(sum(r["prediction"] != "human" for r in human), len(human)),
        "strict_fnr": rate(sum(r["prediction"] != "ai" for r in ai), len(ai)),
        "strict_accuracy": sum(r["prediction"] == r["label"] for r in binary)
        / len(binary)
        if binary
        else None,
        "ai_f1": 2 * tp / (2 * tp + fp + len(ai) - tp) if ai else None,
        "mixed_recall": rate(
            sum(r["prediction"] == "mixed" for r in mixed), len(mixed)
        ),
        "polish_fully_ai_rate": rate(
            sum(r["prediction"] == "ai" for r in polish), len(polish)
        ),
        "ai_or_mixed_recall": rate(
            sum(r["prediction"] in ("ai", "mixed") for r in ai), len(ai)
        ),
        "ranking": ranking([(int(r["label"] == "ai"), r["score"]) for r in binary]),
    }
    loc = [r["localization"] for r in ok if r.get("localization")]
    if loc:
        c = {k: sum(v[k] for v in loc) for k in ["tp", "fp", "tn", "fn"]}
        n = sum(c.values())
        out["localization"] = {
            **c,
            "unit": "whitespace-delimited words, character-midpoint alignment",
            "accuracy": (c["tp"] + c["tn"]) / n if n else None,
            "precision": c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None,
            "recall": c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None,
            "document_fraction_mae": sum(v["fraction_error"] for v in loc) / len(loc),
            "documents": len(loc),
        }
    fractions = [
        r
        for r in ok
        if r.get("target_fraction") is not None
        and r.get("predicted_fraction") is not None
    ]
    out["fraction_mae"] = (
        sum(abs(r["target_fraction"] - r["predicted_fraction"]) for r in fractions)
        / len(fractions)
        if fractions
        else None
    )
    out["fraction_n"] = len(fractions)
    out["mean_target_fraction"] = (
        sum(r["target_fraction"] for r in fractions) / len(fractions)
        if fractions
        else None
    )
    out["mean_predicted_fraction"] = (
        sum(r["predicted_fraction"] for r in fractions) / len(fractions)
        if fractions
        else None
    )
    return out


def report(rows):
    groups = defaultdict(list)
    for r in rows:
        for key in [
            "dataset",
            "cohort",
            "generator",
            "domain",
            "language",
            "attack",
            "version",
            "block_size",
            "length_bucket",
        ]:
            if r.get(key) is not None:
                groups[(r["dataset"], key, str(r[key]))].append(r)
    summaries = {" / ".join(k): summarize(v) for k, v in sorted(groups.items())}
    for (dataset, key, value), subset in groups.items():
        if key not in ("generator", "attack", "cohort"):
            continue
        positives = [r for r in subset if r["label"] == "ai" and "error" not in r]
        controls = [
            r
            for r in rows
            if r["dataset"] == dataset
            and r["label"] == "human"
            and "error" not in r
            and r.get("task") != "polish"
        ]
        # Do not count duplicate human controls from overlapping test files more than once in this supplemental curve.
        controls = list({r["text_sha256"]: r for r in controls}.values())
        if positives and controls:
            summaries[f"{dataset} / {key} / {value}"][
                "ranking_against_dataset_human_controls"
            ] = ranking(
                [(1, r["score"]) for r in positives]
                + [(0, r["score"]) for r in controls]
            )
    return {
        "overall": summarize(rows),
        "groups": summaries,
        "limitations": [
            "Do not compare pooled rates to Pangram private-corpus rates.",
            "Wilson intervals assume independent documents; shared sources and variants violate that assumption.",
            "Sampling is stratified and deliberately not population-weighted. Inspect per-cohort results.",
            "Local detector decisions and fraction proxies are not Pangram predictions or calibrated authorship probabilities.",
        ],
    }
