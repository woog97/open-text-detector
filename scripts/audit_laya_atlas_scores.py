"""Independent readout, identities, and sample-count checks before publication."""

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/classifications/laya-atlas-five-per-year"
m = json.loads((OUT / "manifest.json").read_text())
counts = Counter((p["conference"], p["year"]) for p in m["papers"])
assert len(counts) == 25 and set(counts.values()) == {5}
assert len({p["pdf_sha256"] for p in m["papers"]}) == 125
assert len({p["text_sha256"] for p in m["papers"]}) == 125
params = m["recipe"]["model"]
total = 0
error = 0.0
reused = 0
for p in m["papers"]:
    r = json.loads((OUT / "results" / (p["pdf_sha256"] + ".json")).read_text())
    record = json.loads((OUT / "staged" / (p["pdf_sha256"] + ".json")).read_text())
    blob = (OUT / "objects" / Path(record["item"]["detail_key"]).name).read_bytes()
    assert (
        hashlib.sha256(blob).hexdigest()
        == Path(record["item"]["detail_key"]).name.split(".")[0]
    )
    detail = json.loads(gzip.decompress(blob))
    report = next(x for x in detail["reports"] if x["model"].get("id") == "laya")
    text = report["text"]
    assert (
        hashlib.sha256(text.encode()).hexdigest()
        == p["text_sha256"]
        == r["text_sha256"]
    )
    assert r["recipe_sha256"] == m["recipe_sha256"] and record["verified"]
    raw = np.array([s["raw_laya_score"] for s in r["segments"]])
    assert np.isfinite(raw).all() and np.all((raw >= 0) & (raw <= 1))
    raw = np.clip(raw, 1e-6, 1 - 1e-6)
    x = np.log(raw / (1 - raw))
    indices = np.linspace(0, len(raw) - 1, min(24, len(raw)), dtype=int)
    assert r["inference"]["prior_indices"] == indices.tolist()
    prior = float(np.mean(x[indices]))
    assert abs(prior - r["inference"]["document_prior"]) < 1e-12
    features = np.column_stack([x, np.full(len(raw), prior)])[:, m["recipe"]["columns"]]
    z = (features - np.array(params["mean"])) / np.array(params["scale"])
    evidence = params["coefficients"][0] + sum(
        z[:, i] * coef for i, coef in enumerate(params["coefficients"][1:])
    )
    expected = 1 / (1 + np.exp(-np.clip(evidence, -40, 40)))
    actual = np.array([s["score"] for s in r["segments"]])
    error = max(error, float(np.max(abs(actual - expected))))
    assert error < 1e-12
    assert [s["score"] for s in report["result"]["segments"]] == actual.tolist()
    weights = np.array(
        [len("".join(text[s["start"] : s["end"]].split())) for s in r["segments"]]
    )
    assert abs(float(np.average(actual, weights=weights)) - r["score"]) < 1e-12
    reused += r["inference"]["reused_verified_study_scores"]
    total += len(raw)
receipt = {
    "papers": 125,
    "conference_year_pairs": 25,
    "papers_per_pair": 5,
    "phrases": total,
    "reused_verified_study_papers": reused,
    "freshly_inferred_papers": 125 - reused,
    "max_independent_calibration_error": error,
    "passed": True,
}
(OUT / "score-audit.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
