"""Deterministic conference/year sample plus teacher-score extremes; exact text identities."""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from open_text_detector_backend.providers.laya import Laya
from open_text_detector_backend.result_codec import decode
from transformers import PreTrainedTokenizerFast

OUT = ROOT / "research/benchmarks/laya-meld-v8"
OUT.mkdir(exist_ok=True)
metadata = pq.read_table(
    ROOT / "research/exports/paper-text-hf/data/batch-001.parquet",
    columns=["pdf_sha256", "text_sha256", "conference", "year", "title", "page_count"],
).to_pylist()
by_pdf = {r["pdf_sha256"]: r for r in metadata}
cat = json.loads(
    (ROOT / "research/exports/atlas-public/catalogue-with-scores.json").read_text()
)
items = []
for item in cat["items"]:
    pdf = Path(item["pdf_key"]).stem
    if pdf not in by_pdf or "v8" not in item.get("score_summaries", {}):
        continue
    if not 4 <= by_pdf[pdf]["page_count"] <= 40:
        continue
    h = item["score_summaries"]["v8"]["histogram"]
    if sum(h) < 80:
        continue
    items.append(
        dict(
            by_pdf[pdf],
            histogram=h,
            mean_histogram_score=sum(max(0, i - 0.5) * n / 20 for i, n in enumerate(h))
            / sum(h),
            green_fraction=sum(h[:5]) / sum(h),
            red_fraction=sum(h[17:]) / sum(h),
        )
    )
groups = defaultdict(list)
for item in items:
    groups[item["conference"], item["year"]].append(item)
seed = "laya-meld-v8-study-v1"
selected = []
for key, group in sorted(groups.items()):
    item = min(
        group,
        key=lambda x: hashlib.sha256((seed + x["pdf_sha256"]).encode()).hexdigest(),
    ).copy()
    item["selection"] = "conference_year"
    selected.append(item)
used = {r["pdf_sha256"] for r in selected}
for kind, pool, reverse in [
    ("green_older", [r for r in items if r["year"] <= 2022], False),
    ("red_newer", [r for r in items if r["year"] >= 2025], True),
]:
    for item in sorted(pool, key=lambda r: r["mean_histogram_score"], reverse=reverse):
        if item["pdf_sha256"] not in used:
            selected.append(dict(item, selection=kind))
            used.add(item["pdf_sha256"])
            if sum(r["selection"] == kind for r in selected) == 5:
                break
# Paper-level separation fixed before any Laya predictions. Never use venue/year as features.
for kind in ["conference_year", "green_older", "red_newer"]:
    group = sorted(
        [r for r in selected if r["selection"] == kind],
        key=lambda r: hashlib.sha256(
            ("holdout-v1" + r["pdf_sha256"]).encode()
        ).hexdigest(),
    )
    n = 7 if kind == "conference_year" else 2
    for i, r in enumerate(group):
        r["split"] = "holdout" if i < n else "development"
existing_manifest = OUT / "manifest.json"
if existing_manifest.exists():
    previous = json.loads(existing_manifest.read_text())["papers"]
    signature = lambda rows: sorted(
        (r["pdf_sha256"], r["text_sha256"], r["split"], r["selection"]) for r in rows
    )
    if signature(previous) != signature(selected):
        raise ValueError(
            "Study sample is already frozen; use a new study directory for changed inputs"
        )

texts = pq.read_table(
    ROOT / "research/exports/paper-text-hf/data/batch-001.parquet",
    columns=["pdf_sha256", "text"],
    filters=[("pdf_sha256", "in", list(used))],
).to_pylist()
texts = {r["pdf_sha256"]: r["text"] for r in texts}
p = Laya(model_dir=ROOT / "models")
p.tokenizer = PreTrainedTokenizerFast.from_pretrained(
    ROOT / "models/laya/tokenizer", local_files_only=True
)
p.cfg = {"max_len": 512}
p._prepare_prefix()
records = {
    r["text_sha256"]: r
    for r in map(
        json.loads,
        (
            ROOT
            / "research/classifications/vast-complete-20260925/metadata/v8-results.jsonl"
        )
        .read_text()
        .splitlines(),
    )
}
for paper in selected:
    text = texts[paper["pdf_sha256"]]
    assert hashlib.sha256(text.encode()).hexdigest() == paper["text_sha256"]
    blob = (
        ROOT
        / "research/classifications/vast-complete-20260925/results/v8"
        / (paper["text_sha256"] + ".pgf")
    ).read_bytes()
    assert (
        hashlib.sha256(blob).hexdigest()
        == records[paper["text_sha256"]]["result"]["sha256"]
    )
    teacher = decode(blob)
    assert (
        teacher["inference"]["revision"] == "8990324abd92e1fa17072f6887ea1e5c1cef5abc"
    )
    assert teacher["text_sha256"] == paper["text_sha256"]
    rows = p.prepare(text)
    tokens = teacher["tokens"]
    j = 0
    for row in rows:
        a, b = row["start"], row["end"]
        while j < len(tokens) and tokens[j]["end"] <= a:
            j += 1
        k, total, weight = j, 0.0, 0.0
        while k < len(tokens) and tokens[k]["start"] < b:
            t = tokens[k]
            overlap = max(0, min(b, t["end"]) - max(a, t["start"]))
            w = overlap / (t["end"] - t["start"]) * t.get("token_count", 1)
            total += t["raw_score"] * w
            weight += w
            k += 1
        if not weight:
            raise ValueError("No teacher coverage")
        row["teacher_raw"] = total / weight
        row["teacher_score"] = float(1 / (1 + np.exp(-total / weight)))
        row["teacher_token_weight"] = weight
    # Uniform positional screening sample; independent of teacher score (no extreme-span leakage).
    screen = sorted(
        set(np.linspace(0, len(rows) - 1, min(24, len(rows)), dtype=int).tolist())
    )
    paper.update(
        phrases=len(rows),
        words=len(text.split()),
        teacher_score=teacher["score"],
        teacher_raw=teacher["raw_score"],
        teacher_label=teacher["label"],
        screen_indices=screen,
    )
    (OUT / (paper["pdf_sha256"] + ".json")).write_text(
        json.dumps({"paper": paper, "text": text, "rows": rows})
    )
manifest = {
    "seed": seed,
    "eligibility": "4-40 pages and >=80 scored MELD spans; exclude giant appendix/compilation outliers",
    "selection_policy": "1 hash-selected per conference/year; 5 lowest histogram means from <=2022, 5 highest from >=2025, excluding already selected",
    "screen_policy": "24 uniformly spaced phrases per paper, no teacher-dependent span selection",
    "target_policy": "sigmoid of token-count/overlap-weighted MELD v8 raw evidence per exact Laya phrase",
    "papers": selected,
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(
    json.dumps(
        {
            "papers": len(selected),
            "tuples": len(groups),
            "splits": {
                s: sum(p["split"] == s for p in selected)
                for s in ["development", "holdout"]
            },
            "phrases": sum(p["phrases"] for p in selected),
            "groups": list(map(list, sorted(groups))),
            "extremes": [
                {
                    k: r[k]
                    for k in ["conference", "year", "mean_histogram_score", "selection"]
                }
                for r in selected
                if r["selection"] != "conference_year"
            ],
        }
    )
)
