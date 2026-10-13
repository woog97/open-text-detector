"""Frozen five-per-conference/year Laya publication batch; resumable local results."""

import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from calibrate_laya_document_context import apply
from tune_laya_meld import OUT as STUDY
from tune_laya_meld import Laya, logit, predict

OUT = ROOT / "research/classifications/laya-atlas-five-per-year"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "results").mkdir(exist_ok=True)


def atomic(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    temp.replace(path)


def main():
    profile = json.loads((STUDY / "frozen-profile.json").read_text())
    recipe = json.loads((STUDY / "document-context-followup.json").read_text())
    recipe = {k: recipe[k] for k in ["columns", "model", "variant", "status"]}
    fingerprint = hashlib.sha256(
        json.dumps([profile, recipe], sort_keys=True).encode()
    ).hexdigest()
    metadata = pq.read_table(
        ROOT / "research/exports/paper-text-hf/data/batch-001.parquet",
        columns=[
            "pdf_sha256",
            "text_sha256",
            "conference",
            "year",
            "title",
            "page_count",
        ],
    ).to_pylist()
    by_pdf = {r["pdf_sha256"]: r for r in metadata}
    catalogue = json.loads(
        (ROOT / "research/exports/atlas-public/catalogue-with-scores.json").read_text()
    )
    groups = defaultdict(dict)
    for item in catalogue["items"]:
        pdf = Path(item["pdf_key"]).stem
        if pdf not in by_pdf or "v8" not in item.get("score_summaries", {}):
            continue
        row = by_pdf[pdf]
        if (
            4 <= row["page_count"] <= 40
            and sum(item["score_summaries"]["v8"]["histogram"]) >= 80
        ):
            groups[row["conference"], row["year"]][pdf] = row
    selected = []
    for key, pool in sorted(groups.items()):
        ranked = sorted(
            pool.values(),
            key=lambda r: hashlib.sha256(
                ("laya-meld-v8-study-v1" + r["pdf_sha256"]).encode()
            ).hexdigest(),
        )
        assert len(ranked) >= 5, key
        selected.extend(ranked[:5])
    manifest = {
        "model_id": "laya",
        "model_name": "Laya · Experimental MELD alignment",
        "mode": "document_context",
        "profile": profile,
        "recipe": recipe,
        "recipe_sha256": fingerprint,
        "selection": "First five by the existing study hash order within each conference/year; 4–40 pages and >=80 MELD scored spans. Selection does not use high/low score.",
        "papers": selected,
    }
    path = OUT / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest, "Frozen batch changed"
    else:
        atomic(path, manifest)
    print(
        json.dumps(
            {
                "papers": len(selected),
                "tuples": len(groups),
                "recipe_sha256": fingerprint,
            }
        ),
        flush=True,
    )
    texts = {
        r["pdf_sha256"]: r["text"]
        for r in pq.read_table(
            ROOT / "research/exports/paper-text-hf/data/batch-001.parquet",
            columns=["pdf_sha256", "text"],
            filters=[("pdf_sha256", "in", [r["pdf_sha256"] for r in selected])],
        ).to_pylist()
    }
    p = Laya(
        model_dir=ROOT / "models", runtime="mlx", precision="float16", batch_size=4
    )
    p._load()
    for n, meta in enumerate(selected, 1):
        dest = OUT / "results" / (meta["pdf_sha256"] + ".json")
        if dest.exists():
            saved = json.loads(dest.read_text())
            assert (
                saved["text_sha256"] == meta["text_sha256"]
                and saved["recipe_sha256"] == fingerprint
            )
            print(
                json.dumps({"done": n, "total": len(selected), "cached": True}),
                flush=True,
            )
            continue
        begin = time.perf_counter()
        text = texts[meta["pdf_sha256"]]
        assert hashlib.sha256(text.encode()).hexdigest() == meta["text_sha256"]
        rows = p.targets(text)
        prior_path = STUDY / "evaluation" / ("full-" + meta["pdf_sha256"] + ".json")
        reused = False
        if prior_path.exists():
            prior = json.loads(prior_path.read_text())
            assert (
                prior["text_sha256"] == meta["text_sha256"]
                and prior["variant"] == profile["variant"]
                and prior["profile"] == profile["id"]
            )
            assert [(r["start"], r["end"]) for r in rows] == [
                (s["start"], s["end"]) for s in prior["segments"]
            ]
            raw = np.array([s["raw_laya_score"] for s in prior["segments"]])
            reused = True
        else:
            raw = np.array(
                predict(
                    p,
                    {"text": text, "rows": rows},
                    list(range(len(rows))),
                    profile["input"],
                )
            )
        indices = sorted(
            set(np.linspace(0, len(rows) - 1, min(24, len(rows)), dtype=int).tolist())
        )
        document_prior = float(logit(raw[indices]).mean())
        features = np.column_stack([logit(raw), np.full(len(raw), document_prior)])
        scores = apply(features[:, recipe["columns"]], recipe["model"])
        assert np.isfinite(scores).all() and np.isfinite(raw).all()
        weights = [len("".join(text[r["start"] : r["end"]].split())) for r in rows]
        result = {
            **meta,
            "recipe_sha256": fingerprint,
            "score": float(np.average(scores, weights=weights)),
            "score_type": "experimental_meld_v8_alignment",
            "notice": "Experimental Laya alignment to MELD v8. Weak measured agreement; not validated authorship detection.",
            "inference": {
                "revision": profile["student_revision"],
                "mode": "document_context",
                "profile": profile["id"],
                "runtime": "mlx",
                "precision": "float16",
                "head_precision": "float32",
                "batch_size": 4,
                "pipeline_depth": 2,
                "document_prior": document_prior,
                "prior_indices": indices,
                "reused_verified_study_scores": reused,
            },
            "segments": [
                {
                    "start": r["start"],
                    "end": r["end"],
                    "score": float(s),
                    "raw_laya_score": float(v),
                }
                for r, s, v in zip(rows, scores, raw, strict=True)
            ],
            "seconds": time.perf_counter() - begin,
        }
        atomic(dest, result)
        print(
            json.dumps(
                {
                    "done": n,
                    "total": len(selected),
                    "conference": meta["conference"],
                    "year": meta["year"],
                    "phrases": len(rows),
                    "seconds": result["seconds"],
                    "reused": reused,
                }
            ),
            flush=True,
        )
    atomic(
        OUT / "complete.json",
        {
            "papers": len(selected),
            "tuples": len(groups),
            "recipe_sha256": fingerprint,
            "completed_at": time.time(),
        },
    )


if __name__ == "__main__":
    main()
