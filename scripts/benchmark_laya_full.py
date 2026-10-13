"""Alternating full-paper throughput and score checks against the previous execution path."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from open_text_detector_backend.providers.laya_mlx import LayaMLX
from tune_laya_meld import OUT, VARIANTS, Laya, build_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--pipeline", type=int, default=2)
    ap.add_argument("--head", default="float32")
    args = ap.parse_args()
    manifest = json.loads((OUT / "manifest.json").read_text())
    holdout = [m for m in manifest["papers"] if m["split"] == "holdout"]
    chosen = [min(holdout, key=lambda m: m["phrases"])]
    chosen += [
        min((m for m in holdout if m["selection"] == kind), key=lambda m: m["phrases"])
        for kind in ["green_older", "red_newer"]
    ]
    p = Laya(
        model_dir=ROOT / "models",
        runtime="mlx",
        precision="float16",
        batch_size=4,
        pipeline_depth=0,
    )
    p._load()
    old = p.model
    new = LayaMLX(ROOT / "models/laya", "float16", head_precision=args.head)
    output = []
    for meta in chosen:
        paper = json.loads((OUT / (meta["pdf_sha256"] + ".json")).read_text())
        start = time.perf_counter()
        original_rows = p.prepare(paper["text"])
        rows, markers = build_rows(
            p,
            {"text": paper["text"], "rows": original_rows},
            list(range(len(original_rows))),
            VARIANTS["chatgpt_local"],
        )
        previous_preparation = time.perf_counter() - start
        start = time.perf_counter()
        targets = p.targets(paper["text"])
        optimized_rows, optimized_markers = build_rows(
            p,
            {"text": paper["text"], "rows": targets},
            list(range(len(targets))),
            VARIANTS["chatgpt_local"],
        )
        preparation = time.perf_counter() - start
        assert markers == optimized_markers
        assert [r["ids"] for r in rows] == [r["ids"] for r in optimized_rows]
        p.markers = markers
        p.model, p.batch_size, p.pipeline_depth = old, 4, 0
        reference = np.array(p.score_rows(rows))
        p.model, p.batch_size, p.pipeline_depth = new, args.batch, args.pipeline
        actual = np.array(p.score_rows(rows))
        error = float(np.max(abs(actual - reference)))
        assert error < 0.005
        measurements = {"previous": [], "optimized": []}
        for repeat in range(3):
            for name in (
                ["previous", "optimized"]
                if repeat % 2 == 0
                else ["optimized", "previous"]
            ):
                p.model, p.batch_size, p.pipeline_depth = (
                    (old, 4, 0)
                    if name == "previous"
                    else (new, args.batch, args.pipeline)
                )
                start = time.perf_counter()
                scores = np.array(p.score_rows(rows))
                measurements[name].append(time.perf_counter() - start)
                assert (
                    np.max(abs(scores - (reference if name == "previous" else actual)))
                    < 1e-6
                )
        result = {
            "pdf_sha256": meta["pdf_sha256"],
            "title": meta["title"],
            "selection": meta["selection"],
            "phrases": len(rows),
            "preparation_seconds": preparation,
            "previous_preparation_seconds": previous_preparation,
            "max_raw_score_difference": error,
            "raw_color_changes": int(
                np.sum(
                    np.digitize(actual, [0.2, 0.8], right=True)
                    != np.digitize(reference, [0.2, 0.8], right=True)
                )
            ),
            "seconds": measurements,
            "speedup": float(
                np.median(measurements["previous"])
                / np.median(measurements["optimized"])
            ),
            "speedup_including_preparation": float(
                (previous_preparation + np.median(measurements["previous"]))
                / (preparation + np.median(measurements["optimized"]))
            ),
            "papers_per_minute_including_preparation": 60
            / (preparation + float(np.median(measurements["optimized"]))),
        }
        output.append(result)
        print(json.dumps(result), flush=True)
        (OUT / "optimization-full.json").write_text(
            json.dumps(
                {
                    "hardware": "Apple M4 Pro, 16 GPU cores, 48 GB",
                    "candidate": vars(args),
                    "papers": output,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
