"""Paired kernel and scheduling comparison on frozen 256/512-token inputs."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from open_text_detector_backend.providers.laya_mlx import LayaMLX
from tune_laya_meld import OUT, VARIANTS, Laya, build_rows

p = Laya(
    model_dir=ROOT / "models",
    runtime="mlx",
    precision="float16",
    batch_size=4,
    pipeline_depth=0,
)
p._load()
manifest = json.loads((OUT / "manifest.json").read_text())
results = []
for variant in ["chatgpt_local", "baseline"]:
    rows = []
    for m in manifest["papers"][::6]:
        paper = json.loads((OUT / (m["pdf_sha256"] + ".json")).read_text())
        new, markers = build_rows(p, paper, m["screen_indices"][::3], VARIANTS[variant])
        rows.extend(new)
    p.markers = markers
    p.model = LayaMLX(ROOT / "models/laya", "float16")
    p.pipeline_depth, p.batch_size = 0, 4
    reference = np.array(p.score_rows(rows))
    for head in ["float32", "float16"]:
        p.model = LayaMLX(ROOT / "models/laya", "float16", head_precision=head)
        for batch, pipeline in [(4, 0), (4, 2), (8, 2), (16, 2)]:
            p.batch_size, p.pipeline_depth = batch, pipeline
            scores = np.array(p.score_rows(rows))
            timings = []
            for _ in range(3):
                start = time.perf_counter()
                p.score_rows(rows)
                timings.append(time.perf_counter() - start)
            record = {
                "variant": variant,
                "head_precision": head,
                "batch_size": batch,
                "pipeline": pipeline,
                "seconds": timings,
                "phrases_per_second": len(rows) / float(np.median(timings)),
                "max_score_error": float(np.max(abs(scores - reference))),
            }
            results.append(record)
            print(json.dumps(record), flush=True)
            (OUT / "optimization-pipeline.json").write_text(
                json.dumps(
                    {"phrases_per_variant": len(rows), "results": results}, indent=2
                )
            )
