"""Hardware sweep on fixed real inputs; never changes prompts or targets."""

import gc
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from open_text_detector_backend.providers.laya_mlx import LayaMLX
from tune_laya_meld import OUT, VARIANTS, Laya, build_rows

p = Laya(model_dir=ROOT / "models", runtime="mlx", precision="float16", batch_size=4)
p._load()
manifest = json.loads((OUT / "manifest.json").read_text())
rows = []
for m in manifest["papers"][::3]:
    paper = json.loads((OUT / (m["pdf_sha256"] + ".json")).read_text())
    new, markers = build_rows(
        p, paper, m["screen_indices"][::4], VARIANTS["chatgpt_local"]
    )
    rows.extend(new)
p.markers = markers
reference = np.array(p.score_rows(rows))
results = []
for attention, compile_graph in [("dense", True), ("tiled", True), ("dense", False)]:
    del p.model
    gc.collect()
    mx.clear_cache()
    p.model = LayaMLX(
        ROOT / "models/laya", "float16", attention=attention, compile=compile_graph
    )
    for batch in [1, 2, 4, 8, 16, 32]:
        p.batch_size = batch
        scores = np.array(p.score_rows(rows))
        times = []
        for _ in range(3):
            start = time.perf_counter()
            p.score_rows(rows)
            times.append(time.perf_counter() - start)
        record = {
            "attention": attention,
            "compiled": compile_graph,
            "batch_size": batch,
            "max_score_error": float(np.max(abs(scores - reference))),
            "seconds": times,
            "phrases_per_second": len(rows) / float(np.median(times)),
        }
        results.append(record)
        print(json.dumps(record), flush=True)
        (OUT / "optimization-sweep.json").write_text(
            json.dumps(
                {
                    "phrases": len(rows),
                    "lengths": [len(r["ids"]) for r in rows],
                    "results": results,
                },
                indent=2,
            )
        )
