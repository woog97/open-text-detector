"""Alternate old/new scheduling on one warm model to control thermal/time drift."""

import json
import os
import sys
import time
from pathlib import Path

for key in ["MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER", "MLX_METAL_FAST_SYNCH"]:
    os.environ.pop(key, None)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import mlx.core as mx
import numpy as np
from open_text_detector_backend.pdf_extraction import read_text, text_hash
from open_text_detector_backend.providers.meld_v5_optimized import MeldV5Runner
from open_text_detector_backend.result_codec import decode, encode

OUT = ROOT / "research/benchmarks/meld-v5-max"
OLD = ROOT / "research/benchmarks/meld-v5"
model = MeldV5Runner(
    ROOT / "models/meld-v5", runtime="mlx", precision="float16", attention="tiled"
)
model.predict(
    "The scientific method is based on observations and controlled experiments. " * 400
)
rows = []
for i, paper in enumerate(json.loads((OLD / "papers.json").read_text())):
    text = read_text(ROOT / paper["text_file"])
    assert text_hash(text) == paper["text_sha256"]
    reference = decode((OLD / f"optimized-{i}.pgf").read_bytes())
    times = {"control": [], "pipeline": []}
    results = {}
    for repeat in range(3):
        order = (
            ["control", "pipeline"]
            if (i + repeat) % 2 == 0
            else ["pipeline", "control"]
        )
        for mode in order:
            model.pipeline_depth = 0 if mode == "control" else 2
            mx.set_cache_limit(1 << 30)
            mx.synchronize()
            t = time.perf_counter()
            result = model.predict(text)
            seconds = time.perf_counter() - t
            times[mode].append(seconds)
            assert (
                result["raw_score"] == reference["raw_score"]
                and result["label"] == reference["label"]
            )
            assert (
                result["tokens"] == reference["tokens"]
                and result["segments"] == reference["segments"]
            )
            result["source_maps"] = reference["source_maps"]
            results[mode] = result
    row = {
        "paper": paper["title"],
        "source_tokens": result["inference"]["source_tokens"],
        "times": times,
        "control_seconds": float(np.median(times["control"])),
        "pipeline_seconds": float(np.median(times["pipeline"])),
        "all_scores_and_offsets_identical": True,
        "peak_active_bytes": mx.get_peak_memory(),
        "cache_bytes": mx.get_cache_memory(),
    }
    rows.append(row)
    for mode, result in results.items():
        (OUT / f"paired-{mode}-{i}.pgf").write_bytes(encode(result, level=9))
    (OUT / "paired.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(row), flush=True)
