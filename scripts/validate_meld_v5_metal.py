"""Validate and benchmark the fused GELU/gate kernel against compiled MLX."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import mlx.core as mx
from open_text_detector_backend.providers.meld_v5_metal import geglu
from open_text_detector_backend.providers.meld_v8_mlx import MeldV8MLX

reference = mx.compile(lambda a, b: MeldV8MLX.activation(None, a, b))
rows = []
for dtype in [mx.float32, mx.float16]:
    mx.random.seed(42)
    a = mx.random.normal((2048, 2624)).astype(dtype)
    g = mx.random.normal(a.shape).astype(dtype)
    ref = reference(a, g)
    actual = geglu(a, g)
    mx.eval(ref, actual)
    error = float(mx.max(mx.abs(ref.astype(mx.float32) - actual.astype(mx.float32))))
    assert error < (1e-5 if dtype == mx.float32 else 0.016), error
    row = {"dtype": str(dtype), "max_error": error}
    for name, f in [("compiled", reference), ("custom", geglu)]:
        for _ in range(3):
            mx.eval(f(a, g))
        t = time.perf_counter()
        for _ in range(30):
            mx.eval(f(a, g))
        row[name + "_ms"] = (time.perf_counter() - t) * 1000 / 30
    rows.append(row)
(ROOT / "research/benchmarks/meld-v5/geglu-micro.json").write_text(
    json.dumps(rows, indent=2)
)
print(json.dumps(rows))
