"""Tune custom GEMM dispatch on v5's real dimensions; verify each against MLX."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import mlx.core as mx
import numpy as np
from open_text_detector_backend.providers.meld_v5_gemm import matmul

OUT = ROOT / "research/benchmarks/meld-v5-max"
prior = json.loads((OUT / "gemm-sweep.json").read_text())
rows = []
for k, n in [(1024, 3072), (1024, 1024), (1024, 5248), (2624, 1024)]:
    best = min(
        (
            r
            for r in prior
            if r["k"] == k and r["n"] == n and r.get("tile") and "ms" in r
        ),
        key=lambda r: r["ms"],
    )["tile"]
    tiles = [[*best, sw] for sw in [0, 1, 2, 3]]
    mx.random.seed(17)
    x = mx.random.normal((2048, k)).astype(mx.float16)
    w = (mx.random.normal((n, k)) * 0.03).astype(mx.float16)
    ref = x @ w.T
    mx.eval(x, w, ref)
    for tile in [None] + tiles:
        try:
            f = (
                (lambda x=x, w=w: x @ w.T)
                if tile is None
                else (lambda x=x, w=w, tile=tile: matmul(x, w, tile))
            )
            y = f()
            mx.eval(y)
            error = float(mx.max(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32))))
            assert error < 0.02, error
            for _ in range(3):
                mx.eval(f())
            ts = []
            for _ in range(12):
                t = time.perf_counter()
                mx.eval(f())
                ts.append(time.perf_counter() - t)
            row = {
                "k": k,
                "n": n,
                "tile": tile,
                "ms": float(np.median(ts)) * 1000,
                "max_error": error,
            }
        except (RuntimeError, ValueError, AssertionError) as e:
            row = {"k": k, "n": n, "tile": tile, "error": str(e)[-1200:]}
        rows.append(row)
        (OUT / "gemm-swizzle.json").write_text(json.dumps(rows, indent=2))
        print(row, flush=True)
