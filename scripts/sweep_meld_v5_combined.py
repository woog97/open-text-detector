"""Combine promising scheduler and kernel settings; no reduced-precision weights."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/benchmarks/meld-v5-max"
# Further GPU-threadgroup ordering tests run before full-paper candidates.
with (OUT / "swizzle.log").open("w") as log:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/tune_meld_v5_swizzle.py")],
        stdout=log,
        stderr=subprocess.STDOUT,
        check=True,
    )
gemm = json.loads((OUT / "gemm-swizzle.json").read_text())
tiles = {}
for k, n in [(1024, 3072), (1024, 1024), (1024, 5248), (2624, 1024)]:
    r = min(
        (
            r
            for r in gemm
            if r["k"] == k and r["n"] == n and r.get("tile") and "ms" in r
        ),
        key=lambda r: r["ms"],
    )
    tiles[f"{k}x{n}"] = r["tile"]
env = {"MLX_MAX_OPS_PER_BUFFER": "256", "MLX_MAX_MB_PER_BUFFER": "256"}
settings = [
    ("paired-control", 0, 1, {}, {}),
    ("combo-pipeline", 2, 1, {}, env),
    ("combo-cache", 2, 1, {"cache_mb": 4096}, env),
    ("custom-gemm", 2, 1, {"gemm_tiles": tiles}, env),
    ("streams-2", 4, 2, {}, {}),
    ("streams-4", 8, 4, {}, {}),
    ("streams-2-cache", 4, 2, {"cache_mb": 4096}, env),
]
rows = []
for tag, depth, streams, advanced, env in settings:
    args = [
        sys.executable,
        str(ROOT / "scripts/benchmark_meld_v5_max.py"),
        "--runtime",
        "mlx",
        "--precision",
        "float16",
        "--attention",
        "tiled",
        "--papers",
        "2",
        "--repeats",
        "2",
        "--tag",
        tag,
        "--pipeline",
        str(depth),
        "--streams",
        str(streams),
        "--advanced",
        json.dumps(advanced),
    ]
    with (OUT / (tag + ".log")).open("w") as log:
        code = subprocess.run(
            args,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            env={**os.environ, **env},
        ).returncode
    if code:
        row = {"tag": tag, "error": code}
    else:
        data = json.loads((OUT / (tag + ".json")).read_text())
        row = {
            "tag": tag,
            "seconds": sum(r["seconds"] for r in data),
            "depth": depth,
            "streams": streams,
            "advanced": advanced,
            "env": env,
            "doc_changes": sum(not r["document_label_matches"] for r in data),
            "sentence_changes": sum(r["sentence_label_changes"] for r in data),
            "max_doc_error": max(r["document_score_error"] for r in data),
        }
    rows.append(row)
    (OUT / "combined-sweep.json").write_text(json.dumps(rows, indent=2))
    print(row, flush=True)
