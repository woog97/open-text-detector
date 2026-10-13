import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/benchmarks/meld-v5"
rows = []
settings = [("torch", "float16", 1, "dense", {})]
settings += [
    ("torch", dtype, b, "tiled", {})
    for dtype in ["float32", "float16", "bfloat16"]
    for b in [1, 2, 4, 8]
]
settings += [
    ("mlx", dtype, b, "tiled", {})
    for dtype in ["float32", "float16", "bfloat16"]
    for b in [1, 2, 4, 8]
]
settings += [("mlx", "float16", 1, "dense", {}), ("torch", "float16", 1, "metal", {})]
settings += [
    (
        "torch",
        "float16",
        1,
        "tiled",
        {"PYTORCH_MPS_PREFER_METAL": "1", "PYTORCH_MPS_FAST_MATH": "1"},
    )
]
for runtime, dtype, batch, attention, env in settings:
    tag = (
        f"screen-{runtime}-{dtype}-b{batch}-{attention}"
        + ("-metalmatmul" if env else "")
        + ("-fast" if "PYTORCH_MPS_FAST_MATH" in env else "")
    )
    with (OUT / (tag + ".log")).open("w") as log:
        r = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/benchmark_meld_v5.py"),
                "--runtime",
                runtime,
                "--precision",
                dtype,
                "--attention",
                attention,
                "--batch",
                str(batch),
                "--papers",
                "2",
                "--tag",
                tag,
            ],
            cwd=ROOT,
            check=False,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, **env},
        )
    if r.returncode:
        row = {"tag": tag, "error": r.returncode}
    else:
        data = json.loads((OUT / (tag + ".json")).read_text())
        row = {
            "tag": tag,
            "seconds": sum(r["seconds"] for r in data),
            "tokens": sum(r["source_tokens"] for r in data),
        }
    rows.append(row)
    (OUT / "sweep.json").write_text(json.dumps(rows, indent=2))
    print(row, flush=True)
