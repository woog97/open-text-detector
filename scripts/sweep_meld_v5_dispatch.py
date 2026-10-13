"""Pipeline and Metal command-buffer scheduling sweep; use sequential GPU jobs."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/benchmarks/meld-v5-max"
settings = [(f"pipeline-{d}", d, {}, {}) for d in [1, 2, 4, 8]]
settings += [
    (
        f"dispatch-{n}",
        2,
        {},
        {"MLX_MAX_OPS_PER_BUFFER": str(n), "MLX_MAX_MB_PER_BUFFER": str(n)},
    )
    for n in [10, 80, 256, 1024]
]
settings += [
    ("fast-sync", 2, {}, {"MLX_METAL_FAST_SYNCH": "1"}),
    ("pipeline-strided", 2, {"strided_attention": True}, {}),
]
rows = []
for tag, depth, advanced, env in settings:
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
        "--tag",
        tag,
        "--pipeline",
        str(depth),
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
            "advanced": advanced,
            "env": env,
            "doc_changes": sum(not r["document_label_matches"] for r in data),
            "sentence_changes": sum(r["sentence_label_changes"] for r in data),
            "max_doc_error": max(r["document_score_error"] for r in data),
        }
    rows.append(row)
    (OUT / "dispatch-sweep.json").write_text(json.dumps(rows, indent=2))
    print(row, flush=True)
