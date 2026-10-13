"""Screen native execution changes on two complete papers, preserving prior reports."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/benchmarks/meld-v5-max"
settings = [
    ("control", {}),
    ("strided", {"strided_attention": True}),
    ("transpose", {"transpose_weights": True}),
]
settings += [(f"pad-{n}", {"pad_mlp": n}) for n in [2688, 2816, 3072]]
settings += [(f"chunk-{n}", {"chunk_rows": n}) for n in [256, 512, 1024]]
settings += [
    (f"quant-{bits}-{scope}", {"quant_bits": bits, "quant_scope": scope})
    for bits in [8, 6, 4]
    for scope in ["mlp", "all"]
]
settings += [("cache-0", {"cache_mb": 0}), ("cache-4096", {"cache_mb": 4096})]
rows = []
for tag, advanced in settings:
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
        "--advanced",
        json.dumps(advanced),
    ]
    with (OUT / (tag + ".log")).open("w") as log:
        code = subprocess.run(
            args,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            env=dict(os.environ),
        ).returncode
    if code:
        row = {"tag": tag, "error": code, "advanced": advanced}
    else:
        data = json.loads((OUT / (tag + ".json")).read_text())
        row = {
            "tag": tag,
            "advanced": advanced,
            "seconds": sum(r["seconds"] for r in data),
            "doc_changes": sum(not r["document_label_matches"] for r in data),
            "sentence_changes": sum(r["sentence_label_changes"] for r in data),
            "max_doc_error": max(r["document_score_error"] for r in data),
        }
    rows.append(row)
    (OUT / "sweep.json").write_text(json.dumps(rows, indent=2))
    print(row, flush=True)
