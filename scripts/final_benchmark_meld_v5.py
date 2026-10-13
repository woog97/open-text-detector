"""Choose from measured screening candidates, then benchmark all ten full papers."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/benchmarks/meld-v5"
candidates = []
for p in (
    list(OUT.glob("screen-mlx-float16-*.json"))
    + list(OUT.glob("tile-*-fusion-*.json"))
    + list(OUT.glob("fast-rope*.json"))
):
    rows = json.loads(p.read_text())
    if len(rows) != 2 or not all(x["document_label_matches"] for x in rows):
        continue
    # Keep model precision/head fixed. Candidates with larger score drift remain experiments.
    if max(x["document_score_error"] for x in rows) > 0.01:
        continue
    candidates.append((sum(x["seconds"] for x in rows), p, rows[0]["inference"]))
seconds, path, settings = min(candidates, key=lambda x: x[0])
settings = {
    k: settings[k]
    for k in ["runtime", "attention", "batch_size", "tile_size", "fast_rope"]
} | {"precision": "float16", "fusion": settings["custom_geglu"]}
(OUT / "selected-settings.json").write_text(
    json.dumps(
        {
            "screening_file": path.name,
            "screening_seconds": seconds,
            "settings": settings,
        },
        indent=2,
    )
)


def run(tag, settings, repeats):
    args = [
        "benchmark_meld_v5.py",
        "--tag",
        tag,
        "--repeats",
        str(repeats),
        "--runtime",
        settings.get("runtime", "torch"),
        "--precision",
        settings.get("precision", "float16"),
        "--attention",
        settings.get("attention", "dense"),
        "--batch",
        str(settings.get("batch_size", 1)),
        "--tile",
        str(settings.get("tile_size", 128)),
    ]
    if settings.get("fusion"):
        args += ["--fusion"]
    if settings.get("fast_rope"):
        args += ["--fast-rope"]
    with (OUT / (tag + ".log")).open("w") as log:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / args[0]), *args[1:]],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print("Completed", tag, flush=True)


run("current", {}, 2)
run("optimized", settings, 2)
run("strict", {"runtime": "mlx", "precision": "float32", "attention": "tiled"}, 1)
run("current-repeat", {}, 1)
run(
    "bfloat16-experimental",
    {"runtime": "mlx", "precision": "bfloat16", "attention": "tiled", "batch_size": 8},
    2,
)
