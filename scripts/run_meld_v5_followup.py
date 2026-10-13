"""Run correctness and tile experiments sequentially after the main sweep."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/benchmarks/meld-v5"
steps = [
    ["validate_meld_v5_native.py"],
    ["validate_meld_v5_metal.py"],
    ["sweep_meld_v5_tiles.py"],
]
steps += [
    [
        "benchmark_meld_v5.py",
        "--runtime",
        "mlx",
        "--precision",
        "float16",
        "--attention",
        "tiled",
        "--papers",
        "2",
        "--fast-rope",
        "--tag",
        "fast-rope" + ("-fusion" if fusion else ""),
    ]
    + (["--fusion"] if fusion else [])
    for fusion in [False, True]
]
for i, args in enumerate(steps):
    with (OUT / f"followup-{i}.log").open("w") as log:
        code = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / args[0]), *args[1:]],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    print(args, "exit", code, flush=True)
    if code and i < 2:
        raise SystemExit(code)
