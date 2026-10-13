import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/benchmarks/meld-v8"
variants = [
    [
        "--runtime",
        "mlx",
        "--precision",
        "float16",
        "--attention",
        "tiled",
        "--repeats",
        "2",
        "--tag",
        "optimized",
    ],
    ["--tag", "baseline-repeat"],
    [
        "--runtime",
        "mlx",
        "--precision",
        "float32",
        "--attention",
        "tiled",
        "--batch",
        "4",
        "--tag",
        "strict-mlx",
    ],
]
for args in variants:
    tag = args[-1]
    print("Starting", tag, flush=True)
    with (OUT / (tag + ".log")).open("w") as log:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/benchmark_meld_v8.py"), *args],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print("Finished", tag, flush=True)
