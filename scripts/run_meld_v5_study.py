"""Serialize all GPU work; model outputs from the baseline are the reference."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/benchmarks/meld-v5"
steps = [
    ["benchmark_meld_v5.py", "--tag", "baseline", "--repeats", "2"],
    ["profile_meld_v5.py", "dense"],
    ["profile_meld_v5.py", "tiled"],
    ["sweep_meld_v5.py"],
]
for args in steps:
    with (OUT / ("study-" + args[0] + ".log")).open("w") as log:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / args[0]), *args[1:]],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print("Completed", args, flush=True)
