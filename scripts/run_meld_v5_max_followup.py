import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/benchmarks/meld-v5-max"
for script in ["sweep_meld_v5_dispatch.py", "tune_meld_v5_gemm.py"]:
    with (OUT / (script + ".log")).open("w") as log:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print("completed", script, flush=True)
