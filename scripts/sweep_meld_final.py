import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for precision, batch in [
    ("float32", 1),
    ("float32", 2),
    ("float16", 1),
    ("float16", 2),
    ("float16", 4),
    ("float16", 8),
    ("bfloat16", 1),
]:
    print("Final comparison", precision, batch, flush=True)
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/tune_meld.py"),
            "--precision",
            precision,
            "--batch",
            str(batch),
            "--repeats",
            "3",
            "--tag=-final",
        ],
        check=True,
        cwd=root,
    )
