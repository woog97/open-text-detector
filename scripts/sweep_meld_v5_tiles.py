"""Follow-up local-tile and custom activation screening, on two full papers."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/benchmarks/meld-v5"
rows = []
for tile in [32, 64, 128, 256, 512]:
    for fusion in [False, True]:
        tag = f"tile-{tile}-fusion-{int(fusion)}"
        args = [
            sys.executable,
            str(ROOT / "scripts/benchmark_meld_v5.py"),
            "--runtime",
            "mlx",
            "--precision",
            "float16",
            "--attention",
            "tiled",
            "--papers",
            "2",
            "--tile",
            str(tile),
            "--tag",
            tag,
        ]
        if fusion:
            args += ["--fusion"]
        with (OUT / (tag + ".log")).open("w") as log:
            code = subprocess.run(
                args, stdout=log, stderr=subprocess.STDOUT, check=False
            ).returncode
        if code:
            row = {"tag": tag, "error": code}
        else:
            data = json.loads((OUT / (tag + ".json")).read_text())
            row = {
                "tag": tag,
                "seconds": sum(r["seconds"] for r in data),
                "tile": tile,
                "fusion": fusion,
                "sentence_changes": sum(r["sentence_label_changes"] for r in data),
            }
        rows.append(row)
        (OUT / "tile-sweep.json").write_text(json.dumps(rows, indent=2))
        print(row, flush=True)
