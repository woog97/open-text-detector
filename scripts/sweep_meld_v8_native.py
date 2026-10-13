import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/benchmarks/meld-v8"
for script in ["validate_meld_v8_reference.py", "profile_meld_v8.py"]:
    with (OUT / (script + ".log")).open("w") as log:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
rows = []
settings = [
    ("mlx", dtype, b, "tiled", {})
    for dtype in ["float32", "float16"]
    for b in [1, 2, 4, 8]
]
settings += [
    ("torch", "float16", 1, "tiled", {"PYTORCH_MPS_PREFER_METAL": "1"}),
    (
        "torch",
        "float16",
        1,
        "tiled",
        {"PYTORCH_MPS_PREFER_METAL": "1", "PYTORCH_MPS_FAST_MATH": "1"},
    ),
]
for runtime, dtype, batch, attention, env in settings:
    tag = (
        f"native-{runtime}-{dtype}-b{batch}"
        + ("-metalmatmul" if env else "")
        + ("-fast" if "PYTORCH_MPS_FAST_MATH" in env else "")
    )
    with (OUT / (tag + ".log")).open("w") as log:
        r = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/benchmark_meld_v8.py"),
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
                "--micro",
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
    (OUT / "native-sweep.json").write_text(json.dumps(rows, indent=2))
    print(row, flush=True)
