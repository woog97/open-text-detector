import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / "research/benchmarks/meld-v8"
settings = (
    [("float32", 1, "dense"), ("float32", 1, "tiled")]
    + [
        (dtype, batch, attn)
        for dtype in ["float16", "bfloat16"]
        for attn in ["dense", "tiled"]
        for batch in [1, 2, 4, 8]
    ]
    + [("float16", 1, "metal")]
)
rows = []
for dtype, batch, attn in settings:
    tag = f"micro-{dtype}-{attn}-b{batch}"
    with (out / (tag + ".log")).open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                str(root / "scripts/benchmark_meld_v8.py"),
                "--micro",
                "--papers",
                "2",
                "--precision",
                dtype,
                "--batch",
                str(batch),
                "--attention",
                attn,
                "--tag",
                tag,
            ],
            cwd=root,
            check=False,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        rows.append({"tag": tag, "error": result.returncode})
    else:
        r = json.loads((out / (tag + ".json")).read_text())
        rows.append(
            {
                "tag": tag,
                "seconds": sum(x["seconds"] for x in r),
                "tokens": sum(x["source_tokens"] for x in r),
                "driver_bytes": max(x["driver_bytes"] for x in r),
            }
        )
    (out / "sweep.json").write_text(json.dumps(rows, indent=2))
    print(rows[-1], flush=True)
