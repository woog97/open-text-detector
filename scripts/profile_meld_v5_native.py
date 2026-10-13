"""Profile actual native operations; synchronized instrumentation is separate from throughput."""

import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import mlx.core as mx
import numpy as np
from open_text_detector_backend.pdf_extraction import read_text
from open_text_detector_backend.providers.meld_v5_optimized import MeldV5Runner
from open_text_detector_backend.providers.meld_v8_mlx import MeldV8MLX

m = MeldV5Runner(
    ROOT / "models/meld-v5", runtime="mlx", precision="float16", attention="tiled"
)
n = m.native
p = json.loads((ROOT / "research/benchmarks/meld-v5/papers.json").read_text())[0]
ids = m.tokenizer(
    read_text(ROOT / p["text_file"]), add_special_tokens=False, verbose=False
)["input_ids"][:2046]
a = mx.array([[m.tokenizer.cls_token_id, *ids, m.tokenizer.sep_token_id]])
mx.eval(n.forward(a))
times = []
for _ in range(5):
    t = time.perf_counter()
    mx.eval(n.forward(a))
    times.append(time.perf_counter() - t)
profile = collections.defaultdict(list)
for name in ["linear", "norm", "rotate", "local", "activation"]:
    original = getattr(n, name)

    def wrapped(*args, _name=name, _fn=original, **kwargs):
        mx.synchronize()
        t = time.perf_counter()
        y = _fn(*args, **kwargs)
        mx.eval(y)
        key = _name + (
            ":" + args[1].rsplit(".", 1)[-1] if _name in ["linear", "norm"] else ""
        )
        profile[key].append(time.perf_counter() - t)
        return y

    setattr(n, name, wrapped)
s = time.perf_counter()
mx.eval(MeldV8MLX.forward(n, a))
eager = time.perf_counter() - s
out = {
    "compiled_window_seconds": float(np.median(times)),
    "instrumented_eager_window_seconds": eager,
    "operations": {k: {"seconds": sum(v), "calls": len(v)} for k, v in profile.items()},
}
(ROOT / "research/benchmarks/meld-v5-max/profile.json").write_text(
    json.dumps(out, indent=2)
)
print(json.dumps(out, indent=2))
