"""Synchronized component timings on one full window, separate from throughput runs."""

import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import torch
from open_text_detector_backend.pdf_extraction import read_text
from open_text_detector_backend.providers.meld_v8 import MeldV8

m = MeldV8(ROOT / "models/meld-v8", precision="float16", attention="tiled")
p = json.loads((ROOT / "research/benchmarks/meld-v8/papers.json").read_text())[0]
encoded = m.encode(read_text(ROOT / p["text_file"]))
m.scores(encoded, max_windows=1)
times = collections.defaultdict(list)
handles = []
for i, layer in enumerate(m.model.backbone.layers):
    for name, module in [
        ("mlp", layer.mlp),
        ("global_attention" if i % 3 == 0 else "local_attention", layer.attn),
    ]:

        def pre(module, args):
            torch.mps.synchronize()
            module._bench_start = time.perf_counter()

        def post(module, args, output, name=name):
            torch.mps.synchronize()
            times[name].append(time.perf_counter() - module._bench_start)

        handles.extend(
            [module.register_forward_pre_hook(pre), module.register_forward_hook(post)]
        )
m.scores(encoded, max_windows=1)
for handle in handles:
    handle.remove()
report = {k: {"seconds": sum(v), "calls": len(v)} for k, v in times.items()}
(ROOT / "research/benchmarks/meld-v8/profile.json").write_text(
    json.dumps(report, indent=2)
)
print(json.dumps(report))
