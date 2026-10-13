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
from open_text_detector_backend.providers.meld_v5_optimized import MeldV5Runner

mode = sys.argv[1] if len(sys.argv) > 1 else "dense"
m = MeldV5Runner(ROOT / "models/meld-v5", precision="float16", attention=mode)
p = json.loads((ROOT / "research/benchmarks/meld-v5/papers.json").read_text())[0]
text = read_text(ROOT / p["text_file"])
ids = m.tokenizer(text, add_special_tokens=False)["input_ids"][:2046]
inputs = torch.tensor(
    [[m.tokenizer.cls_token_id, *ids, m.tokenizer.sep_token_id]], device="mps"
)


def run():
    with torch.inference_mode():
        return m.model.token_scores(inputs, torch.ones_like(inputs)).cpu()


run()
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
run()
for handle in handles:
    handle.remove()
report = {k: {"seconds": sum(v), "calls": len(v)} for k, v in times.items()}
(ROOT / f"research/benchmarks/meld-v5/profile-{mode}.json").write_text(
    json.dumps(report, indent=2)
)
print(json.dumps(report))
