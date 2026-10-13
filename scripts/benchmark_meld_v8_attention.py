"""Isolated local-attention correctness and throughput checks."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import torch
from open_text_detector_backend.providers.meld_v8_attention import tiled
from open_text_detector_backend.providers.meld_v8_metal import attention as metal

rows = []
for dtype in (torch.float32, torch.float16):
    for length in (17, 129, 2048):
        torch.manual_seed(9)
        q, k, v = [
            torch.randn(1, 28, length, 64, device="mps", dtype=dtype) for _ in range(3)
        ]
        pos = torch.arange(length, device="mps")
        mask = (pos[:, None] - pos[None, :]).abs() <= 64
        fn = {
            "dense": lambda q=q, k=k, v=v, mask=mask: (
                torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask
                )
            ),
            "tiled": lambda q=q, k=k, v=v: tiled(q, k, v),
            "metal": lambda q=q, k=k, v=v: metal(q, k, v),
        }
        ref = fn["dense"]()
        for name, call in fn.items():
            out = call()
            delta = (ref.float() - out.float()).abs()
            maximum = delta.max().item()
            assert maximum < (1e-5 if dtype == torch.float32 else 0.005), (
                name,
                length,
                maximum,
            )
            for _ in range(3):
                call()
            torch.mps.synchronize()
            t = time.perf_counter()
            for _ in range(30):
                call()
            torch.mps.synchronize()
            seconds = (time.perf_counter() - t) / 30
            row = {
                "dtype": str(dtype),
                "length": length,
                "method": name,
                "seconds": seconds,
                "max_error": maximum,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
Path("research/benchmarks/meld-v8/attention-microbench.json").write_text(
    json.dumps(rows, indent=2)
)
