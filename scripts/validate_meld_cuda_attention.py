"""Numerical and boundary tests plus device-specific local-attention timings."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import torch
import torch.nn.functional as F
import triton
from open_text_detector_backend.providers.meld_cuda_attention import local_attention

torch.manual_seed(7)
rows = []
for dtype in [torch.float16, torch.bfloat16]:
    for length in [1, 63, 64, 65, 127, 129, 257, 2048]:
        q, k, v = [torch.randn(1, 16, length, 64, device="cuda", dtype=dtype) for _ in range(3)]
        idx = torch.arange(length, device="cuda")
        mask = (idx[:, None] - idx[None, :]).abs() <= 64
        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=mask)
        for block, warps in [(32, 4), (64, 4), (128, 4), (64, 8)]:
            out = local_attention(q, k, v, 64, block, warps)
            error = float((out.float() - ref).abs().max())
            assert error < (.006 if dtype == torch.float16 else .04), (dtype, length, block, error)
            row = {"dtype": str(dtype), "length": length, "block": block, "warps": warps,
                   "max_error": error}
            if length == 2048:
                row["milliseconds"] = triton.testing.do_bench(lambda: local_attention(q, k, v, 64, block, warps), warmup=100, rep=200)
            rows.append(row)
            print(json.dumps(row), flush=True)
Path("research/benchmarks/meld-cuda/kernel-validation.json").write_text(json.dumps(rows, indent=2))
