"""Validate bounded-tail masking on real v8 weights, separately from timing runs."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import mlx.core as mx
from open_text_detector_backend.providers.meld_v8_mlx import MeldV8MLX

model = MeldV8MLX(ROOT / "models/meld-v8", precision="float32")
rows = []
for length in [17, 65, 129, 511]:
    ids = mx.array(
        [
            [model.tokenizer.cls_token_id]
            + [1000] * (length - 2)
            + [model.tokenizer.sep_token_id]
        ]
    )
    width = ((length + 127) // 128) * 128
    padded = mx.pad(
        ids, [(0, 0), (0, width - length)], constant_values=model.tokenizer.pad_token_id
    )
    changed = mx.pad(ids, [(0, 0), (0, width - length)], constant_values=1001)
    reference = model.forward(ids, None)
    actual = model.forward(padded, mx.array(length, dtype=mx.int32))[:, :length]
    alternate = model.forward(changed, mx.array(length, dtype=mx.int32))[:, :length]
    mx.eval(reference, actual, alternate)
    error = float(mx.max(mx.abs(reference - actual)).item())
    leak = float(mx.max(mx.abs(actual - alternate)).item())
    assert error < 0.001, (length, error)
    assert leak == 0, (length, leak)
    row = {
        "length": length,
        "bucket": width,
        "max_padding_difference": error,
        "padding_content_effect": leak,
    }
    rows.append(row)
    print(row, flush=True)
(ROOT / "research/benchmarks/meld-v8/padding-validation.json").write_text(
    json.dumps(rows, indent=2)
)
