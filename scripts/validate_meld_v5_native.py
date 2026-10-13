"""Real-weight numerical checks for short documents and shared v8 code."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import numpy as np
import torch
from open_text_detector_backend.pdf_extraction import read_text
from open_text_detector_backend.providers.meld_v5_optimized import MeldV5Runner
from open_text_detector_backend.providers.meld_v8_mlx import MeldV8MLX
from open_text_detector_backend.result_codec import decode

ref = MeldV5Runner(ROOT / "models/meld-v5")
mlx = MeldV5Runner(ROOT / "models/meld-v5", runtime="mlx", attention="tiled")
rows = []
for length in [1, 17, 65, 129, 511, 2048]:
    ids = torch.tensor([[ref.tokenizer.cls_token_id] + [100] * (length - 1)])
    with torch.inference_mode():
        a = (
            ref.model.token_scores(ids.to("mps"), torch.ones_like(ids).to("mps"))
            .cpu()
            .numpy()
        )
    b = mlx.model.token_scores(ids, torch.ones_like(ids)).numpy()
    error = float(np.max(np.abs(a - b)))
    assert error < 0.0003, (length, error)
    rows.append({"length": length, "max_error": error})
# A single shared-code v8 regression checks entire prediction/offsets against saved results.
v8 = MeldV8MLX(ROOT / "models/meld-v8")
p = json.loads((ROOT / "research/benchmarks/meld-v8/papers.json").read_text())[0]
r = v8.predict(read_text(ROOT / p["text_file"]))
b = decode((ROOT / "research/benchmarks/meld-v8/optimized-bounded-0.pgf").read_bytes())
assert (
    r["raw_score"] == b["raw_score"]
    and r["segments"] == b["segments"]
    and r["tokens"] == b["tokens"]
)
report = {"v5_short_window_checks": rows, "v8_full_paper_regression": "exact match"}
(ROOT / "research/benchmarks/meld-v5/native-validation.json").write_text(
    json.dumps(report, indent=2)
)
print(json.dumps(report))
