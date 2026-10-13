"""Validate pipeline boundaries, repeated requests, and source offsets with real weights."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/benchmarks/meld-v5-max"
selection = json.loads((OUT / "selection.json").read_text())
os.environ.update(selection["env"])
sys.path.insert(0, str(ROOT / "backend"))
from open_text_detector_backend.pdf_extraction import read_text
from open_text_detector_backend.providers.meld_v5_optimized import MeldV5Runner

ref = MeldV5Runner(
    ROOT / "models/meld-v5", runtime="mlx", precision="float16", attention="tiled"
)
fast = MeldV5Runner(
    ROOT / "models/meld-v5",
    runtime="mlx",
    precision="float16",
    attention="tiled",
    pipeline_depth=selection["depth"],
    parallel_streams=selection["streams"],
    advanced=selection["advanced"],
)
p = json.loads((ROOT / "research/benchmarks/meld-v5/papers.json").read_text())[0]
text = read_text(ROOT / p["text_file"])
ids = ref.tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
rows = []
for length in [17, 127, 128, 129, 2046, 2047, 3836, 4093, 17]:
    sample = ref.tokenizer.decode(ids[:length])
    a = ref.predict(sample)
    b = fast.predict(sample)
    assert a["raw_score"] == b["raw_score"] and a["label"] == b["label"]
    assert a["tokens"] == b["tokens"] and a["segments"] == b["segments"]
    rows.append(
        {
            "requested_tokens": length,
            "actual_tokens": b["inference"]["source_tokens"],
            "windows": b["inference"]["windows"],
            "exact_match": True,
        }
    )
(OUT / "pipeline-validation.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows))
