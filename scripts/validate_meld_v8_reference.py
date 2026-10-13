"""Check the local implementation against the reviewed upstream scorer on identical tokens."""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import torch
from open_text_detector_backend.providers.meld_v8 import MeldV8

spec = importlib.util.spec_from_file_location(
    "reviewed_meld_v8", ROOT / "models/meld-v8/meld.py"
)
upstream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)
local = MeldV8(ROOT / "models/meld-v8")
local.model._sqdist = upstream.Meld._sqdist
ids = local.tokenizer(
    "The experiment compares predictions with observations and reports uncertainty. "
    * 8,
    return_tensors="pt",
).to("mps")
with torch.inference_mode():
    expected = upstream.Meld.token_scores(local.model, **ids)[1]
    actual = local.model.token_scores(**ids)
    error = (expected - actual).abs().max().item()
    assert error == 0, error
print(json.dumps({"official_v8_head_max_error": error, "tokens": actual.numel()}))
