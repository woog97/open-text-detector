"""Pinned MELD v8 full-paper scoring. Raw text/offsets retained; no input cap.

Full-paper mode deliberately extends the upstream 16,384-token calibration contract.
"""

import json
import math
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from .meld_model import MeldModel
from .localization import localize

REVISION = "8990324abd92e1fa17072f6887ea1e5c1cef5abc"


class MeldV8:
    def __init__(self, directory, precision="float32", batch_size=1, attention="dense"):
        directory = Path(directory)
        assert json.loads((directory / "download-manifest.json").read_text())["revision"] == REVISION
        self.tokenizer = AutoTokenizer.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False
        )
        self.model = MeldModel(directory).eval()
        buffers = [
            (m, k, v)
            for m in self.model.backbone.modules()
            for k, v in m.named_buffers(recurse=False)
            if v.is_floating_point()
        ]
        self.model.backbone.to(dtype=getattr(torch, precision))
        for m, k, v in buffers:
            setattr(m, k, v)
        self.model.to("mps")
        self.batch_size = batch_size
        self.precision = precision
        self.attention = attention
        if attention != "dense":
            from .meld_v8_attention import install

            install(self.model.backbone, attention)
        self.threshold = self.model.cfg["score_offsets"]["overall"]["fpr_0.01"]

    def encode(self, text):
        return self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)

    @torch.inference_mode()
    def scores(self, encoded, max_windows=None):
        ids = encoded["input_ids"]
        windows = [ids[i : i + 2046] for i in range(0, len(ids), 2046)]
        if max_windows:
            windows = windows[:max_windows]
        values = []
        i = 0
        while i < len(windows):
            group = windows[i : i + self.batch_size]
            # Avoid padding entirely: only equal-length windows share a batch.
            if len(group[-1]) != len(group[0]):
                group = group[:-1]
            if not group:
                group = [windows[i]]
            inp = torch.tensor(
                [[self.tokenizer.cls_token_id, *w, self.tokenizer.sep_token_id] for w in group], device="mps"
            )
            mask = (
                torch.ones_like(inp)
                if self.attention == "dense"
                else {"full_attention": None, "sliding_attention": None}
            )
            values.append(self.model.token_scores(inp, mask)[:, 1:-1].float().reshape(-1).cpu())
            i += len(group)
        return torch.cat(values)

    def predict(self, text):
        start = time.perf_counter()
        encoded = self.encode(text)
        tokenized = time.perf_counter()
        values = self.scores(encoded)
        inferred = time.perf_counter()
        assert len(values) == len(encoded["input_ids"]) and torch.isfinite(values).all(), (
            "Incomplete or invalid evidence"
        )
        score = float(values.topk(max(1, math.ceil(len(values) * self.model.cfg["rho"]))).values.mean())
        tokens, segments = localize(text, encoded["offset_mapping"], values.tolist(), self.threshold)
        return {
            "raw_score": score,
            "score": 1 / (1 + math.exp(-score)),
            "score_type": "ai_evidence",
            "label": "ai_evidence" if score > self.threshold else "below_threshold",
            "thresholds": {
                "raw_ai_above": self.threshold,
                "scope": "Upstream capped-document threshold; full-paper and span calibration not established",
            },
            "tokens": tokens,
            "segments": segments,
            "inference": {
                "revision": REVISION,
                "precision": self.precision,
                "batch_size": self.batch_size,
                "runtime": "mlx" if type(self).__name__ == "MeldV8MLX" else "torch",
                "attention": self.attention,
                "source_tokens": len(values),
                "windows": math.ceil(len(values) / 2046),
                "input_cap": None,
                "preprocessing": "raw positioned text, unchanged",
                "calibration": "full-paper extension beyond upstream capped input contract",
                "seconds": {
                    "tokenize": tokenized - start,
                    "inference": inferred - tokenized,
                    "localize": time.perf_counter() - inferred,
                },
            },
        }
