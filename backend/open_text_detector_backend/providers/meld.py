"""Pinned local MELD inference with full-document token and sentence evidence."""

import json
import math
import threading
import time
from pathlib import Path
from .checkpoints import MELD_ID, MELD_REVISION
from .errors import ProviderError
from .localization import localize, sigmoid, window_starts


class Meld:
    def __init__(self, device="auto", model_dir=None, precision="float32", batch_size=1):
        self.precision = precision
        self.batch_size = max(1, int(batch_size))
        self.device_request = device
        self.directory = Path(model_dir) / "meld-v5"
        self.model = None
        self.lock = threading.Lock()

    def _load(self):
        if self.model is not None:
            return
        manifest = self.directory / "download-manifest.json"
        if not manifest.is_file() or json.loads(manifest.read_text()).get("revision") != MELD_REVISION:
            raise ProviderError("model_not_downloaded", "Run scripts/download_meld.py to install MELD v5")
        try:
            import torch
            from transformers import AutoTokenizer
            from .meld_model import MeldModel

            device = self.device_request
            if device == "auto":
                device = (
                    "cuda"
                    if torch.cuda.is_available()
                    else "mps"
                    if torch.backends.mps.is_available()
                    else "cpu"
                )
            tokenizer = AutoTokenizer.from_pretrained(
                self.directory, local_files_only=True, trust_remote_code=False
            )
            if (
                tokenizer.num_special_tokens_to_add(pair=False) != 2
                or tokenizer.cls_token_id is None
                or tokenizer.sep_token_id is None
            ):
                raise ValueError("Unexpected MELD tokenizer special-token template")
            model = MeldModel(self.directory).eval()
            if self.precision not in {"float32", "float16", "bfloat16"}:
                raise ValueError("Unsupported MELD backbone precision")
            # Keep the evidence head in float32 even when the backbone is reduced precision.
            # Rotary frequencies must stay float32: rounding those buffers changes
            # positional phases much more than rounding the learned weights.
            buffers = [
                (module, name, buffer)
                for module in model.backbone.modules()
                for name, buffer in module.named_buffers(recurse=False)
                if buffer.is_floating_point()
            ]
            model.backbone.to(dtype=getattr(torch, self.precision))
            for module, name, buffer in buffers:
                setattr(module, name, buffer)
            model = model.to(device)
            self.tokenizer, self.device, self.model = tokenizer, device, model
        except ImportError as e:
            raise ProviderError(
                "model_dependencies_missing", "Install the optional models dependencies for MELD"
            ) from e
        except Exception as e:
            raise ProviderError("model_load_failed", f"MELD could not be loaded ({type(e).__name__})") from e

    def _window_batches(self, ids, starts, capacity):
        import torch

        batch_size = getattr(self, "batch_size", 1)
        for group_start in range(0, len(starts), batch_size):
            group = starts[group_start : group_start + batch_size]
            windows = [
                [self.tokenizer.cls_token_id, *ids[start : start + capacity], self.tokenizer.sep_token_id]
                for start in group
            ]
            # The final long-document window is full width; short documents have one window.
            batch_ids = torch.tensor(windows, device=self.device)
            values = (
                self.model.token_scores(input_ids=batch_ids, attention_mask=torch.ones_like(batch_ids))[
                    :, 1:-1
                ]
                .float()
                .cpu()
                .tolist()
            )
            yield group, values

    def predict(self, config, text):
        with self.lock:
            begin = time.perf_counter()
            self._load()
            loaded = time.perf_counter()
            import torch

            encoded = self.tokenizer(
                text, add_special_tokens=False, return_offsets_mapping=True, verbose=False
            )
            ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
            if not ids:
                raise ProviderError("empty_tokens", "The tokenizer found no usable text")
            capacity = self.model.cfg["max_length"] - self.tokenizer.num_special_tokens_to_add(pair=False)
            starts = window_starts(len(ids), capacity, 256)
            tokenized = time.perf_counter()
            scores, quality = [0.0] * len(ids), [-1] * len(ids)
            with torch.inference_mode():
                for group, batch_values in self._window_batches(ids, starts, capacity):
                    for start, values in zip(group, batch_values, strict=True):
                        if len(values) != len(ids[start : start + capacity]):
                            raise ProviderError("invalid_response", "MELD token alignment failed")
                        for local, value in enumerate(values):
                            centrality = min(local + 1, len(values) - local)
                            if centrality > quality[start + local]:
                                scores[start + local], quality[start + local] = value, centrality
            inferred = time.perf_counter()
            if not all(math.isfinite(s) for s in scores) or min(quality) < 0:
                raise ProviderError("invalid_response", "MELD returned incomplete or non-finite evidence")
            k = max(1, math.ceil(len(scores) * self.model.cfg["rho"]))
            raw_score = sum(sorted(scores, reverse=True)[:k]) / k
            threshold = self.model.cfg["score_offsets"]["overall"]["fpr_0.01"]
            tokens, segments = localize(text, offsets, scores, threshold)
            localized = time.perf_counter()
            short = len(text.split()) < 100
            memory = {}
            if self.device == "mps":
                memory = {
                    "allocated_bytes": torch.mps.current_allocated_memory(),
                    "driver_bytes": torch.mps.driver_allocated_memory(),
                }
            return {
                "score": sigmoid(raw_score),
                "raw_score": raw_score,
                "score_type": "ai_evidence",
                "label": "uncertain"
                if short
                else "ai_evidence"
                if raw_score > threshold
                else "below_threshold",
                "segments": segments,
                "tokens": tokens,
                "thresholds": {
                    "raw_ai_above": threshold,
                    "calibration": "upstream_document_1pct_fpr",
                    "scope": "upstream validation documents; not guaranteed for this corpus, long-document aggregation or individual spans",
                },
                "localization": {
                    "method": "contextual_token_margins_sentence_mean",
                    "granularity": "token_and_sentence",
                    "offset_unit": "unicode_code_points",
                    "span_threshold_calibrated": False,
                    "notice": "Highlights show local AI evidence. Sentence and token thresholds are exploratory, not calibrated authorship labels.",
                },
                "notice": (
                    "Under 100 words: insufficient text for a reliable MELD verdict. " if short else ""
                )
                + "MELD scores are evidence, not probabilities or percentages of AI authorship. Low evidence does not establish human authorship.",
                "inference": {
                    "model_id": MELD_ID,
                    "revision": MELD_REVISION,
                    "device": self.device,
                    "dtype": "torch." + getattr(self, "precision", "float32"),
                    "head_dtype": "torch.float32",
                    "batch_size": getattr(self, "batch_size", 1),
                    "memory_after_run": memory,
                    "phase_seconds": {
                        "load": loaded - begin,
                        "tokenize": tokenized - loaded,
                        "inference": inferred - tokenized,
                        "localization": localized - inferred,
                    },
                    "max_length": self.model.cfg["max_length"],
                    "windows": len(starts),
                    "source_tokens": len(ids),
                    "overlap": 256,
                    "aggregation": "most_context_token_stitch_then_global_top_quartile_mean",
                    "preprocessing": "original text and line breaks preserved",
                },
            }
