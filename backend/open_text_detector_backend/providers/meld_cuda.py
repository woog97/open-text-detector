"""Opt-in CUDA MELD runner; full text and original offsets are retained."""
import math
import time
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from .meld_model import MeldModel
from .localization import localize, window_starts


class MeldCudaRunner:
    def __init__(self, directory, precision, batch, attention, compile_mode, block=64, warps=4, quantization="none"):
        if not torch.cuda.is_available():
            raise RuntimeError("This opt-in runner requires an NVIDIA CUDA GPU")
        self.revision = json.loads((Path(directory) / "download-manifest.json").read_text())["revision"]
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self.model = MeldModel(directory).eval()
        self.cfg = self.model.cfg
        from .checkpoints import MELD_REVISION
        expected = {"v5": MELD_REVISION, "v8": "8990324abd92e1fa17072f6887ea1e5c1cef5abc"}
        if self.revision != expected[self.cfg["version"]]:
            raise ValueError("Checkpoint revision does not match the reviewed MELD release")
        self.batch = batch
        self.attention = attention
        buffers = [(m, k, v) for m in self.model.backbone.modules()
                   for k, v in m.named_buffers(recurse=False) if v.is_floating_point()]
        self.model.backbone.to(dtype=getattr(torch, precision))
        for m, k, v in buffers:
            setattr(m, k, v)
        self.model.cuda()
        self.quantization = quantization
        if quantization != "none":
            if quantization not in ("fp8-row", "fp8-tensor"):
                raise ValueError(quantization)
            from .meld_cuda_fp8 import install as install_fp8
            install_fp8(self.model.backbone, rowwise=quantization == "fp8-row")
        if attention != "sdpa":
            from open_text_detector_backend.providers.meld_cuda_attention import install
            install(self.model.backbone, attention, block, warps)
        self.forward = self.model.token_scores
        self.eager = self.forward
        self.graphs = {}
        self.graph_mode = compile_mode == "graphs"
        self.compiled = compile_mode not in ("none", "graphs")
        if compile_mode not in ("none", "graphs"):
            self.forward = torch.compile(self.forward, mode=compile_mode, dynamic=False)

    def execute(self, inp, mask):
        # Avoid paying compilation cost for every distinct short-paper/tail length.
        if inp.shape[1] != self.cfg["max_length"]:
            return self.eager(inp, mask)
        if not self.graph_mode or inp.shape[1] != self.cfg["max_length"]:
            if self.compiled:
                torch.compiler.cudagraph_mark_step_begin()
                # Compiled graph outputs can alias storage reused by its next replay.
                return self.forward(inp, mask).clone()
            return self.forward(inp, mask)
        shape = tuple(inp.shape)
        if shape not in self.graphs:
            static = inp.clone()
            static_mask = torch.ones_like(static) if self.attention == "sdpa" else mask
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.forward(static, static_mask)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self.forward(static, static_mask)
            self.graphs[shape] = (static, output, graph)
        static, output, graph = self.graphs[shape]
        static.copy_(inp)
        graph.replay()
        return output.clone()

    def encode(self, text):
        return self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)

    @torch.inference_mode()
    def warmup(self):
        """Compile both full and remainder batch shapes before a long queue starts."""
        for count in sorted({1, self.batch}, reverse=True):
            inp = torch.full((count, self.cfg["max_length"]), self.tokenizer.cls_token_id,
                             dtype=torch.long, device="cuda")
            mask = torch.ones_like(inp) if self.attention == "sdpa" else {
                "full_attention": None, "sliding_attention": None}
            for _ in range(3):
                self.execute(inp, mask)
        torch.cuda.synchronize()

    def finish(self, text, encoded, starts, windows, flat):
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        scores = np.zeros(len(ids), dtype=np.float32)
        quality = np.full(len(ids), -1, dtype=np.int32)
        cursor = 0
        for start, window in zip(starts, windows, strict=True):
            length = len(window) - 2
            centrality = np.minimum(np.arange(1, length + 1), np.arange(length, 0, -1))
            take = centrality > quality[start:start + length]
            scores[start:start + length][take] = flat[cursor:cursor + length][take]
            quality[start:start + length][take] = centrality[take]
            cursor += length
        assert np.isfinite(scores).all() and (quality >= 0).all()
        threshold = self.cfg["score_offsets"]["overall"]["fpr_0.01"]
        k = max(1, math.ceil(len(scores) * self.cfg["rho"]))
        raw = float(np.partition(scores, len(scores) - k)[-k:].astype(np.float64).mean())
        tokens, segments = localize(text, offsets, scores.tolist(), threshold)
        short = self.cfg["version"] == "v5" and len(text.split()) < 100
        return {"raw_score": raw, "score": 1 / (1 + math.exp(-max(-80, min(80, raw)))),
                "score_type": "ai_evidence", "label": "uncertain" if short else
                "ai_evidence" if raw > threshold else "below_threshold",
                "tokens": tokens, "segments": segments,
                "source_tokens": len(ids), "windows": len(windows),
                "thresholds": {"raw_ai_above": threshold,
                    "scope": "Upstream document threshold; full-paper and span calibration not established"},
                "localization": {"method": "contextual_token_margins_sentence_mean",
                    "granularity": "token_and_sentence", "offset_unit": "unicode_code_points",
                    "span_threshold_calibrated": False,
                    "notice": "Highlights show local AI evidence, not calibrated authorship labels."},
                "notice": "MELD scores are evidence, not probabilities or percentages of AI authorship.",
                "inference": {"revision": self.revision, "version": self.cfg["version"],
                    "runtime": "cuda", "precision": str(next(self.model.backbone.parameters()).dtype),
                    "batch_size": self.batch, "attention": self.attention, "quantization": self.quantization,
                    "source_tokens": len(ids), "windows": len(windows),
                    "overlap": 256 if self.cfg["version"] == "v5" else 0,
                    "input_cap": None, "preprocessing": "Original text and character offsets preserved"}}

    @torch.inference_mode()
    def predict_many(self, texts):
        """Batch tokenization and equal-length windows across independent papers."""
        batch_encoded = self.tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        capacity = self.cfg["max_length"] - 2
        records, groups = [], {}
        for paper, text in enumerate(texts):
            encoded = {k: v[paper] for k, v in batch_encoded.items()}
            ids = encoded["input_ids"]
            if not ids:
                raise ValueError("The tokenizer found no usable text")
            starts = (window_starts(len(ids), capacity, 256) if self.cfg["version"] == "v5"
                      else list(range(0, len(ids), capacity)))
            windows = [[self.tokenizer.cls_token_id, *ids[s:s + capacity], self.tokenizer.sep_token_id]
                       for s in starts]
            records.append((encoded, starts, windows))
            for index, window in enumerate(windows):
                groups.setdefault(len(window), []).append((paper, index, window))
        values = [[None] * len(r[2]) for r in records]
        for length, group in groups.items():
            # Materialize and transfer token IDs once per length bucket.
            ids_gpu = torch.from_numpy(np.asarray([w for _, _, w in group], dtype=np.int64)).cuda()
            chunks = []
            for start in range(0, len(group), self.batch):
                inp = ids_gpu[start:start + self.batch]
                mask = torch.ones_like(inp) if self.attention == "sdpa" else {
                    "full_attention": None, "sliding_attention": None}
                chunks.append(self.execute(inp, mask)[:, 1:-1].float())
            flat = torch.cat(chunks).cpu().numpy()
            for index, (paper, window_index, _) in enumerate(group):
                values[paper][window_index] = flat[index]
        return [self.finish(text, *record, np.concatenate(scores)) for text, record, scores in
                zip(texts, records, values, strict=True)]

    @torch.inference_mode()
    def predict(self, text, encoded=None):
        begin = time.perf_counter()
        encoded = self.encode(text) if encoded is None else encoded
        ids = encoded["input_ids"]
        if not ids:
            raise ValueError("The tokenizer found no usable text")
        capacity = self.cfg["max_length"] - 2
        starts = (window_starts(len(ids), capacity, 256) if self.cfg["version"] == "v5"
                  else list(range(0, len(ids), capacity)))
        windows = [[self.tokenizer.cls_token_id, *ids[s:s + capacity], self.tokenizer.sep_token_id]
                   for s in starts]
        tokenized = time.perf_counter()
        outputs = []
        index = 0
        while index < len(windows):
            group = windows[index:index + self.batch]
            if len(group[-1]) != len(group[0]):
                group = group[:-1]
            if not group:
                group = [windows[index]]
            inp = torch.tensor(group, device="cuda")
            mask = torch.ones_like(inp) if self.attention == "sdpa" else {
                "full_attention": None, "sliding_attention": None}
            scores = self.execute(inp, mask)[:, 1:-1].float()
            outputs.extend(scores.unbind(0))
            index += len(group)
        # One device-to-host transfer for the document, rather than each window batch.
        flat = torch.cat(outputs).cpu().numpy()
        inferred = time.perf_counter()
        result = self.finish(text, encoded, starts, windows, flat)
        result["phase_seconds"] = {"tokenize": tokenized - begin, "gpu": inferred - tokenized,
                                  "stitch_localize": time.perf_counter() - inferred}
        return result
