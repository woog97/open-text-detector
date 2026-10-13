"""Native MLX execution of the reviewed MELD v8 encoder and FP32 evidence head."""

import json
import math
from pathlib import Path
import numpy as np
import torch
import mlx.core as mx
from transformers import AutoTokenizer
from .meld_v8 import MeldV8, REVISION


class MeldV8MLX(MeldV8):
    revision = REVISION
    version = "v8"
    hidden_size = 1792
    heads = 28
    tile_size = 128

    def __init__(self, directory, precision="float16", batch_size=1, attention="tiled"):
        directory = Path(directory)
        # Bound allocator retention for corpora with thousands of different tails.
        mx.set_cache_limit(1 << 30)
        assert json.loads((directory / "download-manifest.json").read_text())["revision"] == self.revision
        self.cfg = json.loads((directory / "meld_config.json").read_text())
        self.config = json.loads((directory / "config.json").read_text())
        assert (
            self.cfg["version"] == self.version
            and self.config["hidden_size"] == self.hidden_size
            and self.config["num_attention_heads"] == self.heads
        )
        assert self.config["local_attention"] == 128 and self.config["num_hidden_layers"] == 28
        self.weights = mx.load(str(directory / "model.safetensors"))
        self.precision = precision
        self.batch_size = batch_size
        self.attention = attention
        self.weights = {
            k: v.astype(getattr(mx, precision)) if k.startswith("backbone.") else v.astype(mx.float32)
            for k, v in self.weights.items()
        }
        mx.eval(self.weights)
        self.tokenizer = AutoTokenizer.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False
        )
        self.threshold = self.cfg["score_offsets"]["overall"]["fpr_0.01"]
        # Parent's result formatter only needs cfg, not a second model.
        from types import SimpleNamespace

        self.model = SimpleNamespace(cfg=self.cfg)
        self.forward = mx.compile(self.forward)

    def norm(self, x, name, eps=1e-5):
        return mx.fast.layer_norm(x, self.weights[name + ".weight"], self.weights.get(name + ".bias"), eps)

    def linear(self, x, name):
        return x @ self.weights[name + ".weight"].T

    def local(self, q, k, v, valid_length=None):
        b, h, length, d = q.shape
        tile = self.tile_size
        radius = 64
        n = (length + tile - 1) // tile
        padded = n * tile
        qt = (
            mx.pad(q, [(0, 0), (0, 0), (0, padded - length), (0, 0)])
            .reshape(b, h, n, tile, d)
            .transpose(0, 2, 1, 3, 4)
            .reshape(b * n, h, tile, d)
        )
        indices = mx.arange(n)[:, None] * tile + mx.arange(tile + 2 * radius)[None, :]

        def halo(x):
            x = mx.pad(x, [(0, 0), (0, 0), (radius, padded - length + radius), (0, 0)])
            return x[:, :, indices, :].transpose(0, 2, 1, 3, 4).reshape(b * n, h, tile + 2 * radius, d)

        kp = mx.arange(tile + 2 * radius)[None, :] - radius
        qp = mx.arange(tile)[:, None]
        absolute = mx.arange(n)[:, None, None] * tile + kp
        mask = (mx.abs(qp - kp) <= radius)[None, :, :] & (absolute >= 0) & (absolute < length)
        if valid_length is not None:
            mask = mask & (absolute < valid_length)
        mask = mx.tile(mask, (b, 1, 1))[:, None, :, :]
        out = mx.fast.scaled_dot_product_attention(qt, halo(k), halo(v), scale=0.125, mask=mask)
        return (
            out.reshape(b, n, h, tile, d).transpose(0, 2, 1, 3, 4).reshape(b, h, padded, d)[:, :, :length, :]
        )

    def activation(self, a, gate):
        gelu = (0.5 * a.astype(mx.float32) * (1 + mx.erf(a.astype(mx.float32) / math.sqrt(2)))).astype(
            a.dtype
        )
        return gelu * gate

    def rotate(self, t, cos, sin):
        return t * cos + mx.concatenate([-t[..., 32:], t[..., :32]], -1) * sin

    def forward(self, ids, valid_length=None):
        w = self.weights
        x = self.norm(w["backbone.embeddings.tok_embeddings.weight"][ids], "backbone.embeddings.norm")
        b, length, _ = x.shape
        frequency = mx.arange(0, 64, 2, dtype=mx.float32) / 64
        angles = mx.arange(length, dtype=mx.float32)[:, None] * (1.0 / (160000.0**frequency))[None, :]
        cos = mx.concatenate([mx.cos(angles)] * 2, -1).astype(x.dtype)[None, None, :, :]
        sin = mx.concatenate([mx.sin(angles)] * 2, -1).astype(x.dtype)[None, None, :, :]

        def rope(t):
            return self.rotate(t, cos, sin)

        for i in range(28):
            name = f"backbone.layers.{i}"
            a = x if i == 0 else self.norm(x, name + ".attn_norm")
            qkv = self.linear(a, name + ".attn.Wqkv").reshape(b, length, 3, self.heads, 64)
            q, k, v = [qkv[:, :, j].transpose(0, 2, 1, 3) for j in range(3)]
            q, k = rope(q), rope(k)
            if i % 3 == 0:
                mask = None if valid_length is None else mx.arange(length)[None, None, None, :] < valid_length
                out = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, mask=mask)
            elif self.attention == "dense":
                pos = mx.arange(length)
                mask = mx.abs(pos[:, None] - pos[None, :]) <= 64
                if valid_length is not None:
                    mask = mask & (pos[None, :] < valid_length)
                out = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, mask=mask)
            else:
                out = self.local(q, k, v, valid_length)
            x = x + self.linear(
                out.transpose(0, 2, 1, 3).reshape(b, length, self.hidden_size), name + ".attn.Wo"
            )
            a, gate = mx.split(self.linear(self.norm(x, name + ".mlp_norm"), name + ".mlp.Wi"), 2, -1)
            x = x + self.linear(self.activation(a, gate), name + ".mlp.Wo")

        x = self.norm(x, "backbone.final_norm").astype(mx.float32)
        style = self.norm(self.linear(x, "style_proj"), "style_ln")
        tau = mx.exp(mx.clip(w["log_tau"], -4, 4))

        def dist(p):
            return (
                mx.sum(style * style, -1, keepdims=True) - 2 * style @ p.T + mx.sum(p * p, -1)[None, None, :]
            )

        human_logits = -tau * dist(w["human_anchors"])
        human = (
            mx.max(human_logits, -1, keepdims=True)
            if self.cfg.get("null_reduction") == "max"
            else mx.logsumexp(human_logits, -1, keepdims=True)
        )
        family = -tau * dist(w["family_protos"]) + w["family_bias"][None, None, :]
        return self.cfg["tau_agg"] * mx.logsumexp(mx.clip(family - human, -30, 30) / self.cfg["tau_agg"], -1)

    def scores(self, encoded, max_windows=None):
        ids = encoded["input_ids"]
        windows = [ids[i : i + 2046] for i in range(0, len(ids), 2046)]
        if max_windows:
            windows = windows[:max_windows]
        values = []
        i = 0
        while i < len(windows):
            group = windows[i : i + self.batch_size]
            if len(group[-1]) != len(group[0]):
                group = group[:-1]
            if not group:
                group = [windows[i]]
            real_width = len(group[0]) + 2
            width = math.ceil(real_width / 128) * 128
            inputs = mx.array(
                [
                    [self.tokenizer.cls_token_id, *w, self.tokenizer.sep_token_id]
                    + [self.tokenizer.pad_token_id] * (width - real_width)
                    for w in group
                ]
            )
            valid = None if width == real_width else mx.array(real_width, dtype=mx.int32)
            result = self.forward(inputs, valid)[:, 1 : real_width - 1].reshape(-1)
            mx.eval(result)
            values.append(np.array(result))
            i += len(group)
        return torch.from_numpy(np.concatenate(values))
