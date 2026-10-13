"""Exact local attention with bounded tiles; no change to attended token pairs."""

import torch
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.integrations.sdpa_attention import sdpa_attention_forward

_MASKS = {}


def tiled(q, k, v, radius=64, tile=128):
    b, h, length, d = q.shape
    blocks = (length + tile - 1) // tile
    padded = blocks * tile
    qt = (
        F.pad(q, (0, 0, 0, padded - length))
        .reshape(b, h, blocks, tile, d)
        .permute(0, 2, 1, 3, 4)
        .reshape(b * blocks, h, tile, d)
    )

    def halos(x):
        x = F.pad(x, (0, 0, radius, padded - length + radius))
        return (
            x.unfold(2, tile + 2 * radius, tile)
            .permute(0, 2, 1, 4, 3)
            .reshape(b * blocks, h, tile + 2 * radius, d)
        )

    kt, vt = halos(k), halos(v)
    key = (b, length, radius, tile, str(q.device))
    if key not in _MASKS:
        qp = torch.arange(tile, device=q.device)[:, None]
        kp = torch.arange(tile + 2 * radius, device=q.device)[None, :] - radius
        local = (qp - kp).abs() <= radius
        absolute = torch.arange(blocks, device=q.device)[:, None, None] * tile + kp
        mask = local[None, :, :] & (absolute >= 0) & (absolute < length)
        if len(_MASKS) >= 16:
            _MASKS.pop(next(iter(_MASKS)))
        _MASKS[key] = mask.repeat(b, 1, 1).unsqueeze(1)
    y = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=_MASKS[key], dropout_p=0.0)
    return y.reshape(b, blocks, h, tile, d).permute(0, 2, 1, 3, 4).reshape(b, h, padded, d)[:, :, :length, :]


def attention(module, query, key, value, attention_mask, **kwargs):
    if module.sliding_window is None:
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
    assert attention_mask is None, "Optimized scorer only batches unpadded windows"
    if module.config._attn_implementation == "meld_v8_metal":
        from .meld_v8_metal import attention as metal

        output = metal(query, key, value)
    else:
        output = tiled(query, key, value, radius=module.sliding_window - 1)
    return output.transpose(1, 2).contiguous(), None


def install(backbone, mode):
    if mode not in ("tiled", "metal"):
        raise ValueError(mode)
    name = "meld_v8_" + mode
    ALL_ATTENTION_FUNCTIONS.register(name, attention)
    backbone.config._attn_implementation = name
