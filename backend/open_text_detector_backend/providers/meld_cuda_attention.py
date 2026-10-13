"""Opt-in CUDA attention for unpadded MELD windows; local radius is unchanged.

Global layers use PyTorch SDPA. Local layers use a fused Triton kernel with
FP32 online softmax and accumulation. Imports are lazy so CPU/Mac use is unaffected.
"""

import torch
import torch.nn.functional as F


def install(backbone, mode="triton", block=64, warps=4):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def attention(module, query, key, value, attention_mask, **kwargs):
        if attention_mask is not None:
            raise ValueError("CUDA optimized attention requires unpadded windows")
        radius = module.sliding_window - 1 if module.sliding_window is not None else None
        if radius is None:
            result = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        elif mode == "triton":
            result = local_attention(query, key, value, radius, block, warps)
        elif mode == "tiled":
            from .meld_v8_attention import tiled
            result = tiled(query, key, value, radius=radius, tile=128)
        else:
            raise ValueError(mode)
        return result.transpose(1, 2).contiguous(), None

    name = "meld_cuda_" + mode
    ALL_ATTENTION_FUNCTIONS.register(name, attention)
    backbone.config._attn_implementation = name


_kernel = None


def local_attention(q, k, v, radius=64, block=64, warps=4):
    global _kernel
    import triton
    if _kernel is None:
        from .meld_cuda_kernel import sliding_attention_kernel
        _kernel = sliding_attention_kernel
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Triton attention requires FP16 or BF16")
    b, h, n, d = q.shape
    out = torch.empty((b, h, n, d), device=q.device, dtype=q.dtype)
    if n != 2048:
        from .meld_cuda_kernel import sliding_attention_tail_kernel
        sliding_attention_tail_kernel[(triton.cdiv(n, block), b * h)](
            q.contiguous(), k.contiguous(), v.contiguous(), out, n,
            D=d, R=radius, BM=block, BN=64, SCALE=d ** -0.5, num_warps=warps)
        return out
    _kernel[(triton.cdiv(n, block), b * h)](
        q, k, v, out, *q.stride(), *k.stride(), *v.stride(),
        N=n, H=h, D=d, R=radius, BM=block, BN=64,
        SCALE=d ** -0.5, num_warps=warps,
    )
    return out
