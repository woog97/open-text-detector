"""Measured, opt-in native v5 execution experiments. No checkpoint files are modified."""

import mlx.core as mx
from .meld_v8_mlx import MeldV8MLX


def strided_local(native, q, k, v, valid_length=None):
    b, h, length, d = q.shape
    tile = native.tile_size
    blocks = (length + tile - 1) // tile
    padded = blocks * tile
    qt = mx.contiguous(mx.pad(q, [(0, 0), (0, 0), (0, padded - length), (0, 0)])).reshape(
        b * h, blocks, tile, d
    )

    def halo(x):
        x = mx.contiguous(mx.pad(x, [(0, 0), (0, 0), (64, padded - length + 64), (0, 0)]))
        return mx.as_strided(
            x, shape=(b * h, blocks, tile + 128, d), strides=((padded + 128) * d, tile * d, d, 1)
        )

    kp = mx.arange(tile + 128)[None, :] - 64
    qp = mx.arange(tile)[:, None]
    absolute = mx.arange(blocks)[:, None, None] * tile + kp
    mask = (mx.abs(qp - kp) <= 64)[None, :, :] & (absolute >= 0) & (absolute < length)
    if valid_length is not None:
        mask = mask & (absolute < valid_length)
    out = mx.fast.scaled_dot_product_attention(qt, halo(k), halo(v), scale=0.125, mask=mask[None, :, :, :])
    return out.reshape(b, h, padded, d)[:, :, :length, :]


def configure(native, options):
    options = dict(options)
    unknown = set(options) - {
        "strided_attention",
        "pad_mlp",
        "transpose_weights",
        "quant_bits",
        "quant_group",
        "quant_scope",
        "chunk_rows",
        "cache_mb",
        "gemm_tiles",
    }
    if unknown:
        raise ValueError(f"Unknown native options: {unknown}")
    mx.set_cache_limit(int(options.get("cache_mb", 1024)) * 1024 * 1024)
    if options.get("strided_attention"):
        native.local = lambda q, k, v, valid_length=None: strided_local(native, q, k, v, valid_length)
    padding = int(options.get("pad_mlp", 0))
    if padding:
        if padding < 2624 or padding % 64:
            raise ValueError("Invalid padded MLP width")
        for i in range(28):
            prefix = f"backbone.layers.{i}.mlp"
            a, g = mx.split(native.weights[prefix + ".Wi.weight"], 2, axis=0)
            native.weights[prefix + ".Wi.weight"] = mx.concatenate(
                [mx.pad(x, [(0, padding - 2624), (0, 0)]) for x in (a, g)], axis=0
            )
            native.weights[prefix + ".Wo.weight"] = mx.pad(
                native.weights[prefix + ".Wo.weight"], [(0, 0), (0, padding - 2624)]
            )
    original = native.linear
    transposed = {}
    quantized = {}
    bits = int(options.get("quant_bits", 0))
    group = int(options.get("quant_group", 64))
    if bits not in (0, 4, 6, 8):
        raise ValueError("Invalid quantization precision")
    for name, w in list(native.weights.items()):
        if not name.startswith("backbone.layers.") or not name.endswith(".weight") or w.ndim != 2:
            continue
        key = name[:-7]
        if bits and (options.get("quant_scope", "all") == "all" or ".mlp." in name):
            quantized[key] = mx.quantize(w, group_size=group, bits=bits)
        elif options.get("transpose_weights"):
            transposed[key] = mx.contiguous(w.T)
    mx.eval(native.weights, transposed, quantized)
    chunk = int(options.get("chunk_rows", 0))
    gemm_tiles = options.get("gemm_tiles", {})

    def linear(x, name):
        shape = x.shape
        y = x.reshape(-1, shape[-1])

        def multiply(z):
            if name in quantized:
                w, s, b = quantized[name]
                return mx.quantized_matmul(z, w, s, b, group_size=group, bits=bits)
            weights = native.weights[name + ".weight"]
            shape_key = f"{weights.shape[1]}x{weights.shape[0]}"
            if shape_key in gemm_tiles:
                from .meld_v5_gemm import matmul

                return matmul(z, weights, gemm_tiles[shape_key])
            if name in transposed:
                return z @ transposed[name]
            return original(z, name)

        if chunk and y.shape[0] > chunk:
            out = mx.concatenate([multiply(y[i : i + chunk]) for i in range(0, y.shape[0], chunk)], axis=0)
        else:
            out = multiply(y)
        return out.reshape(*shape[:-1], out.shape[-1])

    if transposed or quantized or chunk or gemm_tiles:
        native.linear = linear
    native.forward = mx.compile(lambda ids, valid_length=None: MeldV8MLX.forward(native, ids, valid_length))
