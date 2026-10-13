"""Apple GPU Laya: fused MLX primitives, compiled graph, exact marker-only final head.

The encoder preserves separate local/global RoPE frequencies. The last decision
layer needs queries and feed-forward work only at option markers: all keys and
values remain present, so this pruning is mathematically exact at inference.
"""
import math
import numpy as np
import mlx.core as mx


class LayaMLX:
    def __init__(self, directory, precision='float16', attention='dense', prune_head=True, compile=True, activation='compiled', head_precision='float32'):
        if precision not in {'float16', 'float32', 'bfloat16'}:
            raise ValueError('Unsupported precision')
        if head_precision not in {'float16', 'float32'}:
            raise ValueError('Unsupported head precision')
        self.attention, self.prune_head = attention, prune_head
        self.activation = activation
        self.head_dtype = getattr(mx, head_precision)
        self.weights = {k: v.astype(getattr(mx, precision) if k.startswith('encoder.') else self.head_dtype) for k, v in
                        mx.load(str(directory / 'model.safetensors')).items()}
        mx.eval(self.weights)
        self.run = mx.compile(self.forward) if compile else self.forward

    def norm(self, x, name):
        return mx.fast.layer_norm(x, self.weights[name + '.weight'], self.weights.get(name + '.bias'), 1e-5)

    def linear(self, x, name):
        y = x @ self.weights[name + '.weight'].T
        bias = self.weights.get(name + '.bias')
        return y if bias is None else y + bias

    def gelu(self, x):
        f = x.astype(mx.float32)
        return (.5 * f * (1 + mx.erf(f / math.sqrt(2)))).astype(x.dtype)

    def local(self, q, k, v, lengths):
        b, h, length, d = q.shape
        tile, radius = 128, 64
        n = (length + tile - 1) // tile
        padded = n * tile
        qt = mx.pad(q, [(0, 0), (0, 0), (0, padded - length), (0, 0)])
        qt = qt.reshape(b, h, n, tile, d).transpose(0, 2, 1, 3, 4).reshape(b * n, h, tile, d)
        indices = mx.arange(n)[:, None] * tile + mx.arange(tile + 2 * radius)[None, :]
        def halo(x):
            x = mx.pad(x, [(0, 0), (0, 0), (radius, padded - length + radius), (0, 0)])
            return x[:, :, indices, :].transpose(0, 2, 1, 3, 4).reshape(b * n, h, tile + 2 * radius, d)
        kp = mx.arange(tile + 2 * radius)[None, :] - radius
        qp = mx.arange(tile)[:, None]
        absolute = mx.arange(n)[:, None, None] * tile + kp
        mask = ((mx.abs(qp - kp) <= radius)[None, None, :, :] & (absolute[None] >= 0)
                & (absolute[None] < lengths[:, None, None, None]))
        out = mx.fast.scaled_dot_product_attention(qt, halo(k), halo(v), scale=.125,
                                                  mask=mask.reshape(b * n, 1, tile, tile + 2 * radius))
        return out.reshape(b, n, h, tile, d).transpose(0, 2, 1, 3, 4).reshape(b, h, padded, d)[:, :, :length]

    def forward(self, ids, lengths, markers):
        x = self.norm(self.weights['encoder.embeddings.tok_embeddings.weight'][ids], 'encoder.embeddings.norm')
        b, length, d = x.shape
        pos = mx.arange(length)
        valid = pos[None, None, None, :] < lengths[:, None, None, None]
        local_mask = (mx.abs(pos[:, None] - pos[None, :]) <= 64)[None, None] & valid
        for i in range(28):
            name = f'encoder.layers.{i}'
            a = x if i == 0 else self.norm(x, name + '.attn_norm')
            qkv = self.linear(a, name + '.attn.Wqkv').reshape(b, length, 3, 16, 64)
            q, k, v = [qkv[:, :, j].transpose(0, 2, 1, 3) for j in range(3)]
            theta = 160000. if i % 3 == 0 else 10000.
            q = mx.fast.rope(q, 64, traditional=False, base=theta, scale=1., offset=0)
            k = mx.fast.rope(k, 64, traditional=False, base=theta, scale=1., offset=0)
            if i % 3 and self.attention == 'tiled':
                a = self.local(q, k, v, lengths)
            else:
                a = mx.fast.scaled_dot_product_attention(q, k, v, scale=.125,
                                                         mask=valid if i % 3 == 0 else local_mask)
            x = x + self.linear(a.transpose(0, 2, 1, 3).reshape(b, length, d), name + '.attn.Wo')
            a, gate = mx.split(self.linear(self.norm(x, name + '.mlp_norm'), name + '.mlp.Wi'), 2, -1)
            if self.activation == 'metal':
                from .meld_v5_metal import geglu
                activated = geglu(a, gate)
            else:
                activated = self.gelu(a) * gate
            x = x + self.linear(activated, name + '.mlp.Wo')
        x = self.norm(x, 'encoder.final_norm').astype(self.head_dtype) + self.weights['type_emb.weight'][0]
        for i in range(2):
            name = f'head.layers.{i}'
            a = self.norm(x, name + '.norm1')
            w = self.weights
            qkv = a @ w[name + '.self_attn.in_proj_weight'].T + w[name + '.self_attn.in_proj_bias']
            qkv = qkv.reshape(b, length, 3, 16, 64)
            q, k, v = [qkv[:, :, j].transpose(0, 2, 1, 3) for j in range(3)]
            prune = i == 1 and self.prune_head
            if prune:
                q = q[:, :, markers, :]
                x = x[:, markers, :]
            width = 2 if prune else length
            a = mx.fast.scaled_dot_product_attention(q, k, v, scale=.125, mask=valid)
            x = x + self.linear(a.transpose(0, 2, 1, 3).reshape(b, width, d), name + '.self_attn.out_proj')
            x = x + self.linear(mx.maximum(self.linear(self.norm(x, name + '.norm2'), name + '.linear1'), 0),
                                name + '.linear2')
        if not self.prune_head:
            x = x[:, markers, :]
        return self.linear(self.gelu(self.linear(self.norm(x, 'scorer.0'), 'scorer.1')), 'scorer.3')[..., 0].astype(mx.float32)

    def probability_array(self, ids, lengths, markers, temperature):
        logits = self.run(mx.array(ids), mx.array(lengths), mx.array(markers))
        return mx.softmax(logits / temperature, axis=-1)[:, 1]

    def submit_probabilities(self, ids, lengths, markers, temperature):
        """Queue one batch so CPU preparation can overlap GPU execution."""
        values = self.probability_array(ids, lengths, markers, temperature)
        mx.async_eval(values)
        return values

    def probabilities(self, ids, lengths, markers, temperature):
        values = self.probability_array(ids, lengths, markers, temperature)
        mx.eval(values)
        return np.array(values)
