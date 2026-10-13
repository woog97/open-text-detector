"""Experimental FP8 backbone linear layers; never enabled by default.

This changes numerical precision and requires full-paper accuracy validation.
The evidence head, embeddings, normalization and attention retain their dtypes.
"""
import torch


class FP8Linear(torch.nn.Module):
    def __init__(self, source, rowwise=True):
        super().__init__()
        weight = source.weight.detach().float()
        scale = (weight.abs().amax(dim=-1, keepdim=True) if rowwise else weight.abs().amax().reshape(1))
        scale = scale.clamp_min(1e-12) / 448.0
        self.register_buffer("weight8", (weight / scale).to(torch.float8_e4m3fn).t())
        self.register_buffer("weight_scale", scale.t().contiguous() if rowwise else scale)
        self.bias = source.bias
        self.rowwise = rowwise
        self.in_features = source.in_features
        self.out_features = source.out_features

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        scale = (flat.float().abs().amax(dim=-1, keepdim=True) if self.rowwise else flat.float().abs().amax().reshape(1))
        scale = scale.clamp_min(1e-12) / 448.0
        quantized = (flat.float() / scale).to(torch.float8_e4m3fn)
        output = torch._scaled_mm(quantized, self.weight8, scale_a=scale, scale_b=self.weight_scale,
                                  out_dtype=x.dtype, use_fast_accum=False)
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(*shape[:-1], self.out_features)


def install(backbone, rowwise=True):
    for name, child in list(backbone.named_children()):
        if isinstance(child, torch.nn.Linear):
            setattr(backbone, name, FP8Linear(child, rowwise))
        else:
            install(child, rowwise)
