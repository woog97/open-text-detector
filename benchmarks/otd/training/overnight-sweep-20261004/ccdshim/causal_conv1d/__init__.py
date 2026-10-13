"""Stand-in for the causal-conv1d package, backed by fla's Triton kernel (no CUDA build needed).

transformers' gated-delta-net layers call causal_conv1d_fn(x[B, D, T], weight[D, W], bias, activation, **kwargs) when a
package named causal_conv1d imports; otherwise they fall back to a slow F.conv1d path. For packed sequences
(packing.py) the kwargs carry cu_seq_lens_q, and the convolution restarts at every sequence boundary; the CUDA
causal-conv1d build ignores that argument, so packing needs this shim. Training only: no decode-time update op.
"""
from fla.modules.conv import causal_conv1d as _fla_causal_conv1d

__version__ = 'fla-shim'


def causal_conv1d_fn(x, weight, bias=None, activation=None, cu_seq_lens_q=None, **kwargs):
    # cu_seq_lens_q must be a named parameter: transformers drops kwargs the implementation's signature does not name.
    y, _ = _fla_causal_conv1d(x.transpose(1, 2), weight, bias, activation=activation, cu_seqlens=cu_seq_lens_q)
    return y.transpose(1, 2)


def causal_conv1d_update(*args, **kwargs):
    raise NotImplementedError('fla shim: training path only')
