"""Fused bidirectional sliding attention, specialized for MELD's head dimensions."""
import triton
import triton.language as tl


@triton.jit
def sliding_attention_kernel(
    Q, K, V, OUT,
    qb: tl.constexpr, qh: tl.constexpr, qn: tl.constexpr, qd: tl.constexpr,
    kb: tl.constexpr, kh: tl.constexpr, kn: tl.constexpr, kd: tl.constexpr,
    vb: tl.constexpr, vh: tl.constexpr, vn: tl.constexpr, vd: tl.constexpr,
    N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, SCALE: tl.constexpr,
):
    tile = tl.program_id(0)
    bh = tl.program_id(1)
    batch, head = bh // H, bh % H
    m = tile * BM + tl.arange(0, BM)
    dim = tl.arange(0, D)
    q = tl.load(Q + batch * qb + head * qh + m[:, None] * qn + dim[None, :] * qd,
                mask=m[:, None] < N, other=0)
    maxv = tl.full((BM,), -float("inf"), tl.float32)
    denom = tl.full((BM,), 0., tl.float32)
    acc = tl.full((BM, D), 0., tl.float32)
    lo = tl.maximum(0, tile * BM - R) // BN * BN
    hi = tl.minimum(N, (tile + 1) * BM + R)
    for start in range(lo, hi, BN):
        n = start + tl.arange(0, BN)
        k = tl.load(K + batch * kb + head * kh + n[None, :] * kn + dim[:, None] * kd,
                    mask=n[None, :] < N, other=0)
        score = tl.dot(q, k) * SCALE
        valid = (n[None, :] < N) & (tl.abs(m[:, None] - n[None, :]) <= R)
        score = tl.where(valid, score, -1.0e6)
        newmax = tl.maximum(maxv, tl.max(score, axis=1))
        prob = tl.exp(score - newmax[:, None])
        alpha = tl.exp(maxv - newmax)
        denom = denom * alpha + tl.sum(prob, axis=1)
        acc = acc * alpha[:, None]
        v = tl.load(V + batch * vb + head * vh + n[:, None] * vn + dim[None, :] * vd,
                    mask=n[:, None] < N, other=0)
        acc += tl.dot(prob.to(v.dtype), v)
        maxv = newmax
    result = acc / denom[:, None]
    tl.store(OUT + ((batch * H + head) * N + m[:, None]) * D + dim[None, :],
             result, mask=m[:, None] < N)


@triton.jit(do_not_specialize=["N"])
def sliding_attention_tail_kernel(Q, K, V, OUT, N, D: tl.constexpr, R: tl.constexpr,
                                  BM: tl.constexpr, BN: tl.constexpr, SCALE: tl.constexpr):
    """One compiled kernel serves arbitrary short tails, instead of one per length."""
    tile = tl.program_id(0)
    bh = tl.program_id(1)
    m = tile * BM + tl.arange(0, BM)
    dim = tl.arange(0, D)
    base = bh * N * D
    q = tl.load(Q + base + m[:, None] * D + dim[None, :], mask=m[:, None] < N, other=0)
    maxv = tl.full((BM,), -float("inf"), tl.float32)
    denom = tl.full((BM,), 0., tl.float32)
    acc = tl.full((BM, D), 0., tl.float32)
    lo = tl.maximum(0, tile * BM - R) // BN * BN
    hi = tl.minimum(N, (tile + 1) * BM + R)
    for start in range(lo, hi, BN):
        n = start + tl.arange(0, BN)
        k = tl.load(K + base + n[None, :] * D + dim[:, None], mask=n[None, :] < N, other=0)
        score = tl.dot(q, k) * SCALE
        valid = (n[None, :] < N) & (tl.abs(m[:, None] - n[None, :]) <= R)
        score = tl.where(valid, score, -1.0e6)
        newmax = tl.maximum(maxv, tl.max(score, axis=1))
        prob = tl.exp(score - newmax[:, None])
        alpha = tl.exp(maxv - newmax)
        denom = denom * alpha + tl.sum(prob, axis=1)
        acc = acc * alpha[:, None]
        v = tl.load(V + base + n[:, None] * D + dim[None, :], mask=n[:, None] < N, other=0)
        acc += tl.dot(prob.to(v.dtype), v)
        maxv = newmax
    tl.store(OUT + base + m[:, None] * D + dim[None, :], acc / denom[:, None], mask=m[:, None] < N)
