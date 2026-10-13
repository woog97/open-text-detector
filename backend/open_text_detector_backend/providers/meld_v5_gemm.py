"""Shape-specialized Metal GEMM using pinned MLX Steel primitives and custom dispatch.

The installed MLX headers retain their Apple/MIT and third-party license notices.
FP32 accumulation, exact unquantized weights; fallback handles unsupported shapes.
"""

from functools import lru_cache
from pathlib import Path
import re
import mlx.core as mx


@lru_cache(None)
def header():
    root = Path(mx.__file__).parent / "include"
    included = set()
    pattern = re.compile(r'^\s*#include "([^"]+)"')

    def visit(name, emit):
        if name in included:
            return ""
        included.add(name)
        lines = []
        for line in (root / name).read_text().splitlines():
            m = pattern.match(line)
            if m:
                lines.append(visit(m[1], emit))
            elif line.strip() != "#pragma once":
                lines.append(line)
        return "\n".join(lines) + "\n" if emit else ""

    # MLX custom kernels automatically include common utils and its dependencies.
    visit("mlx/backend/metal/kernels/utils.h", False)
    return visit("mlx/backend/metal/kernels/steel/gemm/gemm.h", True)


@lru_cache(None)
def kernel():
    return mx.fast.metal_kernel(
        name="meld_specialized_gemm",
        input_names=["A", "B"],
        output_names=["D"],
        header=header(),
        source="""
        using G = mlx::steel::GEMMKernel<T,T,BM,BN,BK,WM,WN,false,true,true,true,float>;
        uint lane=thread_index_in_simdgroup;
        uint sg=simdgroup_index_in_threadgroup;
        uint tm=(threadgroup_position_in_grid.y<<SW)+(threadgroup_position_in_grid.x & ((1<<SW)-1));
        uint tn=threadgroup_position_in_grid.x>>SW;
        if(tm>=M/BM || tn>=N/BN) return;
        uint m=tm*BM;
        uint n=tn*BN;
        threadgroup T As[G::tgp_mem_size_a];
        threadgroup T Bs[G::tgp_mem_size_b];
        thread typename G::mma_t mma(sg,lane);
        thread typename G::loader_a_t la(A+m*K,K,As,sg,lane);
        thread typename G::loader_b_t lb(B+n*K,K,Bs,sg,lane);
        for(int k=0;k<K/BK;++k){
            threadgroup_barrier(mem_flags::mem_threadgroup);
            la.load_unsafe(); lb.load_unsafe();
            threadgroup_barrier(mem_flags::mem_threadgroup);
            mma.mma(As,Bs);
            la.next(); lb.next();
        }
        threadgroup_barrier(mem_flags::mem_none);
        mma.store_result(D+m*N+n,N);
    """,
    )


def matmul(x, w, tiles):
    bm, bn, bk, wm, wn = tiles[:5]
    sw = tiles[5] if len(tiles) > 5 else 0
    m, k = x.shape
    n = w.shape[0]
    if x.dtype != mx.float16 or m % bm or n % bn or k % bk:
        return x @ w.T
    return kernel()(
        inputs=[x, w],
        template=[
            ("T", mx.float16),
            ("BM", bm),
            ("BN", bn),
            ("BK", bk),
            ("WM", wm),
            ("WN", wn),
            ("K", k),
            ("N", n),
            ("M", m),
            ("SW", sw),
        ],
        grid=(n // bn * (1 << sw) * 32, ((m // bm + (1 << sw) - 1) >> sw) * wm * wn, 1),
        threadgroup=(32, wm * wn, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )[0]
