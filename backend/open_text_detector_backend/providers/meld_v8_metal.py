"""Fused 64-dimensional sliding attention for Apple GPUs; FP32 accumulation."""

import torch

_LIBRARY = None


def shader():
    global _LIBRARY
    if _LIBRARY is None:
        source = "#include <metal_stdlib>\nusing namespace metal;\n"
        for dtype, name in [("float", "f32"), ("half", "f16")]:
            source += """
   kernel void attention_NAME(const device TYPE* q, const device TYPE* k,
       const device TYPE* v, device TYPE* output, constant uint& length,
       uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
     uint position=row % length;
     uint base=(row/length)*length*64;
     uint offset=base+position*64+lane;
     float q0=float(q[offset]),q1=float(q[offset+32]);
     float maximum=-INFINITY, normalizer=0.0f, a0=0.0f,a1=0.0f;
     uint first=position>64 ? position-64 : 0;
     uint last=min(length-1,position+64);
     for(uint j=first;j<=last;++j){
       uint ki=base+j*64+lane;
       float score=simd_sum(q0*float(k[ki])+q1*float(k[ki+32]))*0.125f;
       float next=max(maximum,score);
       float old=exp(maximum-next), weight=exp(score-next);
       a0=a0*old+weight*float(v[ki]);a1=a1*old+weight*float(v[ki+32]);
       normalizer=normalizer*old+weight;maximum=next;
     }
     output[offset]=TYPE(a0/normalizer);output[offset+32]=TYPE(a1/normalizer);
   }
   """.replace("NAME", name).replace("TYPE", dtype)
        _LIBRARY = torch.mps.compile_shader(source)
    return _LIBRARY


def attention(q, k, v):
    assert q.shape[-1] == 64 and q.dtype in (torch.float32, torch.float16)
    q, k, v = [x.contiguous() for x in (q, k, v)]
    out = torch.empty_like(q)
    length = torch.tensor([q.shape[2]], device="mps", dtype=torch.int32)
    kernel = shader().attention_f32 if q.dtype == torch.float32 else shader().attention_f16
    kernel(q, k, v, out, length, threads=[q.numel() // 64 * 32], group_size=[32])
    return out
