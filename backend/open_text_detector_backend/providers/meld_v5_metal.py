"""Experimental fused GELU/gate kernel; preserve the reference FP16 rounding point."""

from pathlib import Path
import mlx.core as mx

_KERNEL = None


def geglu(a, gate):
    global _KERNEL
    if _KERNEL is None:
        # MLX custom sources have no filesystem include search path. Inline the
        # installed, pinned MLX math helpers, retaining their license notices.
        headers = Path(mx.__file__).parent / "include/mlx/backend/metal/kernels"
        header = (
            "\n".join(
                line
                for name in ("expm1f.h", "erf.h")
                for line in (headers / name).read_text().splitlines()
                if not line.startswith("#")
            )
            + "\n"
        )
        _KERNEL = mx.fast.metal_kernel(
            name="meld_geglu",
            input_names=["a", "gate"],
            output_names=["out"],
            header=header,
            source="""
                uint i = thread_position_in_grid.x;
                float x = float(a[i]);
                T activated = T(0.5f * x * (1.0f + erf(x * 0.7071067811865475f)));
                out[i] = activated * gate[i];
            """,
        )
    return _KERNEL(
        inputs=[a, gate],
        template=[("T", a.dtype)],
        grid=(a.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
    )[0]
