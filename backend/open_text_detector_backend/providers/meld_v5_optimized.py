"""Opt-in optimized v5 runner, preserving the existing overlap/stitching contract."""

import numpy as np
import torch
from .meld import Meld
from .checkpoints import MELD_REVISION


class MeldV5Runner(Meld):
    def __init__(
        self,
        directory,
        precision="float32",
        batch_size=1,
        attention="dense",
        runtime="torch",
        tile_size=128,
        fusion=False,
        fast_rope=False,
        advanced=None,
        pipeline_depth=0,
        parallel_streams=1,
    ):
        if runtime not in {"torch", "mlx"} or precision not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported runtime or precision")
        if batch_size not in {1, 2, 4, 8} or tile_size not in {32, 64, 128, 256, 512}:
            raise ValueError("Unsupported batch or tile size")
        if attention not in ({"dense", "tiled"} if runtime == "mlx" else {"dense", "tiled", "metal"}):
            raise ValueError("Unsupported attention implementation")
        if runtime != "mlx" and (fusion or fast_rope):
            raise ValueError("Native fusion options require MLX")
        super().__init__("mps", directory.parent, precision, batch_size)
        self.runtime, self.attention = runtime, attention
        self.tile_size, self.fusion = tile_size, fusion
        self.fast_rope = fast_rope
        self.advanced = dict(advanced or {})
        if self.advanced and runtime != "mlx":
            raise ValueError("Advanced settings require MLX")
        if pipeline_depth not in (0, 1, 2, 4, 8):
            raise ValueError("Invalid pipeline depth")
        if pipeline_depth and runtime != "mlx":
            raise ValueError("Pipelining requires MLX")
        self.pipeline_depth = pipeline_depth
        if parallel_streams not in (1, 2, 4):
            raise ValueError("Invalid stream count")
        if parallel_streams > 1 and pipeline_depth < parallel_streams:
            raise ValueError("Stream count requires adequate pipeline depth")
        self.parallel_streams = parallel_streams
        if runtime == "mlx":
            import mlx.core as mx
            from .meld_v8_mlx import MeldV8MLX

            class NativeV5(MeldV8MLX):
                revision = MELD_REVISION
                version = "v5"
                hidden_size = 1024
                heads = 16

            NativeV5.tile_size = tile_size
            if fast_rope:
                NativeV5.rotate = lambda self, t, cos, sin: mx.fast.rope(
                    t, 64, traditional=False, base=160000.0, scale=1.0, offset=0
                )
            if fusion:
                from .meld_v5_metal import geglu

                NativeV5.activation = staticmethod(geglu)
            native = NativeV5(directory, precision, batch_size, attention)
            if self.advanced:
                from .meld_v5_advanced import configure

                configure(native, self.advanced)

            class Adapter:
                cfg = native.cfg

                def token_scores(self, input_ids, attention_mask):
                    # Real windows share a length; short inputs get masked bucket padding.
                    real_width = input_ids.shape[1]
                    width = ((real_width + 127) // 128) * 128
                    inputs = mx.array(input_ids.numpy())
                    valid = None
                    if real_width != width:
                        inputs = mx.pad(
                            inputs,
                            [(0, 0), (0, width - real_width)],
                            constant_values=native.tokenizer.pad_token_id,
                        )
                        valid = mx.array(real_width, dtype=mx.int32)
                    result = native.forward(inputs, valid)[:, :real_width]
                    mx.eval(result)
                    return torch.from_numpy(np.array(result))

            self.model = Adapter()
            self.native = native
            self._streams = (
                [mx.new_stream(mx.gpu) for _ in range(parallel_streams)] if parallel_streams > 1 else []
            )
            self._stream_forwards = [
                mx.compile(lambda ids, valid=None: MeldV8MLX.forward(native, ids, valid))
                for _ in self._streams
            ]
            self.tokenizer = native.tokenizer
            self.device = "cpu"  # Only input/output staging; the encoder runs on the Apple GPU.
        else:
            self._load()
            if attention != "dense":
                from .meld_v8_attention import install

                install(self.model.backbone, attention)
                original = self.model.token_scores
                self.model.token_scores = lambda input_ids, attention_mask: original(
                    input_ids, {"full_attention": None, "sliding_attention": None}
                )
        self.threshold = self.model.cfg["score_offsets"]["overall"]["fpr_0.01"]

    def _window_batches(self, ids, starts, capacity):
        if not self.pipeline_depth:
            yield from super()._window_batches(ids, starts, capacity)
            return
        import mlx.core as mx
        from collections import deque

        groups = [starts[i : i + self.batch_size] for i in range(0, len(starts), self.batch_size)]
        pending = deque()

        def enqueue(group, index):
            windows = [
                [self.tokenizer.cls_token_id, *ids[start : start + capacity], self.tokenizer.sep_token_id]
                for start in group
            ]
            width = len(windows[0])
            padded = ((width + 127) // 128) * 128
            values = mx.array([w + [self.tokenizer.pad_token_id] * (padded - width) for w in windows])
            valid = None if width == padded else mx.array(width, dtype=mx.int32)
            if self._streams:
                slot = index % len(self._streams)
                with mx.stream(self._streams[slot]):
                    result = self._stream_forwards[slot](values, valid)[:, 1 : width - 1]
                    mx.async_eval(result)
            else:
                result = self.native.forward(values, valid)[:, 1 : width - 1]
                mx.async_eval(result)
            pending.append((group, result))

        next_group = 0
        while next_group < len(groups) or pending:
            while next_group < len(groups) and len(pending) < self.pipeline_depth:
                enqueue(groups[next_group], next_group)
                next_group += 1
            group, result = pending.popleft()
            yield group, np.array(result).tolist()

    def predict(self, text):
        result = super().predict({}, text)
        result["inference"].update(
            runtime=self.runtime,
            attention=self.attention,
            tile_size=self.tile_size,
            custom_geglu=self.fusion,
            fast_rope=self.fast_rope,
            advanced=self.advanced,
            pipeline_depth=self.pipeline_depth,
            parallel_streams=self.parallel_streams,
        )
        if self.runtime == "mlx":
            import mlx.core as mx

            result["inference"].update(
                device="mlx-gpu",
                dtype="mlx." + self.precision,
                head_dtype="mlx.float32",
                memory_after_run={
                    "active_bytes": mx.get_active_memory(),
                    "peak_bytes": mx.get_peak_memory(),
                    "cache_bytes": mx.get_cache_memory(),
                },
            )
        return result
