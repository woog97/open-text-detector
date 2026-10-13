"""Steady-state full pipeline including compressed output, with explicit warmup."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import torch
from open_text_detector_backend.providers.meld_cuda import MeldCudaRunner
from open_text_detector_backend.result_codec import encode

p = argparse.ArgumentParser()
p.add_argument("--version", default="v5")
args = p.parse_args()
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
data = json.loads((ROOT / "research/benchmarks/meld-cuda/holdout.json").read_text())
runner = MeldCudaRunner(ROOT / f"models/meld-{args.version}", "float16", 2, "triton", "max-autotune")
t = time.perf_counter()
runner.warmup()
runner.predict(data[0]["text"])
warmup = time.perf_counter() - t
repeats = []
reference = None
for repeat in range(2):
    seconds, serialization, size = 0., 0., 0
    scores = []
    for offset in range(0, len(data), 10):
        group = data[offset:offset + 10]
        t = time.perf_counter()
        results = runner.predict_many([x["text"] for x in group])
        seconds += time.perf_counter() - t
        scores += [(r["raw_score"], r["label"]) for r in results]
        t = time.perf_counter()
        for item, result in zip(group, results, strict=True):
            result["text_sha256"] = item["text_sha256"]
            result["pdf_sha256"] = item["pdf_sha256"]
            size += len(encode(result, level=3))
        serialization += time.perf_counter() - t
    if reference is not None:
        assert scores == reference, "Repeated classifications changed"
    reference = scores
    repeats.append({"inference_seconds": seconds, "serialization_seconds": serialization,
                    "total_seconds": seconds + serialization, "compressed_bytes": size})
    print(args.version, repeat, repeats[-1], flush=True)
out = {"version": args.version, "papers": len(data), "warmup_seconds": warmup, "repeats": repeats}
(ROOT / f"research/benchmarks/meld-cuda/{args.version}/steady-pipeline.json").write_text(json.dumps(out, indent=2))
