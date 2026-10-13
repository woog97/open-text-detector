"""Measure total pipeline throughput when papers share tokenizer/GPU batches."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import numpy as np
import torch
from open_text_detector_backend.providers.meld_cuda import MeldCudaRunner

p = argparse.ArgumentParser()
p.add_argument("--version", default="v5")
p.add_argument("--batch", type=int, default=1)
p.add_argument("--precision", default="float16")
p.add_argument("--compile", default="default")
p.add_argument("--tag", required=True)
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--quantization", default="none")
args = p.parse_args()
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
data = json.loads((ROOT / "research/benchmarks/meld-cuda/papers.json").read_text())
out = ROOT / "research/benchmarks/meld-cuda" / args.version
runner = MeldCudaRunner(ROOT / f"models/meld-{args.version}", args.precision, args.batch, "triton", args.compile,
                        quantization=args.quantization)
texts = [x["text"] for x in data]
t = time.perf_counter()
runner.predict_many(texts)
torch.cuda.synchronize()
warmup = time.perf_counter() - t
timings = []
for _ in range(args.repeats):
    torch.cuda.synchronize()
    t = time.perf_counter()
    results = runner.predict_many(texts)
    torch.cuda.synchronize()
    timings.append(time.perf_counter() - t)
rows = []
for i, result in enumerate(results):
    prior = json.loads((out / f"baseline-{i}.json").read_text())
    for kind in ["tokens", "segments"]:
        assert [(x["start"], x["end"]) for x in prior[kind]] == [(x["start"], x["end"]) for x in result[kind]]
    delta = np.abs(np.array([x["raw_score"] for x in prior["tokens"]]) - np.array([x["raw_score"] for x in result["tokens"]]))
    rows.append({"paper": data[i]["title"], "source_tokens": result["source_tokens"],
                 "document_label_matches": prior["label"] == result["label"],
                 "document_score_error": abs(prior["raw_score"] - result["raw_score"]),
                 "mean_token_error": float(delta.mean()), "max_token_error": float(delta.max()),
                 "sentence_label_changes": sum(a["label"] != b["label"] for a, b in zip(prior["segments"], result["segments"], strict=True)),
                 "sentences": len(prior["segments"])})
    (out / f"{args.tag}-{i}.json").write_text(json.dumps(result))
report = {"args": vars(args), "total_seconds": float(np.median(timings)), "repeat_seconds": timings,
          "warmup_seconds": warmup, "papers": rows, "peak_bytes": torch.cuda.max_memory_allocated()}
(out / f"{args.tag}.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
