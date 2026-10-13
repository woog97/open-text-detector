"""Independent 100-paper FP32 versus optimized validation, including PGF output."""
import argparse
import gc
import hashlib
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
from open_text_detector_backend.result_codec import decode, encode

p = argparse.ArgumentParser()
p.add_argument("--version", default="v5")
args = p.parse_args()
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
data = json.loads((ROOT / "research/benchmarks/meld-cuda/holdout.json").read_text())
out = ROOT / "research/benchmarks/meld-cuda" / args.version
reference = []
runner = MeldCudaRunner(ROOT / f"models/meld-{args.version}", "float32", 1, "sdpa", "none")
runner.predict(data[0]["text"])
baseline_seconds = 0


def compact(result):
    positions = [(x["start"], x["end"]) for name in ["tokens", "segments"] for x in result[name]]
    return {"offset_hash": hashlib.sha256(json.dumps(positions).encode()).hexdigest(),
            "scores": np.array([x["raw_score"] for x in result["tokens"]]),
            "labels": np.array([x["label"] == "ai_evidence" for x in result["segments"]]),
            "label": result["label"], "raw_score": result["raw_score"],
            "source_tokens": result["source_tokens"], "windows": result["windows"]}


for i, paper in enumerate(data):
    assert hashlib.sha256(paper["text"].encode()).hexdigest() == paper["text_sha256"]
    start = time.perf_counter()
    result = runner.predict(paper["text"])
    baseline_seconds += time.perf_counter() - start
    reference.append(compact(result))
    if (i + 1) % 10 == 0:
        print("BASELINE", args.version, i + 1, baseline_seconds, flush=True)
del runner, result
gc.collect()
torch.cuda.empty_cache()
runner = MeldCudaRunner(ROOT / f"models/meld-{args.version}", "float16", 2, "triton", "max-autotune")
start = time.perf_counter()
runner.predict_many([x["text"] for x in data[:10]])
warmup = time.perf_counter() - start
rows = []
optimized_seconds = 0
serialization_seconds = 0
serialized_bytes = 0
for offset in range(0, len(data), 10):
    group = data[offset:offset + 10]
    start = time.perf_counter()
    results = runner.predict_many([x["text"] for x in group])
    optimized_seconds += time.perf_counter() - start
    for i, (paper, result) in enumerate(zip(group, results, strict=True), offset):
        actual, prior = compact(result), reference[i]
        assert actual["offset_hash"] == prior["offset_hash"]
        assert actual["source_tokens"] == prior["source_tokens"]
        assert actual["windows"] == prior["windows"]
        delta = np.abs(actual["scores"] - prior["scores"])
        rows.append({"text_sha256": paper["text_sha256"], "source_tokens": actual["source_tokens"],
            "document_label_matches": actual["label"] == prior["label"],
            "document_score_error": abs(actual["raw_score"] - prior["raw_score"]),
            "mean_token_error": float(delta.mean()), "max_token_error": float(delta.max()),
            "sentence_label_changes": int((actual["labels"] != prior["labels"]).sum()),
            "sentences": len(prior["labels"]), "offsets_match": True})
        result["text_sha256"] = paper["text_sha256"]
        result["pdf_sha256"] = paper["pdf_sha256"]
        start = time.perf_counter()
        blob = encode(result, level=3)
        serialization_seconds += time.perf_counter() - start
        serialized_bytes += len(blob)
        assert decode(blob) == result
    report = {"version": args.version, "papers": rows, "baseline_seconds": baseline_seconds,
              "optimized_seconds": optimized_seconds, "warmup_seconds": warmup,
              "serialization_seconds": serialization_seconds, "serialized_bytes": serialized_bytes,
              "document_label_changes": sum(not x["document_label_matches"] for x in rows),
              "sentence_label_changes": sum(x["sentence_label_changes"] for x in rows),
              "sentences": sum(x["sentences"] for x in rows),
              "source_tokens": sum(x["source_tokens"] for x in rows)}
    (out / "holdout-validation.json").write_text(json.dumps(report, indent=2))
    print("OPTIMIZED", args.version, len(rows), optimized_seconds,
          "doc_changes", report["document_label_changes"], flush=True)
