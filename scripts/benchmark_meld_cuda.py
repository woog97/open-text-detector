"""Isolated CUDA benchmark with full-document coverage and offset parity checks."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import numpy as np
import torch
from open_text_detector_backend.providers.meld_cuda import MeldCudaRunner as Runner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v5", choices=["v5", "v8"])
    p.add_argument("--precision", default="float16")
    p.add_argument("--batch", default=1, type=int)
    p.add_argument("--attention", default="sdpa", choices=["sdpa", "tiled", "triton"])
    p.add_argument("--compile", default="none")
    p.add_argument("--block", default=64, type=int)
    p.add_argument("--warps", default=4, type=int)
    p.add_argument("--papers", default=10, type=int)
    p.add_argument("--repeats", default=1, type=int)
    p.add_argument("--tag", required=True)
    p.add_argument("--reference", default="baseline")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--quantization", default="none")
    args = p.parse_args()
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = ROOT / "research/benchmarks/meld-cuda" / args.version
    out.mkdir(parents=True, exist_ok=True)
    dataset = json.loads((ROOT / "research/benchmarks/meld-cuda/papers.json").read_text())[:args.papers]
    t = time.perf_counter()
    runner = Runner(ROOT / f"models/meld-{args.version}", args.precision, args.batch,
                    args.attention, args.compile, args.block, args.warps, args.quantization)
    load = time.perf_counter() - t
    # Warm the actual full-window and batch shapes before timing.
    t = time.perf_counter()
    runner.predict(dataset[0]["text"])
    torch.cuda.synchronize()
    warmup = time.perf_counter() - t
    rows = []
    for i, paper in enumerate(dataset):
        text = paper["text"]
        assert hashlib.sha256(text.encode()).hexdigest() == paper["text_sha256"]
        timings = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = runner.predict(text)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
        ref = out / f"{args.reference}-{i}.json"
        row = {"paper": paper["title"], "seconds": float(np.median(timings)),
               "repeat_seconds": timings, "source_tokens": result["source_tokens"],
               "windows": result["windows"], "phase_seconds": result["phase_seconds"],
               "raw_score": result["raw_score"], "label": result["label"]}
        if ref.exists() and args.tag != args.reference:
            prior = json.loads(ref.read_text())
            for kind in ["tokens", "segments"]:
                assert [(x["start"], x["end"]) for x in prior[kind]] == [
                    (x["start"], x["end"]) for x in result[kind]], kind
            delta = np.abs(np.array([x["raw_score"] for x in prior["tokens"]]) -
                           np.array([x["raw_score"] for x in result["tokens"]]))
            row.update(max_token_error=float(delta.max()), mean_token_error=float(delta.mean()),
                       document_label_matches=prior["label"] == result["label"],
                       document_score_error=abs(prior["raw_score"] - result["raw_score"]),
                       sentence_label_changes=sum(a["label"] != b["label"] for a, b in
                                                  zip(prior["segments"], result["segments"], strict=True)),
                       sentences=len(prior["segments"]))
        (out / f"{args.tag}-{i}.json").write_text(json.dumps(result))
        rows.append(row)
        report = {"args": vars(args), "load_seconds": load, "warmup_seconds": warmup,
                  "torch": torch.__version__, "cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(), "peak_bytes": torch.cuda.max_memory_allocated(),
                  "total_seconds": sum(r["seconds"] for r in rows), "papers": rows}
        (out / f"{args.tag}.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(row), flush=True)
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True) as prof:
            runner.predict(dataset[0]["text"])
        prof.export_chrome_trace(str(out / f"{args.tag}-trace.json"))
        (out / f"{args.tag}-profile.txt").write_text(prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=35))
    print('TOTAL', report["total_seconds"], flush=True)


if __name__ == "__main__":
    main()
