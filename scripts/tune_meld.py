"""Isolated device benchmark; never writes or replaces cached classifications."""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import torch
from open_text_detector_backend.providers.meld import Meld

p = argparse.ArgumentParser()
p.add_argument("--precision", default="float32")
p.add_argument("--batch", type=int, default=1)
p.add_argument("--all", action="store_true")
p.add_argument("--tag", default="")
p.add_argument("--repeats", type=int, default=1)
p.add_argument("--batch-list", default="")
a = p.parse_args()
out = ROOT / "research/benchmarks/meld-device"
selected = json.loads((out / "papers.json").read_text())
texts = {
    r["id"]: r["text"]
    for r in map(json.loads, (ROOT / "research/data/iclr_2023/papers.jsonl").open())
}
model = Meld("mps", ROOT / "models", precision=a.precision, batch_size=a.batch)
load_started = time.perf_counter()
model._load()
load_seconds = time.perf_counter() - load_started
# Equal warm-up for each setting; model loading is measured separately in the real queue.
model.predict({}, "A short ordinary sentence about a quiet library. " * 20)
results = []
for paper_index, paper in enumerate(selected if a.all else selected[:1]):
    text = texts[paper["paper_id"].split(":")[1]]
    batches = [int(v) for v in a.batch_list.split(",")] if a.batch_list else [a.batch]
    if paper_index % 2:
        batches.reverse()
    for batch_size in batches:
        model.batch_size = batch_size
        timings = []
        for repeat in range(a.repeats):
            start = time.perf_counter()
            r = model.predict({}, text)
            timings.append(time.perf_counter() - start)
        seconds = sorted(timings)[len(timings) // 2]
        original = json.loads((out / (paper["scan_id"] + ".json")).read_text())[
            "result"
        ]
        assert all(
            (x["start"], x["end"]) == (y["start"], y["end"])
            for x, y in zip(r["tokens"], original["tokens"], strict=True)
        )
        deltas = [
            abs(x["raw_score"] - y["raw_score"])
            for x, y in zip(r["tokens"], original["tokens"], strict=True)
        ]
        flips = sum(
            x["label"] != y["label"]
            for x, y in zip(r["segments"], original["segments"], strict=True)
        )
        metrics = {
            "paper_id": paper["paper_id"],
            "seconds": seconds,
            "repeat_seconds": timings,
            "model_load_seconds": load_seconds,
            "raw_score": r["raw_score"],
            "baseline_raw_score": original["raw_score"],
            "same_label": r["label"] == original["label"],
            "token_max_error": max(deltas),
            "token_mean_error": sum(deltas) / len(deltas),
            "token_label_changes": sum(
                x["label"] != y["label"]
                for x, y in zip(r["tokens"], original["tokens"], strict=True)
            ),
            "token_error_p99": sorted(deltas)[int(0.99 * len(deltas))],
            "sentence_label_changes": flips,
            "sentences": len(r["segments"]),
            "inference": r["inference"],
            "mps_allocated_bytes": torch.mps.current_allocated_memory(),
            "mps_driver_bytes": torch.mps.driver_allocated_memory(),
            "process_peak_rss_bytes": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
        }
        results.append(metrics)
        print(json.dumps(metrics), flush=True)
(
    out / f"tune-{a.precision}-b{a.batch}{'-all' if a.all else ''}{a.tag}.json"
).write_text(json.dumps(results, indent=2))
