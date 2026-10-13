"""Reproducible isolated full-paper benchmark; never replaces existing classifications."""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import numpy as np
import torch
from open_text_detector_backend.pdf_extraction import find_maps, read_text, text_hash
from open_text_detector_backend.providers.meld_v5_optimized import MeldV5Runner
from open_text_detector_backend.result_codec import decode, encode

OUT = ROOT / "research/benchmarks/meld-v5"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", default="torch")
    p.add_argument("--precision", default="float32")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--attention", default="dense")
    p.add_argument("--papers", type=int, default=10)
    p.add_argument("--tag", required=True)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--micro", action="store_true")
    p.add_argument("--tile", type=int, default=128)
    p.add_argument("--fusion", action="store_true")
    p.add_argument("--fast-rope", action="store_true")
    args = p.parse_args()
    assert not args.micro, "v5 benchmarks always process complete papers"
    selected = json.loads((OUT / "papers.json").read_text())[: args.papers]
    start = time.perf_counter()
    model = MeldV5Runner(
        ROOT / "models/meld-v5",
        args.precision,
        args.batch,
        args.attention,
        args.runtime,
        args.tile,
        args.fusion,
        args.fast_rope,
    )
    torch.mps.synchronize()
    load = time.perf_counter() - start
    model.predict("This is a warmup sentence about the scientific method. " * 30)
    rows = []
    for i, paper in enumerate(selected):
        text = read_text(ROOT / paper["text_file"])
        assert text_hash(text) == paper["text_sha256"]
        torch.mps.synchronize()
        start = time.perf_counter()
        timings = []
        for repeat in range(args.repeats):
            t = time.perf_counter()
            result = model.predict(text)
            timings.append(time.perf_counter() - t)
        scores = np.array([x["raw_score"] for x in result["tokens"]], dtype=np.float32)
        torch.mps.synchronize()
        seconds = time.perf_counter() - start
        if not args.micro:
            seconds = float(np.median(timings))
        raw = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else scores
        np.save(OUT / f"{args.tag}-{i}.npy", raw)
        row = {
            "paper": paper["title"],
            "seconds": seconds,
            "source_tokens": result["inference"]["source_tokens"],
            "raw_score": result["raw_score"],
            "load_seconds": load,
            "driver_bytes": torch.mps.driver_allocated_memory(),
            "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "inference": result["inference"],
            "runtime": args.runtime,
        }
        if not args.micro:
            row["repeat_seconds"] = timings
        if args.runtime == "mlx":
            import mlx.core as mx

            row["mlx_peak_bytes"] = mx.get_peak_memory()
        baseline = OUT / f"baseline-{i}.npy"
        if baseline.exists() and args.tag != "baseline" and not args.micro:
            ref = np.load(baseline)
            assert len(ref) == len(raw)
            delta = np.abs(ref - raw)
            row.update(
                max_token_error=float(delta.max()),
                mean_token_error=float(delta.mean()),
                token_threshold_flips=int(
                    ((ref > model.threshold) != (raw > model.threshold)).sum()
                ),
            )
        if not args.micro:
            result["source_maps"] = find_maps(text, ROOT / "research/data")
            prior = OUT / f"baseline-{i}.pgf"
            if prior.exists() and args.tag != "baseline":
                reference = decode(prior.read_bytes())
                assert [(r["start"], r["end"]) for r in result["tokens"]] == [
                    (r["start"], r["end"]) for r in reference["tokens"]
                ]
                assert [(r["start"], r["end"]) for r in result["segments"]] == [
                    (r["start"], r["end"]) for r in reference["segments"]
                ]
                row["document_label_matches"] = reference["label"] == result["label"]
                row["document_score_error"] = abs(
                    reference["raw_score"] - result["raw_score"]
                )
                row["sentence_label_changes"] = sum(
                    a["label"] != b["label"]
                    for a, b in zip(
                        reference["segments"], result["segments"], strict=True
                    )
                )
                row["sentences"] = len(result["segments"])
            (OUT / f"{args.tag}-{i}.pgf").write_bytes(encode(result, level=9))
        rows.append(row)
        (OUT / f"{args.tag}.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps(row), flush=True)
    print("TOTAL", sum(r["seconds"] for r in rows), flush=True)


if __name__ == "__main__":
    main()
