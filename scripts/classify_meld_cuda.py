"""Resumable CUDA classification of JSON bundles containing text and identity hashes.

Each input item has text and text_sha256, with optional pdf_sha256/source_maps.
Position maps remain in their existing store and are joined by those hashes.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import torch
from open_text_detector_backend.providers.meld_cuda import MeldCudaRunner
from open_text_detector_backend.result_codec import decode, encode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--version", choices=["v5", "v8"], default="v5")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--profile", type=Path)
    p.add_argument("--stop-at", type=float, help="Unix deadline: finish the current batch, then save and exit")
    p.add_argument("--paper-batch", type=int, default=10)
    args = p.parse_args()
    if args.paper_batch < 1 or args.paper_batch > 64:
        p.error("--paper-batch must be between 1 and 64")
    profile_path = args.profile or ROOT / f"models/meld-{args.version}/cuda-runtime-profile.json"
    profile = json.loads(profile_path.read_text())
    assert profile["version"] == args.version
    profile_sha = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    data = json.loads(args.input.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for item in data:
        digest = hashlib.sha256(item["text"].encode()).hexdigest()
        if digest != item["text_sha256"]:
            raise ValueError("Input text hash mismatch")
        dest = args.output_dir / (digest + ".pgf")
        if dest.exists():
            prior = decode(dest.read_bytes())
            if prior.get("text_sha256") != digest or prior.get("cuda_profile_sha256") != profile_sha:
                raise ValueError("Existing output belongs to a different input/profile; choose a new output directory")
        else:
            pending.append(item)
    if not pending:
        print("All inputs already completed and verified")
        return
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runner = MeldCudaRunner(ROOT / f"models/meld-{args.version}", **profile["settings"])
    started = time.perf_counter()
    runner.warmup()
    warmup_seconds = time.perf_counter() - started

    def save(group, results):
        rows = []
        for item, result in zip(group, results, strict=True):
            for key in ["text_sha256", "pdf_sha256", "source_maps"]:
                if key in item:
                    result[key] = item[key]
            result["cuda_profile_sha256"] = profile_sha
            blob = encode(result, level=3)
            dest = args.output_dir / (item["text_sha256"] + ".pgf")
            tmp = dest.with_suffix(".tmp")
            tmp.write_bytes(blob)
            tmp.replace(dest)
            rows.append({"text_sha256": item["text_sha256"], "bytes": len(blob), "label": result["label"]})
        return rows

    def finish(future):
        for row in future.result():
            print(json.dumps(row), flush=True)

    # Bound in-flight outputs while compression and disk writes overlap GPU work.
    completed_count = 0
    with ThreadPoolExecutor(max_workers=1) as writer:
        writes = deque()
        for offset in range(0, len(pending), args.paper_batch):
            if args.stop_at and time.time() >= args.stop_at:
                break
            group = pending[offset:offset + args.paper_batch]
            results = runner.predict_many([x["text"] for x in group])
            completed_count += len(group)
            writes.append(writer.submit(save, group, results))
            if len(writes) >= 2:
                finish(writes.popleft())
        while writes:
            finish(writes.popleft())
    print(json.dumps({"completed": completed_count, "seconds_including_compile_and_serialization":
                      time.perf_counter() - started, "warmup_seconds": warmup_seconds}), flush=True)


if __name__ == "__main__":
    main()
