"""Classify full extracted papers with the optimized, pinned MELD v5 runner.

Usage: backend/.venv/bin/python scripts/classify_meld_v5.py paper.pgf [more.pgf ...]
Use --reference for the original full-precision PyTorch execution path.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from open_text_detector_backend.pdf_extraction import find_maps, read_text, text_hash
from open_text_detector_backend.providers.meld_v5_optimized import MeldV5Runner
from open_text_detector_backend.result_codec import decode, encode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("papers", nargs="+", type=Path)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "research/classifications/meld-v5-local",
    )
    parser.add_argument(
        "--profile", type=Path, default=ROOT / "models/meld-v5/runtime-profile.json"
    )
    args = parser.parse_args()
    profile = args.profile
    document = {} if args.reference else json.loads(profile.read_text())
    environment = document.get("environment", {})
    supported = {
        "MLX_MAX_OPS_PER_BUFFER",
        "MLX_MAX_MB_PER_BUFFER",
        "MLX_METAL_FAST_SYNCH",
    }
    if set(environment) - supported:
        raise ValueError("Unsupported runtime environment settings")
    for name, value in environment.items():
        os.environ[name] = str(value)
    settings = document.get("settings", {})
    model = MeldV5Runner(ROOT / "models/meld-v5", **settings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in args.papers:
        text = read_text(path)
        digest = text_hash(text)
        result = model.predict(text)
        result["text_sha256"] = digest
        result["source_maps"] = find_maps(text, ROOT / "research/data")
        result["source_file"] = str(path.resolve())
        blob = encode(result)
        assert decode(blob) == result
        destination = args.output_dir / (
            digest + ("-reference" if args.reference else "") + ".pgf"
        )
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(blob)
        temporary.replace(destination)
        print(
            json.dumps(
                {
                    "input": str(path),
                    "output": str(destination),
                    "raw_score": result["raw_score"],
                    "label": result["label"],
                    "source_tokens": result["inference"]["source_tokens"],
                    "bytes": len(blob),
                    "sha256": hashlib.sha256(blob).hexdigest(),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
