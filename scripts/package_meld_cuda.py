"""Build a small CUDA worker bundle without model weights or credentials."""
import argparse
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--papers", type=Path, default=ROOT / "research/benchmarks/meld-cuda/papers.json")
p.add_argument("--output", type=Path, default=Path("/tmp/meld-cuda-worker.tar.gz"))
args = p.parse_args()
files = [ROOT / "scripts" / name for name in ["setup_meld_cuda.sh", "download_meld_r2.py",
    "run_meld_complete_gpu.py", "classify_meld_cuda.py", "classify_meld_r2_queue.py", "benchmark_meld_cuda.py", "benchmark_meld_cuda_bulk.py",
    "validate_meld_cuda_attention.py"]]
files += [ROOT / "backend/open_text_detector_backend" / name for name in ["__init__.py", "network.py", "result_codec.py"]]
files += list((ROOT / "backend/open_text_detector_backend/providers").glob("*.py"))
for version in ["v5", "v8"]:
    files += list((ROOT / f"models/meld-{version}").glob("*.json"))
    files.append(ROOT / f"research/benchmarks/vast-meld/r2-{version}-manifest.json")
with tarfile.open(args.output, "w:gz") as archive:
    for path in files:
        archive.add(path, arcname=path.relative_to(ROOT))
    archive.add(args.papers, arcname="research/benchmarks/meld-cuda/papers.json")
print(f"Worker bundle: {args.output} ({args.output.stat().st_size} bytes); no weights or credentials")
