"""Verify final performance and output identity before promoting the local runtime profile."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from open_text_detector_backend.result_codec import decode

OUT = ROOT / "research/benchmarks/meld-v5-max"
OLD = ROOT / "research/benchmarks/meld-v5"
selection = json.loads((OUT / "selection.json").read_text())
summary = {}
for tag in ["control-full", "max-full", "control-repeat", "max-repeat"]:
    rows = json.loads((OUT / (tag + ".json")).read_text())
    assert len(rows) == 10
    tokens = sum(r["source_tokens"] for r in rows)
    seconds = sum(r["seconds"] for r in rows)
    summary[tag] = {
        "seconds": seconds,
        "tokens": tokens,
        "tokens_per_second": tokens / seconds,
        "papers_per_hour": 36000 / seconds,
        "peak_gpu_active_bytes": max(r["mlx_peak_bytes"] for r in rows),
        "max_retained_cache_bytes": max(
            r["inference"]["memory_after_run"]["cache_bytes"] for r in rows
        ),
    }
    for i in range(10):
        a = decode((OUT / f"{tag}-{i}.pgf").read_bytes())
        b = decode((OLD / f"optimized-{i}.pgf").read_bytes())
        assert a["raw_score"] == b["raw_score"] and a["label"] == b["label"], (
            tag,
            i,
            "document",
        )
        assert a["tokens"] == b["tokens"] and a["segments"] == b["segments"], (
            tag,
            i,
            "spans",
        )
        assert a["source_maps"] and a["source_maps"] == b["source_maps"], (
            tag,
            i,
            "maps",
        )
    summary[tag]["all_results_identical_to_previous_optimized"] = True
old = (summary["control-full"]["seconds"] + summary["control-repeat"]["seconds"]) / 2
new = (summary["max-full"]["seconds"] + summary["max-repeat"]["seconds"]) / 2
summary["throughput_gain_percent"] = (old / new - 1) * 100
summary["warm_papers_per_hour"] = 36000 / new
assert new < old, "Do not promote an optimization without a measured full-paper gain"
profile_path = ROOT / "models/meld-v5/runtime-profile.json"
backup = OUT / "previous-runtime-profile.json"
if not backup.exists():
    backup.write_bytes(profile_path.read_bytes())
profile = json.loads(backup.read_text())
profile["settings"].update(
    advanced=selection["advanced"],
    pipeline_depth=selection["depth"],
    parallel_streams=selection["streams"],
)
profile["environment"] = selection["env"]
profile["cache_limit_bytes"] = selection["advanced"].get("cache_mb", 1024) * 1024**2
profile["benchmark"] = "research/benchmarks/meld-v5-max/README.md"
profile_path.write_text(json.dumps(profile, indent=2))
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
main = json.loads((OUT / "sweep.json").read_text())
dispatch = json.loads((OUT / "dispatch-sweep.json").read_text())
combined = json.loads((OUT / "combined-sweep.json").read_text())
gemm = json.loads((OUT / "gemm-sweep.json").read_text()) + json.loads(
    (OUT / "gemm-swizzle.json").read_text()
)
lines = [
    "# MELD v5: second optimization round",
    "",
    f"The selected native runner gains **{summary['throughput_gain_percent']:.1f}% throughput** over the previous optimized MLX version on this Mac. All ten complete paper outputs are **identical**, including every token score, sentence score, document score, label, and PDF position-map reference.",
    "",
    "## Repeated full-paper measurements",
    "",
    "| Run | Ten-paper seconds | Unique tokens/second | Papers/hour |",
    "|---|---:|---:|---:|",
]
for tag, r in summary.items():
    if not isinstance(r, dict):
        continue
    lines.append(
        f"| {tag} | {r['seconds']:.2f} | {r['tokens_per_second']:,.0f} | {r['papers_per_hour']:,.0f} |"
    )
lines += [
    "",
    "Each run covers the same 236,816 source tokens from ten papers. Two repetitions per paper were used for the first control/candidate runs; independent control/candidate repeats used one. Timings include tokenization, model inference, overlapping-window stitching and localization. Model loading, extraction and result compression are excluded. GPU experiments ran sequentially. Rates depend on paper lengths and this Mac's conditions.",
    "",
    "## Selected changes",
    "",
    "- Queue upcoming windows asynchronously with a bounded pipeline. The GPU can work while CPU code reads and stitches the preceding output. No window, layer, or token is skipped.",
    f"- Pipeline depth: {selection['depth']}; GPU streams: {selection['streams']}. Native settings: `{json.dumps(selection['advanced'])}`.",
    f"- Metal command scheduling: `{json.dumps(selection['env'])}`. These process settings must be applied before MLX first initializes; the classification CLI does this from the saved profile.",
    "- Model weights, FP16 encoder / FP32 head, 2,046-content-token windows, 256-token overlap, most-context stitching and full-paper aggregation are unchanged.",
    "",
    "## Further experiments",
    "",
    f"- {len(main) + len(dispatch) + len(combined)} additional full-paper screening configurations, each using two complete papers; combined candidates use two repetitions per paper.",
    f"- {len(gemm)} matrix-kernel microbenchmark entries covering four actual v5 GEMM dimensions, multiple block shapes and GPU-threadgroup orderings. Built a custom Metal dispatch kernel using the pinned MLX Steel primitives, with FP32 accumulation and verified outputs. Isolated improvements did not improve whole-paper throughput beyond the selected library kernels.",
    "- Tested strided attention views, transposed weights, zero-padded feed-forward widths, and row-chunked matrix operations. These did not improve the final candidate.",
    "- Tested 8-, 6-, and 4-bit quantization of feed-forward or all encoder matrices. None improved throughput here; lower bit counts introduced larger score differences. Original weights remain intact and quantization is not enabled.",
    "- Tested pipeline depths 1/2/4/8, multiple command-buffer limits, fast synchronization, 1/2/4 GPU streams, and allocator-cache limits. Deeper pipelines and multiple GPU streams did not beat the selected setting.",
    "- Synchronized native profiling and matrix microbenchmarks identify dense matrix operations as the remaining main cost. Instrumentation adds synchronization overhead; its timings are kept separate from whole-paper results.",
    "",
    "## Validation and use",
    "",
    "- Full-paper result equality is asserted against the previous saved optimized outputs in `summarize_meld_v5_max.py`.",
    "- Dedicated boundary/repeated-request checks and regression-test status are recorded in `pipeline-validation.json` and `verification.json`.",
    "- The saved local CLI profile is updated. The running app and public website have not been switched.",
    "",
    "```sh",
    "backend/.venv/bin/python scripts/classify_meld_v5.py path/to/positioned-paper.pgf",
    "```",
    "",
    "Previous settings are preserved in `previous-runtime-profile.json`. The higher-fidelity FP32 profile remains available at `models/meld-v5/runtime-full-precision.json`.",
    "",
    "Reproduce sequentially with `scripts/profile_meld_v5_native.py`, `scripts/sweep_meld_v5_max.py`, `scripts/run_meld_v5_max_followup.py`, `scripts/sweep_meld_v5_combined.py`, and `scripts/final_meld_v5_max.py`. Then run the validation and summary scripts. The pinned MLX version and hardware match the first v5 report.",
    "",
    "Implementation references: [MLX asynchronous evaluation](https://ml-explore.github.io/mlx/build/html/python/transforms.html), [MLX scheduler controls](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/utils.h), [MLX Metal GEMM implementation](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/matmul.cpp).",
]
(OUT / "README.md").write_text("\n".join(lines) + "\n")
print(json.dumps(summary, indent=2))
