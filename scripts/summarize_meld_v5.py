"""Summarize the completed full-paper runs and preserve the selected runtime profile."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/benchmarks/meld-v5"
sys.path.insert(0, str(ROOT / "backend"))
from open_text_detector_backend.result_codec import decode

localized_counts = [len(np.load(OUT / f"baseline-{i}.npy")) for i in range(10)]
tags = [
    "baseline",
    "current",
    "current-repeat",
    "optimized",
    "strict",
    "bfloat16-experimental",
]
rows = {tag: json.loads((OUT / (tag + ".json")).read_text()) for tag in tags}
assert all(len(v) == 10 for v in rows.values())
tokens = sum(r["source_tokens"] for r in rows["baseline"])
assert all(sum(r["source_tokens"] for r in v) == tokens for v in rows.values())
summary = {}
for tag, rs in rows.items():
    seconds = sum(r["seconds"] for r in rs)
    summary[tag] = {
        "seconds": seconds,
        "tokens_per_second": tokens / seconds,
        "papers_per_hour": 36000 / seconds,
        "load_seconds": rs[0]["load_seconds"],
        "peak_process_rss_bytes": max(r["peak_rss"] for r in rs),
    }
    if tag != "baseline":
        summary[tag].update(
            document_labels_matching=sum(r["document_label_matches"] for r in rs),
            max_document_score_error=max(r["document_score_error"] for r in rs),
            sentence_label_changes=sum(r["sentence_label_changes"] for r in rs),
            sentences=sum(r["sentences"] for r in rs),
            token_label_changes=sum(r["token_threshold_flips"] for r in rs),
            max_token_error=max(r["max_token_error"] for r in rs),
            mean_token_error=sum(
                r["mean_token_error"] * n
                for r, n in zip(rs, localized_counts, strict=True)
            )
            / sum(localized_counts),
        )
direct = {
    "document_labels_matching": 0,
    "sentence_label_changes": 0,
    "token_label_changes": 0,
}
for i in range(10):
    a = decode((OUT / f"current-{i}.pgf").read_bytes())
    b = decode((OUT / f"optimized-{i}.pgf").read_bytes())
    direct["document_labels_matching"] += a["label"] == b["label"]
    for kind, field in [
        ("segments", "sentence_label_changes"),
        ("tokens", "token_label_changes"),
    ]:
        direct[field] += sum(
            x["label"] != y["label"] for x, y in zip(a[kind], b[kind], strict=True)
        )
summary["optimized_vs_current"] = direct
selection = json.loads((OUT / "selected-settings.json").read_text())
settings = selection["settings"]
assert summary["optimized"]["document_labels_matching"] == 10
profile = {
    "version": "v5",
    "revision": rows["baseline"][0]["inference"]["revision"],
    "settings": settings,
    "cache_limit_bytes": 1 << 30,
    "short_window_bucket": 128,
    "overlap": 256,
    "input_cap": None,
    "benchmark": "research/benchmarks/meld-v5/README.md",
}
(ROOT / "models/meld-v5/runtime-profile.json").write_text(json.dumps(profile, indent=2))
strict_profile = dict(
    profile,
    settings={
        "runtime": "mlx",
        "precision": "float32",
        "attention": "tiled",
        "batch_size": 1,
    },
)
(ROOT / "models/meld-v5/runtime-full-precision.json").write_text(
    json.dumps(strict_profile, indent=2)
)
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
fast = summary["optimized"]
strict = summary["strict"]
bf = summary["bfloat16-experimental"]
ratios = [
    summary[t]["seconds"] / fast["seconds"] for t in ["current", "current-repeat"]
]
main = json.loads((OUT / "sweep.json").read_text())
tiles = json.loads((OUT / "tile-sweep.json").read_text())
count = len(main) + len(tiles) + len(list(OUT.glob("fast-rope*.json")))
lines = [
    "# MELD v5 throughput optimization",
    "",
    "Benchmarked the existing pinned v5 checkpoint on the same ten complete papers as the v8 study. The app was already configured for PyTorch FP16, batch size 1; that is the practical performance baseline. FP32 is retained separately as the numerical reference.",
    "",
    f"All ten papers contain **{tokens:,} source tokens**. No input truncation, dropped spans, changed overlap, or skipped model layers. Every implementation retains the existing 256-token overlap, most-context token selection (including its tie rule), global top-quartile pooling, v5 log-sum-exp human-anchor reduction, and exact character offsets. Window overlap causes additional model work beyond the unique-token counts below.",
    "",
    "| Implementation | Seconds / 10 papers | Unique tokens/s | Papers/hour |",
    "|---|---:|---:|---:|",
]
for tag, label in [
    ("baseline", "Original FP32 reference"),
    ("current", "Existing FP16 configuration"),
    ("current-repeat", "Existing FP16 isolated repeat"),
    ("optimized", "Optimized native MLX FP16"),
    ("strict", "Native MLX FP32"),
    ("bfloat16-experimental", "Native MLX BF16, batch 8 (experimental)"),
]:
    r = summary[tag]
    lines.append(
        f"| {label} | {r['seconds']:.2f} | {r['tokens_per_second']:,.0f} | {r['papers_per_hour']:,.0f} |"
    )
lines += [
    "",
    f"Best measured settings: `{json.dumps(settings, sort_keys=True)}`. Throughput is **{min(ratios):.2f}–{max(ratios):.2f}×** the current v5 settings. These are hardware/workload-specific measurements, not a universal maximum.",
    "",
    "Times include tokenization, synchronized model inference, overlapping-window stitching, document pooling, and localization. They exclude loading, extraction, input decompression and result compression. Current/optimized runs each use two repetitions per paper and sum the medians; the full-precision alternative and final current-setting repeat use one. Model load times are retained in the JSON. GPU jobs ran sequentially for the reported runs. An initial exploratory baseline had a briefly overlapping job and was replaced entirely by the isolated reference run.",
    "",
    "## Fidelity and position maps",
    "",
    f"- Optimized: **{fast['document_labels_matching']}/10 document labels** match FP32. Maximum document-score difference: {fast['max_document_score_error']:.6f}.",
    f"- Optimized: **{fast['sentence_label_changes']}/{fast['sentences']:,} sentence labels** and {fast['token_label_changes']} localized token labels differ from FP32. Maximum localized token-score difference: {fast['max_token_error']:.6f}; token-weighted mean absolute difference: {fast['mean_token_error']:.6f}.",
    f"- Existing FP16 also differs from FP32: sentence-label changes = {summary['current']['sentence_label_changes']}; token-label changes = {summary['current']['token_label_changes']}. Reduced precision is not bit-identical.",
    f"- MLX FP32 alternative: {strict['document_labels_matching']}/10 document labels match; {strict['sentence_label_changes']} sentence-label changes; maximum document-score difference {strict['max_document_score_error']:.8f}.",
    f"- Compared directly with current FP16: {direct['document_labels_matching']}/10 document labels match, {direct['sentence_label_changes']} sentence-label changes, {direct['token_label_changes']} token-label changes.",
    f"- BF16 experiment: {bf['document_labels_matching']}/10 document labels match FP32; {bf['sentence_label_changes']} sentence changes, {bf['token_label_changes']} token changes, and maximum document-score difference {bf['max_document_score_error']:.6f}.",
    "- Every run asserts that all token and sentence offsets match the reference. Outputs retain content-addressed links to the original PDF position maps.",
    "- Agreement checks verify implementation fidelity, not classifier accuracy. Existing long-document and span-calibration limitations remain unchanged.",
    "- Final verification: 83 backend tests passed, lint passed, and the reusable CLI reproduced the first benchmark paper's token scores and segments exactly. Compressed output, source text hash, and complete PDF position-map references were verified. See `verification.json`.",
    "",
    "## Profiling and experiments",
    "",
    f"- Screened **{count} configurations** using two entire papers (30,041 source tokens), then validated selected configurations on all ten. Detailed logs and failed candidates are retained.",
    "- Compared PyTorch and MLX, FP32/FP16/BF16, and batches 1/2/4/8. Kept evidence-head arithmetic in FP32.",
    "- Profiled attention and feed-forward layers using GPU-synchronized hooks. Tiled attention attends to precisely the same ±64 tokens while avoiding a dense local attention matrix. Timings are in `profile-dense.json` and `profile-tiled.json`.",
    "- Tested local attention tile widths 32/64/128/256/512, the custom fused Metal sliding-attention kernel, and PyTorch Metal/fast-math switches.",
    "- Wrote and tested a fused Metal GELU/gating kernel, preserving the half-precision rounding point between activation and gating. Compared it with compiled MLX elementwise fusion; see `geglu-micro.json` and `tile-sweep.json`.",
    "- Tested native fused rotary-position encoding as a separate numerical/performance experiment. Selection and complete settings are recorded in `selected-settings.json`.",
    "- Neither custom kernel beat compiled MLX in the selected configuration. BF16 was no faster in the full-paper confirmation and produced more label changes, so it was not selected.",
    "- Kept the allocator cache bounded to approximately 1 GiB; short windows use masked buckets of 128 tokens. Long v5 papers already use full-width final windows, avoiding a new compiled shape per paper.",
    "- Reused the reviewed v8 encoder implementation with explicit v5 dimensions and head reduction. Short-window validation compares real weights against PyTorch; the shared-code v8 regression must reproduce an entire saved paper exactly.",
    "",
    "## Use",
    "",
    "```sh",
    "backend/.venv/bin/python scripts/classify_meld_v5.py path/to/positioned-paper.pgf",
    "```",
    "",
    "The runner loads the model once for all supplied paths, reads `models/meld-v5/runtime-profile.json`, and writes verified compressed results to `research/classifications/meld-v5-local`. Add `--reference` to use the original PyTorch FP32 implementation. For the native FP32 alternative, pass `--profile models/meld-v5/runtime-full-precision.json --output-dir research/classifications/meld-v5-full-precision`. This is opt-in: the running app, cached classifications, and public website have not been switched.",
    "",
    "Reproduce with `scripts/run_meld_v5_study.py`, then `scripts/run_meld_v5_followup.py`, then `scripts/final_benchmark_meld_v5.py`, sequentially using the backend Python environment. Hardware and package versions are recorded in `environment.json`. Full-paper results are `optimized-0.pgf` through `optimized-9.pgf`.",
    "",
    "Custom kernel API references: [MLX Metal kernels](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html), [MLX rotary encoding](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.rope.html).",
]
(OUT / "README.md").write_text("\n".join(lines) + "\n")
print(json.dumps(summary, indent=2))
