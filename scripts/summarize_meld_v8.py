"""Write the reproducible v8 benchmark report and selected runtime profile."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/benchmarks/meld-v8"


def load(name):
    return json.loads((OUT / (name + ".json")).read_text())


variants = {
    name: load(name)
    for name in ["baseline", "baseline-repeat", "optimized-bounded", "strict-mlx"]
}
for rows in variants.values():
    assert len(rows) == 10
fast = variants["optimized-bounded"]
assert all(r["document_label_matches"] for r in fast)
assert all(
    sum(r["source_tokens"] for r in rows) == 236816 for rows in variants.values()
)
summary = {
    name: {
        "seconds": sum(r["seconds"] for r in rows),
        "tokens_per_second": 236816 / sum(r["seconds"] for r in rows),
        "papers_per_hour": 36000 / sum(r["seconds"] for r in rows),
        "peak_process_rss_bytes": max(r["peak_rss"] for r in rows),
    }
    for name, rows in variants.items()
}
summary["quality"] = {
    "same_document_labels": sum(r["document_label_matches"] for r in fast),
    "max_document_score_error": max(r["document_score_error"] for r in fast),
    "sentence_label_changes": sum(r["sentence_label_changes"] for r in fast),
    "sentences": sum(r["sentences"] for r in fast),
    "token_label_changes": sum(r["token_threshold_flips"] for r in fast),
    "max_token_error": max(r["max_token_error"] for r in fast),
}
summary["speedup_range"] = [
    summary[t]["seconds"] / summary["optimized-bounded"]["seconds"]
    for t in ["baseline", "baseline-repeat"]
]
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
profile = {
    "revision": "8990324abd92e1fa17072f6887ea1e5c1cef5abc",
    "runtime": "mlx",
    "backbone_precision": "float16",
    "head_precision": "float32",
    "batch_size": 1,
    "attention": "tiled",
    "window_content_tokens": 2046,
    "overlap": 0,
    "input_cap": None,
    "preprocessing": "raw text, unchanged",
    "tail_bucket_tokens": 128,
    "mlx_cache_limit_bytes": 1073741824,
    "hardware": "Apple M4 Pro, 48 GiB unified memory",
    "benchmark_summary": str((OUT / "summary.json").relative_to(ROOT)),
}
(ROOT / "models/meld-v8/runtime-profile.json").write_text(json.dumps(profile, indent=2))
lines = [
    "# MELD v8 throughput benchmark",
    "",
    f"Downloaded and checksum-verified the official v8 release: **1,028,137,822 parameters**, 4.11 GB of weights. Pinned revision: `{profile['revision']}`.",
    "",
    "Source: [official MELD release](https://huggingface.co/anon-review-meld-2026/meld/tree/8990324abd92e1fa17072f6887ea1e5c1cef5abc).",
    "",
    "## Full-paper results",
    "",
    "Ten fixed papers spanning ten venue/year groups and 6,857–16,611 words, totaling **236,816 tokenizer tokens**. Selection and original input hashes are in `papers.json`. Every paper is processed in full; tokenizer offsets are retained for PDF position maps.",
    "",
    "| Implementation | Total seconds | Tokens/second | Papers/hour |",
    "|---|---:|---:|---:|",
]
labels = {
    "baseline": "Original PyTorch FP32, first run",
    "baseline-repeat": "Original PyTorch FP32, isolated repeat",
    "optimized-bounded": "Optimized MLX FP16, bounded memory",
    "strict-mlx": "MLX FP32, bounded memory",
}
for tag, r in summary.items():
    if tag not in labels:
        continue
    lines.append(
        f"| {labels[tag]} | {r['seconds']:.2f} | {r['tokens_per_second']:,.0f} | {r['papers_per_hour']:,.0f} |"
    )
lo, hi = summary["speedup_range"]
q = summary["quality"]
lines += [
    "",
    f"Best measured configuration: **MLX FP16, FP32 evidence head, tiled attention, batch size 1**. Throughput is **{lo:.2f}–{hi:.2f}×** the observed reference runs. The optimized measurement uses two repetitions per paper and the median time per paper. Baseline variation is shown explicitly; these are measurements on this Mac, not a universal hardware limit.",
    "",
    "Warm timings include tokenization, all model windows, global score pooling, and token/sentence localization. They exclude model download/loading, extraction, input decompression and output compression. Model load time and per-phase timings are retained in each JSON result. GPU experiments ran sequentially; the initial baseline had a brief overlapping attention validation, so the subsequent uncontended baseline is included.",
    "",
    "## Numerical checks",
    "",
    f"- All **10/10 document labels** match the FP32 reference. Maximum raw document-score difference: **{q['max_document_score_error']:.6f}**.",
    f"- **{q['sentence_label_changes']}/{q['sentences']:,} sentence highlight labels** changed. {q['token_label_changes']} localized token labels changed. FP16 is not bit-identical; maximum localized-token evidence difference was {q['max_token_error']:.6f}.",
    "- Token and sentence character offsets match the baseline exactly for every paper. Result files contain references to the original text/PDF position maps.",
    f"- The full-precision MLX alternative had {sum(r['sentence_label_changes'] for r in variants['strict-mlx'])} sentence-label changes and maximum document-score error {max(r['document_score_error'] for r in variants['strict-mlx']):.8f}.",
    "- The local FP32 evidence head matched the reviewed official v8 implementation exactly on the reference check. Padding tests verify that changing padded token values has zero effect on real-token scores.",
    "- Final verification: all 83 backend tests passed; lint passed. The reusable CLI reproduced the first benchmark paper's scores and segments exactly, with verified text hashes and complete position-map references.",
    "",
    "## Optimization work",
    "",
    "- Screened 29 precision/runtime/batch/kernel configurations using the same 28,978-token workload, with detailed results in `sweep.json` and `native-sweep.json`. The screening workload uses up to eight windows from each of two papers; final full-paper measurements above do not apply that limit.",
    "- Replaced dense local attention with exact 128-query tiles and 64-token halos. Every valid token attends to the same ±64 neighbors, including boundaries.",
    "- Wrote and validated a custom fused Metal attention kernel with FP32 accumulation. It improved on dense attention but lost to tiled library attention, so it remains experimental rather than the default. Microbenchmark data and numeric errors are in `attention-microbench.json`.",
    "- Ported the reviewed encoder and evidence head to MLX, with compiled operations and native Apple GPU primitives. Kept the evidence head in FP32.",
    "- Tried FP32, FP16, BF16, window batches 1/2/4/8, and PyTorch Metal matmul/fast-math switches. Larger batches and fast-math did not beat the selected configuration.",
    "- Bounded retained allocator memory to 1 GiB and bucketed final windows to multiples of 128, with explicit key-padding masks. This limits shape specializations and avoids cache growth across thousands of papers.",
    "- Profiled the full encoder separately: after attention tiling, projections and feed-forward matrix operations dominate. Additional attention-kernel work has limited end-to-end headroom.",
    "",
    "## Scoring scope",
    "",
    "The upstream v8 canonical scoring contract uses FP32, consecutive non-overlapping 2,046-content-token windows, and an input cap of 16,384 tokens. Its default CLI also normalizes input text. This project keeps the raw extracted text and processes **all tokens** to retain complete PDF alignment. Those choices are explicit extensions beyond the shipped capped-input calibration. Speed comparisons here use the same full-paper text and window rules throughout. Agreement with the reference measures implementation fidelity, not real-world detector accuracy.",
    "",
    "## Use the optimized runner",
    "",
    "```sh",
    "backend/.venv/bin/python scripts/classify_meld_v8.py path/to/positioned-paper.pgf",
    "```",
    "",
    "It loads the model once for all supplied papers and writes checksum-verifiable compressed results under `research/classifications/meld-v8-local`. Use `--reference` for the original FP32 PyTorch path. The optional `apple-gpu` dependency group pins MLX; tested package versions and hardware are in `environment.json`. The public website and the existing v5 model selection are unchanged.",
    "",
    "Final benchmark classifications: `optimized-bounded-0.pgf` through `optimized-bounded-9.pgf`. The matching NumPy arrays and JSON timing/quality records are retained for reproduction.",
    "",
    "Reproduce:",
    "",
    "```sh",
    "backend/.venv/bin/python scripts/download_meld_v8.py",
    "backend/.venv/bin/python scripts/benchmark_meld_v8.py --runtime mlx --precision float16 --attention tiled --repeats 2 --tag optimized-bounded",
    "```",
    "",
]
(OUT / "README.md").write_text("\n".join(lines))
print(json.dumps(summary, indent=2))
