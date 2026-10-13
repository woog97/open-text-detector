import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / "research/benchmarks/meld-device"
papers = json.loads((out / "papers.json").read_text())
baselines = {
    p["paper_id"]: json.loads((out / (p["scan_id"] + ".json")).read_text())
    for p in papers
}
pairs = json.loads((out / "tune-float16-b1-all-paired.json").read_text())
groups = {b: [r for r in pairs if r["inference"]["batch_size"] == b] for b in (1, 2)}
summary = {}
for batch, rows in groups.items():
    summary[str(batch)] = {
        "total_seconds": sum(r["seconds"] for r in rows),
        "median_seconds": statistics.median(r["seconds"] for r in rows),
        "same_document_labels": sum(r["same_label"] for r in rows),
        "sentence_label_changes": sum(r["sentence_label_changes"] for r in rows),
        "sentences": sum(r["sentences"] for r in rows),
        "token_label_changes": sum(r["token_label_changes"] for r in rows),
        "max_document_raw_score_delta": max(
            abs(r["raw_score"] - r["baseline_raw_score"]) for r in rows
        ),
        "max_token_raw_score_delta": max(r["token_max_error"] for r in rows),
    }
first = papers[0]["paper_id"]
warm_old = sum(
    s["result"]["performance"]["classification_seconds"]
    for pid, s in baselines.items()
    if pid != first
)
for batch, rows in groups.items():
    summary[str(batch)]["matched_warm_speedup"] = warm_old / sum(
        r["seconds"] for r in rows if r["paper_id"] != first
    )
(out / "summary.json").write_text(json.dumps(summary, indent=2))
lines = [
    "# MELD v5 device benchmark",
    "",
    "Apple M4 Pro, 48 GB unified memory; PyTorch 2.14.0, MPS. Ten previously unclassified ICLR 2023 papers, 128,923 words. Original full-text scans are retained in the paper library.",
    "",
    "## Ten-paper paired comparison",
    "",
    "| FP16 window batch | Total inference time | Median per paper | Matched warm speedup over original FP32 |",
    "|---|---:|---:|---:|",
]
for b, v in summary.items():
    lines.append(
        f"| {b} | {v['total_seconds']:.2f} s | {v['median_seconds']:.2f} s | {v['matched_warm_speedup']:.2f}× |"
    )
lines += [
    "",
    "Batch order alternated by paper. Full text, context size, overlap, weights and classification threshold were unchanged. Matched warm speedup excludes the first original paper, whose timing included the old cold model load. Paired experiment timing excludes loading; these are local measurements, not guarantees.",
    "",
    "## Repeated single-paper batch and precision sweep",
    "",
    "| Backbone | Window batch | Median of 3 runs | GPU tensor memory | GPU driver allocation after run |",
    "|---|---:|---:|---:|---:|",
]
for path in sorted(out.glob("tune-*-final.json")):
    r = json.loads(path.read_text())[0]
    info = r["inference"]
    lines.append(
        f"| {info['dtype']} | {info['batch_size']} | {r['seconds']:.2f} s | {r['mps_allocated_bytes'] / 1e9:.2f} GB | {r['mps_driver_bytes'] / 1e9:.2f} GB |"
    )
lines += [
    "",
    "GPU allocations are snapshots after inference, not peak usage or energy measurements. The source head and positional-frequency buffers remain FP32. No document-level parallelism was introduced; the batches contain independent sliding windows.",
    "",
    "## Numerical validation",
    "",
]
for b, v in summary.items():
    lines.append(
        f"- Batch {b}: {v['same_document_labels']}/10 document labels unchanged; {v['sentence_label_changes']}/{v['sentences']} sentence labels changed; {v['token_label_changes']} token labels changed. Maximum document evidence difference: {v['max_document_raw_score_delta']:.6f}; maximum individual token evidence difference: {v['max_token_raw_score_delta']:.6f}."
    )
lines += [
    "",
    "FP16 is not bit-identical to FP32. Matching this small historical sample does not establish general detector accuracy. All ten original document verdicts were below the model’s AI-evidence threshold; that does not verify authorship.",
    "",
    "## Papers",
    "",
    "| Paper | Original FP32 runtime | FP16 batch 1 | FP16 batch 2 |",
    "|---|---:|---:|---:|",
]
for p in papers:
    pid = p["paper_id"]
    r1 = next(r for r in groups[1] if r["paper_id"] == pid)
    r2 = next(r for r in groups[2] if r["paper_id"] == pid)
    old = baselines[pid]["result"]["performance"]["classification_seconds"]
    lines.append(
        f"| [{p['title']}](http://localhost:3000/papers?paper={pid}) | {old:.2f} s{' (cold)' if pid == first else ''} | {r1['seconds']:.2f} s | {r2['seconds']:.2f} s |"
    )
(out / "REPORT.md").write_text("\n".join(lines) + "\n")
print(json.dumps(summary, indent=2))
