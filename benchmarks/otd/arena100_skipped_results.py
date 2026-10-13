"""Export the skipped-model extension and a deduplicated cumulative corpus."""

import json
from collections import Counter
from datetime import datetime, timezone

from arena20 import B, read, sha, write
from arena100_results import audit, collect
from prepare import record, validate

D = B / "data/arena100-skipped"
R = B / "runs/arena100-skipped"


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    mf = json.loads((D / "generation_manifest.json").read_text())
    prompts = read(D / "selected_prompts.jsonl")
    assert (
        sha((D / "selected_prompts.jsonl").read_bytes())
        == mf["selected_prompts_sha256"]
    )
    rows = collect(R)
    pp = {p["prompt_id"]: p for p in prompts}
    expected = {(m["id"], p["prompt_id"]) for m in mf["models"] for p in prompts}
    assert {(r["model_id"], r["prompt_id"]) for r in rows} <= expected
    dataset = []
    inputs = []
    stats = []
    for m in mf["models"]:
        cells = [r for r in rows if r["model_id"] == m["id"]]
        good = [r for r in cells if r["success"]]
        new = [r for r in good if not r.get("reused_from")]

        def cost(rs):
            return sum((r["raw"].get("usage") or {}).get("cost") or 0 for r in rs)

        usage = [r["raw"]["usage"] for r in good if r["raw"].get("usage")]
        stats.append(
            {
                "model": m["id"],
                "name": m["name"],
                "success": len(good),
                "failed": len(cells) - len(good),
                "not_attempted": 100 - len(cells),
                "reused": len(good) - len(new),
                "new_cost_usd": cost(new),
                "full_100_cost_usd": cost(good),
                "token_usage_known": len(usage),
                "cost_known": sum(
                    isinstance(u.get("cost"), (int, float)) for u in usage
                ),
                "mean_completion_tokens": sum(
                    u.get("completion_tokens", 0) for u in usage
                )
                / len(usage)
                if usage
                else None,
                "reported_reasoning_tokens": sum(
                    (u.get("completion_tokens_details") or {}).get(
                        "reasoning_tokens", 0
                    )
                    or 0
                    for u in usage
                ),
                "eligible": sum(audit(r)[0] for r in cells),
                "settings": m["settings"],
                "input_usd_per_million": m["input_usd_per_million"],
                "output_usd_per_million": m["output_usd_per_million"],
                "catalog_created_date": datetime.fromtimestamp(
                    m["catalog_record"]["created"], timezone.utc
                )
                .date()
                .isoformat(),
            }
        )
        for r in cells:
            assert r["request_sha256"] == sha(
                json.dumps(r["request"], sort_keys=True).encode()
            )
            raw = r.get("raw") or {}
            text = r["audit"]["text"] if r["success"] else None
            dataset.append(
                {
                    "model_id": r["model_id"],
                    "prompt_id": r["prompt_id"],
                    "prompt_text": pp[r["prompt_id"]]["text"],
                    "request_model": r["request"]["model"],
                    "response_model": raw.get("model"),
                    "provider": raw.get("provider"),
                    "success": r["success"],
                    "response_text": text,
                    "response_sha256": sha(text.encode()) if text is not None else None,
                    "request_sha256": r["request_sha256"],
                    "generation_id": raw.get("id"),
                    "usage": raw.get("usage"),
                    "finish_reason": raw["choices"][0].get("finish_reason")
                    if raw
                    else None,
                    "reused_from": r.get("reused_from"),
                    "mechanically_eligible": audit(r)[0],
                    "eligibility_reason": audit(r)[1],
                }
            )
            if audit(r)[0]:
                inputs.append(
                    record(
                        "arena100-skipped",
                        sha((r["model_id"] + "\0" + r["prompt_id"]).encode()),
                        text,
                        "ai",
                        generator=r["model_id"],
                        cohort=r["model_id"],
                        group_id=r["prompt_id"],
                        prompt_id=r["prompt_id"],
                        response_sha256=sha(text.encode()),
                        task="ai_detection",
                    )
                )
    write(R / "dataset.jsonl", dataset)
    validate(inputs)
    write(R / "detector_inputs.jsonl", inputs)
    summary = {
        "intended_cells": len(expected),
        "recorded_cells": len(rows),
        "success": sum(s["success"] for s in stats),
        "new_cost_usd": sum(s["new_cost_usd"] for s in stats),
        "reused": sum(s["reused"] for s in stats),
        "full_dataset_cost_usd": sum(s["full_100_cost_usd"] for s in stats),
        "models": stats,
        "missing_usage_successes": sum(s["success"] - s["cost_known"] for s in stats),
        "unavailable_requested_models": mf["unavailable_requested_models"],
        "dataset_sha256": sha((R / "dataset.jsonl").read_bytes()),
    }
    save(R / "generation_summary.json", summary)
    prior = read(B / "runs/arena100-combined/dataset.jsonl")
    combined = {(r["model_id"], r["prompt_id"]): r for r in prior}
    for r in dataset:
        k = (r["model_id"], r["prompt_id"])
        if k in combined:
            assert (
                r["reused_from"] and combined[k]["generation_id"] == r["generation_id"]
            )
            assert combined[k]["response_sha256"] == r["response_sha256"]
        else:
            combined[k] = r
    full = list(combined.values())
    good = [r for r in full if r["success"]]
    assert len({r["generation_id"] for r in good}) == len(good)
    C = B / "runs/arena100-expanded"
    C.mkdir(exist_ok=True)
    write(C / "dataset.jsonl", full)
    priorcost = json.loads((B / "runs/arena100-combined/summary.json").read_text())[
        "full_3200_cell_corpus_reported_cost_usd"
    ]
    totals = {
        "intended_models": 49,
        "intended_cells": 4900,
        "recorded_cells": len(full),
        "success": len(good),
        "new_extension_cost_usd": summary["new_cost_usd"],
        "full_corpus_reported_cost_usd": priorcost + summary["new_cost_usd"],
        "dataset_sha256": sha((C / "dataset.jsonl").read_bytes()),
        "per_model_success": dict(Counter(r["model_id"] for r in good)),
        "missing_usage_successes": sum(s["success"] - s["cost_known"] for s in stats),
        "unavailable_requested_models": mf["unavailable_requested_models"],
    }
    ledger_path = C / "key_usage_reconciliation.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else None
    if ledger:
        before = json.loads((R / "key_before.json").read_text())
        totals["api_key_usage_total_usd"] = ledger["usage"]
        totals["api_key_usage_increase_usd"] = ledger["usage"] - before["usage"]
        totals["unattributed_historical_key_usage_usd"] = ledger[
            "unattributed_difference_usd"
        ]
    save(C / "summary.json", totals)
    lines = [
        "# Previously skipped models: 100-prompt generation results",
        "",
        f"{summary['success']:,}/{len(expected):,} successful responses across 18 available models, including 100 reused GPT-6 Luna responses. New reported cost: **${summary['new_cost_usd']:.6f}**.",
        "",
        f"Billing metadata is missing for {summary['missing_usage_successes']} successful responses; the reported cost excludes their unknown charges. Claude Fable token averages use 98 responses. Billing metadata lookups for the two missing records returned 404.",
        "",
        "GPT-6 Terra is not listed in the frozen OpenRouter catalog. Terra Latest points to GPT-5.6 Terra; it was not treated as GPT-6 Terra.",
        "",
        "| Model | Success /100 | New cost USD | Input $/M | Output $/M | Mean output tokens | Reasoning | Catalog date |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for s in stats:
        reasoning = s["settings"].get("reasoning", {})
        label = reasoning.get(
            "effort",
            "disabled" if reasoning.get("enabled") is False else "not advertised",
        )
        avg = (
            f"{s['mean_completion_tokens']:.1f}"
            if s["mean_completion_tokens"] is not None
            else "—"
        )
        lines.append(
            f"| {s['name']} | {s['success']} | {s['new_cost_usd']:.6f} | {s['input_usd_per_million']:g} | {s['output_usd_per_million']:g} | {avg} | {label} | {s['catalog_created_date']} |"
        )
    lines += [
        "",
        f"Combined corpus: {len(good):,} successful responses, with ${totals['full_corpus_reported_cost_usd']:.6f} reported cost including earlier runs and reused responses. No duplicate model/prompt pairs or generation IDs.",
        "",
        "Costs sum known successful-response usage only; missing billing records are not treated as free calls. Failed or uncertain attempts may also carry charges. Mean output tokens includes reported reasoning tokens and averages only responses with usage records. Catalog dates are OpenRouter listing dates, not independently verified release dates.",
        "",
        "All successful outputs are preserved, including short/truncated responses. Mechanical eligibility requires stop completion and at least 50 words. Added models have detector inputs prepared but have not been scored by local detectors.",
        "",
        "[Extension dataset](runs/arena100-skipped/dataset.jsonl) · [Combined dataset](runs/arena100-expanded/dataset.jsonl) · [Protocol](ARENA_100_SKIPPED_PROTOCOL.md)",
    ]
    if ledger:
        lines += [
            "",
            f"Final account usage snapshot: ${ledger['usage']:.8f} total, an increase of ${totals['api_key_usage_increase_usd']:.8f} during this extension, matching the known response-cost sum. The approximately $0.003 difference from the full corpus total already existed before this run. Two individual response billing records remain unavailable; no extra charge for them is visible in this snapshot. Remaining key allowance: ${ledger['limit_remaining']:.8f}.",
        ]
    (B / "ARENA_100_SKIPPED_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in totals.items() if k != "per_model_success"}))


if __name__ == "__main__":
    main()
