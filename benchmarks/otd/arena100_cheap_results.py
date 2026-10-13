"""Account for the cheap-roster expansion without double-counting reused calls."""

import json
from collections import Counter

from arena20 import B, read, sha, write
from arena100_results import audit, collect
from prepare import record, validate

D = B / "data/arena100-cheap"
R = B / "runs/arena100-cheap"


def main():
    manifest = json.loads((D / "generation_manifest.json").read_text())
    prompts = read(D / "selected_prompts.jsonl")
    rows = collect(R)
    expected = {(m["id"], p["prompt_id"]) for m in manifest["models"] for p in prompts}
    assert (
        len(rows) == 2000
        and {(r["model_id"], r["prompt_id"]) for r in rows} == expected
    )
    pp = {p["prompt_id"]: p for p in prompts}
    dataset, stats, detector_inputs = [], [], []
    for m in manifest["models"]:
        cells = [r for r in rows if r["model_id"] == m["id"]]
        success = [r for r in cells if r["success"]]
        new = [r for r in success if not r.get("reused_from")]
        reused = [r for r in success if r.get("reused_from")]

        def cost(xs):
            return sum((r["raw"].get("usage") or {}).get("cost") or 0 for r in xs)

        stats.append(
            {
                "model": m["id"],
                "name": m["name"],
                "request_routes": sorted({r["request"]["model"] for r in cells}),
                "success": len(success),
                "new_success": len(new),
                "reused_success": len(reused),
                "new_reported_cost_usd": cost(new),
                "reused_historical_cost_usd": cost(reused),
                "full_dataset_reported_cost_usd": cost(success),
                "cost_known_successes": sum(
                    isinstance((r["raw"].get("usage") or {}).get("cost"), (int, float))
                    for r in success
                ),
                "mechanically_eligible": sum(audit(r)[0] for r in cells),
                "input_usd_per_million": m["input_usd_per_million"],
                "output_usd_per_million": m["output_usd_per_million"],
                "requested_settings": m["settings"],
                "failure_statuses": dict(
                    Counter(
                        str(r["attempts"][-1].get("status"))
                        for r in cells
                        if not r["success"]
                    )
                ),
            }
        )
        for r in cells:
            raw = r.get("raw") or {}
            text = r["audit"]["text"] if r["success"] else None
            dataset.append(
                {
                    "prompt_id": r["prompt_id"],
                    "prompt_text": pp[r["prompt_id"]]["text"],
                    "model_id": r["model_id"],
                    "request_model": r["request"]["model"],
                    "response_model": raw.get("model"),
                    "provider": raw.get("provider"),
                    "success": r["success"],
                    "response_text": text,
                    "response_sha256": sha(text.encode()) if text is not None else None,
                    "request_sha256": r["request_sha256"],
                    "generation_id": raw.get("id"),
                    "finish_reason": raw["choices"][0].get("finish_reason")
                    if raw
                    else None,
                    "usage": raw.get("usage"),
                    "reused_from": r.get("reused_from"),
                    "mechanically_eligible": audit(r)[0],
                    "eligibility_reason": audit(r)[1],
                }
            )
            if audit(r)[0]:
                detector_inputs.append(
                    record(
                        "arena100-cheap",
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
    summary = {
        "cells": 2000,
        "success": sum(s["success"] for s in stats),
        "models": stats,
        "new_reported_cost_usd": sum(s["new_reported_cost_usd"] for s in stats),
        "reused_historical_cost_usd": sum(
            s["reused_historical_cost_usd"] for s in stats
        ),
        "full_dataset_reported_cost_usd": sum(
            s["full_dataset_reported_cost_usd"] for s in stats
        ),
        "cost_note": "Reported usage only; failed/uncertain attempts may have unreported charges. Reused calls are historical, not charged again.",
        "route_overrides": json.loads((D / "qwen_route_override.json").read_text()),
    }
    write(R / "dataset.jsonl", dataset)
    validate(detector_inputs)
    write(R / "detector_inputs.jsonl", detector_inputs)
    summary["dataset_sha256"] = sha((R / "dataset.jsonl").read_bytes())
    (R / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    base = json.loads((B / "runs/arena100/generation_summary.json").read_text())
    combined = B / "runs/arena100-combined"
    combined.mkdir(exist_ok=True)
    base_rows = [
        {**r, "success": True} for r in read(B / "runs/arena100/dataset.jsonl")
    ]
    all_rows = base_rows + dataset
    assert (
        len(all_rows)
        == len({(r["model_id"], r["prompt_id"]) for r in all_rows})
        == 3200
    )
    assert len({r["generation_id"] for r in all_rows if r["success"]}) == sum(
        r["success"] for r in all_rows
    )
    write(combined / "dataset.jsonl", all_rows)
    totals = {
        "cells": 3200,
        "successful_responses": 1200 + summary["success"],
        "new_reported_cost_for_both_100_prompt_passes_usd": base["reported_cost_usd"]
        + summary["new_reported_cost_usd"],
        "historical_reused_cost_usd": summary["reused_historical_cost_usd"],
        "full_3200_cell_corpus_reported_cost_usd": base["reported_cost_usd"]
        + summary["full_dataset_reported_cost_usd"],
    }
    ledger_path = combined / "key_usage_reconciliation.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else None
    if ledger:
        totals["api_key_usage_total_usd"] = ledger["usage"]
        totals["unattributed_key_usage_difference_usd"] = ledger[
            "key_usage_minus_known_corpus_usd"
        ]
    (combined / "summary.json").write_text(json.dumps(totals, indent=2) + "\n")
    lines = [
        "# Earlier cheap models on 100 prompts",
        "",
        f"{summary['success']:,} successful responses out of 2,000 cells across 20 models. New reported charges: **${summary['new_reported_cost_usd']:.8f}**. Historical charges for 365 reused responses: ${summary['reused_historical_cost_usd']:.8f}; those calls were not billed again.",
        "",
        "| Model | Success /100 | Reused | New cost USD | Full 100-prompt cost USD | Eligible |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in stats:
        name = s["name"]
        if len(s["request_routes"]) > 1:
            name = name.replace(" (free)", "") + " (free + capped paid route)"
        lines.append(
            f"| {name} | {s['success']} | {s['reused_success']} | {s['new_reported_cost_usd']:.8f} | {s['full_dataset_reported_cost_usd']:.8f} | {s['mechanically_eligible']} |"
        )
    lines += [
        "",
        summary["cost_note"],
        "",
        f"Together with the original 12-model Arena-100 pass: {totals['successful_responses']:,} successful responses across 3,200 intended cells. New charges across both 100-prompt passes: ${totals['new_reported_cost_for_both_100_prompt_passes_usd']:.8f}.",
        "",
        "[Cheap-model dataset](runs/arena100-cheap/dataset.jsonl) · [Combined 32-model dataset](runs/arena100-combined/dataset.jsonl) · [Protocol](ARENA_100_CHEAP_PROTOCOL.md)",
        "",
        "Eligibility is mechanical only: successful, stop-finished, at least 50 words. No exhaustive substantive-prose/refusal adjudication or native output-longer-than-input token check. Generation completion does not imply completed detector scoring.",
        "Qwen 27B's free-route successes are retained. Failed free calls may use the paid route for the identical canonical version, with the same output cap below $2/M. Actual request route and provider are stored per response; this is a serving-route change, not a different model version.",
    ]
    if ledger:
        lines += [
            "",
            f"API-key usage reconciliation: the key reports **${ledger['usage']:.8f} total usage**, versus ${totals['full_3200_cell_corpus_reported_cost_usd']:.8f} attached to saved successful responses. The ${ledger['key_usage_minus_known_corpus_usd']:.8f} difference remains unattributed; failed/timed-out calls or other key activity cannot be ruled out. Billing-metadata lookups for eight saved failed generation IDs returned 404. The key total includes previous activity and is not an invoice for only this extension.",
        ]
    (B / "ARENA_100_CHEAP_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(totals))


if __name__ == "__main__":
    main()
