"""Report every Arena cell, exploratory detector FNR and paired comparisons."""

import json
from collections import Counter
from itertools import combinations

from arena20 import B, D, read, write
from arena_review import R, collect
from metrics import rate


def pct(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def main():
    source = json.loads((D / "source_manifest.json").read_text())
    primary = json.loads((D / "generation_manifest.json").read_text())
    supplement = json.loads((D / "replacement_manifest.json").read_text())
    models = primary["models"] + supplement["models"]
    original_replaced = {m["replaces"] for m in supplement["models"]}
    active = [m for m in models if m["id"] not in original_replaced]
    responses = collect()
    audits = read(R / "response_audit.jsonl")
    assert (
        len(responses) == 440
        and len({(r["model_id"], r["prompt_id"]) for r in responses}) == 440
    )
    lookup = {(r["model_id"], r["prompt_id"]): r for r in responses}
    audit = {(r["model_id"], r["prompt_id"]): r for r in audits}
    # A useful provider-usage proxy, kept separate from the unverified native user-only gate.
    token_proxies = {}
    for key, response in lookup.items():
        usage = (response.get("raw") or {}).get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        reasoning_tokens = (usage.get("completion_tokens_details") or {}).get(
            "reasoning_tokens"
        )
        valid = all(
            isinstance(v, int) and not isinstance(v, bool) and v >= 0
            for v in [prompt_tokens, completion_tokens, reasoning_tokens]
        )
        visible = completion_tokens - reasoning_tokens if valid else None
        token_proxies[key] = {
            "reported_prompt_tokens_including_system": prompt_tokens,
            "reported_completion_tokens": completion_tokens,
            "reported_reasoning_tokens": reasoning_tokens,
            "reported_visible_tokens": visible,
            "visible_gt_total_prompt_proxy": visible > prompt_tokens
            if visible is not None and visible >= 0
            else None,
            "notice": "Diagnostic only: includes system/input formatting; user-only native counts are not verified. Not used to select prompts or responses.",
        }
    # Generic pooled Wilson intervals are inappropriate for repeated prompt clusters.
    for detector in ["meld", "editlens"]:
        path = R / detector / "metrics.json"
        values = json.loads(path.read_text())
        pooled = [values["overall"]] + [
            v
            for k, v in values["groups"].items()
            if " / generator / " not in k and " / cohort / " not in k
        ]
        for group in pooled:
            for value in group.values():
                if isinstance(value, dict) and "wilson95" in value:
                    value["wilson95"] = None
                    value["interval_note"] = (
                        "Suppressed: pooled outputs share 20 prompt clusters."
                    )
        values["limitations"] = [
            "The sample is a fixed-seed uniform sample of eligible unique prompts in a historical English opening-request frame, conditional on the single-reviewer rubric.",
            "Per-generator Wilson intervals are exploratory; pooled intervals are suppressed because responses share prompts.",
            "Response-scope decisions used local excerpts plus full text for flagged cases, not exhaustive independent review.",
            "Native user-only input/output token gate is unverified; not exact Pangram replication.",
            "Detector thresholds are local baselines, not Pangram decisions or calibrated authorship probabilities.",
        ]
        path.write_text(json.dumps(values, indent=2) + "\n")
    predictions = {}
    for detector in ["meld", "editlens"]:
        predictions[detector] = {
            (r["generator"], r["prompt_id"]): r
            for r in read(R / detector / "predictions.jsonl")
        }
        expected = {key for key, r in audit.items() if r["eligible"]}
        assert set(predictions[detector]) == expected
        for key, r in predictions[detector].items():
            assert r["response_sha256"] == audit[key]["response_sha256"]
    summary = {}
    matrix = []
    prompts = read(D / "selected_prompts.jsonl")
    for m in models:
        mid = m["id"]
        cells = [lookup[(mid, p["prompt_id"])] for p in prompts]
        scopes = [audit[(mid, p["prompt_id"])] for p in prompts]
        stats = {
            "attempted": 20,
            "successful": sum(r["success"] for r in cells),
            "scope_eligible": sum(r["eligible"] for r in scopes),
            "exclusions": dict(
                Counter(r["reason"] for r in scopes if not r["eligible"])
            ),
            "provider_counts": dict(
                Counter((r.get("raw") or {}).get("provider", "none") for r in cells)
            ),
            "reported_cost_usd": sum(
                float((r.get("raw") or {}).get("usage", {}).get("cost") or 0)
                for r in cells
            ),
            "detectors": {},
        }
        for detector, preds in predictions.items():
            good = [
                r
                for (model, pid), r in preds.items()
                if model == mid and "error" not in r
            ]
            misses = sum(r["prediction"] != "ai" for r in good)
            detections = len(good) - misses
            stats["detectors"][detector] = {
                "fnr": rate(misses, len(good)),
                "detections": detections,
                "yield_per_20": detections / 20,
                "failures": sum(
                    "error" in r for (model, pid), r in preds.items() if model == mid
                ),
            }
        summary[mid] = stats
        for p in prompts:
            key = (mid, p["prompt_id"])
            r = lookup[key]
            matrix.append(
                {
                    "model": mid,
                    "prompt_id": p["prompt_id"],
                    "frame_position": p["position"],
                    "success": r["success"],
                    "audit": audit[key],
                    "token_usage_proxy": token_proxies[key],
                    "predictions": {d: ps.get(key) for d, ps in predictions.items()},
                    "http_statuses": [a.get("status") for a in r["attempts"]],
                    "finish_reason": (r.get("audit") or {}).get("finish_reason"),
                }
            )
    pairs = []
    for detector, preds in predictions.items():
        for left, right in combinations(active, 2):
            a, b = left["id"], right["id"]
            common = [
                p["prompt_id"]
                for p in prompts
                if (a, p["prompt_id"]) in preds
                and (b, p["prompt_id"]) in preds
                and "error" not in preds[(a, p["prompt_id"])]
                and "error" not in preds[(b, p["prompt_id"])]
            ]
            pairs.append(
                {
                    "detector": detector,
                    "left": a,
                    "right": b,
                    "common_prompt_ids": common,
                    "n": len(common),
                    "left_fnr": rate(
                        sum(preds[(a, p)]["prediction"] != "ai" for p in common),
                        len(common),
                    ),
                    "right_fnr": rate(
                        sum(preds[(b, p)]["prediction"] != "ai" for p in common),
                        len(common),
                    ),
                }
            )
    successes = sum(r["success"] for r in responses)
    eligible = sum(r["eligible"] for r in audits)
    totalcost = sum(r["reported_cost_usd"] for r in summary.values())
    proxy_pass = sum(
        x["visible_gt_total_prompt_proxy"] is True for x in token_proxies.values()
    )
    proxy_known = sum(
        x["visible_gt_total_prompt_proxy"] is not None for x in token_proxies.values()
    )
    data = {
        "source": source,
        "cells": len(responses),
        "successful": successes,
        "scope_eligible": eligible,
        "reported_cost_usd": totalcost,
        "models": summary,
        "strict_pangram_length_gate_metrics": None,
        "native_token_gate_verified": 0,
        "provider_usage_proxy": {"pass": proxy_pass, "known": proxy_known},
        "notice": "Scope-screened exploratory variant. Native user-only token comparison unverified; excerpt review is not exhaustive originality validation.",
    }
    (R / "summary.json").write_text(json.dumps(data, indent=2) + "\n")
    write(R / "cell_matrix.jsonl", matrix)
    write(R / "paired_comparisons.jsonl", pairs)
    lines = [
        "# Arena-20 local benchmark results",
        "",
        f"Completed **{len(responses)} recorded cells**: 400 original slots plus 40 availability replacements. **{successes} successful responses; {eligible} passed the local response screen.** Both local detectors were applied to every eligible response. Reported successful-request usage cost: **${totalcost:.4f}** (not an invoice; failed attempts may have unreported charges).",
        "",
        "The intended active roster has 20 model variants. Two account-blocked original routes remain separately visible below. Service failures are retained; no response was regenerated because of refusal, length, content quality or detector score.",
        "",
        "## Sampling and scope",
        "",
        f"All {source['source_rows']:,} source rows were read. Exact opening-prompt agreement across both arms yielded {source['frame_count']:,} unique normalized requests. The precommitted seed ordered the complete frame; screening the first {source['reviewed']} candidates produced the first 20 eligible prompts, from 20 distinct anonymized users. No topic quotas or post-generation prompt substitutions.",
        "",
        f"Sample SHA-256: `{source['selected_sha256']}`. Categories: "
        + ", ".join(f"{k} {v}" for k, v in source["category_counts"].items())
        + ".",
        "",
        "This is a 2023 English opening-request sample, weighted by unique prompt rather than traffic frequency. The public source may appear in generator training data, so unseen-prompt generalization is not established. Prompt eligibility and response scope were reviewed by one assistant. Response review used beginning/end excerpts and format counts, with full text for flagged cases; it was not an exhaustive independent human review or plagiarism check. Bullet paragraphs count as prose; bare data/verse and refusal-only answers do not. All scope decisions preceded scoring of that response. No generator names or scores appeared in review packets.",
        "",
        "**Exact Pangram-style results are unavailable:** provider usage does not consistently supply user-only native token counts. The output-longer-than-input token gate is unverified for every response. Results below explicitly omit that gate and use successful, completed, locally scope-screened answers of at least 50 words. They are exploratory local-baseline results, not a Pangram 4 reproduction.",
        "",
        f"Provider-usage diagnostic: {proxy_pass}/{proxy_known} responses with complete reported counts have visible completion tokens (reported completion minus reasoning) greater than the entire reported prompt count, including the system message. This is a separate proxy, not proof of native user-only tokenization; it did not filter the sample or accuracy denominator.",
        "",
        "## Active roster results",
        "",
        "| Generator | API success /20 | Eligible | MELD misses /N (FNR) | EditLens misses /N (FNR) | MELD detections /20 | EditLens detections /20 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m in active:
        s = summary[m["id"]]
        a = s["detectors"]["meld"]
        b = s["detectors"]["editlens"]
        x = a["fnr"]
        y = b["fnr"]
        lines.append(
            f"| {m['name']} | {s['successful']}/20 | {s['scope_eligible']} | {x['count']}/{x['n']} ({pct(x['rate'])}) | {y['count']}/{y['n']} ({pct(y['rate'])}) | {a['detections']}/20 | {b['detections']}/20 |"
        )
    lines += [
        "",
        "Mixed counts as a miss under the strict document decision convention. MELD uses its shipped threshold (50–99-word results are below its recommended length); EditLens uses exploratory 0.1/0.8 thresholds. Detections/20 is end-to-end yield, not detector recall. Zero-denominator FNR is unavailable, not zero.",
        "",
        "## Uncertainty and paired comparisons",
        "",
        "| Generator | MELD FNR Wilson 95% interval | EditLens FNR Wilson 95% interval |",
        "|---|---:|---:|",
    ]
    for m in active:
        intervals = []
        for d in ["meld", "editlens"]:
            ci = summary[m["id"]]["detectors"][d]["fnr"]["wilson95"]
            intervals.append("—" if ci is None else f"{pct(ci[0])}–{pct(ci[1])}")
        lines.append(f"| {m['name']} | {intervals[0]} | {intervals[1]} |")
    common = set(p["prompt_id"] for p in prompts)
    for m in active:
        common &= {
            r["prompt_id"] for r in audits if r["model_id"] == m["id"] and r["eligible"]
        }
    lines += [
        "",
        f"Common scope-eligible prompts across all 20 intended active models: **{len(common)}**. `paired_comparisons.jsonl` contains common-prompt denominators and rates for every pair of active generators, separately for each detector. Do not rank models using unmatched denominators alone.",
        "",
        "Twenty prompts give coarse estimates: one miss is 5 percentage points when N=20; zero misses still has a 16.1% Wilson upper bound. Responses share prompt clusters, so 400 outputs are not 400 independent prompt draws. No pooled independence-based interval or significance ranking is claimed. There are no human answer controls here: FPR and AUROC are unavailable.",
        "",
        "## Availability and response exclusions",
        "",
        "| Route | Exclusions |",
        "|---|---|",
    ]
    for m in models:
        s = summary[m["id"]]
        if s["exclusions"]:
            lines.append(
                f"| `{m['id']}` | "
                + "; ".join(f"{k}: {v}" for k, v in s["exclusions"].items())
                + " |"
            )
    lines += [
        "",
        "The two original account restrictions were preserved: Muse Spark 1.3 required age attestation; Liquid free conflicted with account privacy preferences. Muse Glimmer and Nemotron were chosen from the next eligible catalog versions, based on availability rather than response quality. Exact errors/retries/providers are in the response files.",
        "",
        "## Audit artifacts",
        "",
        "- `source_manifest.json`, `frame.jsonl`, `screening.jsonl`, `selected_prompts.jsonl` under `data/arena20/`: full acquisition and sampling provenance.",
        "- `generation_manifest.json` and `replacement_manifest.json`: frozen versions, price caps and settings.",
        "- `responses.jsonl` in each run folder: original requests, raw responses, usage and retry attempts.",
        "- `content_reviews.jsonl`, `response_audit.jsonl`, `review_packets/`: local scope decisions before scoring.",
        "- `cell_matrix.jsonl`: every attempted prompt/model cell and both detector outcomes.",
        "- `paired_comparisons.jsonl`: all pairwise common-prompt comparisons.",
        "- `meld/` and `editlens/`: checkpoint manifests, code snapshots, predictions, aggregate metrics and scored-input hashes.",
        "",
        "The separate proposed API-based judging step was rejected by automatic approval review and never ran. All content review and detector scoring used local files. The website was not modified.",
    ]
    (R / "REPORT.md").write_text("\n".join(lines) + "\n")
    (B / "ARENA_20_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "cells": len(responses),
                "successful": successes,
                "scope_eligible": eligible,
                "reported_cost_usd": totalcost,
            }
        )
    )


if __name__ == "__main__":
    main()
