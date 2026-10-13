"""Mechanical response accounting and descriptive reporting for Arena-100."""

import json
from collections import Counter

from arena20 import B, read, sha, write

D = B / "data/arena100"
R = B / "runs/arena100"


def collect(run_dir=None):
    root = run_dir or R
    paths = (
        [root / "responses.jsonl"]
        + sorted(root.glob("responses_resume_*.jsonl"))
        + [root / "responses_batch.jsonl"]
    )
    latest = {}
    for path in paths:
        for row in read(path) if path.exists() else []:
            key = (row["model_id"], row["prompt_id"])
            assert key not in latest or not latest[key]["success"], (
                "Successful response was regenerated"
            )
            latest[key] = row
    return list(latest.values())


def audit(r):
    if not r["success"]:
        return False, "api_failure"
    if not r["audit"]["not_truncated"]:
        return False, "non_stop_finish"
    if not r["audit"]["length_ok"]:
        return False, "below_50_words"
    return True, "mechanically_eligible"


def export_dataset():
    assert (R / "sync_complete.json").exists(), "Generation is still running"
    rs = collect()
    mf = json.loads((D / "generation_manifest.json").read_text())
    ps = read(D / "selected_prompts.jsonl")
    assert (
        len(rs) == 1200 and len({(r["model_id"], r["prompt_id"]) for r in rs}) == 1200
    )
    assert {(r["model_id"], r["prompt_id"]) for r in rs} == {
        (m["id"], p["prompt_id"]) for m in mf["models"] for p in ps
    }
    assert all(r["success"] for r in rs), "User requested all 1200 successful cells"
    prompt_by_id = {p["prompt_id"]: p for p in ps}
    dataset = []
    for r in rs:
        text = r["audit"]["text"]
        dataset.append(
            {
                "prompt_id": r["prompt_id"],
                "prompt_text": prompt_by_id[r["prompt_id"]]["text"],
                "model_id": r["model_id"],
                "response_text": text,
                "response_sha256": sha(text.encode()),
                "request_sha256": r["request_sha256"],
                "generation_id": r["raw"].get("id"),
                "finish_reason": r["raw"]["choices"][0].get("finish_reason"),
                "usage": r["raw"].get("usage"),
                "mechanically_eligible": audit(r)[0],
                "eligibility_reason": audit(r)[1],
            }
        )
    write(R / "dataset.jsonl", dataset)
    return rs, mf, ps


def main():
    rs, mf, ps = export_dataset()
    preds = {}
    detector_configuration = {}
    eligible_hashes = {
        (r["model_id"], r["prompt_id"]): sha(r["audit"]["text"].encode())
        for r in rs
        if audit(r)[0]
    }
    for model in ["meld", "editlens"]:
        path = R / model / "predictions.jsonl"
        manifest = json.loads((R / model / "manifest.json").read_text())
        assert manifest.get("finished_utc"), f"{model} is still scoring"
        predictions = read(path)
        preds[model] = {(r["generator"], r["prompt_id"]): r for r in predictions}
        assert len(predictions) == len(preds[model]), "Duplicate detector predictions"
        assert set(preds[model]) == set(eligible_hashes), "Incomplete detector coverage"
        assert all(
            r["response_sha256"] == eligible_hashes[key]
            for key, r in preds[model].items()
        ), "Detector response hash mismatch"
        detector_configuration[model] = {
            "device": manifest["device"],
            "effective_dtypes": sorted(
                {
                    str((r.get("inference") or {}).get("dtype"))
                    for r in predictions
                    if "error" not in r
                }
            ),
            "precision_note": "The shared runner's precision argument applies to MELD. EditLens selects its own dtype; inference metadata records the effective dtype.",
        }
    rows = []
    matrix = []
    for m in mf["models"]:
        cells = [r for r in rs if r["model_id"] == m["id"]]
        success = [r for r in cells if r["success"]]
        usage = [(r["raw"].get("usage") or {}) for r in success]
        known = [u for u in usage if isinstance(u.get("cost"), (int, float))]
        stat = {
            "model": m["id"],
            "name": m["name"],
            "mode": m["mode"],
            "requested_settings": m["settings"],
            "attempted": 100,
            "successful": len(success),
            "eligible": sum(audit(r)[0] for r in cells),
            "exclusions": dict(Counter(audit(r)[1] for r in cells if not audit(r)[0])),
            "reported_cost_usd": sum(u["cost"] for u in known),
            "cost_known_responses": len(known),
            "input_tokens": sum(u.get("prompt_tokens") or 0 for u in usage),
            "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usage),
            "completion_count_known": sum(
                isinstance(u.get("completion_tokens"), int) for u in usage
            ),
            "reasoning_tokens": sum(
                (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
                for u in usage
            ),
            "reasoning_count_known": sum(
                isinstance(
                    (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                    int,
                )
                for u in usage
            ),
            "detectors": {},
        }
        for det, pp in preds.items():
            pp = [r for (mid, pid), r in pp.items() if mid == m["id"]]
            stat["detectors"][det] = {
                "scored": sum("error" not in r for r in pp),
                "errors": sum("error" in r for r in pp),
                "misses": sum(
                    r.get("prediction") != "ai" for r in pp if "error" not in r
                ),
            }
        rows.append(stat)
        for r in cells:
            key = (r["model_id"], r["prompt_id"])
            eligible, reason = audit(r)
            matrix.append(
                {
                    "model_id": key[0],
                    "prompt_id": key[1],
                    "successful": r["success"],
                    "eligible": eligible,
                    "reason": reason,
                    "substantive_prose_review": "unverified",
                    "native_user_only_token_gate": "unverified",
                    "response_sha256": sha(r["audit"]["text"].encode())
                    if r["success"]
                    else None,
                    "detectors": {d: p.get(key) for d, p in preds.items()},
                }
            )
    batch = (
        json.loads((R / "batch_latest.json").read_text())
        if (R / "batch_latest.json").exists()
        else {}
    )
    batch_cost = (batch.get("usage") or {}).get("cost")
    # Prefer aggregate batch cost for total if supplied; never count it twice.
    sync_cost = sum(x["reported_cost_usd"] for x in rows if x["mode"] != "batch")
    bstat = next(x for x in rows if x["mode"] == "batch")
    total = sync_cost + (
        batch_cost
        if isinstance(batch_cost, (float, int))
        else bstat["reported_cost_usd"]
    )
    summary = {
        "models": rows,
        "cells": len(rs),
        "success": sum(r["success"] for r in rs),
        "eligible": sum(audit(r)[0] for r in rs),
        "reported_cost_usd": total,
        "batch_aggregate_cost_usd": batch_cost,
        "detector_configuration": detector_configuration,
        "limitations": [
            "Mechanical screen only (successful stop finish and >=50 whitespace words); no exhaustive original-prose or refusal-only adjudication.",
            "Native user-only output-longer-than-input token gate unverified.",
            "Descriptive AI-only pilot; no human FPR/AUROC; repeated prompt clusters invalidate pooled independent intervals.",
            "Costs are provider-reported, not invoice reconciliation; failed attempts may be billed.",
        ],
    }
    (R / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write(R / "cell_matrix.jsonl", matrix)
    lines = [
        "# Arena-100 results",
        "",
        f"{len(rs)} recorded prompt/model pairs, {summary['success']} successful responses, {summary['eligible']} mechanically eligible responses. Reported usage cost: **${total:.6f}**.",
        "",
        "100 prompts, first 100 eligible in the original seeded order after screening 320 candidates. Original 20 unchanged. 100 distinct source users. No new dataset or model-weight downloads.",
        "",
        "Reasoning disabled when optional; GPT-OSS 120B low; Llama has no reasoning control. Grok 4.3 uses one asynchronous batch; the other 11 models use concurrent requests.",
        "",
        "| Model | Mode | Success /100 | Mechanically eligible | Cost $ | Reported reasoning tokens (coverage) | MELD misses/scored | EditLens misses/scored |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for s in rows:
        det = []
        for name in ["meld", "editlens"]:
            v = s["detectors"].get(name)
            det.append(f"{v['misses']}/{v['scored']}" if v else "pending")
        cost = (
            batch_cost
            if s["mode"] == "batch" and isinstance(batch_cost, (float, int))
            else s["reported_cost_usd"]
        )
        lines.append(
            f"| {s['name']} | {s['mode']} | {s['successful']} | {s['eligible']} | {cost:.6f} | {s['reasoning_tokens']} ({s['reasoning_count_known']}/{s['successful']}) | {' | '.join(det)} |"
        )
    lines += (
        ["", "## Interpretation", ""]
        + ["- " + x for x in summary["limitations"]]
        + [
            "- Reasoning totals are provider reports; missing counts do not mean zero. See summary.json for coverage.",
            "- Mixed detector decisions count as misses. MELD uses its shipped threshold; EditLens 0.1/0.8 is exploratory. No threshold tuning.",
            "- The earlier Arena-20 response screen included manual excerpts. This run reports a separate mechanical-screen variant; do not conflate the two protocols.",
            "",
            "Artifacts: runs/arena100/dataset.jsonl contains all 1200 prompt/response pairs in one file. data/arena100/ contains frozen prompts, screening and manifests; runs/arena100/ also contains every raw attempt/response, batch request/status, detector outputs and cell matrix.",
        ]
    )
    text = "\n".join(lines) + "\n"
    (R / "REPORT.md").write_text(text)
    (B / "ARENA_100_RESULTS.md").write_text(text)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ["cells", "success", "eligible", "reported_cost_usd"]
            }
        )
    )


if __name__ == "__main__":
    main()
