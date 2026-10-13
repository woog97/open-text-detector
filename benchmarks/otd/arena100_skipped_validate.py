"""Final completeness and provenance validation; no API calls or generation."""

import json
from collections import Counter
from datetime import datetime, timezone

from arena20 import B, read, sha
from arena100_generate import body
from arena100_results import collect
from arena100_skipped_results import main as export

D = B / "data/arena100-skipped"
R = B / "runs/arena100-skipped"


def main():
    mf = json.loads((D / "generation_manifest.json").read_text())
    prompts = read(D / "selected_prompts.jsonl")
    pp = {p["prompt_id"]: p for p in prompts}
    models = {m["id"]: m for m in mf["models"]}
    rows = collect(R)
    assert len(rows) == 1800 and all(r["success"] for r in rows)
    assert {(r["model_id"], r["prompt_id"]) for r in rows} == {
        (m, p) for m in models for p in pp
    }
    assert (D / "selected_prompts.jsonl").read_bytes() == (
        B / "data/arena100/selected_prompts.jsonl"
    ).read_bytes()
    assert sum(bool(r.get("reused_from")) for r in rows) == 100
    missing_usage = [
        r["raw"]["id"]
        for r in rows
        if not isinstance((r["raw"].get("usage") or {}).get("cost"), (int, float))
    ]
    prior = {
        r["generation_id"]: r for r in read(B / "runs/arena100-combined/dataset.jsonl")
    }
    disabled_reasoning = []
    token_cap_exceptions = []
    for r in rows:
        m = models[r["model_id"]]
        assert r["request"] == body(m, pp[r["prompt_id"]], mf)
        assert r["request_sha256"] == sha(
            json.dumps(r["request"], sort_keys=True).encode()
        )
        if r.get("reused_from"):
            saved = prior[r["raw"]["id"]]
            assert saved["response_sha256"] == sha(r["audit"]["text"].encode())
        u = r["raw"].get("usage") or {}
        reasoning = (u.get("completion_tokens_details") or {}).get(
            "reasoning_tokens", 0
        ) or 0
        if m["settings"].get("reasoning", {}).get("enabled") is False and reasoning:
            disabled_reasoning.append(
                {"model": m["id"], "prompt": r["prompt_id"], "tokens": reasoning}
            )
        if u.get("completion_tokens", 0) > 4096:
            token_cap_exceptions.append(
                [m["id"], r["prompt_id"], u["completion_tokens"]]
            )
    export()
    full = read(B / "runs/arena100-expanded/dataset.jsonl")
    assert len(full) == 4900 and len({r["generation_id"] for r in full}) == 4900
    assert set(Counter(r["model_id"] for r in full).values()) == {100}
    receipts = {}
    for name in [
        "arena100_generate.py",
        "arena100_skipped_prepare.py",
        "arena100_skipped_results.py",
        "arena100_skipped_validate.py",
    ]:
        receipts[name] = sha((B / name).read_bytes())
    result = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "extension_cells": 1800,
        "new_cells": 1700,
        "reused_cells": 100,
        "combined_models": 49,
        "combined_cells": 4900,
        "sample_sha256": mf["selected_prompts_sha256"],
        "script_sha256": receipts,
        "disabled_reasoning_reported_tokens_exceptions": disabled_reasoning,
        "token_cap_exceptions": token_cap_exceptions,
        "missing_usage_generation_ids": missing_usage,
        "unavailable_models": mf["unavailable_requested_models"],
    }
    (R / "VALIDATION.json").write_text(json.dumps(result, indent=2) + "\n")
    (R / "sync_complete.json").write_text(
        json.dumps(
            {"at": result["completed_utc"], "success": 1800, "new_success": 1700}
        )
        + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
