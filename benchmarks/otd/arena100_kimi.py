"""Freeze and export the authorized Kimi K3 addition without changing prior rows."""

import json
import sys
from datetime import datetime, timezone

from arena20 import B, read, sha, write
from arena100_generate import body
from arena100_results import audit, collect

D = B / "data/arena100-kimi"
R = B / "runs/arena100-kimi"


def prepare():
    assert not (D / "generation_manifest.json").exists(), "Already frozen"
    D.mkdir(exist_ok=True)
    R.mkdir(exist_ok=True)
    catalog = B / "sources/openrouter-kimi-k3-models.json"
    m = next(
        m
        for m in json.loads(catalog.read_text())["data"]
        if m["id"] == "moonshotai/kimi-k3"
    )
    assert not m["reasoning"]["mandatory"]
    settings = {
        "max_tokens": 4096,
        "temperature": 1,
        "reasoning": {"enabled": False, "exclude": True},
    }
    model = {
        "id": m["id"],
        "catalog_id": m["id"],
        "canonical_slug": m["canonical_slug"],
        "name": m["name"],
        "mode": "sync",
        "settings": settings,
        "catalog_record": m,
        "provider": {
            "sort": "price",
            "require_parameters": True,
            "max_price": {
                "prompt": float(m["pricing"]["prompt"]) * 1e6,
                "completion": float(m["pricing"]["completion"]) * 1e6,
                "request": 0,
            },
        },
    }
    for name in ["selected_prompts.jsonl", "screening.jsonl", "source_manifest.json"]:
        (D / name).write_bytes((B / "data/arena100" / name).read_bytes())
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "models": [model],
        "system": "You are a helpful assistant.",
        "planned_models": 1,
        "planned_requests": 100,
        "prompt_count": 100,
        "selected_prompts_sha256": sha((D / "selected_prompts.jsonl").read_bytes()),
        "catalog_sha256": sha(catalog.read_bytes()),
        "concurrency": 32,
        "request_timeout": 600,
    }
    (D / "generation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Frozen Kimi K3 × 100; reasoning disabled; input $3/M, output $15/M.")


def export():
    mf = json.loads((D / "generation_manifest.json").read_text())
    ps = {p["prompt_id"]: p for p in read(D / "selected_prompts.jsonl")}
    rows = collect(R)
    assert len(rows) == 100 and all(r["success"] for r in rows)
    assert {r["prompt_id"] for r in rows} == set(ps)
    dataset = []
    for r in rows:
        assert r["request"] == body(mf["models"][0], ps[r["prompt_id"]], mf)
        raw = r["raw"]
        text = r["audit"]["text"]
        u = raw.get("usage") or {}
        assert not (
            (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        )
        dataset.append(
            {
                "model_id": r["model_id"],
                "prompt_id": r["prompt_id"],
                "prompt_text": ps[r["prompt_id"]]["text"],
                "request_model": r["request"]["model"],
                "response_model": raw.get("model"),
                "provider": raw.get("provider"),
                "success": True,
                "response_text": text,
                "response_sha256": sha(text.encode()),
                "request_sha256": r["request_sha256"],
                "generation_id": raw["id"],
                "usage": raw.get("usage"),
                "finish_reason": raw["choices"][0].get("finish_reason"),
                "mechanically_eligible": audit(r)[0],
                "eligibility_reason": audit(r)[1],
            }
        )
    write(R / "dataset.jsonl", dataset)
    prior = read(B / "runs/arena100-expanded/dataset.jsonl")
    assert len(prior) == 4900
    combined = prior + dataset
    assert (
        len({(r["model_id"], r["prompt_id"]) for r in combined})
        == len({r["generation_id"] for r in combined})
        == 5000
    )
    C = B / "runs/arena100-expanded-kimi"
    C.mkdir(exist_ok=True)
    write(C / "dataset.jsonl", combined)
    summary = {
        "new_responses": 100,
        "models": 50,
        "total_responses": 5000,
        "eligible_new": sum(r["mechanically_eligible"] for r in dataset),
        "new_reported_cost_usd": sum(
            (r["usage"] or {}).get("cost") or 0 for r in dataset
        ),
        "missing_costs": sum((r["usage"] or {}).get("cost") is None for r in dataset),
        "dataset_sha256": sha((C / "dataset.jsonl").read_bytes()),
        "prior_dataset_sha256": sha(
            (B / "runs/arena100-expanded/dataset.jsonl").read_bytes()
        ),
    }
    (C / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (R / "sync_complete.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    prepare() if sys.argv[1] == "prepare" else export()
