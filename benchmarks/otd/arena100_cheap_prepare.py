"""Freeze the authorized expansion of the earlier cheap roster to 100 prompts."""

import json
from copy import deepcopy
from datetime import datetime, timezone

from arena20 import B, read, sha, write
from arena100_generate import body


def main():
    d = B / "data/arena100-cheap"
    r = B / "runs/arena100-cheap"
    d.mkdir(exist_ok=True)
    r.mkdir(exist_ok=True)
    assert not (d / "generation_manifest.json").exists(), "Already frozen"
    old = json.loads((B / "data/arena20/generation_manifest.json").read_text())
    replacements = json.loads(
        (B / "data/arena20/replacement_manifest.json").read_text()
    )
    replaced = {m["replaces"] for m in replacements["models"]}
    models = deepcopy(
        [m for m in old["models"] if m["id"] not in replaced] + replacements["models"]
    )
    catalog_path = B / "sources/openrouter-arena100-cheap-models.json"
    cat = {m["id"]: m for m in json.loads(catalog_path.read_text())["data"]}
    for m in models:
        current = cat[m["id"]]
        assert current["canonical_slug"] == m["canonical_slug"]
        assert float(current["pricing"]["completion"]) * 1e6 < 2
        assert float(m["output_usd_per_million"]) < 2
        m.update(
            mode="sync",
            catalog_id=m["id"],
            provider=old["provider"],
            current_catalog_record=current,
        )
        reasoning = m["settings"].get("reasoning", {})
        assert (
            reasoning.get("enabled") is False
            or reasoning.get("effort") == "low"
            or not reasoning
        )
    assert len(models) == 20 and len({m["id"] for m in models}) == 20
    base = B / "data/arena100"
    for name in ["selected_prompts.jsonl", "screening.jsonl", "source_manifest.json"]:
        (d / name).write_bytes((base / name).read_bytes())
    prompts = read(d / "selected_prompts.jsonl")
    assert len(prompts) == 100
    manifest = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "Run all previously selected models under $2/M output on the larger prompt dataset, in parallel",
        "models": models,
        "planned_models": 20,
        "planned_prompts": 100,
        "planned_requests": 2000,
        "system": old["system"],
        "selected_prompts_sha256": sha((d / "selected_prompts.jsonl").read_bytes()),
        "catalog_sha256": sha(catalog_path.read_bytes()),
        "reasoning_policy": "Preserve identical earlier settings: disabled optional reasoning; low where mandatory",
        "concurrency": 32,
        "request_timeout": 600,
        "reuse_policy": "Reuse only successful responses to the original 20 whose complete request matches exactly; retain source paths and hashes",
    }
    by_model = {m["id"]: m for m in models}
    by_prompt = {p["prompt_id"]: p for p in prompts}
    reused = []
    receipts = []
    for source in [
        B / "runs/arena20/responses.jsonl",
        B / "runs/arena20-replacements/responses.jsonl",
    ]:
        for row in read(source):
            if row["model_id"] not in by_model or not row["success"]:
                continue
            expected = body(
                by_model[row["model_id"]], by_prompt[row["prompt_id"]], manifest
            )
            assert expected == row["request"], "Cannot reuse changed request settings"
            row = deepcopy(row)
            row["reused_from"] = str(source.relative_to(B))
            reused.append(row)
        receipts.append(
            {"source": str(source.relative_to(B)), "sha256": sha(source.read_bytes())}
        )
    assert len(reused) == len({(x["model_id"], x["prompt_id"]) for x in reused}) == 365
    manifest.update(
        reused_successes=len(reused), new_planned_requests=2000 - len(reused)
    )
    (d / "generation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write(r / "responses.jsonl", reused)
    (r / "reuse_receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
    (r / "arena100_cheap_prepare.py").write_bytes(
        (B / "arena100_cheap_prepare.py").read_bytes()
    )
    print(
        json.dumps(
            {
                "models": 20,
                "prompts": 100,
                "reused": len(reused),
                "new_requests": 2000 - len(reused),
                "output_rate_sum": sum(
                    float(m["output_usd_per_million"]) for m in models
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
