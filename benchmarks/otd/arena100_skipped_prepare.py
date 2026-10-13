"""Freeze previously skipped Pangram generators plus requested GPT releases."""

import json
from datetime import datetime, timezone

from arena20 import B, read, sha, write
from arena100_generate import body
from arena100_results import collect

SKIPPED = [
    "anthropic/claude-fable-5",
    "anthropic/claude-opus-5",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-opus-4.8",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-haiku-4.5",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3.5-flash",
    "mistralai/mistral-medium-3-5",
    "openai/gpt-5.6-sol",
    "openai/gpt-5.6-terra",
    "openai/gpt-5.5",
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
]
EXTRA = [
    "openai/gpt-5.6-luna",
    "openai/gpt-6-luna",
    "openai/gpt-6-sol",
    "openai/gpt-6-astra",
]
D = B / "data/arena100-skipped"
R = B / "runs/arena100-skipped"


def main():
    assert not (D / "generation_manifest.json").exists(), "Already frozen"
    D.mkdir(exist_ok=True)
    R.mkdir(exist_ok=True)
    catalog_path = B / "sources/openrouter-arena100-skipped-models.json"
    cat = json.loads(catalog_path.read_text())
    lookup = {m["id"]: m for m in cat["data"]}
    assert "openai/gpt-6-terra" not in lookup
    old = json.loads((B / "data/arena100-cheap/generation_manifest.json").read_text())
    luna = next(m for m in old["models"] if m["id"] == "openai/gpt-6-luna")
    models = []
    for mid in SKIPPED + EXTRA:
        m = lookup[mid]
        settings = {"max_tokens": 4096}
        if "temperature" in m["supported_parameters"]:
            settings["temperature"] = 1
        if "reasoning" in m["supported_parameters"]:
            if (m.get("reasoning") or {}).get("mandatory"):
                assert "low" in m["reasoning"]["supported_efforts"]
                settings["reasoning"] = {"effort": "low", "exclude": True}
            else:
                settings["reasoning"] = {"enabled": False, "exclude": True}
        rates = {k: float(m["pricing"][k]) * 1e6 for k in ["prompt", "completion"]}
        provider = {
            "sort": "price",
            "require_parameters": True,
            "max_price": {**rates, "request": 0},
        }
        if mid == luna["id"]:
            assert m["canonical_slug"] == luna["canonical_slug"]
            settings = luna["settings"]
            provider = luna["provider"]
        models.append(
            {
                "id": mid,
                "catalog_id": mid,
                "canonical_slug": m["canonical_slug"],
                "name": m["name"],
                "mode": "sync",
                "settings": settings,
                "provider": provider,
                "catalog_record": m,
                "input_usd_per_million": rates["prompt"],
                "output_usd_per_million": rates["completion"],
            }
        )
    for name in ["selected_prompts.jsonl", "screening.jsonl", "source_manifest.json"]:
        (D / name).write_bytes((B / "data/arena100" / name).read_bytes())
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "system": old["system"],
        "planned_models": len(models),
        "planned_requests": 100 * len(models),
        "prompt_count": 100,
        "selected_prompts_sha256": sha((D / "selected_prompts.jsonl").read_bytes()),
        "catalog_sha256": sha(catalog_path.read_bytes()),
        "concurrency": 32,
        "request_timeout": 600,
        "unavailable_requested_models": ["openai/gpt-6-terra"],
        "scope": "All 14 omitted Table 3 generators plus GPT-5.6 Luna and available GPT-6 releases; standard variants.",
        "reuse_policy": "Reuse successful cells only when full request body matches, including routing/settings.",
    }
    prompts = read(D / "selected_prompts.jsonl")
    pp = {p["prompt_id"]: p for p in prompts}
    reused = []
    for row in collect(B / "runs/arena100-cheap"):
        if row["model_id"] == luna["id"] and row["success"]:
            assert row["request"] == body(luna, pp[row["prompt_id"]], manifest)
            reused.append({**row, "reused_from": "runs/arena100-cheap"})
    assert len(reused) == 100
    write(R / "responses.jsonl", reused)
    (D / "generation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (R / "arena100_skipped_prepare.py").write_bytes(
        (B / "arena100_skipped_prepare.py").read_bytes()
    )
    prior = read(B / "runs/arena100-combined/dataset.jsonl")
    inp = sum(r["usage"]["prompt_tokens"] for r in prior) / len(prior)
    out = sum(r["usage"]["completion_tokens"] for r in prior) / len(prior)
    new = [m for m in models if m["id"] != luna["id"]]
    input_rate = sum(m["input_usd_per_million"] for m in new)
    output_rate = sum(m["output_usd_per_million"] for m in new)
    estimate = {
        "new_models": len(new),
        "new_requests": 100 * len(new),
        "reused": 100,
        "mean_prior_input_tokens": inp,
        "mean_prior_output_tokens": out,
        "estimated_usd": 100 * (input_rate * inp + output_rate * out) / 1e6,
        "output_4096_cap_scenario_usd": 100
        * (input_rate * inp + output_rate * 4096)
        / 1e6,
        "summed_input_rate": input_rate,
        "summed_output_rate": output_rate,
    }
    (D / "cost_estimate.json").write_text(json.dumps(estimate, indent=2) + "\n")
    lines = [
        "# Skipped-model expansion on the frozen 100 prompts",
        "",
        "All 14 previously omitted Pangram Table 3 models plus GPT-5.6 Luna, GPT-6 Luna, GPT-6 Sol, and GPT-6 Astra. GPT-6 Terra is absent from the current OpenRouter catalog; the Terra Latest alias is GPT-5.6 Terra. No substitute is labelled GPT-6 Terra.",
        "",
        "Reuse 100 existing GPT-6 Luna responses with identical complete requests. Generate 1,700 additional responses with 32 concurrent requests across models. This uses concurrent chat completions, not the discounted asynchronous batch endpoint. Preserve all successes; retry failed requests only. Reasoning disabled when optional, low when mandatory. 4,096-token cap includes reasoning where applicable. Account privacy settings unchanged.",
        "",
        f"Historical response lengths suggest ${estimate['estimated_usd']:.2f}; the 4,096-output-token cap scenario is ${estimate['output_4096_cap_scenario_usd']:.2f} before any retry/failed-call charges. Neither is a billing guarantee.",
        "",
        "| Model | Input $/M | Output $/M | Reasoning |",
        "|---|---:|---:|---|",
    ]
    for m in models:
        lines.append(
            f"| {m['name']} | {m['input_usd_per_million']:g} | {m['output_usd_per_million']:g} | {json.dumps(m['settings'].get('reasoning', 'not advertised'))} |"
        )
    lines += [
        "",
        "Same prompt checksum and sampling protocol as ARENA_100_PROTOCOL.md. No resampling or replacement based on outputs. Mechanical detector eligibility does not imply substantive prose review. New detector inputs will be prepared; detector scoring is separate.",
    ]
    (B / "ARENA_100_SKIPPED_PROTOCOL.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(estimate, indent=2))


if __name__ == "__main__":
    main()
