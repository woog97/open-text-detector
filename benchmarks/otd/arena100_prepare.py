"""Freeze the pre-reviewed extension of the original Arena sample and 12-model roster."""

import json
from collections import Counter
from datetime import datetime, timezone

from arena20 import B, read, sha, write

D = B / "data/arena100"
ids = [
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-pro",
    "google/gemma-4-31b-it",
    "meta-llama/llama-3.3-70b-instruct",
    "moonshotai/kimi-k2.6",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.7-max",
    "tencent/hy3-preview",
    "thinkingmachines/inkling",
    "x-ai/grok-4.3",
    "z-ai/glm-5.2",
]


def main():
    assert not (D / "generation_manifest.json").exists(), "Already frozen"
    frame = read(B / "data/arena20/frame.jsonl")
    ds = read(D / "screening.jsonl")
    original = json.loads((B / "data/arena20/source_manifest.json").read_text())
    assert (
        sha((B / "data/arena20/frame.jsonl").read_bytes()) == original["frame_sha256"]
    )
    assert ds[:79] == read(B / "data/arena20/screening.jsonl")
    assert [d["position"] for d in ds] == list(range(1, len(ds) + 1))
    assert all(d["prompt_id"] == r["prompt_id"] for d, r in zip(ds, frame))
    selected = [
        {**frame[d["position"] - 1], **d} for d in ds if d["decision"] == "eligible"
    ]
    assert len(selected) == 100 and ds[-1]["decision"] == "eligible"
    assert selected[:20] == read(B / "data/arena20/selected_prompts.jsonl")
    assert len({p["prompt_id"] for p in selected}) == 100
    write(D / "selected_prompts.jsonl", selected)
    source = {
        **original,
        "selected": 100,
        "reviewed": len(ds),
        "selected_sha256": sha((D / "selected_prompts.jsonl").read_bytes()),
        "category_counts": dict(Counter(p["category"] for p in selected)),
        "selected_unique_users": len({u for p in selected for u in p["users"]}),
        "rubric": "Original ARENA_20_PROTOCOL.md, expanded to first 100 eligible; no quotas or new seed",
        "reviewer": "Single assistant; new candidates screened before generation; initial 20 locked despite previous observed outputs",
    }
    (D / "source_manifest.json").write_text(json.dumps(source, indent=2) + "\n")
    catalog = json.loads((B / "sources/openrouter-arena100-models.json").read_text())
    lookup = {m["id"]: m for m in catalog["data"]}
    models = []
    for mid in ids:
        batch = mid == "x-ai/grok-4.3"
        m = lookup[mid + (":batch" if batch else "")]
        assert float(m["pricing"]["completion"]) * 1e6 <= 7
        settings = {"max_tokens": 4096}
        if "temperature" in m["supported_parameters"]:
            settings["temperature"] = 1
        if "reasoning" in m["supported_parameters"]:
            if (m.get("reasoning") or {}).get("mandatory"):
                assert "low" in m["reasoning"].get("supported_efforts", [])
                settings["reasoning"] = {"effort": "low", "exclude": True}
            else:
                settings["reasoning"] = {"enabled": False, "exclude": True}
        models.append(
            {
                "id": mid,
                "catalog_id": m["id"],
                "canonical_slug": m["canonical_slug"],
                "name": m["name"],
                "mode": "batch" if batch else "concurrent",
                "settings": settings,
                "catalog_record": m,
                "provider": {
                    "max_price": {
                        "prompt": float(m["pricing"]["prompt"]) * 1e6,
                        "completion": float(m["pricing"]["completion"]) * 1e6,
                        "request": 0,
                    },
                    "sort": "price",
                    "require_parameters": True,
                },
            }
        )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "system": "You are a helpful assistant.",
        "prompt_count": 100,
        "planned_requests": 1200,
        "selected_prompts_sha256": source["selected_sha256"],
        "sample_path": str(D / "selected_prompts.jsonl"),
        "batch_provider": {"only": ["xai"]},
        "retry_policy": "At most 2 retries for explicit 429/5xx sync responses. No retry for uncertain transport completion, refusals, truncation, quality, or batch submission.",
        "catalog_sha256": sha(
            (B / "sources/openrouter-arena100-models.json").read_bytes()
        ),
    }
    (D / "generation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = [
        "# Arena-100 frozen sample",
        "",
        f"First 100 eligible prompts among {len(ds)} candidates in the original full-frame seeded order. Original 20 unchanged; {source['selected_unique_users']} distinct anonymized source users. Single-reviewer screening. No topic quotas.",
        "",
        f"Sample SHA256: `{source['selected_sha256']}`.",
    ]
    for i, p in enumerate(selected, 1):
        lines += [
            "",
            f"## {i}. {p['category']} — position {p['position']}",
            "",
            p["text"],
            "",
            p["rationale"],
        ]
    (B / "ARENA_100_SAMPLE.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {"sample": source, "models": [(m["id"], m["settings"]) for m in models]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
