import datetime
import json
import re
from decimal import Decimal

from arena20 import B, D, sha

if (D / "generation_manifest.json").exists():
    raise SystemExit(
        "Generation manifest already frozen; create a separate experiment to select again."
    )
ms = json.loads((B / "sources/openrouter-models.json").read_text())["data"]
excluded = []
candidates = []
for m in ms:
    reason = None
    mid = m["id"]
    price = Decimal(m["pricing"].get("completion", "999")) * 1000000
    if price >= 2:
        reason = "Output price is not strictly below $2/M"
    elif mid.startswith("~") or "latest" in mid or mid.startswith("openrouter/"):
        reason = "Moving/router alias"
    elif ":batch" in mid:
        reason = "Batch-only duplicate/route, outside synchronous experiment"
    elif mid.endswith(":free") and any(
        n["id"] == mid[:-5]
        and Decimal(n["pricing"].get("completion", "999")) * 1000000 < 2
        for n in ms
    ):
        reason = "Duplicate free route of an affordable paid catalog model"
    elif mid.startswith("openai/gpt-6-luna-pro"):
        reason = "Same underlying Luna model, pro reasoning mode"
    elif mid.startswith(
        (
            "inference-net/schematron",
            "tencent/hy-mt",
            "inclusionai/ling-3.0-flash-fin",
            "inclusionai/ling-3.0-flash-sante",
        )
    ):
        reason = "Extraction, translation, finance or health specialist"
    elif "image" in m["architecture"].get("output_modalities", []) or "text" not in m[
        "architecture"
    ].get("output_modalities", []):
        reason = "Not a text-only output generator"
    elif any(
        Decimal(x.get("completion", "0")) * 1000000 >= 2
        for x in m["pricing"].get("overrides", [])
    ):
        reason = "Published price override reaches or exceeds $2/M"
    if reason:
        excluded.append({"id": mid, "reason": reason})
        continue
    slug = m.get("canonical_slug", "")
    date = re.search(r"-(20\d{6})(?:$|:)", slug)
    if date:
        version = datetime.datetime.strptime(date[1], "%Y%m%d").date().isoformat()
        basis = "canonical slug date (catalog version proxy, not independently verified launch date)"
    elif mid == "cohere/command-a-plus":
        version = "2026-05-01"
        basis = "canonical slug month 05-2026; day unknown, first day only for ordering"
    else:
        version = (
            datetime.datetime.fromtimestamp(m["created"], datetime.timezone.utc)
            .date()
            .isoformat()
        )
        basis = "OpenRouter listing date (release date unavailable)"
    candidates.append((version, m["created"], mid, m, basis))
candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
selected = []
families = {
    "openai": "GPT",
    "xiaomi": "MiMo",
    "qwen": "Qwen",
    "prism-ml": "Qwen (Bonsai derivative)",
    "z-ai": "GLM",
    "deepseek": "DeepSeek",
    "inclusionai": "Ling",
    "inception": "Mercury",
    "nex-agi": "Nex",
    "meta": "Muse",
    "ibm-granite": "Granite",
    "dots-studio": "Dots",
    "liquid": "LFM",
    "upstage": "Solar",
}
for version, created, mid, m, basis in candidates[:20]:
    r = m.get("reasoning") or {}
    reasoning = None
    if "reasoning" in m["supported_parameters"]:
        if not r.get("mandatory"):
            reasoning = {"enabled": False, "exclude": True}
        else:
            efforts = r.get("supported_efforts") or []
            lowest = next(
                (
                    e
                    for e in ["minimal", "low", "medium", "high", "xhigh", "max"]
                    if e in efforts
                ),
                None,
            )
            reasoning = {
                "exclude": True,
                **({"effort": lowest} if lowest else {"enabled": True}),
            }
    settings = {"max_tokens": 4096}
    if "temperature" in m["supported_parameters"]:
        settings["temperature"] = 1
    if reasoning:
        settings["reasoning"] = reasoning
    selected.append(
        dict(
            id=mid,
            canonical_slug=m["canonical_slug"],
            name=m["name"],
            family=families.get(mid.split("/")[0], mid.split("/")[0]),
            version_date_proxy=version,
            date_basis=basis,
            listed_utc=datetime.datetime.fromtimestamp(
                created, datetime.timezone.utc
            ).isoformat(),
            input_usd_per_million=str(Decimal(m["pricing"]["prompt"]) * 1000000),
            output_usd_per_million=str(Decimal(m["pricing"]["completion"]) * 1000000),
            settings=settings,
            catalog_record=m,
        )
    )
manifest = {
    "catalog_url": "https://openrouter.ai/api/v1/models",
    "catalog_sha256": sha((B / "sources/openrouter-models.json").read_bytes()),
    "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "selected_prompts_sha256": sha((D / "selected_prompts.jsonl").read_bytes()),
    "models": selected,
    "planned_prompts": 20,
    "planned_requests": 400,
    "system": "You are a helpful assistant.",
    "provider": {
        "max_price": {"prompt": 2, "completion": 1.999999, "request": 0},
        "sort": "price",
        "require_parameters": True,
    },
    "budget_note": "4096 API completion tokens per call, generally includes reasoning; $3.28 maximum completion allowance across 400 initial calls at routing ceiling, plus input and any billed retries. Not a guarantee across nonconforming providers.",
    "retry_policy": "At most two retries only HTTP429, 5xx, or transport failure; no retries for successful refusals, short text, truncation or quality.",
    "excluded_candidates": excluded,
    "eligible_not_selected": [
        {"id": x[2], "version_date_proxy": x[0]} for x in candidates[20:]
    ],
}
(D / "generation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
lines = [
    "# Arena-20 generator roster",
    "",
    f"Frozen {manifest['frozen_utc']}. 20 catalog model variants across {len(set(m['family'].split(' (')[0] for m in selected))} families. All text-only requests; multimodal models remain eligible.",
    "",
    "Recency uses canonical version dates where provided, otherwise listing dates. These are catalog proxies, not verified original release dates. Select newest eligible versions, breaking date ties by listing timestamp. Free-only models are included; duplicate free/batch routes and moving aliases are excluded. Agentic models such as Nex can produce prose and remain eligible; narrow extraction, translation and domain specialists do not. GLM FlashX and DeepSeek vision are distinct catalog variants, not claims of independent training/backbones. Bonsai counts with Qwen for family diversity.",
    "",
    "| Model / API route | Family | Version-date proxy | Input $/M | Output $/M |",
    "|---|---|---|---:|---:|",
]
for m in selected:
    lines.append(
        f"| `{m['id']}` | {m['family']} | {m['version_date_proxy']} | {float(m['input_usd_per_million']):g} | {float(m['output_usd_per_million']):g} |"
    )
lines += [
    "",
    "Sources: [live OpenRouter catalog](https://openrouter.ai/api/v1/models), saved in `sources/openrouter-models.json`; all selected catalog records and exclusion decisions are in `data/arena20/generation_manifest.json`.",
    "",
    "Price is the catalog starting price; the actual request enforces `provider.max_price.completion=1.999999` dollars per million, so higher-priced routes cannot be used. No model fallback is enabled. Provider fallback within the same model is allowed and the returned provider/model are recorded. Canonical slugs are recorded, but API route names do not cryptographically pin model weights. Free routes may be rate limited; failed cells stay in the matrix rather than being replaced.",
    "",
    "Settings: 4096 API completion tokens, temperature 1 only where supported, no tools. Disable optional reasoning; for mandatory reasoning use the lowest advertised effort, or the model default if no effort control is advertised. Reasoning is excluded from visible text and can consume the completion allowance. Exact per-model settings are frozen in the manifest.",
    "",
    manifest["budget_note"],
]
(B / "ARENA_20_MODELS.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines[6:29]))
