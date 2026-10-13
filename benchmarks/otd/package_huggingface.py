"""Build a credential-free Hugging Face dataset package from frozen generations."""

import json
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

from arena20 import B, read, sha
from arena100_results import collect

OUT = B / "exports/arena-prose-100-49-models"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data").mkdir(exist_ok=True)
    original_path = B / "runs/arena100-expanded-kimi/dataset.jsonl"
    original = read(original_path)
    prompts = {
        p["prompt_id"]: p for p in read(B / "data/arena100/selected_prompts.jsonl")
    }
    manifests = {}
    raw = {}
    for phase in ["arena100", "arena100-cheap", "arena100-skipped", "arena100-kimi"]:
        manifest = json.loads(
            (B / f"data/{phase}/generation_manifest.json").read_text()
        )
        for m in manifest["models"]:
            manifests.setdefault(m["id"], m)
        for r in collect(B / f"runs/{phase}"):
            k = (r["model_id"], r["prompt_id"])
            if k in raw:
                assert r["raw"]["id"] == raw[k]["raw"]["id"]
            else:
                raw[k] = r
    assert len(original) == len(raw) == 5000
    assert len(manifests) == 50
    schema = pa.schema(
        [
            ("id", pa.string()),
            ("prompt_id", pa.string()),
            ("model_id", pa.string()),
            ("model_name", pa.string()),
            ("canonical_model", pa.string()),
            ("prompt", pa.string()),
            ("response", pa.string()),
            ("label", pa.string()),
            ("prompt_category", pa.string()),
            ("language", pa.string()),
            ("source_dataset", pa.string()),
            ("source_revision", pa.string()),
            ("source_question_ids", pa.list_(pa.string())),
            ("screening_position", pa.int64()),
            ("request_model", pa.string()),
            ("response_model", pa.string()),
            ("provider", pa.string()),
            ("generation_id", pa.string()),
            ("finish_reason", pa.string()),
            ("success", pa.bool_()),
            ("mechanically_eligible", pa.bool_()),
            ("eligibility_reason", pa.string()),
            ("response_words", pa.int64()),
            ("prompt_tokens", pa.int64()),
            ("completion_tokens", pa.int64()),
            ("reasoning_tokens", pa.int64()),
            ("reported_cost_usd", pa.float64()),
            ("usage_available", pa.bool_()),
            ("reasoning_setting", pa.string()),
            ("max_tokens", pa.int64()),
            ("temperature", pa.float64()),
            ("request_settings_json", pa.string()),
            ("usage_json", pa.string()),
            ("request_sha256", pa.string()),
            ("response_sha256", pa.string()),
        ]
    )
    rows = []
    source = json.loads((B / "data/arena100/source_manifest.json").read_text())
    for row in sorted(original, key=lambda r: (r["model_id"], r["prompt_id"])):
        p = prompts[row["prompt_id"]]
        m = manifests[row["model_id"]]
        r = raw[(row["model_id"], row["prompt_id"])]
        response = r["raw"]
        usage = response.get("usage")
        u = usage or {}
        settings = m["settings"]
        reasoning = settings.get("reasoning", {})
        assert sha(row["response_text"].encode()) == row["response_sha256"]
        assert row["prompt_text"] == p["text"]
        rows.append(
            {
                "id": sha((row["model_id"] + "\0" + row["prompt_id"]).encode()),
                "prompt_id": row["prompt_id"],
                "model_id": row["model_id"],
                "model_name": m["name"],
                "canonical_model": m["canonical_slug"],
                "prompt": p["text"],
                "response": row["response_text"],
                "label": "ai",
                "prompt_category": p["category"],
                "language": "en",
                "source_dataset": "lmsys/chatbot_arena_conversations",
                "source_revision": source["revision"],
                "source_question_ids": p["source_ids"],
                "screening_position": p["position"],
                "request_model": r["request"].get("model", row["model_id"]),
                "response_model": response.get("model"),
                "provider": response.get("provider"),
                "generation_id": response.get("id"),
                "finish_reason": row["finish_reason"],
                "success": row["success"],
                "mechanically_eligible": row["mechanically_eligible"],
                "eligibility_reason": row["eligibility_reason"],
                "response_words": len(row["response_text"].split()),
                "prompt_tokens": u.get("prompt_tokens"),
                "completion_tokens": u.get("completion_tokens"),
                "reasoning_tokens": (u.get("completion_tokens_details") or {}).get(
                    "reasoning_tokens"
                ),
                "reported_cost_usd": u.get("cost"),
                "usage_available": usage is not None,
                "reasoning_setting": reasoning.get(
                    "effort",
                    "disabled"
                    if reasoning.get("enabled") is False
                    else "not_advertised",
                ),
                "max_tokens": settings["max_tokens"],
                "temperature": settings.get("temperature"),
                "request_settings_json": json.dumps(settings, sort_keys=True),
                "usage_json": json.dumps(usage, sort_keys=True)
                if usage is not None
                else None,
                "request_sha256": row["request_sha256"],
                "response_sha256": row["response_sha256"],
            }
        )
    table = pa.Table.from_pylist(rows, schema=schema)
    path = OUT / "data/test-00000-of-00001.parquet"
    pq.write_table(table, path, compression="zstd", version="2.6")
    loaded = pq.read_table(path)
    assert loaded.equals(table)
    assert len(set(loaded["id"].to_pylist())) == 5000
    assert set(Counter(loaded["model_id"].to_pylist()).values()) == {100}
    eligible = sum(r["mechanically_eligible"] for r in rows)
    model_rows = []
    for mid, m in sorted(manifests.items()):
        model_rows.append(
            {
                "model_id": mid,
                "name": m["name"],
                "canonical_model": m["canonical_slug"],
                "settings": m["settings"],
                "serving_mode": m["mode"],
                "catalog_pricing": m["catalog_record"]["pricing"],
                "catalog_created": m["catalog_record"]["created"],
                "note": "Catalog prices are routing-dependent snapshots, not guaranteed final billed rates.",
            }
        )
    (OUT / "models.json").write_text(json.dumps(model_rows, indent=2) + "\n")
    safe_source = {k: v for k, v in source.items() if k not in ["url"]}
    provenance = {
        "source": safe_source,
        "corpus_sha256": sha(original_path.read_bytes()),
        "rows": len(rows),
        "models": len(manifests),
        "distinct_prompts": len(prompts),
        "mechanically_eligible_rows": eligible,
        "usage_missing_rows": sum(not r["usage_available"] for r in rows),
        "unavailable_requested_model": "openai/gpt-6-terra",
        "system_message": "You are a helpful assistant.",
        "schema": {f.name: str(f.type) for f in schema},
        "parquet_sha256": sha(path.read_bytes()),
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    roster = "\n".join(f"| {m['name']} | `{m['model_id']}` | 100 |" for m in model_rows)
    card = f"""---
pretty_name: Arena Prose — 100 Prompts × 50 Models
language:
- en
license: other
license_name: mixed-source-and-model-output-terms
license_link: LICENSE.md
task_categories:
- text-generation
- text-classification
tags:
- ai-text-detection
- synthetic
- evaluation
- chatbot-arena
size_categories:
- 1K<n<10K
configs:
- config_name: default
  default: true
  data_files:
  - split: test
    path: data/test-*.parquet
---

# Arena Prose: 100 prompts × 50 models

A paired exploratory AI-text-detection corpus: **5,000 successful generated responses** from **50 models**, each answering the same **100 English prose prompts**. Generation was performed through OpenRouter in September 2026 with optional reasoning disabled and mandatory reasoning set to low. This is an independent local benchmark inspired by [Pangram 4 §5.2](https://arxiv.org/html/2607.27183v1#S5.SS2), not an official Pangram dataset or exact replication.

## Loading

```python
from datasets import load_dataset

ds = load_dataset("woog/arena-prose-100-49-models", split="test")
# Preserve every response for generation analysis; filter only for the defined detector gate.
scorable = ds.filter(lambda row: row["mechanically_eligible"])
```

There is a single `test` split. Every model/prompt pair has a stable unique ID. This release does not invent training/validation splits: the same prompt appears 50 times, so any future partitioning must group by `prompt_id` to avoid prompt leakage.

## Sampling and source attribution

Prompts come from [LMSYS Chatbot Arena Conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations), revision `{source["revision"]}`. Only opening user requests are used; archived assistant responses and votes are not included. The source contains 33,000 pairs collected in April–June 2023. Exact-agreement opening requests were deduplicated using NFC and whitespace normalization, producing 26,679 unique requests. Original wording was preserved for generation.

Candidates were ordered globally by SHA-256 of `pangram4-arena20-v1`, a NUL separator, and the normalized prompt ID. A single assistant reviewer screened the contiguous prefix of 320 candidates; the first 100 eligible prompts were retained. Eligible requests are self-contained English requests for substantial original prose. Coding, mathematical problem-solving, multiple-choice, rewriting/editing, translation, extraction, copied-text requests and short factual answers were excluded. No topic quotas were imposed. Category counts: 44 explanatory, 15 creative, 11 argumentative, 26 practical, 4 professional.

The original 20 selected prompts were retained when expanding to 100. The 80 additions were screened before their generation; outcomes from the original 20 had already been observed. All generators received the identical frozen sample. Reused successful generations were retained only when the full request matched. Responses were not replaced for refusal, quality, length, truncation or detector outcomes. Source question IDs permit tracing requests; anonymized source-user IDs and timestamps are not exported.

## Generation and fields

The fixed system message was `You are a helpful assistant.` Calls used a 4,096-token cap and temperature 1 where advertised; unsupported controls were omitted. Reasoning was disabled when optional and set to low when mandatory. Llama advertised no reasoning control. The token cap includes reasoning where the provider counts it toward completion tokens. Grok 4.3 used the asynchronous batch endpoint; other calls used parallel chat completions. Failed requests were retried without regenerating successes.

`model_id` is the logical roster ID, while `request_model`, `response_model`, and `provider` identify the actual route where recorded. Qwen 3.8 27B retains the logical `:free` ID; 67 responses used the paid DeepInfra BF16 route for the identical canonical model after free-route rate limits. Pricing and availability differ across providers. Missing provider/usage fields remain null.

Main columns:

- `prompt`, `response`, `prompt_id`, `model_id`, `model_name`, `canonical_model`, `label` (always `ai`).
- `prompt_category`, source dataset/revision/question IDs, and `screening_position` for traceability.
- `finish_reason`, `success`, `mechanically_eligible`, `eligibility_reason`, and `response_words`.
- Nullable input/completion/reasoning token counts and `reported_cost_usd`; `usage_available` distinguishes missing records from zero usage.
- Requested reasoning/temperature/token settings, complete usage JSON, and request/response hashes.

`models.json` lists the 50 generators and settings; `provenance.json` contains source hashes, sample checksums, and the full schema. Costs are response-level usage when available. Grok's batch aggregate charge is not allocated to individual rows; null row costs must not be treated as zero. Two Claude Fable responses also omitted usage entirely. The repository contains no credentials or account billing details.

## Eligibility and limitations

All 5,000 successful responses are retained; **{eligible:,}** pass the mechanical gate (stop completion and at least 50 whitespace-separated words). Short/truncated/refusal-like outputs are not regenerated. The gate does not establish substantive original prose or adjudicate refusals; native user-only input versus output token length was not verified. Local detector predictions are not included in this release.

These are only 100 distinct, historical, English prompts with single-reviewer screening, not 5,000 independent prompt draws. Model selection evolved over several phases and combines the 26 Pangram report generators with newer inexpensive models and requested GPT variants, followed by Kimi K3. The repository slug retains its original 49-model name for URL stability; the current release has 50 models. GPT-6 Terra was not available in the frozen OpenRouter catalog and is not included. This AI-only corpus cannot establish human false-positive rate or AUROC. It is suitable for descriptive paired comparisons, not fine-grained population ranking; uncertainty calculations should cluster by prompt. Generated statements are unverified model outputs.

## Models

| Model | OpenRouter logical ID | Responses |
|---|---|---:|
{roster}

## Licensing and attribution

See [LICENSE.md](LICENSE.md). The upstream card assigns CC BY 4.0 to user prompts. Its separate license for its archived assistant outputs is not applied to these newly generated completions. New outputs may be subject to the applicable model/provider terms; this release does not represent them as covered by one permissive blanket license. Source text is unchanged; selection, organization, metadata and newly generated responses are additions.

Source citation: Zheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, [arXiv:2306.05685](https://arxiv.org/abs/2306.05685). Dataset source: LMSYS Chatbot Arena Conversations. Do not attempt to identify source users.
"""
    (OUT / "README.md").write_text(card)
    (OUT / "LICENSE.md").write_text("""# Component licensing

User prompts originate from LMSYS Chatbot Arena Conversations, licensed by that source under Creative Commons Attribution 4.0 International: https://creativecommons.org/licenses/by/4.0/

Attribution and pinned source: https://huggingface.co/datasets/lmsys/chatbot_arena_conversations/tree/1b6335d42a1d2c7e34870c905d03ab964f7f2bd8 . See README.md for authors, citation and the sampling transformation. Selected prompt strings are unchanged.

The source's archived model answers are not included. The answers here are new generations from the models named in models.json. No blanket license grant or warranty is made for these third-party model outputs; applicable model/provider terms remain relevant. This notice does not grant rights beyond those available from the respective rightsholders.

The Hugging Face `license: other` metadata describes this component-specific notice, not a standard permissive license for the entire corpus.
""")
    allowed = {
        "README.md",
        "LICENSE.md",
        "models.json",
        "provenance.json",
        "data/test-00000-of-00001.parquet",
    }
    assert {
        p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file()
    } == allowed
    secrets = [
        s.split("=", 1)[1].strip().strip('"\x27')
        for s in (B / ".env.secrets").read_text().splitlines()
        if "=" in s and not s.startswith("#")
    ]
    for p in OUT.rglob("*"):
        if p.is_file():
            assert not any(v and v.encode() in p.read_bytes() for v in secrets)
    assert not any(v and v in json.dumps(rows) for v in secrets)
    print(
        json.dumps(
            {
                "path": str(OUT),
                "rows": len(rows),
                "models": len(manifests),
                "eligible": eligible,
                "bytes": sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()),
                "parquet_sha256": sha(path.read_bytes()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
