# Arena-20 selected models

The final intended roster contains **20 model variants across 13 named families**. **19 returned responses**; Nex Mini was unavailable for all 20 requests. Two initially selected routes were replaced because of account restrictions, which were preserved. All original failures remain in the audit trail.

Selection: newest eligible catalog versions with output pricing strictly below $2 per million tokens. Version dates below come from canonical catalog slugs, not independently verified original launch dates. Tie-break by OpenRouter listing timestamp. Text input/output is required; multimodal general models are allowed.

| Model / API route | Family | Version-date proxy | Input $/M | Output $/M | Successful /20 |
|---|---|---|---:|---:|---:|
| `openai/gpt-6-luna` | GPT | 2026-09-22 | 0.1 | 0.5 | 20/20 |
| `xiaomi/mimo-v2.6-flash` | MiMo | 2026-09-21 | 0.14 | 0.28 | 20/20 |
| `xiaomi/mimo-v2.6-pro` | MiMo | 2026-09-21 | 0.435 | 0.87 | 20/20 |
| `qwen/qwen3.8-omni-flash` | Qwen | 2026-09-18 | 0.15 | 0.47 | 20/20 |
| `prism-ml/ternary-bonsai-2-27b` | Qwen (Bonsai derivative) | 2026-09-18 | 0.075 | 0.5 | 20/20 |
| `z-ai/glm-5.3-flashx` | GLM | 2026-09-18 | 0.37 | 1.25 | 20/20 |
| `inclusionai/ling-3.0-flash-vl` | Ling | 2026-09-10 | 0.06 | 0.18 | 20/20 |
| `deepseek/deepseek-v4.1-flash` | DeepSeek | 2026-09-10 | 0.1 | 0.5 | 20/20 |
| `inception/mercury-2.5` | Mercury | 2026-09-08 | 0.04 | 0.15 | 20/20 |
| `nex-agi/nex-n2.5-mini` | Nex | 2026-09-08 | 0.025 | 0.1 | 0/20 |
| `nex-agi/nex-n2.5-pro` | Nex | 2026-09-07 | 0.075 | 0.25 | 20/20 |
| `ibm-granite/granite-4.2-8b` | Granite | 2026-08-31 | 0.06 | 0.25 | 20/20 |
| `qwen/qwen3.8-flash` | Qwen | 2026-08-26 | 0.15 | 0.47 | 17/20 |
| `z-ai/glm-5.3-flash` | GLM | 2026-08-26 | 0.15 | 0.5 | 20/20 |
| `deepseek/deepseek-v4-flash-vision-exp` | DeepSeek | 2026-08-21 | 0.22 | 0.66 | 20/20 |
| `qwen/qwen3.8-27b:free` | Qwen | 2026-08-14 | 0 | 0 | 8/20 |
| `dots-studio/dots-3-note-preview:free` | Dots | 2026-08-13 | 0 | 0 | 20/20 |
| `upstage/solar-pro4` | Solar | 2026-08-10 | 0.09 | 0.36 | 20/20 |
| `meta/muse-glimmer-30b` | Muse | 2026-08-10 | 0.3 | 1.2 | 20/20 |
| `nvidia/nemotron-3.5-lightning` | Nemotron | 2026-08-07 | 0.07 | 0.2 | 20/20 |

Prices are the frozen catalog starting prices. Every actual request enforced `provider.max_price.completion=1.999999` dollars per million, so a higher-priced route could not be selected. Prompt pricing was capped at $2/M and per-request fees at zero. There was no cross-model fallback. Same-model provider fallback was allowed and each returned provider was recorded.

## What “20 models” means

These are distinct catalog variants, not a claim of 20 independently trained backbones. GLM FlashX is a speed variant; DeepSeek vision is a multimodal variant. Bonsai is counted with Qwen for family diversity. Nex remains eligible despite its coding emphasis because it supports general prose; narrow extraction, translation and domain specialists were excluded.

Duplicate batch/free routes and moving “latest” aliases were excluded. A free route was retained when there was no affordable paid version: for example, Qwen3.8 27B’s paid route exceeded the price ceiling. Free-route rate limits are recorded as failures. GPT-6 Luna Pro was excluded because the catalog describes it as the same underlying Luna model with a different reasoning mode. Cohere Command A+ was newly listed but its canonical version dates to May, so it did not displace newer versions. DeepSeek V4 Pro 0813 had a time-dependent output price above $2/M and was excluded.

## Availability replacements

| Original route | Restriction | Replacement |
|---|---|---|
| `meta/muse-spark-1.3-contributor` | HTTP 403: age attestation required | `meta/muse-glimmer-30b` |
| `liquid/lfm-2.5-2.6b:free` | HTTP 404: conflicts with existing account data-use preferences | `nvidia/nemotron-3.5-lightning` |

The replacements were the next eligible versions in the frozen recency ordering. Each received all 20 unchanged prompts, adding 40 cells to the original 400. No response quality or detector score informed these replacements. Nex Mini subsequently returned only service failures; those 20 cells remain unavailable rather than triggering unlimited retries or more substitutions.

## Frozen settings and sources

Each model received the same neutral system message and original user prompt. Cap: 4096 API completion tokens, generally shared with reasoning. Temperature 1 where supported; disable optional reasoning, use the lowest advertised mandatory reasoning effort otherwise. No tools, browsing, style/evasion instructions or per-model prompt adjustments. Raw provider defaults and complete settings are saved.

[OpenRouter catalog](https://openrouter.ai/api/v1/models), archived as `sources/openrouter-models.json`; [provider price-cap documentation](https://openrouter.ai/docs/guides/routing/provider-selection). Frozen full records, ranking exclusions and dates are in `data/arena20/generation_manifest.json`; account replacements are in `replacement_manifest.json`. Model routes do not cryptographically pin weights.

Only transport, rate-limit and server failures received at most two retries. Successful refusals, short responses, truncations and detector misses were never regenerated. The maximum initial completion allowance at the routing ceiling was $3.28 for 400 cells plus $0.33 for replacements; actual reported usage was much lower. Input and possibly billed retries are additional.
