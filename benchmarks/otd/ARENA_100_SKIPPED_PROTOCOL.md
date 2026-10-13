# Skipped-model expansion on the frozen 100 prompts

All 14 previously omitted Pangram Table 3 models plus GPT-5.6 Luna, GPT-6 Luna, GPT-6 Sol, and GPT-6 Astra. GPT-6 Terra is absent from the current OpenRouter catalog; the Terra Latest alias is GPT-5.6 Terra. No substitute is labelled GPT-6 Terra.

Reuse 100 existing GPT-6 Luna responses with identical complete requests. Generate 1,700 additional responses with 32 concurrent requests across models. This uses concurrent chat completions, not the discounted asynchronous batch endpoint. Preserve all successes; retry failed requests only. Reasoning disabled when optional, low when mandatory. 4,096-token cap includes reasoning where applicable. Account privacy settings unchanged.

Historical response lengths suggest $22.84; the 4,096-output-token cap scenario is $119.54 before any retry/failed-call charges. Neither is a billing guarantee.

| Model | Input $/M | Output $/M | Reasoning |
|---|---:|---:|---|
| Anthropic: Claude Fable 5 | 10 | 50 | {"effort": "low", "exclude": true} |
| Anthropic: Claude Opus 5 | 5 | 25 | {"enabled": false, "exclude": true} |
| Anthropic: Claude Sonnet 5 | 2 | 10 | {"enabled": false, "exclude": true} |
| Anthropic: Claude Opus 4.8 | 5 | 25 | {"enabled": false, "exclude": true} |
| Anthropic: Claude Sonnet 4.6 | 3 | 15 | {"enabled": false, "exclude": true} |
| Anthropic: Claude Haiku 4.5 | 1 | 5 | {"enabled": false, "exclude": true} |
| Google: Gemini 3.1 Pro Preview | 2 | 12 | {"effort": "low", "exclude": true} |
| Google: Gemini 3.5 Flash | 1.5 | 9 | {"effort": "low", "exclude": true} |
| Mistral: Mistral Medium 3.5 | 1.5 | 7.5 | {"enabled": false, "exclude": true} |
| OpenAI: GPT-5.6 Sol | 2 | 10 | {"enabled": false, "exclude": true} |
| OpenAI: GPT-5.6 Terra | 2 | 12 | {"enabled": false, "exclude": true} |
| OpenAI: GPT-5.5 | 5 | 30 | {"enabled": false, "exclude": true} |
| OpenAI: GPT-5.4 | 2.5 | 15 | {"enabled": false, "exclude": true} |
| OpenAI: GPT-5.4 Mini | 0.75 | 4.5 | {"enabled": false, "exclude": true} |
| OpenAI: GPT-5.6 Luna | 0.2 | 1.2 | {"enabled": false, "exclude": true} |
| OpenAI: GPT-6 Luna | 0.1 | 0.5 | {"enabled": false, "exclude": true} |
| OpenAI: GPT-6 Sol | 2 | 10 | {"enabled": false, "exclude": true} |
| OpenAI: GPT-6 Astra | 10 | 50 | {"effort": "low", "exclude": true} |

Same prompt checksum and sampling protocol as ARENA_100_PROTOCOL.md. No resampling or replacement based on outputs. Mechanical detector eligibility does not imply substantive prose review. New detector inputs will be prepared; detector scoring is separate.
