# Previously skipped models: 100-prompt generation results

1,800/1,800 successful responses across 18 available models, including 100 reused GPT-6 Luna responses. New reported cost: **$20.317845**.

Billing metadata is missing for 2 successful responses; the reported cost excludes their unknown charges. Claude Fable token averages use 98 responses. Billing metadata lookups for the two missing records returned 404.

GPT-6 Terra is not listed in the frozen OpenRouter catalog. Terra Latest points to GPT-5.6 Terra; it was not treated as GPT-6 Terra.

| Model | Success /100 | New cost USD | Input $/M | Output $/M | Mean output tokens | Reasoning | Catalog date |
|---|---:|---:|---:|---:|---:|---|---|
| Anthropic: Claude Fable 5 | 100 | 3.437170 | 10 | 50 | 692.4 | low | 2026-06-09 |
| Anthropic: Claude Opus 5 | 100 | 2.879440 | 5 | 25 | 1142.8 | disabled | 2026-07-24 |
| Anthropic: Claude Sonnet 5 | 100 | 0.791992 | 2 | 10 | 783.0 | disabled | 2026-06-30 |
| Anthropic: Claude Opus 4.8 | 100 | 1.799855 | 5 | 25 | 710.9 | disabled | 2026-05-27 |
| Anthropic: Claude Sonnet 4.6 | 100 | 0.834288 | 3 | 15 | 549.3 | disabled | 2026-02-17 |
| Anthropic: Claude Haiku 4.5 | 100 | 0.216141 | 1 | 5 | 425.6 | disabled | 2025-10-15 |
| Google: Gemini 3.1 Pro Preview | 100 | 1.763872 | 2 | 12 | 1465.6 | low | 2026-02-19 |
| Google: Gemini 3.5 Flash | 100 | 1.497264 | 1.5 | 9 | 1659.4 | low | 2026-05-19 |
| Mistral: Mistral Medium 3.5 | 100 | 0.716083 | 1.5 | 7.5 | 946.2 | disabled | 2026-04-30 |
| OpenAI: GPT-5.6 Sol | 100 | 0.398458 | 2 | 10 | 391.4 | disabled | 2026-07-09 |
| OpenAI: GPT-5.6 Terra | 100 | 0.652478 | 2 | 12 | 537.9 | disabled | 2026-07-09 |
| OpenAI: GPT-5.5 | 100 | 1.587920 | 5 | 30 | 523.5 | disabled | 2026-04-24 |
| OpenAI: GPT-5.4 | 100 | 0.870003 | 2.5 | 15 | 574.2 | disabled | 2026-03-05 |
| OpenAI: GPT-5.4 Mini | 100 | 0.199913 | 0.75 | 4.5 | 438.4 | disabled | 2026-03-17 |
| OpenAI: GPT-5.6 Luna | 100 | 0.052000 | 0.2 | 1.2 | 427.5 | disabled | 2026-07-09 |
| OpenAI: GPT-6 Luna | 100 | 0.000000 | 0.1 | 0.5 | 307.6 | disabled | 2026-09-22 |
| OpenAI: GPT-6 Sol | 100 | 0.227438 | 2 | 10 | 220.4 | disabled | 2026-09-22 |
| OpenAI: GPT-6 Astra | 100 | 2.393530 | 10 | 50 | 471.7 | low | 2026-09-04 |

Combined corpus: 4,900 successful responses, with $22.781334 reported cost including earlier runs and reused responses. No duplicate model/prompt pairs or generation IDs.

Costs sum known successful-response usage only; missing billing records are not treated as free calls. Failed or uncertain attempts may also carry charges. Mean output tokens includes reported reasoning tokens and averages only responses with usage records. Catalog dates are OpenRouter listing dates, not independently verified release dates.

All successful outputs are preserved, including short/truncated responses. Mechanical eligibility requires stop completion and at least 50 words. Added models have detector inputs prepared but have not been scored by local detectors.

[Extension dataset](runs/arena100-skipped/dataset.jsonl) · [Combined dataset](runs/arena100-expanded/dataset.jsonl) · [Protocol](ARENA_100_SKIPPED_PROTOCOL.md)

Final account usage snapshot: $22.78432888 total, an increase of $20.31784505 during this extension, matching the known response-cost sum. The approximately $0.003 difference from the full corpus total already existed before this run. Two individual response billing records remain unavailable; no extra charge for them is visible in this snapshot. Remaining key allowance: $17.21567112.
