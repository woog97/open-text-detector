# Monthly top-20 model cost estimate

Source: [OpenRouter](https://openrouter.ai/rankings), as of 2026-09-23T01:03:27.570Z. Licensed under CC BY 4.0.

Window: 2026-08-24 through 2026-09-22 inclusive (30 complete UTC days). Rank by prompt plus completion tokens, variants separate. Public traffic only. Summed the documented daily top-50 dataset; bounded missing-day volume by each day’s 50th-place count and verified no omitted volume could alter any of the top-20 ranks. Displayed volumes are rounded observed totals, potentially lower bounds.

Prices from the current catalog saved alongside this report. Estimates use 20 requests with 50 input tokens each, and either 1,000 or 4,096 total completion tokens per request, including reasoning. Formula: (1,000 × input price + 20 × completion allowance × output price) / 1,000,000. No retries, tools, per-request fees, or provider markups included. These are scenarios, not measured costs or an invoice guarantee.

| Rank | Model | Monthly tokens (T) | Input $/M | Output $/M | Cost at 1,000 | Cost at 4,096 |
|---|---|---:|---:|---:|---:|---:|
| 1 | Z.ai: GLM 5.3 Flash | 52.03 | 0.15 | 0.5 | $0.0102 | $0.0411 |
| 2 | OpenAI: GPT-5.6 Luna | 51.40 | 0.2 | 1.2 | $0.0242 | $0.0985 |
| 3 | Tencent: Hy4 preview | 51.04 | 0.834 | 2.501 | $0.0509 | $0.2057 |
| 4 | DeepSeek: DeepSeek V4 Flash 0731 | 48.11 | 0.04 | 0.64 | $0.0128 | $0.0525 |
| 5 | Xiaomi: MiMo-V2.5 | 28.01 | 0.14 | 0.28 | $0.0057 | $0.0231 |
| 6 | DeepSeek: DeepSeek V4.1 Flash | 25.86 | 0.1 | 0.5 | $0.0101 | $0.0411 |
| 7 | Tencent: Hy3 | 20.21 | 0.132 | 0.528 | $0.0107 | $0.0434 |
| 8 | DeepSeek: DeepSeek V4 Flash 0423 | 19.57 | 0.088606 | 0.177212 | $0.0036 | $0.0146 |
| 9 | NVIDIA: Nemotron 3 Ultra (free) | 18.51 | 0 | 0 | $0.0000 | $0.0000 |
| 10 | Ox Alpha — absent from catalog; revealed as GLM 5.3 Flash | 15.68 | — | — | — | — |
| 11 | Z.ai: GLM 5.3 | 10.73 | 0.84 | 2.64 | $0.0536 | $0.2171 |
| 12 | Z.ai: GLM 5.2 | 9.25 | 0.6496 | 2.0416 | $0.0415 | $0.1679 |
| 13 | MiniMax M3 (free) — absent from catalog | 8.17 | — | — | — | — |
| 14 | OpenAI: GPT-5.6 Sol | 7.80 | 2 | 10 | $0.2020 | $0.8212 |
| 15 | Google: Gemini 3.7 Flash | 7.58 | 0.75 | 3.75 | $0.0758 | $0.3080 |
| 16 | MoonshotAI: Kimi K3 | 6.89 | 3 | 15 | $0.3030 | $1.2318 |
| 17 | Google: Gemini 3.8 Flash | 6.55 | 0.75 | 3.75 | $0.0758 | $0.3080 |
| 18 | MiniMax: MiniMax M3 | 6.33 | 0.3 | 1.2 | $0.0243 | $0.0986 |
| 19 | Anthropic: Claude Opus 5 | 6.03 | 5 | 25 | $0.5050 | $2.0530 |
| 20 | Anthropic: Claude Sonnet 5 | 5.90 | 2 | 10 | $0.2020 | $0.8212 |

18 catalog-listed routes total: $1.611130 at 1,000 output tokens or $6.546638 at 4,096. Free MiniMax, if available at its page-advertised zero price, would add $0. Ox Alpha has no current independently routable price.

Ox Alpha is the revealed predecessor of GLM 5.3 Flash: https://openrouter.ai/stealth/ox-alpha . MiniMax M3 free page still advertises zero pricing, but this does not confirm endpoint availability: https://openrouter.ai/minimax/minimax-m3:free . The paid MiniMax M3 is separately ranked #18.

GLM 5.3 Flash (#1) and DeepSeek V4.1 Flash (#6) have already received all 20 prompts in the existing pilot. Those recorded runs can be reused if the same generation settings are retained. No new generation requests were made for this estimate.
