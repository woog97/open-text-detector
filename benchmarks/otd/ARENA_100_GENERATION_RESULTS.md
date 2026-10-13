# Arena-100 generation complete

**1,200 successful responses: 100 prompts × 12 models. Total reported usage cost: $1.66934653.**

Optional reasoning disabled; GPT-OSS 120B low; Llama has no reasoning setting. Grok uses one batch; the other models used concurrent requests. Failed cells were retried following explicit user authorization. Successful responses were preserved.

| Model | Responses | Cost USD | Average completion tokens | Reported reasoning tokens | Mechanically eligible |
|---|---:|---:|---:|---:|---:|
| DeepSeek: DeepSeek V4 Flash 0423 | 100 | 0.01342478 | 742.8 | 0 | 100 |
| DeepSeek: DeepSeek V4 Pro 0423 | 100 | 0.17968401 | 935.6 | 0 | 98 |
| Google: Gemma 4 31B | 100 | 0.02578228 | 746.6 | 0 | 99 |
| Meta: Llama 3.3 70B Instruct | 100 | 0.01776522 | 542.5 | 0 | 100 |
| MoonshotAI: Kimi K2.6 | 100 | 0.16366816 | 550.3 | 0 | 98 |
| NVIDIA: Nemotron 3 Ultra | 100 | 0.36312900 | 1502.3 | 0 | 98 |
| OpenAI: gpt-oss-120b | 100 | 0.02816580 | 1639.3 | 1574 | 95 |
| Qwen: Qwen3.7 Max | 100 | 0.35317400 | 783.9 | 0 | 100 |
| Tencent: Hy3 preview | 100 | 0.02768742 | 449.3 | 0 | 98 |
| Thinking Machines: Inkling | 100 | 0.25024575 | 608.4 | 0 | 95 |
| SpaceXAI: Grok 4.3 (batch) | 100 | 0.08486020 | 373.4 | 0 | 100 |
| Z.ai: GLM 5.2 | 100 | 0.16175991 | 817.3 | 0 | 99 |

[Complete dataset](runs/arena100/dataset.jsonl) · [Frozen prompts](ARENA_100_SAMPLE.md) · [Protocol](ARENA_100_PROTOCOL.md)

Completion-token counts include any reported reasoning tokens. The dataset preserves all 1,200 responses; 1,180 qualify for the separate mechanical detector screen. Exclusions: {'below_50_words': 13, 'non_stop_finish': 7}.

Provider-reported usage, including batch aggregate once. Prior timed-out calls and other failures may have unreported charges; not invoice reconciliation.

Detector scores are separate from generation completion. See runs/arena100/ for per-response predictions and manifests.
