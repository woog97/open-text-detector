# Arena-100 results

1200 recorded prompt/model pairs, 1200 successful responses, 1180 mechanically eligible responses. Reported usage cost: **$1.669347**.

100 prompts, first 100 eligible in the original seeded order after screening 320 candidates. Original 20 unchanged. 100 distinct source users. No new dataset or model-weight downloads.

Reasoning disabled when optional; GPT-OSS 120B low; Llama has no reasoning control. Grok 4.3 uses one asynchronous batch; the other 11 models use concurrent requests.

| Model | Mode | Success /100 | Mechanically eligible | Cost $ | Reported reasoning tokens (coverage) | MELD misses/scored | EditLens misses/scored |
|---|---|---:|---:|---:|---:|---|---|
| DeepSeek: DeepSeek V4 Flash 0423 | concurrent | 100 | 100 | 0.013425 | 0 (100/100) | 2/100 | 18/100 |
| DeepSeek: DeepSeek V4 Pro 0423 | concurrent | 100 | 98 | 0.179684 | 0 (100/100) | 0/98 | 34/98 |
| Google: Gemma 4 31B | concurrent | 100 | 99 | 0.025782 | 0 (100/100) | 1/99 | 14/99 |
| Meta: Llama 3.3 70B Instruct | concurrent | 100 | 100 | 0.017765 | 0 (100/100) | 0/100 | 2/100 |
| MoonshotAI: Kimi K2.6 | concurrent | 100 | 98 | 0.163668 | 0 (100/100) | 3/98 | 3/98 |
| NVIDIA: Nemotron 3 Ultra | concurrent | 100 | 98 | 0.363129 | 0 (100/100) | 1/98 | 58/98 |
| OpenAI: gpt-oss-120b | concurrent | 100 | 95 | 0.028166 | 1574 (100/100) | 0/95 | 52/95 |
| Qwen: Qwen3.7 Max | concurrent | 100 | 100 | 0.353174 | 0 (100/100) | 0/100 | 5/100 |
| Tencent: Hy3 preview | concurrent | 100 | 98 | 0.027687 | 0 (100/100) | 4/98 | 9/98 |
| Thinking Machines: Inkling | concurrent | 100 | 95 | 0.250246 | 0 (100/100) | 2/95 | 17/95 |
| SpaceXAI: Grok 4.3 (batch) | batch | 100 | 100 | 0.084860 | 0 (100/100) | 6/100 | 22/100 |
| Z.ai: GLM 5.2 | concurrent | 100 | 99 | 0.161760 | 0 (100/100) | 1/99 | 24/99 |

## Interpretation

- Mechanical screen only (successful stop finish and >=50 whitespace words); no exhaustive original-prose or refusal-only adjudication.
- Native user-only output-longer-than-input token gate unverified.
- Descriptive AI-only pilot; no human FPR/AUROC; repeated prompt clusters invalidate pooled independent intervals.
- Costs are provider-reported, not invoice reconciliation; failed attempts may be billed.
- Reasoning totals are provider reports; missing counts do not mean zero. See summary.json for coverage.
- Mixed detector decisions count as misses. MELD uses its shipped threshold; EditLens 0.1/0.8 is exploratory. No threshold tuning.
- The earlier Arena-20 response screen included manual excerpts. This run reports a separate mechanical-screen variant; do not conflate the two protocols.

Artifacts: runs/arena100/dataset.jsonl contains all 1200 prompt/response pairs in one file. data/arena100/ contains frozen prompts, screening and manifests; runs/arena100/ also contains every raw attempt/response, batch request/status, detector outputs and cell matrix.
