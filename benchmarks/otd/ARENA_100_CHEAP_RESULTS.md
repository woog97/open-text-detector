# Earlier cheap models on 100 prompts

2,000 successful responses out of 2,000 cells across 20 models. New reported charges: **$0.64111519**. Historical charges for 365 reused responses: $0.15302697; those calls were not billed again.

| Model | Success /100 | Reused | New cost USD | Full 100-prompt cost USD | Eligible |
|---|---:|---:|---:|---:|---:|
| OpenAI: GPT-6 Luna | 100 | 20 | 0.01013570 | 0.01573290 | 93 |
| Xiaomi: MiMo-V2.6-Flash | 100 | 20 | 0.02424240 | 0.02967608 | 100 |
| Xiaomi: MiMo-V2.6-Pro | 100 | 20 | 0.07379253 | 0.09235485 | 100 |
| Qwen: Qwen3.8 Omni Flash | 100 | 20 | 0.03551695 | 0.04520414 | 96 |
| PrismML: Ternary Bonsai 2 27B | 100 | 20 | 0.03885395 | 0.05027057 | 99 |
| Z.ai: GLM 5.3 FlashX | 100 | 20 | 0.04325081 | 0.06097334 | 99 |
| inclusionAI: Ling 3.0 Flash VL | 100 | 20 | 0.01019526 | 0.01327614 | 100 |
| DeepSeek: DeepSeek V4.1 Flash | 100 | 20 | 0.03486984 | 0.04307462 | 99 |
| Inception: Mercury 2.5 | 100 | 20 | 0.00361538 | 0.00447575 | 92 |
| Nex AGI: Nex-N2.5-Mini | 100 | 0 | 0.00520853 | 0.00520853 | 99 |
| Nex AGI: Nex-N2.5-Pro | 100 | 20 | 0.00974645 | 0.01331057 | 98 |
| IBM: Granite 4.2 8B | 100 | 20 | 0.02877159 | 0.03624781 | 96 |
| Qwen: Qwen3.8 Flash | 100 | 17 | 0.03591666 | 0.04270985 | 98 |
| Z.ai: GLM 5.3 Flash | 100 | 20 | 0.01844713 | 0.02511649 | 99 |
| DeepSeek: DeepSeek V4 Flash Vision Exp | 100 | 20 | 0.04856346 | 0.06779960 | 99 |
| Qwen: Qwen3.8 27B (free + capped paid route) | 100 | 8 | 0.11983193 | 0.11983193 | 99 |
| Dots Studio: Dots3-Note Preview (free) | 100 | 20 | 0.00000000 | 0.00000000 | 96 |
| Upstage: Solar Pro 4 | 100 | 20 | 0.02284218 | 0.02969028 | 100 |
| Meta: Muse Glimmer 30B | 100 | 20 | 0.06509012 | 0.08331126 | 100 |
| NVIDIA: Nemotron 3.5 Lightning | 100 | 20 | 0.01222433 | 0.01587745 | 99 |

Reported usage only; failed/uncertain attempts may have unreported charges. Reused calls are historical, not charged again.

Together with the original 12-model Arena-100 pass: 3,200 successful responses across 3,200 intended cells. New charges across both 100-prompt passes: $2.31046172.

[Cheap-model dataset](runs/arena100-cheap/dataset.jsonl) · [Combined 32-model dataset](runs/arena100-combined/dataset.jsonl) · [Protocol](ARENA_100_CHEAP_PROTOCOL.md)

Eligibility is mechanical only: successful, stop-finished, at least 50 words. No exhaustive substantive-prose/refusal adjudication or native output-longer-than-input token check. Generation completion does not imply completed detector scoring.
Qwen 27B's free-route successes are retained. Failed free calls may use the paid route for the identical canonical version, with the same output cap below $2/M. Actual request route and provider are stored per response; this is a serving-route change, not a different model version.

API-key usage reconciliation: the key reports **$2.46648383 total usage**, versus $2.46348869 attached to saved successful responses. The $0.00299513 difference remains unattributed; failed/timed-out calls or other key activity cannot be ruled out. Billing-metadata lookups for eight saved failed generation IDs returned 404. The key total includes previous activity and is not an invoice for only this extension.
