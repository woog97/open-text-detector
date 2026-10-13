# Arena-20 local benchmark results

Completed **440 recorded cells**: 400 original slots plus 40 availability replacements. **365 successful responses; 351 passed the local response screen.** Both local detectors were applied to every eligible response. Reported successful-request usage cost: **$0.1530** (not an invoice; failed attempts may have unreported charges).

The intended active roster has 20 model variants. Two account-blocked original routes remain separately visible below. Service failures are retained; no response was regenerated because of refusal, length, content quality or detector score.

## Sampling and scope

All 33,000 source rows were read. Exact opening-prompt agreement across both arms yielded 26,679 unique normalized requests. The precommitted seed ordered the complete frame; screening the first 79 candidates produced the first 20 eligible prompts, from 20 distinct anonymized users. No topic quotas or post-generation prompt substitutions.

Sample SHA-256: `b8b71dc4eaf3fcac80c6d1ed6355965825486814e56d13f0a23cb3b113581c92`. Categories: explanatory 7, creative 3, argumentative 2, practical 5, professional 3.

This is a 2023 English opening-request sample, weighted by unique prompt rather than traffic frequency. The public source may appear in generator training data, so unseen-prompt generalization is not established. Prompt eligibility and response scope were reviewed by one assistant. Response review used beginning/end excerpts and format counts, with full text for flagged cases; it was not an exhaustive independent human review or plagiarism check. Bullet paragraphs count as prose; bare data/verse and refusal-only answers do not. All scope decisions preceded scoring of that response. No generator names or scores appeared in review packets.

**Exact Pangram-style results are unavailable:** provider usage does not consistently supply user-only native token counts. The output-longer-than-input token gate is unverified for every response. Results below explicitly omit that gate and use successful, completed, locally scope-screened answers of at least 50 words. They are exploratory local-baseline results, not a Pangram 4 reproduction.

Provider-usage diagnostic: 365/365 responses with complete reported counts have visible completion tokens (reported completion minus reasoning) greater than the entire reported prompt count, including the system message. This is a separate proxy, not proof of native user-only tokenization; it did not filter the sample or accuracy denominator.

## Active roster results

| Generator | API success /20 | Eligible | MELD misses /N (FNR) | EditLens misses /N (FNR) | MELD detections /20 | EditLens detections /20 |
|---|---:|---:|---:|---:|---:|---:|
| OpenAI: GPT-6 Luna | 20/20 | 19 | 2/19 (10.5%) | 9/19 (47.4%) | 17/20 | 10/20 |
| Xiaomi: MiMo-V2.6-Flash | 20/20 | 20 | 0/20 (0.0%) | 9/20 (45.0%) | 20/20 | 11/20 |
| Xiaomi: MiMo-V2.6-Pro | 20/20 | 20 | 1/20 (5.0%) | 11/20 (55.0%) | 19/20 | 9/20 |
| Qwen: Qwen3.8 Omni Flash | 20/20 | 18 | 0/18 (0.0%) | 4/18 (22.2%) | 18/20 | 14/20 |
| PrismML: Ternary Bonsai 2 27B | 20/20 | 20 | 1/20 (5.0%) | 7/20 (35.0%) | 19/20 | 13/20 |
| Z.ai: GLM 5.3 FlashX | 20/20 | 19 | 0/19 (0.0%) | 3/19 (15.8%) | 19/20 | 16/20 |
| inclusionAI: Ling 3.0 Flash VL | 20/20 | 20 | 0/20 (0.0%) | 3/20 (15.0%) | 20/20 | 17/20 |
| DeepSeek: DeepSeek V4.1 Flash | 20/20 | 18 | 0/18 (0.0%) | 2/18 (11.1%) | 18/20 | 16/20 |
| Inception: Mercury 2.5 | 20/20 | 18 | 0/18 (0.0%) | 2/18 (11.1%) | 18/20 | 16/20 |
| Nex AGI: Nex-N2.5-Mini | 0/20 | 0 | 0/0 (—) | 0/0 (—) | 0/20 | 0/20 |
| Nex AGI: Nex-N2.5-Pro | 20/20 | 19 | 0/19 (0.0%) | 4/19 (21.1%) | 19/20 | 15/20 |
| IBM: Granite 4.2 8B | 20/20 | 19 | 0/19 (0.0%) | 7/19 (36.8%) | 19/20 | 12/20 |
| Qwen: Qwen3.8 Flash | 17/20 | 15 | 0/15 (0.0%) | 2/15 (13.3%) | 15/20 | 13/20 |
| Z.ai: GLM 5.3 Flash | 20/20 | 19 | 0/19 (0.0%) | 3/19 (15.8%) | 19/20 | 16/20 |
| DeepSeek: DeepSeek V4 Flash Vision Exp | 20/20 | 19 | 0/19 (0.0%) | 4/19 (21.1%) | 19/20 | 15/20 |
| Qwen: Qwen3.8 27B (free) | 8/20 | 8 | 0/8 (0.0%) | 2/8 (25.0%) | 8/20 | 6/20 |
| Dots Studio: Dots3-Note Preview (free) | 20/20 | 20 | 1/20 (5.0%) | 3/20 (15.0%) | 19/20 | 17/20 |
| Upstage: Solar Pro 4 | 20/20 | 20 | 0/20 (0.0%) | 3/20 (15.0%) | 20/20 | 17/20 |
| Meta: Muse Glimmer 30B | 20/20 | 20 | 2/20 (10.0%) | 8/20 (40.0%) | 18/20 | 12/20 |
| NVIDIA: Nemotron 3.5 Lightning | 20/20 | 20 | 0/20 (0.0%) | 5/20 (25.0%) | 20/20 | 15/20 |

Mixed counts as a miss under the strict document decision convention. MELD uses its shipped threshold (50–99-word results are below its recommended length); EditLens uses exploratory 0.1/0.8 thresholds. Detections/20 is end-to-end yield, not detector recall. Zero-denominator FNR is unavailable, not zero.

## Uncertainty and paired comparisons

| Generator | MELD FNR Wilson 95% interval | EditLens FNR Wilson 95% interval |
|---|---:|---:|
| OpenAI: GPT-6 Luna | 2.9%–31.4% | 27.3%–68.3% |
| Xiaomi: MiMo-V2.6-Flash | 0.0%–16.1% | 25.8%–65.8% |
| Xiaomi: MiMo-V2.6-Pro | 0.9%–23.6% | 34.2%–74.2% |
| Qwen: Qwen3.8 Omni Flash | 0.0%–17.6% | 9.0%–45.2% |
| PrismML: Ternary Bonsai 2 27B | 0.9%–23.6% | 18.1%–56.7% |
| Z.ai: GLM 5.3 FlashX | 0.0%–16.8% | 5.5%–37.6% |
| inclusionAI: Ling 3.0 Flash VL | 0.0%–16.1% | 5.2%–36.0% |
| DeepSeek: DeepSeek V4.1 Flash | 0.0%–17.6% | 3.1%–32.8% |
| Inception: Mercury 2.5 | 0.0%–17.6% | 3.1%–32.8% |
| Nex AGI: Nex-N2.5-Mini | — | — |
| Nex AGI: Nex-N2.5-Pro | 0.0%–16.8% | 8.5%–43.3% |
| IBM: Granite 4.2 8B | 0.0%–16.8% | 19.1%–59.0% |
| Qwen: Qwen3.8 Flash | 0.0%–20.4% | 3.7%–37.9% |
| Z.ai: GLM 5.3 Flash | 0.0%–16.8% | 5.5%–37.6% |
| DeepSeek: DeepSeek V4 Flash Vision Exp | 0.0%–16.8% | 8.5%–43.3% |
| Qwen: Qwen3.8 27B (free) | 0.0%–32.4% | 7.1%–59.1% |
| Dots Studio: Dots3-Note Preview (free) | 0.9%–23.6% | 5.2%–36.0% |
| Upstage: Solar Pro 4 | 0.0%–16.1% | 5.2%–36.0% |
| Meta: Muse Glimmer 30B | 2.8%–30.1% | 21.9%–61.3% |
| NVIDIA: Nemotron 3.5 Lightning | 0.0%–16.1% | 11.2%–46.9% |

Common scope-eligible prompts across all 20 intended active models: **0**. `paired_comparisons.jsonl` contains common-prompt denominators and rates for every pair of active generators, separately for each detector. Do not rank models using unmatched denominators alone.

Twenty prompts give coarse estimates: one miss is 5 percentage points when N=20; zero misses still has a 16.1% Wilson upper bound. Responses share prompt clusters, so 400 outputs are not 400 independent prompt draws. No pooled independence-based interval or significance ranking is claimed. There are no human answer controls here: FPR and AUROC are unavailable.

## Availability and response exclusions

| Route | Exclusions |
|---|---|
| `openai/gpt-6-luna` | non_stop_finish: 1 |
| `qwen/qwen3.8-omni-flash` | non_stop_finish: 2 |
| `z-ai/glm-5.3-flashx` | non_stop_finish: 1 |
| `deepseek/deepseek-v4.1-flash` | refusal_only: 1; non_stop_finish: 1 |
| `inception/mercury-2.5` | below_50_words: 1; refusal_only: 1 |
| `nex-agi/nex-n2.5-mini` | api_failure: 20 |
| `nex-agi/nex-n2.5-pro` | non_stop_finish: 1 |
| `meta/muse-spark-1.3-contributor` | api_failure: 20 |
| `ibm-granite/granite-4.2-8b` | non_stop_finish: 1 |
| `qwen/qwen3.8-flash` | non_stop_finish: 2; api_failure: 3 |
| `z-ai/glm-5.3-flash` | non_stop_finish: 1 |
| `deepseek/deepseek-v4-flash-vision-exp` | non_stop_finish: 1 |
| `qwen/qwen3.8-27b:free` | api_failure: 12 |
| `liquid/lfm-2.5-2.6b:free` | api_failure: 20 |

The two original account restrictions were preserved: Muse Spark 1.3 required age attestation; Liquid free conflicted with account privacy preferences. Muse Glimmer and Nemotron were chosen from the next eligible catalog versions, based on availability rather than response quality. Exact errors/retries/providers are in the response files.

## Audit artifacts

- `source_manifest.json`, `frame.jsonl`, `screening.jsonl`, `selected_prompts.jsonl` under `data/arena20/`: full acquisition and sampling provenance.
- `generation_manifest.json` and `replacement_manifest.json`: frozen versions, price caps and settings.
- `responses.jsonl` in each run folder: original requests, raw responses, usage and retry attempts.
- `content_reviews.jsonl`, `response_audit.jsonl`, `review_packets/`: local scope decisions before scoring.
- `cell_matrix.jsonl`: every attempted prompt/model cell and both detector outcomes.
- `paired_comparisons.jsonl`: all pairwise common-prompt comparisons.
- `meld/` and `editlens/`: checkpoint manifests, code snapshots, predictions, aggregate metrics and scored-input hashes.

The separate proposed API-based judging step was rejected by automatic approval review and never ran. All content review and detector scoring used local files. The website was not modified.
