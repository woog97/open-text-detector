# Inferring Pangram 4 model and dataset sizes

23 September 2026. This supplements the reconstruction and revised cost estimate. The disclosed runtime is evidence; the experiment and fresh-data budgets we proposed are not observations about Pangram.

**Working hypothesis:** an efficient MoE with roughly 80B total/3B active parameters and several million training documents is plausible. Under the central assumptions below it corresponds to about 10M document-equivalents. Confidence in the checkpoint identity remains low. A heavier MoE with a smaller dataset fits the same runtime. The evaluation program contains roughly three million reported examples across major sets, but the globally unique count is not disclosed.

## Evidence and limits

The [model card](https://www.pangram.com/research/model-card/pangram-4) reports eight H100s for four days, a sparse causal backbone, two-stage LoRA, and Repeat2. That provides 2,764,800 aggregate accelerator-seconds. The approximately $2,000 compute price adds no separate information about dataset size; it is just runtime multiplied by a rental rate.

The [technical overview](https://www.pangram.com/blog/pangram-4-technical) gives a roughly sixfold parameter increase over Pangram 3.3. The predecessor's production parameter count remains unknown. The historical 12B Pangram model and the 24B research EditLens model are clues, not verified production predecessors.

## Inverting the compute

Use the approximation:

`processed tokens = GPU-seconds × sustained FLOPs/second ÷ (k × active parameters)`

Choose k=6 as a central accounting approximation. Frozen-base LoRA omits most weight-gradient work, but activation-gradient propagation, recomputation, attention, and other work remain. k=4–8 is a useful sensitivity check; it is not measured for Pangram. MoE communication and small matrix inefficiencies also affect sustained throughput.

For the table, assume 50–150 TFLOP/s sustained per H100, inclusive of idle/communication effects across the run. This is an illustrative throughput envelope, not a benchmark or confidence interval. For context, NVIDIA's [H100 SXM specifications](https://www.nvidia.com/en-us/data-center/h100/) quote 1,979 BF16 TFLOP/s with hardware sparsity; the dense counterpart is about half that. MoE expert routing does not itself earn the 2:4 hardware-sparsity multiplier. Peak throughput is not a training-throughput estimate.

Assume one source-token-equivalent pass in stage 1 and half a pass in stage 2, with two copies per stage-2 window. Then processed tokens = 2 × source-token-equivalents. Divide the latter by an assumed 800 retained tokens per document to obtain document-equivalents.

| Active parameters | Processed tokens, 50–150 TFLOP/s | Document-equivalents at 800 tokens | Central value at 100 TFLOP/s |
|---|---:|---:|---:|
| 3B | 7.68–23.04B | 4.8–14.4M | 9.6M documents |
| 13B | 1.77–5.32B | 1.11–3.32M | 2.22M documents |
| 39B | 0.59–1.77B | 0.37–1.11M | 0.74M documents |

The central 3B-active branch gives 7.68B source-token-equivalents, about 15M full 512-token windows, and 15.36B actually processed tokens across stages. Stage-1 and stage-2 exposures are not distinct datasets whose sizes should simply be added.

These are **exposure-equivalent sizes**, not recovered unique-document counts. Repeated sampling, class balancing, multiple epochs, document cropping, and padding can separate them from unique corpus size. If only one 512-token crop per document is seen, use 512 instead of 800. If there are two effective passes over the source corpus rather than one, the unique corpus can be half as large.

Sensitivity:

- k=4 instead of 6 increases inferred tokens by 50%; k=8 decreases them by 25%.
- Stage 2 covering a quarter rather than half of stage 1 increases inferred source tokens by one-third at fixed processed tokens.
- Sustained throughput of 10 rather than 50 TFLOP/s reduces the low-end estimates fivefold. Poorly optimized expert kernels could therefore accommodate a much smaller corpus.
- Runtime including both preliminary and final training would reduce the amount attributed to one final fit. The model card does not explicitly identify every included job.

## What model size fits?

| Hypothesis | Total / active size | Why retain it? |
|---|---|---|
| Qwen3-Next-80B-A3B | 80B / approximately 3B | 80/12 ≈ 6.7; shared expert and efficient sparse backbone fit architectural clues |
| Mixtral-8x7B | Approximately 47B / 13B | Approximately six times an 8B predecessor, if that predecessor assumption were true |
| Mixtral-8x22B | 141B / 39B | 141/24 ≈ 5.9; fits a hypothetical continuation of the 24B Mistral research lineage |

Model specifications: [Qwen](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct), [Mixtral 8x7B](https://mistral.ai/news/mixtral-of-experts/), [Mixtral 8x22B](https://docs.mistral.ai/models/mixtral-8x22b-0-1-0-3).

This is a shortlist, not an exhaustive set. Other open MoEs can fit. The runtime does not preferentially establish the 80B hypothesis; it only tells us what dataset exposure is compatible with that hypothesis under chosen efficiencies.

If the H100s are 80GB models, eight provide 640GB aggregate VRAM. BF16 weights alone occupy roughly 160GB for 80B parameters or 282GB for 141B. Both can plausibly fit a sharded LoRA workflow. Thus eight GPUs do not distinguish these candidates. Quantization, optimizer layout, and activation memory would change the constraint.

My first implementation candidate remains 80B-total/3B-active, but I would label its identity **low confidence**. For that candidate, use approximately 5–15M document-equivalents as a conditional working range, not a verified corpus-size estimate. The competing heavier-backbone branches support roughly 0.4–3M instead.

## Evaluation data: mostly disclosed

The model card reports:

| Set | Reported examples |
|---|---:|
| English human FineWeb | 1,000,000 |
| Multilingual human FineWeb2 | 996,273 |
| English AI generations | 519,993 |
| Additional human domain rows, summed | 473,226 |

The first three sum to 2,516,266. Adding the domain rows gives 2,989,492 reported entries. These are not verified globally unique documents; category or source overlap could change the union.

The [technical report §5.5](https://arxiv.org/html/2607.27183v1) additionally gives 11,363 lightly polished texts, 14,990 substantively edited texts, and 4,826 examples in its WildChat-derived editing evaluation. These add 31,179 entries, taking the arithmetic tally above three million before other mixed and public benchmarks. Do not assume these also belong to training or calibration.

The model card's multilingual-AI row lists 190,149 examples, 23,578 false negatives, and 1.24%. Those numbers are inconsistent: 23,578/190,149 is about 12.4%. I exclude that row from the headline tally rather than infer which field is wrong.

A separate training-validation or calibration set size is not given. A possible reproduction allocation is 50k–200k diverse documents plus a large independent human false-positive audit, but that is a design recommendation, not an estimate recovered from their run. Cheap forward-only evaluation can cover millions of examples without implying that the training set must be larger.

## Implication for the corrected budget

The $1,920 final fit is compatible with a large training corpus. Low accelerator rental cost does not imply a tiny dataset. Conversely, the 100k–250k new-source scenarios in our earlier budget cannot establish the size of Pangram's existing corpus.

For illustration only, our proposed 4.25 document variants per human source converts 9.6M document-equivalents to roughly 2.26M source families, before accounting for repeated exposures. Neither that expansion ratio nor the 800-token mean is disclosed. Under a 3B-active hypothesis, training only 425k–1.06M 800-token documents on the same stage schedule would imply about 4.4–11.1 effective TFLOP/s per H100 over the reported duration. That is possible with substantial inefficiency, but is outside the 50–150 range used in the main table.

I would therefore distinguish **total accumulated training data**, potentially millions of documents, from **new synthetic data purchased for this release**, which may be a much smaller refresh. Runtime constrains the first only conditionally and says little about the second.

Arithmetic is saved in `pangram4-size-inference.json`. The most useful next measurement would be short-window LoRA throughput on the leading backbone candidates. That would narrow the dataset-size range substantially without requiring a full training run.
