# Laya contextual phrase baseline

Laya is the project described in Nandakishor Mukkunnoth's [“Laya the open source version of Jev”](https://laya.convaiinnovations.com/). His earlier research dates to 2025; this is a separate model, not released Jev weights. Its comparisons with Jev do not establish AI-writing detection quality.

The English checkpoint is pinned to `convaiinnovations/laya` revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`. The installer checks SHA-256 for LFS files and Git blob hashes for other files. It downloads no model Python code. The reviewed architecture follows [Laya v0.3.20](https://github.com/NandhaKishorM/laya/tree/v0.3.20), with its original tensor names and strict reference weight loading. Model and code are Apache-2.0; see `licenses/laya-Apache-2.0.txt`.

## Use

```sh
backend/.venv/bin/python scripts/download_laya.py
cd backend
.venv/bin/python -m open_text_detector_backend serve
```

In another terminal, from the project root:

```sh
backend/.venv/bin/python scripts/activate_laya.py
```

Choose **Laya · Experimental contextual phrases** in the workbench model selector, or set it as default in Models & settings. The activation script preserves the existing default; `--default` explicitly changes it. The API accepts `provider: "laya"`, `task: "text"`, and `model_id: "convaiinnovations/laya"`. Other Laya checkpoints and image input are rejected. Weights stay loaded between requests; inference is serialized like MELD.

The existing `models` dependency extra supports the PyTorch reference. Install `apple-gpu` for MLX (`uv sync --frozen --extra models --extra apple-gpu`). No Laya package or remote-code execution is required.

## Phrase and context policy

- Preserve original Unicode code-point offsets, whitespace, and source text. Duplicate tokenizer offsets for multi-byte characters cannot produce overlapping highlights.
- Group short clauses into a useful target, split at sentence/semicolon/colon boundaries after roughly 16 tokens, and cap targets at roughly 64 tokens. Paragraph boundaries always end a target. The token cap may split a long word; it never splits a Unicode character.
- Score each target independently with an explicit two-option question (human writing versus generated/substantially rewritten writing). The target is always retained in full. No token-level attribution is fabricated.
- Supply preceding and following context from the same paragraph, balanced within the checkpoint's 512-token input budget. If one side is shorter, lend the unused budget to the other. The question, answer descriptions, delimiters, and target take priority over context. Each side's candidate text is capped at 4,096 characters to bound preprocessing work.
- The original target, rather than overlapping context, determines the displayed span and its aggregation weight. The document summary is the non-whitespace-character-weighted mean of phrase scores. This is an intervention-style preference score, not MELD's top-quartile evidence statistic, an authorship probability, or a fraction of AI-written words.
- Preserve upstream `choice:2` temperature (about 1.906), but explicitly mark this task uncalibrated. Workspace thresholds are policy settings; no detection accuracy is inferred from upstream calibration claims. Very short text, non-English input, and adversarial instructions are outside a reliable validated detector. Natural-language target markers do not provide an injection guarantee.
- More than 5,000 phrases produces an explicit error rather than silently dropping text.

Independent target questions cost more than MELD's single-pass token evidence. Reusing encoder context across targets would change the bidirectional question-conditioned computation; we therefore batch independent questions rather than incorrectly sharing hidden states. A future fine-tuned span head would be a different model requiring labeled evaluation.

## Apple GPU execution

`OTD_LAYA_RUNTIME=auto` selects MLX on Apple Silicon when installed and the requested device is `auto` or `mps`; otherwise it selects the PyTorch reference. Explicit `torch` and `mlx` values are available. The defaults are `OTD_LAYA_BATCH_SIZE=4` and `OTD_LAYA_PRECISION=float16`. The portable PyTorch reference always uses FP32. Use `OTD_LAYA_PRECISION=float32` with MLX for close numerical reference agreement.

The native implementation uses compiled MLX graphs, fused layer normalization, fused rotary embedding and scaled-dot-product attention, length-sorted batches, padding in 32-token buckets, and an FP32 decision head. It preserves the encoder's distinct global/local rotary frequencies and padding masks. In the final decision layer, only answer-marker queries and feed-forward rows are computed; all context keys and values remain available. The unused escalation output is not computed.

MLX execution queues at most two batches, overlapping CPU batch preparation with GPU work while bounding in-flight activations. Results are collected in original phrase order, including the final partial batch, and checked for finite scores. The synchronous comparison path remains available through `Laya(..., pipeline_depth=0)`. PyTorch execution is unchanged.

Dense versus halo-tiled local attention, precision, marker pruning, and batch sizes are measured rather than presumed faster. The custom Metal GEGLU kernel used elsewhere in this workspace is also tested as a candidate. These are local measurements, not a guarantee of globally maximal throughput.

## Reproduce validation and benchmarks

```sh
cd backend
.venv/bin/python -m pytest tests/test_laya.py tests/test_meld.py -q
cd ..
backend/.venv/bin/python scripts/benchmark_laya.py
backend/.venv/bin/python scripts/benchmark_laya.py \
  --text-file research/extractions/hyperdas/original-reviewbench-ocr.txt \
  --output research/benchmarks/laya/m4-pro-paper.json
backend/.venv/bin/python scripts/profile_laya_kernels.py
```

The sweep validates each candidate against the PyTorch FP32 model before timing. FP32 tolerance is 0.0001; mixed FP16 tolerance is 0.005 in score, and threshold-label changes are recorded separately. GPU timing forces output evaluation and transfer, excludes model load/tokenization, and follows a warm-up. Results retain individual timings, input lengths, reference scores, configuration, and hardware. Peak allocator memory is cumulative across candidates in a process. The short synthetic fixture is a numerical/performance test, not a labeled detection benchmark. Paper benchmarks use the first 48 prepared targets; the full document is used by normal scans.

## Measured result on this M4 Pro

The real-paper sweep (`research/benchmarks/laya/m4-pro-paper.json`) used 48 targets, three warm measured repetitions, and an M4 Pro with 16 GPU cores and 48 GB RAM:

| Runtime | Batch | Phrases/s | Maximum score difference from reference |
| --- | ---: | ---: | ---: |
| PyTorch MPS, FP32 reference | 4 | 10.38 | — |
| MLX FP32, dense attention, marker pruning | 4 | 11.86 | 0.00000453 |
| MLX FP16 encoder / FP32 head, dense attention, marker pruning | 4 | **13.42** | **0.00152** |
| MLX mixed precision, tiled local attention | 8 | 12.39 | 0.00152 |

The selected paper configuration improves measured warm inference throughput by **29.3%** over the reference. No classifications crossed the default 0.2/0.8 thresholds in this sample. Differences could still change decisions near a threshold on other data. Batch 4 is the service default because it won on full paper contexts; batch 8 was slightly faster on the shorter fixture (36.15 versus 35.52 phrases/s). The short fixture is not comparable to full-length paper throughput.

The custom GEGLU candidate was rejected because its score deviation from the selected mixed-precision implementation exceeded the stricter 0.0001 activation-only equivalence tolerance. It remains available only to the profiling script through an explicit constructor option; the service uses the compiled activation. Timing results exclude model loading and preparation; end-to-end saved scans report those separately. These measurements assess execution speed and numerical agreement, **not detection accuracy**.

## MELD v8 agreement study

A subsequent study selected 35 papers: one per conference/year pair (25 pairs), plus five low-scoring older and five high-scoring newer papers. Nine prompt/context formulations, an answer-order ensemble, and calibration alternatives were compared using paper-level development/holdout separation. The generic Laya checkpoint showed weak agreement with MELD v8; score calibration alone often collapsed predictions into the middle band. See [the study report](../research/benchmarks/laya-meld-v8/REPORT.md) for metrics, sampling policy and limitations.

The experimental recipe is reproducible on new text with `scripts/classify_laya_study.py INPUT --output OUTPUT.json`. Its default document-context mode uses only Laya's own local and document-level predictions, with a small development-fitted alignment readout. It does not look up MELD scores, conference, year, or a training-paper identity at inference. The follow-up evaluations are exploratory; the service's baseline has not been promoted to this weak matching candidate.

The study runner now extracts target offsets directly before constructing its selected prompt; it no longer constructs and discards the service's original prompt. All 23,270 study inputs were checked for identical token IDs after this change. Throughput sweeps and paired full-paper measurements for this optimization are saved separately under `research/benchmarks/laya-meld-v8/optimization-*.json`; historical study predictions are not overwritten. Reproduce with `scripts/optimize_laya_study.py`, `scripts/benchmark_laya_pipeline.py`, and `scripts/benchmark_laya_full.py`. The last script alternates previous and optimized execution, includes short, older-green, and newer-red held-out papers, and measures preparation separately. It excludes checkpoint loading and PDF extraction. The default FP32 head and batch size four are retained; experimental head precision is only a benchmark constructor option.

## Published five-per-year sample

The Cloudflare atlas now includes a separate **Laya · Experimental MELD alignment** selection for 125 papers: five per each of 25 conference/year pairs. The frozen document-context recipe scored 86,803 phrases; 28 exact-input study results were reused and 97 papers were freshly inferred. The existing MELD reports and default view are preserved. Open the [Laya sample](https://open-text-detector.woog09.workers.dev/?model=laya#collection).

Selection, provenance, the public-content preservation checks, and publication receipts are retained in `research/classifications/laya-atlas-five-per-year/`. Its README documents the resumable classifier, staging, browser validation, and conditional publication scripts. The website labels this as a completed experimental sample, not a whole-corpus classification run or a prevalence estimate.
