# Frozen detector evaluation suite

One entry point for our ModernBERT classifier and MELD v5/v8. Scoring is offline, uses BF16 on CUDA, and makes **no paid API calls**. No FP32 fallback or test-set threshold fitting.

```bash
# In this workspace (only needed when assembling a new frozen bundle):
python benchmarks/pangram4/eval_suite/suite.py build
python benchmarks/pangram4/eval_suite/suite.py list

# Copy eval_suite/bundle to the A100 alongside the existing model directory.
python bundle/suite.py run --bundle bundle \
  --models-root /data/workspace/paper-v3-modernbert-20260930 \
  --profile workflow --output results
# Change --profile to comparison, assistance, full, or context.
# Optional: --models ours or meld-v5,meld-v8; --limit 8 for a smoke test.
```

`--models-root` contains `run-01/best_model` and `baseline-models/{meld-v5,meld-v8}`, each with the checkpoint, model configuration and tokenizer files. These are existing A100 artifacts; our checkpoint is not automatically downloaded or publicly distributed. Initial model installation remains necessary on another machine. The original A100 environment used torch 2.12 and transformers 5.17; each run records exact installed versions, Python, CUDA, GPU and model/tokenizer hashes. Install compatible torch, transformers, numpy, safetensors and tokenizers; identical environment records are required for cache reuse.

## Profiles and datasets

| Profile | What it contains | Use |
|---|---|---|
| `comparison` | Frozen representative 13,751-row comparison: old v3 paper targets and remaining/context human paragraphs; Arena multi-model outputs; PELIC, Liang, VUB, Perkins, MELD-eval, DetectRL, Epoch, OpAI, Sem-Detect, GEDE, Saha, local binary/length/mixed proxies, exploratory paper pilots | Reproduce the previous three-model comparison at its saved operating points; development-exposed |
| `workflow` | 8,552 rows: 864 reconstruction views, 216 matched originals, 4,901 other human paragraphs, 2,571 ELLIPSE essays | Fresh paper workflow evaluation; inspect each condition/view and clean human controls separately |
| `assistance` | 972 calibration/test views of proofreading, light polish, substantial rewriting | Diagnostic scores; assisted spans have **no binary authorship gold**; separate calibration/test reporting |
| `full` | Original 65,348-row broad diagnostic collection | Larger sensitivity analysis; overlaps `comparison`, not additional independent evidence |
| `context` | Original contextual human-paper control suite | Human false alarms with surrounding context |

Exact dataset counts, selected row IDs, text, labels and SHA-256 hashes are frozen in the bundle. Validation files are included separately for audit; the runner never fits thresholds from them. See the source workspace's `benchmarks/pangram4/COVERAGE.md` for publisher URLs, selection details and missing Pangram benchmarks. These are our subsets/proxies, not Pangram's exact private evaluation sets.

## Labels and interpretation

Regions use Python character offsets `[start,end)`: `0` human, `1` known generated wording, `-100` unknown/assisted. Native document labels are not invented token labels. Human/AI boundary tokens and ambiguous sentences are masked. All models share the ModernBERT reference-token grid; MELD uses overlap-weighted projection of raw evidence. Sentence segmentation is the frozen historical regex, not expert linguistic annotation. Thresholds are the saved validation-calibrated comparison-v1 thresholds, including a separate document threshold.

Results include token/sentence precision, recall, human FPR, document metrics, per-dataset breakdowns and paper/group-cluster bootstrap intervals. Mixed passages do not enter pure document accuracy. Never pool conditions/views as independent papers, or pool full/comparison profiles. Published baseline training contamination is unknown. Assistance score rates measure flagging, not correctness; context-only human metrics do not label assisted wording as human.

## Reproducibility

`validate` checks all frozen code/data hashes, row IDs, text hashes and region validity. Each model run locks checkpoint/tokenizer contents, data, code, thresholds, environment and BF16 settings; incompatible resume attempts fail. Scores are resumable by deterministic text chunks. Exact saved-data evaluation is repeatable in the same environment, subject to normal GPU numerical variability. Regenerating Luna text is **not** bitwise reproducible; archived outputs, selection and prompt protocols are the benchmark. No detector scores selected the fresh workflow test papers.

The bundle is self-contained for scoring **once model artifacts and dependencies are installed**. It includes third-party text for local use; do not republish it wholesale. HF publication separates our paper workflow data (component-specific paper/output terms) from a text-free inventory of third-party benchmark selections. ELLIPSE remains under its upstream CC-BY-NC-SA-4.0 terms. Raw PDFs, credentials, request traces and private model weights are excluded from publication.
