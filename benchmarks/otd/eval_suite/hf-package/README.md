---
pretty_name: AI Paper Workflow Evaluation
language: [en]
license: other
license_name: mixed-paper-and-model-output-terms
license_link: LICENSE.md
task_categories: [token-classification, text-classification]
configs:
- config_name: reconstruction
  default: true
  data_files:
  - split: test
    path: data/reconstruction/test.parquet
  - split: validation
    path: data/reconstruction/validation.parquet
- config_name: human_controls
  data_files:
  - split: test
    path: data/human_controls/test.parquet
  - split: validation
    path: data/human_controls/validation.parquet
- config_name: assistance
  data_files:
  - split: validation
    path: data/assistance/validation.parquet
  - split: test
    path: data/assistance/test.parquet
- config_name: index
  data_files:
  - split: test
    path: data/index/test.parquet
---
# AI Paper Workflow Evaluation

Frozen GPT-6 Luna Flex outputs and historical research-paper controls. **Evaluation only: keep test out of training, prompt development and threshold selection.** 162 main papers: 54 validation and 108 test, stratified across NeurIPS, ICML and ACL, 2013–2021. The 27 development pilot papers are excluded. Author-name connected components do not cross splits; this is not perfect author identity resolution.

```python
from datasets import load_dataset
# Pin revision to the upload commit recorded by your experiment.
ds = load_dataset("woog/ai-paper-workflow-eval", "reconstruction", revision=REVISION)
```

| Config | Validation / test rows | Purpose |
|---|---|---|
| reconstruction | 432 / 864 | One/two sentences, v3 paragraph, concise paragraph; target hidden from writer |
| human_controls | 554 / 5,117 | Matched originals and other human body paragraphs; use clean-novel flags for primary FPR |
| assistance | 324 / 648 | Proofread, light polish, substantial rewrite; diagnostic, no binary target gold |
| index | See suite manifest | Text-free frozen row selections for every local profile; **not a test split to score or train on** |

Paragraph and contextual views are correlated, not additional generations. Main reconstruction counts are 216 validation + 432 test generated targets; assistance counts are 162 + 324. Match originals via `family_id`, cluster by paper (author-component sensitivity), and report condition/view separately. Human controls deliberately retain extraction diagnostics; validation remaining controls are restricted to clean-novel prose.

`regions` are character `[start,end)` spans: 0 human, 1 generated replacement, -100 unknown/assisted. These are provenance labels, not measures of correctness. `paper_id` is stable; `forum_id` is null when not sourced from OpenReview. Quality reviews are by the same Luna model, not human gold; original verdicts and evidence-format corrections are retained in `metadata_json`. All first valid writer outputs remain, including quality failures and sentence-count mismatches. No detector scores selected this test set.

The `index` config inventories the older paper-v3 comparison, Arena, PELIC, Liang, VUB, Perkins, MELD-eval, DetectRL, Epoch, OpAI, Sem-Detect, GEDE, Saha, ELLIPSE and local diagnostic proxies. Those third-party texts are **not mirrored here**. ELLIPSE is upstream CC-BY-NC-SA-4.0. See SOURCES.md for acquisition choices. Existing public generated corpora: [v3 paper pairs](https://huggingface.co/datasets/woog/ai-paper-provenance-v3), [Arena](https://huggingface.co/datasets/woog/arena-prose-100-49-models).

Local frozen bundle + checkpoint artifacts run offline through `runner/suite.py` (see runner/README.md). This HF dataset alone does not hydrate every third-party profile or download our checkpoint. Code/data/model hashes, fixed comparison-v1 thresholds and BF16 inference are recorded; changed inputs invalidate score caches. Same-environment rescoring is reproducible subject to GPU numerics; new Luna generation is stochastic. These are our benchmarks, not Pangram 4's exact private cohorts. See GENERATION.md and GENERATION_RESULTS.md for generation design and quality results.
