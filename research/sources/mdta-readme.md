---
license: cc-by-sa-4.0
language:
  - en
task_categories:
  - text-classification
  - text-generation
tags:
  - ai-generated-text
  - machine-generated-text-detection
  - adversarial
  - llm-generated-text-detection
  - benchmark
pretty_name: MDTA
size_categories:
  - 10K<n<100K
source_datasets:
  - Hello-SimpleAI/HC3
configs:
  - config_name: finance
    data_files: finance.jsonl
  - config_name: medicine
    data_files: medicine.jsonl
  - config_name: open_qa
    data_files: open_qa.jsonl
  - config_name: reddit_eli5
    data_files: reddit_eli5.jsonl
  - config_name: wiki_csai
    data_files: wiki_csai.jsonl
---

# MDTA: Models, Domains, Temperatures, and Adversaries

MDTA is a benchmark for AI-generated text detection. It pairs human-written and LLM-generated answers across five domains, four open-weights models, and three sampling temperatures, and augments each LLM response with three adversarial paraphrases (including constrained letter-avoidance rewrites).

The dataset spans **24,322 prompt-aligned questions** (around **642,000 text samples** when counting all generated and adversarial responses). Questions and human answers are sourced from the [HC3 dataset](https://huggingface.co/datasets/Hello-SimpleAI/HC3); MDTA extends HC3 with modern model coverage, temperature variation, and targeted adversarial augmentation.

## Domains (configs)

| Config | Source | Questions |
|---|---|---|
| finance | finance Q&A | 3,933 |
| medicine | medical Q&A | 1,248 |
| open_qa | open-domain Q&A | 1,187 |
| reddit_eli5 | Reddit ELI5 | 17,112 |
| wiki_csai | Wikipedia (CS / AI) | 842 |

Total: 24,322 questions. MDTA does not ship a fixed train/test split. Evaluation is governed by the seed-based protocol described under [Benchmark protocol](#benchmark-protocol).

## Loading

```python
from datasets import load_dataset

finance = load_dataset("nsp909/MDTA", "finance", split="train")
print(finance[0])
```

Or load a single file directly:

```python
ds = load_dataset("json", data_files="reddit_eli5.jsonl")
```

(Hugging Face exposes a single split called `train` containing all rows. The benchmark splits are constructed at runtime from random seeds, not from file names.)

## Row schema

Each line in a `*.jsonl` file is one question with all associated answers:

```json
{
  "question_index": 0,
  "question": "...",
  "human_answers": ["..."],
  "model_responses": {
    "llama-3.1-8b":  {"temp_0.2": "...", "temp_0.5": "...", "temp_0.8": "..."},
    "gemma-3-12b":   {"temp_0.2": "...", "temp_0.5": "...", "temp_0.8": "..."},
    "qwen2.5-vl-7b": {"temp_0.2": "...", "temp_0.5": "...", "temp_0.8": "..."},
    "ministral-8b":  {"temp_0.2": "...", "temp_0.5": "...", "temp_0.8": "..."}
  },
  "adv_responses": {
    "llama-3.1-8b":  {"adv_letter_x": "Q", "adv_letters_xy": ["Q", "L"], "adv_plain": "...", "adv_no_letter_x": "...", "adv_no_letters_xy": "..."},
    "gemma-3-12b":   {...},
    "qwen2.5-vl-7b": {...},
    "ministral-8b":  {...}
  }
}
```

### Field reference

- `question_index` (int): stable id within a domain.
- `question` (string): the prompt.
- `human_answers` (list of strings): one or more reference human-written answers.
- `model_responses` (dict): standard LLM responses, keyed by model. Each model maps to three sampling temperatures (`temp_0.2`, `temp_0.5`, `temp_0.8`).
- `adv_responses` (dict): adversarially rewritten LLM responses, keyed by model. For each model:
  - `adv_letter_x` (string): the chosen target letter to avoid.
  - `adv_letters_xy` (list of two strings): the two target letters to avoid.
  - `adv_plain` (string): a plain paraphrase baseline.
  - `adv_no_letter_x` (string): rewrite that avoids `adv_letter_x`.
  - `adv_no_letters_xy` (string): rewrite that avoids both `adv_letters_xy` letters.

## Models covered

- `llama-3.1-8b`
- `gemma-3-12b`
- `qwen2.5-vl-7b`
- `ministral-8b`

## Benchmark protocol

MDTA is benchmarked under a small-calibration / large-evaluation protocol that matches how training-free detectors are deployed in practice. Standard 80/10/10 supervised splits would saturate detector AUROC on this dataset and lose discrimination, so they are not used.

### Sample construction

Per domain, expand each row into a list of (text, label) samples:

- For each entry in `human_answers`: one sample with `label = 0` (human).
- For each `(model, temp)` in `model_responses`: one sample with `label = 1` (AI).
- For each `(model, variant)` in `adv_responses` where `variant ∈ {adv_plain, adv_no_letter_x, adv_no_letters_xy}`: one sample with `label = 1` (AI), used for the adversarial track.

### Per-seed train / test split

Pick **5 fixed random seeds** and use them across the entire evaluation. For each seed and each train size `T ∈ {100, 250, 500, 1000}`:

1. Shuffle human samples and AI samples independently with the seed.
2. Take the first `T/2` of each class as the **train** (calibration) set, balanced.
3. The **test set is everything else**: all remaining human and AI samples in the domain. This is the "small calibration, large evaluation" regime standard for training-free detectors.

Report mean and standard deviation over the 5 seeds. The choice of seed values is at the submitter's discretion (any fixed list works, the variance is small either way), but they must be fixed and disclosed for the run to be reproducible.

### Two tracks

**Track 1 - Standard detection.** AI samples come from `model_responses` (clean LLM output across 4 models × 3 temperatures). The test set typically contains tens of thousands of samples, so AUROC is statistically tight.

**Track 2 - Adversarial robustness.** AI samples come from `adv_responses` instead. Each of the three adversarial conditions (`adv_plain`, `adv_no_letter_x`, `adv_no_letters_xy`) is evaluated separately. For the two letter-avoidance conditions, restrict to rows where the model produced a non-error response (`source_valid == 1` in the precomputed score CSVs).

### Metrics

AUROC and F1, mean and standard deviation over the seeds. TPR at 1% FPR is recommended as a secondary metric for high-stakes deployment.

### Reference implementation

The full reference implementation, including the detector scoring pipeline (DNA, Binoculars, FastDetectGPT, LD-Score) and the seed/train/test contract, accompanies the companion paper. The pseudocode is straightforward enough to reproduce from this section: shuffle by seed, slice the first `T/2` of each class as train, evaluate the rest.

## Baselines

The numbers below are taken from the companion paper, *Beyond Perplexity: Character Distribution Signatures and the MDTA Benchmark for AI Text Detection* (Tables 2, 3, 9). Protocol: train on `T = 100` balanced samples (50 human + 50 AI), evaluate, mean ± std over 5 runs. Reference models: Falcon-7b-Instruct + Falcon-7b for DNA-DetectLLM and Binoculars; GPT-J-6B + GPT-Neo-2.7B for FastDetectGPT.

**Method naming.** Three families are reported:

- **Training-free baselines.** `DNA-DetectLLM` (DNA), `Binoculars` (Bino), and `FastDetectGPT` (FDGPT) score text using surrogate-LM log-probabilities. They are threshold-calibrated on the training set.
- **LD-Score augmented (paper contribution).** `LD-DNA`, `LD-Bino`, and `LD-FDGPT` augment each baseline with the **Letter Distribution Score (LD-Score)**, a character-level Jensen-Shannon divergence between the input text and a reference distribution computed from the candidate models. The two scores are stacked into a 2-D feature vector and classified with an RBF-SVM.
- **Perplexity ensembles.** `DNA+Bino` and `DNA+FDGPT` stack the two baseline scores and classify with an RBF-SVM, but without the LD-Score signal. These isolate the gain that comes specifically from the orthogonal letter-distribution feature.

### Track 1 - Standard detection (AUROC, T=100)

| Method | Finance | Medicine | Open QA | Reddit ELI5 | Wiki CSAI | Avg |
|---|---|---|---|---|---|---|
| DNA | 0.958 | 0.996 | 0.812 | 0.971 | 0.992 | 0.946 |
| Bino | 0.946 | 0.995 | 0.817 | 0.966 | 0.990 | 0.943 |
| FDGPT | 0.871 | 0.954 | 0.774 | 0.930 | 0.963 | 0.898 |
| **LD-DNA** | **0.974** | **0.998** | 0.856 | **0.982** | **0.992** | **0.960** |
| LD-Bino | 0.970 | 0.997 | **0.859** | 0.978 | 0.990 | 0.959 |
| LD-FDGPT | 0.888 | 0.961 | 0.808 | 0.934 | 0.956 | 0.909 |
| DNA+Bino | 0.974 | 0.994 | 0.832 | 0.958 | 0.989 | 0.949 |
| DNA+FDGPT | 0.971 | 0.996 | 0.826 | 0.965 | 0.990 | 0.950 |

### Track 1 - Standard detection (F1, T=100)

| Method | Finance | Medicine | Open QA | Reddit ELI5 | Wiki CSAI | Avg |
|---|---|---|---|---|---|---|
| DNA | 0.933 | 0.976 | 0.756 | 0.913 | 0.958 | 0.907 |
| Bino | 0.905 | 0.977 | 0.752 | 0.895 | 0.949 | 0.896 |
| FDGPT | 0.751 | 0.879 | 0.701 | 0.832 | 0.896 | 0.812 |
| **LD-DNA** | 0.929 | **0.982** | **0.788** | **0.949** | 0.955 | **0.921** |
| LD-Bino | 0.918 | 0.976 | 0.784 | 0.936 | 0.956 | 0.914 |
| LD-FDGPT | 0.770 | 0.907 | 0.720 | 0.856 | 0.885 | 0.828 |
| DNA+Bino | 0.927 | 0.979 | 0.764 | 0.911 | 0.958 | 0.908 |
| DNA+FDGPT | 0.925 | 0.978 | 0.757 | 0.914 | 0.957 | 0.906 |

LD-Score augmentation gives the largest absolute gains on domains with specialized vocabulary (Finance, Reddit ELI5, Open QA), where letter-distribution divergence is most pronounced. Medicine and Wiki CSAI show smaller deltas because the perplexity baselines are already near-saturated.

### Unbalanced regime (AUROC, averaged over domains)

When humans are scarce in the calibration sample (natural-ratio sampling rather than balanced), threshold methods improve only gradually with more training data, while LD-augmented variants converge rapidly from the outset.

| Train T | DNA | LD-DNA | Δ | Bino | LD-Bino | Δ |
|---|---|---|---|---|---|---|
| 100  | 0.847 | **0.935** | +0.088 | 0.796 | **0.918** | +0.122 |
| 250  | 0.863 | **0.949** | +0.086 | 0.823 | **0.945** | +0.122 |
| 500  | 0.864 | **0.945** | +0.081 | 0.812 | **0.942** | +0.130 |
| 1000 | 0.886 | **0.945** | +0.059 | 0.801 | **0.935** | +0.134 |

### Track 2 - Adversarial robustness (AUROC, T=100)

For each adversarial condition, AI samples are drawn from `adv_responses[<model>][<condition>]` instead of `model_responses`. The two letter-avoidance conditions are filtered to `source_valid == 1` (rows where the model produced a non-error rewrite).

**Condition (A): `adv_plain` (paraphrase)**

| Method | Finance | Medicine | Open QA | Reddit ELI5 | Wiki CSAI | Avg |
|---|---|---|---|---|---|---|
| DNA | 0.975 | 0.995 | 0.834 | 0.938 | 0.982 | 0.945 |
| Bino | 0.968 | 0.994 | 0.808 | 0.922 | 0.979 | 0.934 |
| **LD-DNA** | **0.979** | **0.996** | **0.845** | **0.948** | **0.984** | **0.950** |
| LD-Bino | 0.978 | 0.995 | 0.828 | 0.937 | 0.983 | 0.944 |

**Condition (B): `adv_no_letter_x` (single-letter avoidance)**

| Method | Finance | Medicine | Open QA | Reddit ELI5 | Wiki CSAI | Avg |
|---|---|---|---|---|---|---|
| DNA | 0.946 | **0.980** | 0.756 | 0.877 | 0.964 | 0.905 |
| Bino | 0.916 | 0.964 | 0.719 | 0.828 | 0.943 | 0.874 |
| **LD-DNA** | **0.961** | 0.979 | **0.761** | **0.886** | **0.968** | **0.911** |
| LD-Bino | 0.938 | 0.962 | 0.727 | 0.857 | 0.954 | 0.888 |

**Condition (C): `adv_no_letters_xy` (two-letter avoidance, hardest)**

| Method | Finance | Medicine | Open QA | Reddit ELI5 | Wiki CSAI | Avg |
|---|---|---|---|---|---|---|
| DNA | **0.938** | 0.959 | 0.712 | 0.853 | 0.944 | 0.881 |
| Bino | 0.898 | 0.943 | 0.674 | 0.798 | 0.914 | 0.845 |
| **LD-DNA** | 0.936 | **0.961** | **0.727** | **0.863** | **0.946** | **0.887** |
| LD-Bino | 0.902 | 0.946 | 0.691 | 0.819 | 0.928 | 0.857 |

**Track 2 summary.** All methods lose ground from Track 1 to Track 2. The drop is largest under condition (C), which forces the most surface-level character-distribution shift. LD-Score augmentation is computed against clean (non-adversarial) reference distributions and is not optimized for these attacks; it nonetheless yields consistent lift across all three conditions, with the largest gains under (B). DNA's perplexity signal already captures most of the distinguishing information under attack, so LD-Score offers smaller marginal lift on DNA. Closing the (C) robustness gap is the hardest open problem on this benchmark.

Letter-avoidance constraints consistently degrade all detectors. The drop is largest on the two-letter condition, where surface character statistics shift most sharply but log-likelihood gaps narrow. Note that LD-Score is computed on clean (non-adversarial) reference distributions and is not optimized against these attacks; the small remaining gains under adversarial conditions therefore reflect transfer rather than direct training. Closing the dual-letter robustness gap is the hardest open problem on this benchmark.

## Intended use

- Training and evaluating AI-generated text detectors.
- Studying robustness of detectors to adversarial paraphrasing, including constrained-vocabulary rewrites (avoiding specific letters).
- Comparing model output distributions across domains and sampling temperatures.

## Out-of-scope use

This dataset is for research. It is not a benchmark of factual correctness: model responses are not fact-checked, and human answers are taken from their original sources without verification.

## Data collection

**Source corpus.** Questions and `human_answers` are taken directly from the [HC3 dataset](https://huggingface.co/datasets/Hello-SimpleAI/HC3). The five MDTA configs (`finance`, `medicine`, `open_qa`, `reddit_eli5`, `wiki_csai`) correspond one-to-one with HC3's five domain subsets. ChatGPT answers from HC3 are not redistributed in MDTA. Average human response length ranges from about 187 to about 1,302 tokens depending on domain.

**Model responses.** For each question we generated answers using four mid-sized open-weights LLMs:

- `llama-3.1-8b`
- `gemma-3-12b`
- `qwen2.5-vl-7b`
- `ministral-8b`

Each model was sampled at three temperatures: 0.2 (deterministic), 0.5 (balanced), and 0.8 (stochastic). This temperature stratification is intentional, since sampling temperature directly modulates vocabulary use and probability distributions.

The chat template used a fixed system message followed by the question as the user turn:

```
system: "You are a helpful assistant."
user:   <question from the dataset>
```

**Adversarial responses.** For each clean model output, the originating model was prompted to produce three adversarial variants. The letters `x` and `y` are randomly sampled per question (with `x ≠ y`) and recorded in the row as `adv_letter_x` and `adv_letters_xy`:

| Variant | Prompt template |
|---|---|
| `adv_plain` | `Paraphrase the following text: {source_text}` |
| `adv_no_letter_x` | `Paraphrase the following text without using the letter {x}: {source_text}` |
| `adv_no_letters_xy` | `Paraphrase the following text without using the letters {x} or {y}: {source_text}` |

Here `{source_text}` is the corresponding clean `model_responses[<model>][<temp>]` output. No system message is used for adversarial generation. The letter-avoidance (lipogrammatic) constraints are designed to stress-test character-level detection signals by forcing measurable shifts in character-level statistics.

**Compliance.** Models do not always follow the letter-avoidance instruction. Some `adv_no_letter_x` and `adv_no_letters_xy` outputs still contain the forbidden letters, and a small number contain explicit refusals or generation errors. Compliance varies by model, by domain, and by the specific letter chosen (rare letters like `q` or `z` are easier to avoid than common ones like `e` or `t`). The data is released as-is so that detector evaluations reflect realistic adversarial conditions rather than a curated subset. Users who need strictly compliant adversarial samples should filter rows themselves using a substring check on `adv_letter_x` / `adv_letters_xy`.

## Limitations and biases

- Domain coverage is limited to the five configs listed above.
- Human reference answers reflect the biases of their source corpora.
- Adversarial rewrites may degrade fluency or factuality compared to the original model output.
- Models do not always comply with letter-avoidance constraints. A non-trivial fraction of `adv_no_letter_x` and `adv_no_letters_xy` outputs still contain the forbidden letter(s); consumers needing strict compliance should filter on `adv_letter_x` / `adv_letters_xy` themselves.

## License

Released under **CC-BY-SA-4.0**, matching the license of the upstream [HC3 dataset](https://huggingface.co/datasets/Hello-SimpleAI/HC3) from which questions and human answers are derived. Any redistribution or derivative work must remain under the same license and provide attribution to both MDTA and HC3.

Any code released alongside this dataset is provided under the MIT license.

## Attribution

MDTA builds on the HC3 dataset:

> Guo, Biyang, et al. *How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection.* arXiv:2301.07597, 2023. [https://huggingface.co/datasets/Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)

Baseline detectors used in the benchmark numbers above:

> Hans, Abhimanyu, et al. *Spotting LLMs with Binoculars: Zero-shot Detection of Machine-Generated Text.* arXiv:2401.12070, 2024.

> Zhu, et al. *DNA-DetectLLM.* 2025.

> Bao, Guangsheng, et al. *Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature.* arXiv:2310.05130, 2024.

## Citation

If you use this dataset, please cite:

```bibtex
@misc{narayanasamy2026perplexitycharacterdistributionsignatures,
  title={Beyond Perplexity: Character Distribution Signatures and the MDTA Benchmark for AI Text Detection},
  author={Narayanasamy, Priyadarshan and Agrawal, Swastik and Faber, Klint and Alam, Fardina Fathmiul},
  year={2026},
  eprint={2605.01647},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2605.01647}
}
```