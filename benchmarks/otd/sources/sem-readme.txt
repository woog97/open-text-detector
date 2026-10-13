# Sem-Detect: Semantic Level Detection of AI Generated Peer-Reviews

Official code for the paper *"Sem-Detect: Semantic Level Detection of AI Generated Peer-Reviews"* (ICML 2026).

**Quick links:**  
[Paper 📄](https://arxiv.org/abs/2605.21713) | [Dataset 🤗](https://huggingface.co/datasets/Sem-Detect/ML_Conferences-Peer-Reviews) | Blog Post (soon) 


Sem-Detect is an authorship attribution method that classifies peer reviews into three categories:
- **Human-written** — original human reviews
- **LLM-refined** — human reviews polished by an LLM
- **AI-generated** — fully AI-written reviews

The key insight is that different AI models converge on similar claims when reviewing the same paper, while human reviewers produce more diverse judgments. Sem-Detect combines semantic features (claim-level similarity analysis) with textual features (perplexity, entropy) to achieve robust three-class detection.


## Installation

```bash
git clone https://github.com/avduarte333/Sem-Detect.git
cd sem-detect
pip install -r requirements.txt
```

Set up your API credentials:
```bash
# Copy .env.example to .env and fill in your API keys.
cp .env.example .env
```

## Self-Hosted Web Demo

A Flask-based web interface is available in the `demo/` directory for interactive review analysis.

```bash
python demo/app.py
```

The demo includes a browser-based UI for linking an OpenReview account, crawling a paper and its reviews, running the full detection pipeline, and visualizing the results with SHAP explanations. Precomputed results for ICLR 2026 are also included in `demo/static/precomputed/`.



## Dataset

The full dataset is available on 
[HuggingFace 🤗](https://huggingface.co/datasets/Sem-Detect/ML_Conferences-Peer-Reviews)<br>
If you want to use the same JSON format we considered for our experiments, run this additional script upon downloading:
```bash
python scripts/hf-to-json.py
```


## Quick Start

Sem-Detect generates AI reference reviews for the same paper and compares them semantically against the target review. To do this, LLMs need to read the paper. There are two modes:

| Mode | How it works | Providers |
|------|-------------|-----------|
| **Native PDF** (`--native-pdf`) | Sends the original PDF directly to LLMs | Gemini, OpenAI, Claude on Bedrock |
| **Parsed text** (default) | Pre-parses PDF to text via [Marker](https://github.com/datalab-to/marker) | All providers |

Native PDF is simpler (fewer steps), but not all providers support it. In particular, **DeepSeek** and **Qwen on Bedrock** and **LiteLLM** do not. If your reference models include those, use parsed text mode so all models can read the paper.

### Option A: From an OpenReview Paper

`fetch_openreview.py` downloads the blind submission PDF, parses it with Marker, and extracts the reviews into a ready-to-use JSON:

```bash
python scripts/fetch_openreview.py \
    --url "https://openreview.net/forum?id={forum_id}" \
    --config configs/config_example.yaml \
    --output output/my_paper \
    --cutoff-date "mm/dd/yyyy"
```

Then run detection:

```bash
python scripts/detect.py \
    --paper-json output/my_paper/paper_data.json \
    --references gemini:gemini-2.5-pro bedrock:deepseek.v3-v1:0 bedrock:qwen.qwen3-235b-a22b-2507-v1:0 \
    --cleaning-provider gemini:gemini-2.5-flash \
    --config configs/config_example.yaml \
    --output-dir output/my_paper
```

You can also add `--native-pdf output/my_paper/paper.pdf` to `detect.py` so that providers that support it (Gemini, OpenAI, Claude) receive the original PDF rather than parsed text. The fetch step downloads `paper.pdf` to the output directory regardless.

> **About `--cutoff-date`:** Recommended for conferences where authors can update blind submissions during rebuttal (e.g. ICLR). Specifying the deadline ensures we download the same version that reviewers saw. See `scripts/fetch_openreview.py` for built-in cutoff dates for a few conferences.

### Option B: From a Local PDF

If your paper is not on OpenReview, fill in `data/paper_data_template.json` with the review(s) and rating, then choose a mode:

**Native PDF** (recommended if your reference models support it):

```bash
python scripts/detect.py \
    --paper-json data/paper_data_template.json \
    --native-pdf my_paper.pdf \
    --references gemini:gemini-2.5-pro gemini:gemini-2.5-flash \
    --cleaning-provider gemini:gemini-2.5-flash \
    --config configs/config_example.yaml \
    --output-dir results/manual-run
```

No need to fill `parsed_pdf_content` in the template — LLMs read the PDF directly.

**Parsed text** (needed if using DeepSeek/Qwen on Bedrock or LiteLLM):

```bash
# Parse the PDF first
python scripts/parse_pdf.py my_paper.pdf --output-dir output/my_paper

# Copy parsed_pdf_content from the output into paper_data_template.json, then:
python scripts/detect.py \
    --paper-json data/paper_data_template.json \
    --references gemini:gemini-2.5-pro bedrock:deepseek.v3-v1:0 bedrock:qwen.qwen3-235b-a22b-2507-v1:0 \
    --cleaning-provider gemini:gemini-2.5-flash \
    --config configs/config_example.yaml \
    --output-dir results/manual-run
```

### What the Pipeline Does

1. Generate 3 AI reference reviews (one per model, score-matched)
2. Clean all reviews from LLM-text artifacts (e.g. *"Sure, here is your Review..."*)
3. Extract structured claims from all reviews
4. Embed claims using Qwen3-Embedding-0.6B
5. Compute textual features using a reference LM
6. Classify using the pre-trained LightGBM model

The `--output-dir` saves all intermediate files plus a detailed analysis report (`report.txt`) showing per-claim similarity breakdowns.

### Pre-trained Models

Two pre-trained classifiers are provided, each paired with a different reference LM for textual features:

| Classifier | Textual Features Model | Notes |
|------------|----------------------|-------|
| `models/lightgbm_best_model.pkl` | Mistral-7B-Instruct-v0.3 | Recommended (higher accuracy) |
| `models/lightgbm_best_model_qwen3_1.7b.pkl` | Qwen3-1.7B | Faster, lower GPU requirements |

The classifier and textual features model must match — each classifier was trained on features computed by its paired reference LM. `detect.py` handles this automatically: passing `--model-path` selects the correct textual features model, so no extra flags are needed:

```bash
# Single paper with the Qwen3-1.7B variant (textual model auto-selected)
python scripts/detect.py \
    --paper-json output/my_paper/paper_data.json \
    --references gemini:gemini-2.5-pro bedrock:deepseek.v3-v1:0 bedrock:qwen.qwen3-235b-a22b-2507-v1:0 \
    --cleaning-provider gemini:gemini-2.5-flash \
    --config configs/config_example.yaml \
    --model-path models/lightgbm_best_model_qwen3_1.7b.pkl \
    --output-dir output/my_paper
```

For the step-by-step pipeline (`compute_textual_features.py`), you must pass `--model` explicitly to match your chosen classifier:
```bash
python scripts/compute_textual_features.py input.pkl output.pkl --model Qwen/Qwen3-1.7B
```



## Step-by-Step Pipeline on Multiple Papers

You can also run Sem-Detect step-by-step independently on a set of multiple papers (see `data/Multiple-Papers-Example/` for an example).<br>
Each step reads the previous output and adds its data. The final `.pkl` has everything needed for evaluation.

```bash
# 1. Generate AI reviews and rewrites (supports multiple providers in one run)
python scripts/generate_reviews.py data/Multiple-Papers-Example/00-papers.json data/Multiple-Papers-Example/01-with_reviews.json \
    --providers gemini:gemini-2.5-flash gemini:gemini-2.5-pro bedrock:deepseek.v3-v1:0 bedrock:qwen.qwen3-235b-a22b-2507-v1:0 \
    --type both --clean
```
```bash
# 2. Extract claims from all reviews
python scripts/extract_claims.py data/Multiple-Papers-Example/01-with_reviews.json data/Multiple-Papers-Example/02-with_claims.json \
    --provider gemini:gemini-2.5-flash
```
```bash
# 3. Generate claim embeddings (GPU recommended)
python scripts/generate_embeddings.py data/Multiple-Papers-Example/02-with_claims.json data/Multiple-Papers-Example/03-with_embeddings.pkl
```
```bash
# 4. Compute textual features (GPU required) — adds textual features to each review
#    Defaults to Mistral-7B (matches lightgbm_best_model.pkl).
#    For the Qwen3 classifier, pass: --model Qwen/Qwen3-1.7B
python scripts/compute_textual_features.py data/Multiple-Papers-Example/03-with_embeddings.pkl data/Multiple-Papers-Example/04-ready.pkl
```
```bash
# 5. 3-class predictions (AI / LLM-Refined / Human)
#    The --model-path must match the textual features model used in step 4
#    (see "Pre-trained Models" section above).
python scripts/evaluate.py \
    --model-path models/lightgbm_best_model.pkl \
    --test-data data/Multiple-Papers-Example/04-ready.pkl \
    --targets human gemini-2.5-pro=ai,rewrite gemini-2.5-flash=ai,rewrite deepseek.v3-v1:0=ai,rewrite qwen.qwen3-235b-a22b-2507-v1:0=ai,rewrite \
    --sources gemini-2.5-pro gemini-2.5-flash deepseek.v3-v1:0 qwen.qwen3-235b-a22b-2507-v1:0 \
    --output-dir results/my_run
```

Running `generate_reviews.py` multiple times is safe — it reads from the output file if it exists and skips reviews that were already generated (dedup by model + score for AI, by model + original review for rewrites).<br>
This step-by-step sequential pipeline is basically how we created our training and test data. However, if you're planning on doing high-volume workloads, consider asynchronous/batch processing (e.g., [Gemini Batch API](https://ai.google.dev/gemini-api/docs/batch-api)) to maximize throughput and minimize cost.



## Project Structure

```
sem-detect/
├── semdetect/                      # Core library
│   ├── pipeline.py                    # End-to-end orchestrator
│   ├── review_generator.py            # Generate AI reviews + rewrites
│   ├── review_cleaner.py              # Clean raw reviews (LLM + markdown stripping)
│   ├── claim_extractor.py             # Extract structured claims (5 categories)
│   ├── embedding_generator.py         # Claim embeddings (Qwen3-0.6B)
│   ├── textual_features.py            # Perplexity, entropy, fast-detect, top-k
│   ├── feature_builder.py             # Build 9-feature vector
│   ├── classifier.py                  # LightGBM inference
│   ├── report.py                      # Analysis report generation
│   ├── config.py                      # Config loader + provider factory
│   └── providers/                     # LLM backends
│       ├── base.py                       # Abstract interface
│       ├── gemini.py                     # Google Gemini (sync)
│       ├── bedrock.py                    # AWS Bedrock (sync)
│       ├── openai_provider.py            # OpenAI (sync)
│       └── litellm.py                    # LiteLLM proxy (100+ providers)
├── scripts/                        # CLI tools
│   ├── fetch_openreview.py            # Fetch paper + reviews from OpenReview
│   ├── parse_pdf.py                   # PDF text extraction + anonymity checking
│   ├── detect.py                      # Classify reviews (single paper)
│   ├── generate_reviews.py            # Generate AI reviews + rewrites for a dataset
│   ├── extract_claims.py              # Extract claims for a dataset
│   ├── generate_embeddings.py         # Generate embeddings for a dataset
│   ├── compute_textual_features.py    # Compute textual features for a dataset
│   ├── evaluate.py                    # 3-class evaluation on a dataset
│   ├── hf-to-json.py                  # Convert Sem-Detect HF Dataset to JSON format
│   └── precompute_demo_results.py     # Pre-compute results for the web demo
├── demo/                           # Flask web demo
│   ├── app.py                         # Web application
│   ├── templates/                     # HTML templates
│   └── static/                        # JS, CSS, pre-computed results
├── models/                         # Pre-trained LightGBM classifiers
├── configs/                        # Configuration templates
└── data/                           # Data templates + multi-paper example
    ├── paper_data_template.json       # Template for manual input
    └── Multiple-Papers-Example/       # Step-by-step pipeline example data
```

## Supported LLM Providers

| Provider | Models | Config section |
|----------|--------|----------------|
| Google Gemini | gemini-2.5-flash, gemini-2.5-pro | `gemini` |
| AWS Bedrock | DeepSeek-V3.1, Qwen3-235B-A22B | `bedrock` |
| OpenAI | GPT-5.4, etc. | `openai` |
| LiteLLM | Any model via proxy gateway | `litellm` |


## Citation

```bibtex
@misc{duarte2026semdetectsemanticleveldetection,
      title={{Sem-Detect: Semantic Level Detection of AI Generated Peer-Reviews}}, 
      author={André V. Duarte and Brian Tufts and Aditya Oke and Fei Fang and Arlindo L. Oliveira and Lei Li},
      year={2026},
      eprint={2605.21713},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.21713}, 
}
```
