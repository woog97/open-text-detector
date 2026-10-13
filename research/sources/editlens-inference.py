"""
Run EditLens inference on a HuggingFace dataset.

Adds two columns to the dataset:
  - editlens_{base_model}_bucket:   predicted bucket (int), 0 = human, n_buckets-1 = AI
  - editlens_{base_model}_score:    continuous score (float), weighted sum of bucket probabilities

Usage:
  python inference.py \
    --checkpoint pangram/editlens_roberta-large \
    --base_model FacebookAI/roberta-large \
    --dataset pangram/editlens_iclr \
    --split test \
    --text_col text \
    --output predictions.jsonl
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from huggingface_hub import hf_hub_download, repo_exists
from safetensors import safe_open
from scipy.special import softmax
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from preprocess import clean_text, count_words
from train import NormedLinear


def is_qlora_checkpoint(checkpoint: str) -> bool:
    """Check whether checkpoint is a QLora adapter (local path or HF repo)."""
    if os.path.isdir(checkpoint):
        return os.path.exists(os.path.join(checkpoint, "adapter_config.json"))
    # HF repo — check if adapter_config.json exists in the repo
    try:
        hf_hub_download(checkpoint, "adapter_config.json")
        return True
    except Exception:
        return False


def infer_n_buckets(checkpoint: str) -> int:
    """Infer the number of classification buckets from a checkpoint.

    For full fine-tuned checkpoints, reads num_labels from config.json.
    For LoRA checkpoints, reads the output dim of the saved score head weights.
    """
    if not is_qlora_checkpoint(checkpoint):
        return AutoConfig.from_pretrained(checkpoint).num_labels

    # LoRA: find n_buckets from the saved score head weight shape
    if os.path.isdir(checkpoint):
        safetensor_files = glob.glob(os.path.join(checkpoint, "*.safetensors"))
        safetensor_path = safetensor_files[0] if safetensor_files else None
        bin_path = os.path.join(checkpoint, "adapter_model.bin")
    else:
        # HF repo — download the adapter weights
        try:
            safetensor_path = hf_hub_download(checkpoint, "adapter_model.safetensors")
        except Exception:
            safetensor_path = None
        try:
            bin_path = hf_hub_download(checkpoint, "adapter_model.bin")
        except Exception:
            bin_path = None

    if safetensor_path and os.path.exists(safetensor_path):
        with safe_open(safetensor_path, framework="pt") as f:
            for key in f.keys():
                if "score" in key and "linear.weight" in key:
                    return f.get_tensor(key).shape[0]

    if bin_path and os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        for key, tensor in state_dict.items():
            if "score" in key and "linear.weight" in key:
                return tensor.shape[0]

    raise ValueError(
        f"Could not infer n_buckets from checkpoint at {checkpoint}. "
        "No score head weights found in adapter."
    )


def run_inference(
    checkpoint_path: str,
    base_model_name: str,
    dataset_name_or_path: str,
    split: str = None,
    text_col: str = "text",
    max_length: int = 1024,
    batch_size: int = 24,
    min_words: int = 0,
):
    """Run inference and return the dataset with bucket_pred / score_pred."""

    n_buckets = infer_n_buckets(checkpoint_path)
    print(f"Inferred n_buckets={n_buckets} from checkpoint")

    # --- tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    # --- model ---
    is_qlora = is_qlora_checkpoint(checkpoint_path)

    if is_qlora:
        from peft import PeftModel
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=n_buckets,
            quantization_config=quantization_config,
        )
        base_model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(base_model, "score") and isinstance(
            base_model.score, torch.nn.Linear
        ):
            hidden_size = base_model.config.hidden_size
            device = next(base_model.parameters()).device
            base_model.score = NormedLinear(hidden_size, n_buckets, device=device)
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path,
        )

    model.eval()

    # --- dataset ---
    if os.path.isdir(dataset_name_or_path):
        ds = load_from_disk(dataset_name_or_path)
        if isinstance(ds, dict):
            if split is None:
                raise ValueError(
                    f"Dataset has multiple splits {list(ds.keys())}. "
                    "Pass --split to select one."
                )
            ds = ds[split]
    else:
        ds = load_dataset(dataset_name_or_path, split=split)

    if text_col not in ds.column_names:
        raise ValueError(
            f"Column '{text_col}' not found. Available columns: {ds.column_names}"
        )

    # optional minimum word filter
    if min_words > 0:
        ds = ds.filter(
            lambda x: x[text_col] is not None
            and count_words(x[text_col]) >= min_words,
            num_proc=4,
        )

    # tokenize (no labels needed for inference)
    def tokenize(example):
        text = clean_text(example[text_col])
        return tokenizer(text, truncation=True, max_length=max_length)

    ds_tokenized = ds.map(tokenize, num_proc=4)

    # --- predict ---
    training_args = TrainingArguments(
        output_dir="/tmp/editlens_inference",
        per_device_eval_batch_size=batch_size if not is_qlora else 4,
        bf16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    output = trainer.predict(ds_tokenized)
    probs = softmax(output.predictions, axis=1)
    bucket_preds = np.argmax(probs, axis=1)

    bucket_labels = np.arange(n_buckets)
    score_preds = (probs @ bucket_labels) / (n_buckets - 1)

    # --- add columns to original (non-tokenized) dataset ---
    # Derive a clean base-model tag: e.g. "FacebookAI/roberta-large" → "roberta_large"
    base_model_tag = base_model_name.split("/")[-1]
    bucket_col = f"editlens_{base_model_tag}_bucket"
    score_col = f"editlens_{base_model_tag}_score"

    ds = ds.add_column(bucket_col, bucket_preds.tolist())
    ds = ds.add_column(score_col, score_preds.tolist())

    return ds


def main():
    parser = argparse.ArgumentParser(
        description="Run EditLens inference on a HuggingFace dataset"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--base_model",
        required=True,
        help="Base model name (e.g. FacebookAI/roberta-large)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        required=True,
        help="Must be <512 for FacebookAI/roberta-large",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="HuggingFace dataset name or local path to a saved dataset",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split (required for multi-split datasets)",
    )
    parser.add_argument(
        "--text_col",
        default="text",
        help="Column name containing the text to classify (default: text)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=24,
        help="Eval batch size (default: 24)",
    )
    parser.add_argument(
        "--min_words",
        type=int,
        default=0,
        help="Minimum word count filter (default: 0, no filter)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save output dataset (jsonl, csv, or directory for arrow format)",
    )

    args = parser.parse_args()

    ds = run_inference(
        checkpoint_path=args.checkpoint,
        base_model_name=args.base_model,
        dataset_name_or_path=args.dataset,
        split=args.split,
        text_col=args.text_col,
        max_length=args.max_length,
        batch_size=args.batch_size,
        min_words=args.min_words,
    )

    base_model_tag = args.base_model.split("/")[-1]
    bucket_col = f"editlens_{base_model_tag}_bucket"
    score_col = f"editlens_{base_model_tag}_score"

    print(f"\nInference complete. {len(ds)} examples processed.")
    print(f"Bucket distribution: {np.bincount(ds[bucket_col]).tolist()}")
    print(f"Score stats: mean={np.mean(ds[score_col]):.4f}, std={np.std(ds[score_col]):.4f}")

    if args.output:
        if args.output.endswith(".jsonl"):
            ds.to_json(args.output)
        elif args.output.endswith(".csv"):
            ds.to_csv(args.output)
        else:
            ds.save_to_disk(args.output)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
