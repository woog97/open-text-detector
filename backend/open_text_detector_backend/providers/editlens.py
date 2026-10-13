"""EditLens inference following Pangram's published NormedLinear/bucket scoring.

Model weights are never downloaded during startup. Optional ML dependencies and
approved Hugging Face access are required only when an enabled model is used.
"""
import gc
import threading
import json
from pathlib import Path
from .checkpoints import QWEN_ID, QWEN_REVISION
from .errors import ProviderError
from .preprocess import preprocess


class EditLens:
    def __init__(self, device="auto", token=None, model_dir=None):
        self.token = token
        self.device_request = device
        self.loaded_id = None
        self.model = self.tokenizer = None
        self.lock = threading.Lock()
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parents[3] / "models"

    def _load(self, config):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from huggingface_hub import hf_hub_download
            from safetensors import safe_open
        except ImportError as e:
            raise ProviderError("model_dependencies_missing", "Install the optional models dependencies before enabling EditLens") from e
        checkpoint = config["model_id"]
        if self.loaded_id == checkpoint:
            return
        self.model = self.tokenizer = None
        self.loaded_id = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        device = self.device_request
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float32 if device == "cpu" else torch.bfloat16 if checkpoint == QWEN_ID else torch.float16
        try:
            tokenizer_source = config["base_model_id"]
            if checkpoint == QWEN_ID:
                tokenizer_source = self.model_dir / "editlens-qwen3-4b-merged-v3"
                manifest = tokenizer_source / "download-manifest.json"
                if not manifest.is_file() or json.loads(manifest.read_text()).get("revision") != QWEN_REVISION:
                    raise ProviderError("model_not_downloaded", "Run scripts/download_model.py to install the pinned Qwen EditLens checkpoint")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=False,
                token=self.token, local_files_only=checkpoint == QWEN_ID)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "left"
            if checkpoint == QWEN_ID:
                from .qwen import load_merged_qwen
                model = load_merged_qwen(tokenizer_source, dtype)
                model.config.pad_token_id = tokenizer.pad_token_id
                buckets = model.config.num_labels
                self.max_length = 1024
            elif "Llama" in checkpoint:
                from peft import PeftModel
                path = hf_hub_download(checkpoint, "adapter_model.safetensors", token=self.token)
                with safe_open(path, framework="pt") as weights:
                    score_keys = [k for k in weights.keys() if "score" in k and "linear.weight" in k]
                    if not score_keys:
                        raise ProviderError("invalid_checkpoint", "EditLens classifier head was not found in the adapter")
                    buckets = weights.get_slice(score_keys[0]).get_shape()[0]
                class NormedLinear(torch.nn.Module):
                    def __init__(self, hidden_size, labels):
                        super().__init__()
                        self.norm = torch.nn.LayerNorm(hidden_size, dtype=dtype)
                        self.linear = torch.nn.Linear(hidden_size, labels, bias=False, dtype=dtype)

                    def forward(self, x):
                        return self.linear(self.norm(x))
                base = AutoModelForSequenceClassification.from_pretrained(
                    config["base_model_id"], num_labels=buckets, torch_dtype=dtype,
                    trust_remote_code=False, use_safetensors=True, token=self.token)
                base.config.pad_token_id = tokenizer.pad_token_id
                base.score = NormedLinear(base.config.hidden_size, buckets)
                model = PeftModel.from_pretrained(base, checkpoint, token=self.token)
                self.max_length = 1024
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    checkpoint, torch_dtype=dtype, trust_remote_code=False, use_safetensors=True, token=self.token)
                buckets = model.config.num_labels
                self.max_length = 512
            if buckets < 2:
                raise ProviderError("invalid_checkpoint", "EditLens requires at least two classification buckets")
            self.model = model.to(device).eval()
            self.tokenizer = tokenizer
            self.device = device
            self.buckets = buckets
            self.dtype = str(dtype)
            self.loaded_id = checkpoint
        except ProviderError:
            raise
        except Exception as e:
            name = type(e).__name__
            message = str(e).lower()
            if "gated" in message or "401" in message or "403" in message or "token" in message:
                raise ProviderError("model_access_required", "Hugging Face access is required for the checkpoint and base model; configure HF_TOKEN after approval") from e
            raise ProviderError("model_load_failed", f"Model could not be loaded ({name}). Verify available memory, device, dependencies, and model access") from e

    def predict(self, config, text):
        with self.lock:
            self._load(config)
            import torch
            cleaned, source_map = preprocess(text)
            tokens = self.tokenizer(cleaned, add_special_tokens=False, return_offsets_mapping=True)
            offsets = tokens["offset_mapping"]
            chunk_size = self.max_length - self.tokenizer.num_special_tokens_to_add(pair=False) - 8
            scores, segments, weights = [], [], []
            with torch.inference_mode():
                index = 0
                while index < len(offsets):
                    end_index = min(index + chunk_size, len(offsets))
                    while end_index < len(offsets) and end_index > index + 1 and source_map[offsets[end_index][0]][0] == source_map[max(0, offsets[end_index][0]-1)][0]:
                        end_index -= 1
                    start = 0 if index == 0 else offsets[index][0]
                    end = len(cleaned) if end_index == len(offsets) else offsets[end_index][0]
                    encoded = self.tokenizer(cleaned[start:end], return_tensors="pt", truncation=True,
                                             max_length=self.max_length).to(self.device)
                    logits = self.model(**encoded).logits.float()
                    probs = torch.softmax(logits, dim=-1)
                    bucket_values = torch.arange(self.buckets, device=probs.device, dtype=probs.dtype)
                    score = float((probs @ bucket_values / (self.buckets - 1)).item())
                    segments.append({"start": source_map[start][0], "end": source_map[end-1][1], "score": score})
                    scores.append(score)
                    weights.append(end_index - index)
                    index = end_index
            if not weights:
                raise ProviderError("empty_tokens", "The tokenizer found no usable text")
            return {"score": sum(s*w for s, w in zip(scores, weights)) / sum(weights), "segments": segments,
                    "inference": {"model_id": self.loaded_id,
                                  "revision": QWEN_REVISION if self.loaded_id == QWEN_ID else None,
                                  "device": self.device, "dtype": self.dtype, "buckets": self.buckets,
                                  "max_length": self.max_length,
                                  "aggregation": "token_weighted_nonoverlapping_chunks",
                                  "preprocessing": "EditLens clean_text with source offset mapping"}}
