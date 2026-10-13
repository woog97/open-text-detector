"""Simple non-authorship controls to distinguish task mismatch from a broken runtime."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from tune_laya_meld import OUT, Laya, predict

p = Laya(model_dir=ROOT / "models", runtime="mlx", precision="float16", batch_size=4)
p._load()
cases = [
    (
        "positive",
        "I love this product. It works perfectly and I am delighted with it!",
        0,
    ),
    (
        "negative",
        "This product is horrible. It broke immediately and I want a refund.",
        1,
    ),
    ("positive", "The staff were kind and helpful. We had a wonderful stay.", 0),
    ("negative", "I hate the service. Nobody helped and the experience was awful.", 1),
]
variant = {
    "instruction": "What is the sentiment of this review?",
    "options": [
        "positive: The reviewer is happy or satisfied.",
        "negative: The reviewer is unhappy or dissatisfied.",
    ],
    "context": False,
    "max_length": 512,
}
result = []
for name, text, label in cases:
    paper = {"text": text, "rows": [{"start": 0, "end": len(text)}]}
    score = predict(p, paper, [0], variant)[0]
    result.append(
        {
            "expected": name,
            "negative_score": score,
            "correct": int(score >= 0.5) == label,
        }
    )
(OUT / "semantic-controls.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result))
