"""Run the experimental MELD-alignment inference recipe on a new local document.

Requires only the frozen recipe, Laya checkpoint and source document. No cached
MELD result, paper identity, publication year or conference is used in prediction.
Not promoted to the service: agreement in the study was weak.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from calibrate_laya_document_context import apply
from open_text_detector_backend.pdf_extraction import read_text, text_hash
from tune_laya_meld import OUT, Laya, calibrate, logit, predict


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--mode",
        choices=["raw", "score_calibrated", "color_calibrated", "document_context"],
        default="document_context",
    )
    ap.add_argument("--runtime", choices=["auto", "torch", "mlx"], default="auto")
    args = ap.parse_args()
    profile = json.loads((OUT / "frozen-profile.json").read_text())
    text = read_text(args.input)
    p = Laya(
        model_dir=ROOT / "models",
        runtime=args.runtime,
        precision="float16",
        batch_size=4,
    )
    p._load()
    rows = p.targets(text)
    raw = np.array(
        predict(
            p, {"text": text, "rows": rows}, list(range(len(rows))), profile["input"]
        )
    )
    metadata = {
        "prompt_profile": profile,
        "mode": args.mode,
        "runtime": p.runtime,
        "precision": p.precision,
        "batch_size": p.batch_size,
        "pipeline_depth": p.pipeline_depth if p.runtime == 'mlx' else 0,
    }
    if args.mode == "raw":
        scores = raw
    elif args.mode == "score_calibrated":
        scores = calibrate(logit(raw), profile["calibration"])
    elif args.mode == "color_calibrated":
        parameters = json.loads((OUT / "color-followup.json").read_text())[
            "calibration"
        ]
        metadata["calibration"] = parameters
        scores = calibrate(logit(raw), parameters)
    else:
        recipe = json.loads((OUT / "document-context-followup.json").read_text())
        indices = sorted(
            set(np.linspace(0, len(rows) - 1, min(24, len(rows)), dtype=int).tolist())
        )
        prior = float(logit(raw[indices]).mean())
        features = np.column_stack([logit(raw), np.full(len(raw), prior)])
        scores = apply(features[:, recipe["columns"]], recipe["model"])
        metadata.update(
            context_model=recipe["model"],
            feature_columns=recipe["columns"],
            document_prior=prior,
            document_prior_indices=indices,
        )
    weights = [sum(not c.isspace() for c in text[r["start"] : r["end"]]) for r in rows]
    output = {
        "text_sha256": text_hash(text),
        "score": float(np.average(scores, weights=weights)),
        "score_type": "experimental_meld_v8_alignment",
        "inference": metadata,
        "notice": "Experimental teacher-alignment study. Weak measured agreement with MELD v8; not validated authorship detection.",
        "segments": [
            {
                "start": r["start"],
                "end": r["end"],
                "score": float(s),
                "raw_laya_score": float(v),
                "band": "green" if s <= 0.2 else "red" if s > 0.8 else "middle",
            }
            for r, s, v in zip(rows, scores, raw, strict=True)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(
        json.dumps(
            {"phrases": len(rows), "score": output["score"], "output": str(args.output)}
        )
    )


if __name__ == "__main__":
    main()
