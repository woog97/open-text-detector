"""Independently verify saved study targets and optional upstream numerical parity."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from open_text_detector_backend.result_codec import decode
from tune_laya_meld import OUT, Laya, build_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", type=Path)
    args = ap.parse_args()
    manifest = json.loads((OUT / "manifest.json").read_text())
    papers = manifest["papers"]
    records = {
        r["text_sha256"]: r
        for r in map(
            json.loads,
            (
                ROOT
                / "research/classifications/vast-complete-20260925/metadata/v8-results.jsonl"
            )
            .read_text()
            .splitlines(),
        )
    }
    assert len({p["pdf_sha256"] for p in papers}) == len(papers) == 35
    assert len({p["text_sha256"] for p in papers}) == 35
    assert (
        len(
            {
                (p["conference"], p["year"])
                for p in papers
                if p["selection"] == "conference_year"
            }
        )
        == 25
    )
    errors, drift, hold_pred, hold_target = [], [], [], []
    count = 0
    for meta in papers:
        paper = json.loads((OUT / (meta["pdf_sha256"] + ".json")).read_text())
        text = paper["text"]
        assert hashlib.sha256(text.encode()).hexdigest() == meta["text_sha256"]
        blob = (
            ROOT
            / "research/classifications/vast-complete-20260925/results/v8"
            / (meta["text_sha256"] + ".pgf")
        ).read_bytes()
        assert (
            hashlib.sha256(blob).hexdigest()
            == records[meta["text_sha256"]]["result"]["sha256"]
        )
        teacher = decode(blob)
        assert (
            teacher["inference"]["revision"]
            == "8990324abd92e1fa17072f6887ea1e5c1cef5abc"
        )
        # Independent character-density integral: no moving token-pointer alignment.
        density = np.zeros(len(text) + 1)
        evidence = np.zeros(len(text) + 1)
        for token in teacher["tokens"]:
            a, b = token["start"], token["end"]
            w = token.get("token_count", 1) / (b - a)
            density[a] += w
            density[b] -= w
            evidence[a] += w * token["raw_score"]
            evidence[b] -= w * token["raw_score"]
        weight_integral = np.r_[0.0, np.cumsum(np.cumsum(density)[:-1])]
        score_integral = np.r_[0.0, np.cumsum(np.cumsum(evidence)[:-1])]
        full = json.loads(
            (OUT / "evaluation" / ("full-" + meta["pdf_sha256"] + ".json")).read_text()
        )
        assert full["text_sha256"] == meta["text_sha256"]
        assert len(full["segments"]) == len(paper["rows"])
        covered = []
        last = 0
        for row, result in zip(paper["rows"], full["segments"]):
            a, b = row["start"], row["end"]
            assert last <= a < b <= len(text)
            assert (a, b) == (result["start"], result["end"])
            last = b
            covered.append(text[a:b])
            w = weight_integral[b] - weight_integral[a]
            raw = (score_integral[b] - score_integral[a]) / w
            expected = 1 / (1 + np.exp(-raw))
            errors.append(abs(expected - result["teacher_score"]))
            assert abs(w - row["teacher_token_weight"]) < 1e-6
            count += 1
        assert "".join("".join(covered).split()) == "".join(text.split())
        indices = meta["screen_indices"]
        assert indices == sorted(
            set(
                np.linspace(
                    0, len(paper["rows"]) - 1, min(24, len(paper["rows"])), dtype=int
                ).tolist()
            )
        )
        if meta["split"] == "holdout":
            screen = json.loads(
                (
                    OUT / "evaluation" / ("holdout-" + meta["pdf_sha256"] + ".json")
                ).read_text()
            )["selected_raw"]
            hold_pred.extend(s["score"] for s in full["segments"])
            hold_target.extend(s["teacher_score"] for s in full["segments"])
        else:
            screen = json.loads(
                (
                    OUT / "screen" / ("chatgpt_local-" + meta["pdf_sha256"] + ".json")
                ).read_text()
            )["scores"]
        drift.extend(
            abs(
                np.array(screen)
                - np.array([full["segments"][i]["raw_laya_score"] for i in indices])
            )
        )
    assert max(errors) < 1e-7
    saved = json.loads((OUT / "full-summary.json").read_text())["holdout"]
    a, b = np.array(hold_pred), np.array(hold_target)
    independent_mae = float(np.mean(abs(a - b)))
    independent_accuracy = float(
        np.mean(
            np.digitize(a, [0.2, 0.8], right=True)
            == np.digitize(b, [0.2, 0.8], right=True)
        )
    )
    assert abs(independent_mae - saved["mae"]) < 1e-12
    assert abs(independent_accuracy - saved["color_accuracy"]) < 1e-12
    report = {
        "papers": len(papers),
        "phrases": count,
        "independent_target_max_error": max(errors),
        "screen_full_max_score_drift": max(drift),
        "screen_full_mean_score_drift": float(np.mean(drift)),
        "holdout_mae": independent_mae,
        "holdout_color_accuracy": independent_accuracy,
    }
    if args.upstream:
        import torch
        from transformers.initialization import no_init_weights

        spec = importlib.util.spec_from_file_location("upstream_laya", args.upstream)
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        p = Laya(model_dir=ROOT / "models", runtime="torch", device="mps", batch_size=4)
        p._load()
        # Independent upstream decision head, identical strictly loaded checkpoint.
        with no_init_weights():
            reference = upstream.DecisionModel(p.model.encoder)
        reference.load_state_dict(p.model.state_dict(), strict=True)
        reference.eval().to("mps")
        profile = json.loads((OUT / "frozen-profile.json").read_text())
        rows = []
        for meta in [
            papers[0],
            *[m for m in papers if m["selection"] != "conference_year"],
        ]:
            paper = json.loads((OUT / (meta["pdf_sha256"] + ".json")).read_text())
            new, markers = build_rows(
                p,
                paper,
                [0, len(paper["rows"]) // 2, len(paper["rows"]) - 1],
                profile["input"],
            )
            rows.extend(new)
        p.markers = markers
        own = np.array(p.score_rows(rows))
        upstream_scores = []
        with torch.inference_mode():
            for row in rows:
                ids = torch.tensor([row["ids"]], device="mps")
                logits, _ = reference(
                    ids,
                    torch.ones_like(ids),
                    torch.tensor([markers], device="mps"),
                    torch.ones((1, 2), dtype=torch.bool, device="mps"),
                    torch.zeros(1, dtype=torch.long, device="mps"),
                )
                upstream_scores.append(
                    torch.softmax(logits / p.temperature, -1)[0, 1].item()
                )
        mlx = Laya(
            model_dir=ROOT / "models", runtime="mlx", precision="float16", batch_size=4
        )
        mlx._load()
        mlx.markers = markers
        optimized = np.array(mlx.score_rows(rows))
        mlx.batch_size = 1
        single = np.array(mlx.score_rows(rows))
        perm = np.arange(len(rows))[::-1]
        reordered = np.array(mlx.score_rows([rows[i] for i in perm]))[::-1]
        report["numerical"] = {
            "examples": len(rows),
            "upstream_source_sha256": hashlib.sha256(
                args.upstream.read_bytes()
            ).hexdigest(),
            "torch_vs_upstream_max": float(np.max(abs(own - upstream_scores))),
            "mlx_vs_upstream_max": float(np.max(abs(optimized - upstream_scores))),
            "mlx_batch_size_max": float(np.max(abs(single - optimized))),
            "mlx_permutation_max": float(np.max(abs(single - reordered))),
        }
        assert report["numerical"]["torch_vs_upstream_max"] < 1e-4
        assert report["numerical"]["mlx_vs_upstream_max"] < 0.005
        assert report["numerical"]["mlx_batch_size_max"] < 0.005
        assert report["numerical"]["mlx_permutation_max"] < 1e-6
    (OUT / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
