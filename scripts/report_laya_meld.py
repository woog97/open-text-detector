"""Summarize teacher agreement without mistaking score shrinkage for detection quality."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from calibrate_laya_document_context import apply as apply_document_context
from tune_laya_meld import OUT, calibrate, logit, metrics

manifest = json.loads((OUT / "manifest.json").read_text())
dev = json.loads((OUT / "development-results.json").read_text())
evaluation = json.loads((OUT / "evaluation-summary.json").read_text())
color_followup = json.loads((OUT / "color-followup.json").read_text())
document_recipe = json.loads((OUT / "document-context-followup.json").read_text())
document_scores = []
document_targets = []
document_ids = []
color_scores = []
color_targets = []
color_ids = []
rows = []
all_scores = []
all_targets = []
all_ids = []
hold_scores = []
hold_targets = []
hold_ids = []
examples = []
for j, paper in enumerate(manifest["papers"]):
    path = OUT / "evaluation" / ("full-" + paper["pdf_sha256"] + ".json")
    if not path.exists():
        continue
    result = json.loads(path.read_text())
    segments = result["segments"]
    pred = np.array([s["score"] for s in segments])
    target = np.array([s["teacher_score"] for s in segments])
    m = metrics(pred, target, np.full(len(pred), j))
    color_pred = calibrate(
        logit(np.array([s["raw_laya_score"] for s in segments])),
        color_followup["calibration"],
    )
    raw_scores = np.array([s["raw_laya_score"] for s in segments])
    prior = float(logit(raw_scores[paper["screen_indices"]]).mean())
    document_features = np.column_stack([logit(raw_scores), np.full(len(pred), prior)])
    document_pred = apply_document_context(
        document_features[:, document_recipe["columns"]], document_recipe["model"]
    )
    if paper["split"] == "holdout":
        document_scores.extend(document_pred)
        document_targets.extend(target)
        document_ids.extend([j] * len(pred))
        color_scores.extend(color_pred)
        color_targets.extend(target)
        color_ids.extend([j] * len(pred))
    rows.append(
        {
            **{
                k: paper[k]
                for k in [
                    "conference",
                    "year",
                    "title",
                    "selection",
                    "split",
                    "pdf_sha256",
                    "text_sha256",
                    "phrases",
                ]
            },
            "meld_mean_phrase_score": float(target.mean()),
            "laya_mean_phrase_score": float(pred.mean()),
            "meld_red_fraction": float(np.mean(target > 0.8)),
            "laya_red_fraction": float(np.mean(pred > 0.8)),
            "document_context_mean": float(document_pred.mean()),
            "document_context_mae": float(np.mean(abs(document_pred - target))),
            "document_context_red_fraction": float(np.mean(document_pred > 0.8)),
            "document_context_green_fraction": float(np.mean(document_pred <= 0.2)),
            "color_followup_red_fraction": float(np.mean(color_pred > 0.8)),
            "color_followup_color_agreement": metrics(
                color_pred, target, np.full(len(pred), j)
            )["color_accuracy"],
            "meld_green_fraction": float(np.mean(target <= 0.2)),
            "laya_green_fraction": float(np.mean(pred <= 0.2)),
            "mae": m["mae"],
            "spearman": m["spearman"],
            "color_agreement": m["color_accuracy"],
            "seconds": result["seconds"],
        }
    )
    all_scores.extend(pred)
    all_targets.extend(target)
    all_ids.extend([j] * len(pred))
    if paper["split"] == "holdout":
        hold_scores.extend(pred)
        hold_targets.extend(target)
        hold_ids.extend([j] * len(pred))
    source = json.loads((OUT / (paper["pdf_sha256"] + ".json")).read_text())["text"]
    for index in np.argsort(abs(pred - target))[-3:][::-1]:
        s = segments[index]
        examples.append(
            {
                "title": paper["title"],
                "conference": paper["conference"],
                "year": paper["year"],
                "split": paper["split"],
                "start": s["start"],
                "end": s["end"],
                "meld": s["teacher_score"],
                "laya": s["score"],
                "raw_laya": s["raw_laya_score"],
                "text": source[s["start"] : s["end"]],
            }
        )
if rows:
    with (OUT / "papers.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "mismatches.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(examples[0]))
        writer.writeheader()
        writer.writerows(examples)
    full = {
        "completed_papers": len(rows),
        "expected_papers": 35,
        "all": metrics(all_scores, all_targets, np.array(all_ids)),
        "holdout": metrics(hold_scores, hold_targets, np.array(hold_ids))
        if hold_scores
        else None,
        "paper_balanced_mae": float(np.mean([r["mae"] for r in rows])),
        "holdout_paper_balanced_mae": float(
            np.mean([r["mae"] for r in rows if r["split"] == "holdout"])
        )
        if hold_scores
        else None,
    }
    full["exploratory_color_followup_holdout"] = (
        metrics(color_scores, color_targets, np.array(color_ids))
        if color_scores
        else None
    )
    full["exploratory_document_context_holdout"] = (
        metrics(document_scores, document_targets, np.array(document_ids))
        if document_scores
        else None
    )
    if document_scores:
        predicted_band = (np.array(document_scores) > 0.2).astype(int) + (
            np.array(document_scores) > 0.8
        ).astype(int)
        teacher_band = (np.array(document_targets) > 0.2).astype(int) + (
            np.array(document_targets) > 0.8
        ).astype(int)
        full["document_context_holdout_confusion_teacher_rows_prediction_columns"] = (
            np.bincount(teacher_band * 3 + predicted_band, minlength=9)
            .reshape(3, 3)
            .tolist()
        )
    (OUT / "full-summary.json").write_text(json.dumps(full, indent=2))
else:
    full = {}
# Paired uncertainty estimate resamples held-out papers, not correlated phrases.
diffs = []
for paper in manifest["papers"]:
    if paper["split"] != "holdout":
        continue
    result = json.loads(
        (OUT / "evaluation" / ("holdout-" + paper["pdf_sha256"] + ".json")).read_text()
    )
    source = json.loads((OUT / (paper["pdf_sha256"] + ".json")).read_text())
    y = np.array([source["rows"][i]["teacher_score"] for i in result["indices"]])
    diffs.append(
        float(
            np.mean(
                abs(np.array(result["selected_calibrated"]) - y)
                - abs(np.array(result["baseline"]) - y)
            )
        )
    )
rng = np.random.default_rng(260925)
bootstrap = np.mean(rng.choice(diffs, size=(5000, len(diffs)), replace=True), axis=1)
uncertainty = {
    "paired_MAE_change_selected_minus_original": float(np.mean(diffs)),
    "paper_bootstrap_95pct": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
}
(OUT / "uncertainty.json").write_text(json.dumps(uncertainty, indent=2))
lines = [
    "# Laya versus MELD v8: inference agreement study",
    "",
    "**Result: the tested fixed-backbone inference changes do not reproduce MELD v8 phrase classification.**",
    "",
    f"Full-paper completion: **{len(rows)}/35 papers**. The held-out screening comparison is complete (11 papers, 264 phrases).",
    "",
    "The selected prompt lowers score error primarily by shrinking predictions toward the middle. On the held-out screen it assigns every phrase to the middle band. Its error is nearly the same as a constant-score baseline, and it does not recover the green/red highlights.",
    "",
    "## Held-out comparison",
    "",
    "| Method | Mean absolute score error ↓ | Spearman rank ↑ | Color agreement ↑ | Balanced color agreement ↑ |",
    "|---|---:|---:|---:|---:|",
]
evaluation["holdout_screen"]["color_followup"] = color_followup["exploratory_holdout"]
evaluation["holdout_screen"]["document_context"] = document_recipe[
    "exploratory_holdout"
]
labels = {
    "baseline": "Original Laya",
    "baseline_calibrated": "Original + score calibration",
    "constant_development_mean": "Constant development-set mean",
    "selected": "Selected prompt + score calibration",
    "color_followup": "Color-focused calibration (exploratory)",
    "document_context": "Local + document signal (exploratory)",
}
for key, label in labels.items():
    m = evaluation["holdout_screen"][key]
    lines.append(
        f"| {label} | {m['mae']:.4f} | {m['spearman']:.4f} | {m['color_accuracy']:.1%} | {m['balanced_color_accuracy']:.1%} |"
    )
lines += [
    "",
    "Color bands follow the reader: green ≤0.20, middle >0.20 to ≤0.80, red >0.80. These are agreement measures against MELD, not authorship accuracy.",
    "",
    "The color-focused follow-up was designed after the initial held-out failure was observed. It uses the same frozen prompt and fits parameters on development data only, but its reuse of the holdout is exploratory. It restores colored predictions at the cost of higher score error; ranking is unchanged. A new untouched set would be needed to validate further changes.",
    "",
    "A second exploratory follow-up adds the mean of Laya's own log-odds over the 24 uniformly spaced phrases as a document-level feature. Its small ridge-regularized readout was fitted on development data; ridge and feature choice used paper-disjoint CV. It improves average error and overall color agreement somewhat, but phrase ordering remains weak. No teacher score, venue, year or document identity enters this inference feature. Its reuse of the already-reviewed holdout is also exploratory.",
    "",
    "## Sample and separation",
    "",
    "- One deterministic hash-selected paper from each of 25 conference/year tuples; five additional lowest-mean-score papers from 2022 or earlier; five highest-mean-score papers from 2025 or later.",
    "- Eligibility: 4–40 pages and at least 80 scored MELD passages, excluding giant compilation/appendix outliers. Extremes ranked using the existing v8 score histogram mean, not year alone.",
    "- 35 distinct PDF hashes and text hashes, checked against pinned source text and checksum-verified MELD v8 outputs.",
    "- 24 development papers, 11 held-out papers. Screening uses 24 uniformly spaced phrases per paper; no teacher-dependent span selection.",
    "- Four-fold development evaluation holds out entire papers. The winning profile and calibration were frozen before held-out scores were read. Full-paper results were not used to choose the profile.",
    "- Target: sigmoid of MELD v8 raw token evidence averaged over each exact Laya phrase, weighted by token count and character overlap. No prompt receives MELD scores, dates, conference names, or selection labels.",
    "- Full-paper selected inference covers every phrase. Original Laya was evaluated on the fixed screen, not rerun across all full papers.",
    "",
    "## Tested inference changes",
    "",
    "| Variant | Raw development MAE | Paper-disjoint calibrated CV MAE | Raw rank correlation | Within-paper rank |",
    "|---|---:|---:|---:|---:|",
]
for name, v in dev["variants"].items():
    lines.append(
        f"| {name} | {v['raw']['mae']:.4f} | {v['paper_disjoint_cv']['mae']:.4f} | {v['raw']['spearman']:.4f} | {v['raw']['mean_within_paper_spearman']:.4f} |"
    )
profile = evaluation["profile"]
lines += [
    "",
    f"Selected: **{profile['variant']}**, {profile['input']['max_length']} tokens, with adjacent context. Calibration is sigmoid(a × logit(Laya score) + b), a={profile['calibration']['slope']:.3f}, b={profile['calibration']['intercept']:.3f}. Parameters fit only on development papers.",
    "",
    "Selection minimizes development paper-disjoint CV MAE + 0.05 × (1 − mean within-paper rank correlation). Because the underlying signal is weak, calibration compresses almost all predictions into the middle. This is a failed matching candidate, not a validated replacement.",
    "",
    "Basic sentiment controls: 4/4 correct. The native runtime also previously matched the PyTorch reference numerically. This supports a task-capability gap rather than an obvious broken forward pass, but the four controls are not a broad model validation.",
    "",
    "## Full-paper results",
    "",
    (
        f"All {len(rows)} complete papers cover {len(all_scores):,} phrases. The 11 held-out full papers cover {len(hold_scores):,} phrases. "
        f"The document-context follow-up has {full['exploratory_document_context_holdout']['color_accuracy']:.1%} three-color agreement, "
        f"{full['exploratory_document_context_holdout']['balanced_color_accuracy']:.1%} balanced color agreement, "
        f"and mean within-paper Spearman {full['exploratory_document_context_holdout']['mean_within_paper_spearman']:.3f}. "
        "These follow-up numbers remain exploratory because the original holdout had already been reviewed."
    ),
    "",
    "The full held-out confusion matrix for the document-context follow-up is recorded in `full-summary.json` (teacher rows, prediction columns, ordered green/middle/red). The candidate predicts no red spans in this run: it fails the central red-region matching goal even where aggregate error improves.",
    "",
    "| Conference/year | Selection | Split | Phrases | MELD mean | Laya mean | MELD red | Laya red | MAE |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['conference']}/{r['year']} | {r['selection']} | {r['split']} | {r['phrases']} | {r['meld_mean_phrase_score']:.3f} | {r['laya_mean_phrase_score']:.3f} | {r['meld_red_fraction']:.1%} | {r['laya_red_fraction']:.1%} | {r['mae']:.3f} |"
    )
lines += [
    "",
    "Paper means summarize aligned phrases, not MELD’s separate top-quartile document statistic. See `papers.csv` for full titles and identities, and `mismatches.csv` for exact text examples.",
    "",
    "## Interpretation",
    "",
    "This small, extreme-enriched sample is not representative of the full corpus. Prompt selection can overfit 24 development papers; phrases within a paper are correlated. The held-out test is the relevant check, and it shows very weak agreement. Lower raw error alone would overstate the improvement.",
    "",
    "The generic Laya decision model lacks demonstrated AI-writing detection ability on this sample. A next quality experiment would require teacher-supervised training/distillation or a task-specific head, followed by another untouched paper-level evaluation. This study does not establish that training would succeed. Laya's original neural weights stayed fixed; only small calibration/readout parameters were fitted to development MELD scores. Kernel optimization was deferred.",
    "",
    "## Reproduce",
    "",
    "```sh",
    "backend/.venv/bin/python scripts/prepare_laya_meld_study.py",
    "backend/.venv/bin/python scripts/tune_laya_meld.py",
    "backend/.venv/bin/python scripts/evaluate_laya_meld.py --full",
    "backend/.venv/bin/python scripts/check_laya_semantics.py",
    "backend/.venv/bin/python scripts/calibrate_laya_colors.py",
    "backend/.venv/bin/python scripts/calibrate_laya_document_context.py",
    "backend/.venv/bin/python scripts/report_laya_meld.py",
    "```",
    "",
    "Predictions are checkpointed per paper/variant. Frozen profile changes require a new study rather than silently reusing the holdout. Model weights and source hashes are recorded in the manifest/profile.",
]
(OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
print(
    json.dumps(
        {
            "complete": len(rows),
            "expected": 35,
            "full": full,
            "uncertainty": uncertainty,
        }
    )
)
