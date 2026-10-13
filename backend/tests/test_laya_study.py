"""Study invariants: metric sanity, calibration and no teacher features in model input."""

import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location(
    "laya_study", Path(__file__).resolve().parents[2] / "scripts/tune_laya_meld.py"
)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_metrics_identity_and_ties():
    scores = np.array([0.01, 0.01, 0.3, 0.9, 0.95, 0.99])
    result = study.metrics(scores, scores, np.array([0, 0, 0, 1, 1, 1]))
    assert result["mae"] == 0
    assert np.isclose(result["spearman"], 1)
    assert result["auc_meld_above_half"] == 1
    assert result["color_accuracy"] == 1
    assert study.correlation(np.ones(6), scores) == 0


def test_monotone_calibration_recovers_known_mapping():
    x = np.linspace(-4, 4, 101)
    y = study.sigmoid(1.2 * x - 0.5)
    fit = study.fit_calibration(x, y)
    assert fit["slope"] >= 0
    assert np.mean(abs(study.calibrate(x, fit) - y)) < 0.005


def test_teacher_scores_do_not_enter_inputs():
    from types import SimpleNamespace

    p = SimpleNamespace(
        tokenizer=SimpleNamespace(cls_token_id=1, sep_token_id=2, mask_token_id=3),
        _encode=lambda text: [ord(c) for c in text],
    )
    paper = {"text": "A short target.", "rows": [{"start": 0, "end": 15, "teacher_score": 0.01}]}
    variant = {"instruction": "Classify.", "options": ["human", "ai"], "context": False, "max_length": 512}
    first, markers = study.build_rows(p, paper, [0], variant)
    paper["rows"][0]["teacher_score"] = 0.99
    second, _ = study.build_rows(p, paper, [0], variant)
    assert first[0]["ids"] == second[0]["ids"]
    assert len(markers) == 2


def test_color_bands_match_reader_boundaries():
    predicted = np.array([0.2, 0.8])
    target = np.array([0.1, 0.5])
    result = study.metrics(predicted, target, np.array([0, 0]))
    assert result["color_accuracy"] == 1


def test_document_context_readout_uses_only_provided_features(monkeypatch):
    import importlib

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    context = importlib.import_module("calibrate_laya_document_context")
    rng = np.random.default_rng(12)
    x = rng.normal(size=(200, 2))
    target = study.sigmoid(0.2 * x[:, 0] + 0.8 * x[:, 1] - 0.4)
    model = context.fit(x, target, 0.1)
    result = context.apply(x, model)
    assert np.mean(abs(result - target)) < 0.005
    assert set(model) == {"mean", "scale", "coefficients", "ridge"}
    assert np.all((result >= 0) & (result <= 1))
