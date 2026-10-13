import pytest

from metrics import ranking, rate, summarize


def test_ties_and_orientation():
    assert ranking([(0, 0), (1, 1)])["auroc"] == 1
    assert ranking([(0, 1), (1, 0)])["auroc"] == 0
    r = ranking([(0, 0.5), (1, 0.5)])
    assert r["auroc"] == 0.5 and r["tpr_at_fpr"]["0.01"] == 0


def test_strict_mixed_and_missing_classes():
    r = summarize(
        [
            {"label": "human", "prediction": "mixed", "score": 0.5},
            {"label": "ai", "prediction": "mixed", "score": 0.5},
        ]
    )
    assert r["strict_fpr"]["rate"] == 1 and r["strict_fnr"]["rate"] == 1
    assert summarize([])["strict_fpr"]["rate"] is None
    assert ranking([(1, 0.1)])["auroc"] is None


def test_zero_errors_is_not_certainty():
    assert rate(0, 100)["wilson95"][1] > 0.03
    assert rate(100, 100)["wilson95"][0] < 0.97


def test_errors_not_silently_scored():
    r = summarize([{"error": "failed", "label": "ai"}])
    assert r["failures"] == 1 and r["strict_fnr"]["rate"] is None


def test_nonfinite():
    with pytest.raises(ValueError):
        ranking([(0, 0), (1, float("nan"))])


def test_localization_midpoints_and_false_negatives():
    from run import localization

    r = localization("abc def ghi", [[4, 7]], [[0, 3]])
    assert r == {"tp": 0, "fp": 1, "tn": 1, "fn": 1, "fraction_error": 0}


def test_stratified_selection_and_span_validation():
    from prepare import record, sample, validate

    rows = [
        record("test", i, "some words here", "human" if i % 2 else "ai", cohort="same")
        for i in range(20)
    ]
    chosen, _ = sample(rows, 2)
    assert len(chosen) == 4 and sample(reversed(rows), 2)[0] == chosen
    with pytest.raises(ValueError):
        validate([rows[0], rows[0]])
    bad = record("test", "bad", "abc", "mixed", ai_spans=[[0, 5]])
    with pytest.raises(ValueError):
        validate([bad])


def test_pilot_keeps_both_classes_and_limits_each_dataset():
    from prepare import record
    from run import select

    rows = [
        record("test", i, "some text", "human" if i < 10 else "ai", cohort="same")
        for i in range(20)
    ]
    chosen = select(rows, 4)
    assert len(chosen) == 4 and {r["label"] for r in chosen} == {"human", "ai"}


def test_many_ai_cohorts_do_not_exclude_human_controls():
    from prepare import record
    from run import select

    rows = [record("test", i, "some text", "ai", cohort=str(i)) for i in range(20)]
    rows += [record("test", "human", "human prose", "human", cohort="zzz")]
    assert {r["label"] for r in select(rows, 2)} == {"human", "ai"}
