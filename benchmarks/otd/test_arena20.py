import json

from arena20 import SEED, D, normalize, read, sha
from arena_generate import audit_signals, request_body


def test_frozen_sample_is_first_twenty_eligible():
    frame = read(D / "frame.jsonl")
    assert frame == sorted(frame, key=lambda r: (r["rank"], r["prompt_id"]))
    for r in frame:
        assert r["prompt_id"] == sha(normalize(r["text"]).encode())
        assert r["rank"] == sha((SEED + "\0" + r["prompt_id"]).encode())
    decisions = read(D / "screening.jsonl")
    assert [r["position"] for r in decisions] == list(range(1, len(decisions) + 1))
    expected = [r["prompt_id"] for r in decisions if r["decision"] == "eligible"]
    selected = read(D / "selected_prompts.jsonl")
    assert len(expected) == len(set(expected)) == 20
    assert expected == [r["prompt_id"] for r in selected]
    assert decisions[-1]["decision"] == "eligible"


def test_unicode_jsonl_and_dedup(tmp_path):
    p = tmp_path / "test.jsonl"
    p.write_text(json.dumps({"text": "a\u2028b"}, ensure_ascii=False) + "\n")
    assert read(p) == [{"text": "a\u2028b"}]
    assert normalize(" e\u0301\t x ") == normalize("é x")
    assert normalize("A") != normalize("a")


def test_grid_and_price_enforcement():
    m = json.loads((D / "generation_manifest.json").read_text())
    assert len(m["models"]) == 20
    assert len(set(x["family"].split(" (")[0] for x in m["models"])) >= 5
    for model in m["models"]:
        assert float(model["output_usd_per_million"]) < 2
        body = request_body(m, model, {"text": "original prompt"})
        assert body["provider"]["max_price"]["completion"] < 2
        assert body["messages"][1]["content"] == "original prompt"
        assert "tools" not in body and "models" not in body


def test_truncation_and_unknown_token_gate():
    result = audit_signals(
        {"choices": [{"finish_reason": "length", "message": {"content": "word " * 80}}]}
    )
    assert result["length_ok"] and not result["not_truncated"]
    assert result["input_output_token_condition"].startswith("unverified")
    assert not audit_signals({"choices": []})["length_ok"]
