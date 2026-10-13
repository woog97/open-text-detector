import json

from arena20 import B, read, sha
from arena100_generate import body, effective_status
from arena100_results import audit
from arena100_score import eligible_ids, fully_scored

D = B / "data/arena100"


def test_inline_provider_error_is_retryable_without_treating_success_as_failure():
    assert effective_status(200, {"error": {"code": 502}}) == 502
    assert effective_status(200, {"choices": [{}]}) == 200
    assert effective_status(429, {"error": {"code": 429}}) == 429


def test_frozen_100_is_first_100_eligible_and_preserves_twenty():
    frame = read(B / "data/arena20/frame.jsonl")
    ds = read(D / "screening.jsonl")
    selected = read(D / "selected_prompts.jsonl")
    assert len(selected) == 100 and len(ds) == 320
    assert [d["position"] for d in ds] == list(range(1, 321))
    assert all(d["prompt_id"] == r["prompt_id"] for d, r in zip(ds, frame))
    assert selected == [
        {**frame[d["position"] - 1], **d} for d in ds if d["decision"] == "eligible"
    ]
    assert selected[:20] == read(B / "data/arena20/selected_prompts.jsonl")
    assert json.loads((D / "generation_manifest.json").read_text())[
        "selected_prompts_sha256"
    ] == sha((D / "selected_prompts.jsonl").read_bytes())


def test_roster_obeys_exclusions_and_reasoning_policy():
    mf = json.loads((D / "generation_manifest.json").read_text())
    ms = mf["models"]
    assert len(ms) == 12
    for m in ms:
        assert not m["id"].startswith("anthropic/")
        assert not m["id"].startswith("openai/") or m["id"] == "openai/gpt-oss-120b"
        assert m["provider"]["max_price"]["completion"] <= 7
        r = m["settings"].get("reasoning")
        assert r is None or r.get("enabled") is False or r.get("effort") == "low"
        p = {"text": 'A literal prompt with "quotes", \\slashes and\nnewlines.'}
        req = body(m, p, mf, batch=m["mode"] == "batch")
        assert req["messages"][1]["content"] == p["text"]
        assert req["max_tokens"] == 4096 and "tools" not in req
        assert ("provider" not in req) == (m["mode"] == "batch")


def test_mechanical_audit_is_not_content_review():
    assert audit({"success": False}) == (False, "api_failure")
    assert audit(
        {"success": True, "audit": {"not_truncated": False, "length_ok": True}}
    ) == (False, "non_stop_finish")
    assert audit(
        {"success": True, "audit": {"not_truncated": True, "length_ok": True}}
    ) == (True, "mechanically_eligible")


def test_scoring_completion_refreshes_successes_added_during_scoring():
    rows = [
        {
            "model_id": "model",
            "prompt_id": str(i),
            "success": True,
            "audit": {"not_truncated": True, "length_ok": True},
        }
        for i in range(1200)
    ]
    stale = [*rows[:-1], {**rows[-1], "success": False}]
    seen = eligible_ids(stale)
    assert not fully_scored(stale, seen)
    assert not fully_scored(rows, seen)
    assert fully_scored(rows, eligible_ids(rows))


def test_cheap_extension_reuses_identical_requests_and_keeps_price_cap():
    from arena100_results import collect

    d = B / "data/arena100-cheap"
    mf = json.loads((d / "generation_manifest.json").read_text())
    models = {m["id"]: m for m in mf["models"]}
    prompts = {p["prompt_id"]: p for p in read(d / "selected_prompts.jsonl")}
    assert len(models) == 20 and len(prompts) == 100
    assert (d / "selected_prompts.jsonl").read_bytes() == (
        D / "selected_prompts.jsonl"
    ).read_bytes()
    reused = [r for r in collect(B / "runs/arena100-cheap") if r.get("reused_from")]
    assert len(reused) == 365
    for r in reused:
        assert r["success"]
        assert r["request"] == body(models[r["model_id"]], prompts[r["prompt_id"]], mf)
    for m in models.values():
        assert m["provider"]["max_price"]["completion"] < 2
    override = json.loads((d / "qwen_route_override.json").read_text())["overrides"][
        "qwen/qwen3.8-27b:free"
    ]
    cat = {
        m["id"]: m
        for m in json.loads(
            (B / "sources/openrouter-arena100-cheap-models.json").read_text()
        )["data"]
    }
    assert (
        cat[override["model"]]["canonical_slug"]
        == models["qwen/qwen3.8-27b:free"]["canonical_slug"]
    )
    assert override["provider"]["max_price"]["completion"] < 2
