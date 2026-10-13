import asyncio
import io
import json

import pytest
from botocore.stub import Stubber
from open_text_detector_backend.config import Settings
from open_text_detector_backend.result_codec import canonical, encode, decode
from open_text_detector_backend.result_store import ResultStore, ResultStorageError
from .conftest import TEXT


def sample():
    return {
        "score": 0.5,
        "label": "uncertain",
        "score_type": "ai_evidence",
        "unicode": "α🙂",
        "tokens": [
            {
                "start": i * 3,
                "end": i * 3 + 2,
                "raw_score": -0.0 if i == 0 else 1.25,
                "token_count": 1,
                "score": 0.2222222222222222,
                "label": "low_evidence",
            }
            for i in range(100)
        ],
        "segments": [
            {
                "start": 0,
                "end": 300,
                "raw_score": 1.125,
                "score": 0.7549149868676283,
                "token_count": 100,
                "label": "low_evidence",
            }
        ],
        "performance": {"classification_seconds": 1.23456789},
        "extra": [True, None, {"x": 2**80}],
    }


def test_lossless_and_corruption():
    original = sample()
    compressed = encode(original)
    assert canonical(decode(compressed)) == canonical(original)
    assert len(compressed) < len(canonical(original)) / 5
    for bad in [compressed[:-4], b"BAD!" + compressed[4:], compressed[:8] + b"x" * 32 + compressed[40:]]:
        with pytest.raises(Exception):
            decode(bad)
    # Unknown schemas and mixed numeric types are retained exactly as JSON.
    original["tokens"] = [{"score": 0.1}, {"score": 1, "unknown": [1, 2]}]
    assert canonical(decode(encode(original))) == canonical(original)


def test_local_cache_independent_copies_and_missing_object(tmp_path):
    settings = Settings(data_dir=tmp_path, _env_file=None)
    store = ResultStore(settings)
    summary, reference = store.put(sample())
    assert "tokens" not in json.loads(summary)
    restored = store.get(reference)
    restored["tokens"].clear()
    assert len(store.get(reference)["tokens"]) == 100
    ref = json.loads(reference)
    path = store.root / ref["sha256"][:2] / (ref["sha256"] + ".pgf")
    path.write_bytes(b"corrupt")
    cold = ResultStore(settings)
    with pytest.raises(ResultStorageError):
        cold.get(reference)
    assert store.cache_size <= settings.result_cache_bytes


def test_s3_contract_and_cold_reads(tmp_path):
    import boto3
    from botocore.response import StreamingBody

    settings = Settings(
        data_dir=tmp_path,
        result_storage="s3",
        result_s3_bucket="test-findings",
        result_s3_endpoint="https://example.r2.cloudflarestorage.com",
        _env_file=None,
    )
    client = boto3.client("s3", region_name="auto", aws_access_key_id="test", aws_secret_access_key="test")
    store = ResultStore(settings, client)
    blob = encode(sample())
    import hashlib

    sha = hashlib.sha256(blob).hexdigest()
    key = f"findings/v1/{sha[:2]}/{sha}.pgf"
    with Stubber(client) as stub:
        stub.add_response(
            "put_object",
            {},
            {
                "Bucket": "test-findings",
                "Key": key,
                "Body": blob,
                "ContentType": "application/octet-stream",
                "Metadata": {"sha256": sha, "format": "pgf1-zstd19"},
            },
        )
        for _ in range(2):
            stub.add_response(
                "get_object",
                {"Body": StreamingBody(io.BytesIO(blob), len(blob)), "ContentLength": len(blob)},
                {"Bucket": "test-findings", "Key": key},
            )
        _, reference = store.put(sample())
        assert canonical(ResultStore(settings, client).get(reference)) == canonical(sample())
        stub.assert_no_pending_responses()
    wrong = ResultStore(settings.model_copy(update={"result_s3_bucket": "different"}), client)
    with pytest.raises(ResultStorageError):
        wrong.get(reference)


def test_api_summary_does_not_fetch_objects(configured, monkeypatch):
    client, admin, app, provider, _, _ = configured
    scan = client.post("/v1/scans", headers=admin, json={"text": TEXT}).json()
    asyncio.run(app.state.service.process_one())
    row = app.state.db.one("SELECT result,result_storage FROM scans WHERE id=?", (scan["id"],))
    assert row["result_storage"] and "segments" not in json.loads(row["result"])
    report = client.get("/v1/scans/" + scan["id"], headers=admin)
    assert report.status_code == 200
    assert report.json()["result"]["segments"]
    assert "result_storage" not in report.json()
    assert report.json()["result"]["performance"]["storage_format"] == "pgf1-zstd19"
    assert report.headers["content-encoding"] == "gzip"

    def unavailable(*args):
        raise ResultStorageError("No object store")

    monkeypatch.setattr(app.state.service.results, "get", unavailable)
    assert client.get("/v1/scans", headers=admin).status_code == 200
    assert client.get("/v1/findings", headers=admin).status_code == 200
    assert client.get("/v1/scans/" + scan["id"], headers=admin).status_code == 503
    assert len(provider.calls) == 1  # Object read failures never rerun inference.


def test_failed_object_write_never_marks_completed(configured, monkeypatch):
    client, admin, app, _, _, _ = configured
    scan = client.post("/v1/scans", headers=admin, json={"text": TEXT}).json()

    def fail(*args):
        raise ResultStorageError("Storage unavailable")

    monkeypatch.setattr(app.state.service.results, "put", fail)
    asyncio.run(app.state.service.process_one())
    row = app.state.db.one("SELECT status,result,result_storage,error FROM scans WHERE id=?", (scan["id"],))
    assert row["status"] == "failed" and row["result"] is None and row["result_storage"] is None
    assert json.loads(row["error"])["code"] == "result_storage_failed"


def test_resumable_migration_and_legacy_compatibility(configured):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts/migrate_result_storage.py"
    spec = importlib.util.spec_from_file_location("migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client, admin, app, _, _, _ = configured
    scan = client.post("/v1/scans", headers=admin, json={"text": TEXT}).json()
    original = sample()
    app.state.db.execute(
        "UPDATE scans SET status='completed',result=? WHERE id=?", (json.dumps(original), scan["id"])
    )
    assert (
        client.get("/v1/scans/" + scan["id"], headers=admin).json()["result"]["tokens"] == original["tokens"]
    )
    stats = module.migrate(app.state.db, app.state.service.results, apply=False)
    assert stats["reports"] == 1
    assert (
        app.state.db.one("SELECT result_storage FROM scans WHERE id=?", (scan["id"],))["result_storage"]
        is None
    )
    stats = module.migrate(app.state.db, app.state.service.results, apply=True)
    assert stats["reports"] == 1 and stats["compressed_bytes"] < stats["original_json_bytes"]
    assert module.migrate(app.state.db, app.state.service.results, apply=True)["reports"] == 0
    assert (
        client.get("/v1/scans/" + scan["id"], headers=admin).json()["result"]["tokens"] == original["tokens"]
    )
