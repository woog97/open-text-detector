import asyncio
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from docx import Document
from .conftest import TEXT
from open_text_detector_backend.providers import ProviderError
from open_text_detector_backend.service import ScanService
from open_text_detector_backend.main import create_app
from fastapi.testclient import TestClient


def submit(client, admin, **kwargs):
    response = client.post("/v1/scans", json={"text": TEXT, **kwargs}, headers=admin)
    assert response.status_code == 202, response.text
    return response.json()


def complete(app):
    return asyncio.run(app.state.service.process_one())


def test_private_by_default_and_pending_model(workspace):
    c, admin, app, provider = workspace
    assert c.get("/health").json()["status"] == "ok"
    assert c.get("/v1/models").status_code == 401
    assert c.get("/v1/models", headers={"Authorization": "Bearer invalid"}).status_code == 401
    model = c.get("/v1/models", headers=admin).json()["items"][0]
    assert model["model_id"] == "pangram/editlens_Llama-3.2-3B"
    assert model["enabled"] is False
    response = c.post("/v1/scans", json={"text": TEXT}, headers=admin)
    assert response.status_code == 409 and response.json()["detail"]["code"] == "model_disabled"
    assert provider.calls == []


def test_model_secret_encrypted_redacted_and_preserved(configured):
    c, admin, app, _, model, payload = configured
    row = app.state.db.one("SELECT * FROM models WHERE id=?", (model["id"],))
    assert row["secret"] != payload["api_key"]
    assert app.state.db.decrypt(row["secret"]) == payload["api_key"]
    assert "secret" not in model and "api_key" not in model and model["has_api_key"]
    payload.pop("api_key")
    payload["name"] = "Renamed model"
    response = c.put("/v1/models/" + model["id"], json=payload, headers=admin)
    assert response.status_code == 200 and response.json()["has_api_key"]
    payload["clear_api_key"] = True
    assert not c.put("/v1/models/" + model["id"], json=payload, headers=admin).json()["has_api_key"]


def test_validation_never_echoes_secret(configured):
    c, admin, _, _, _, payload = configured
    payload["upper_threshold"] = .1
    response = c.post("/v1/models", json=payload, headers=admin)
    assert response.status_code == 422
    assert "upstream-secret-for-testing" not in response.text


def test_model_snapshot_history_notes_reports(configured):
    c, admin, app, provider, model, payload = configured
    row = submit(c, admin, title="Initial report")
    payload["name"] = "Changed name"
    payload["upper_threshold"] = .6
    c.put("/v1/models/" + model["id"], json=payload, headers=admin)
    assert complete(app)
    report = c.get("/v1/scans/" + row["id"], headers=admin).json()
    assert report["status"] == "completed" and report["result"]["label"] == "ai_assisted"
    assert report["model"]["name"] == "Test HTTP classifier"
    assert report["result"]["segments"][0]["end"] == len(TEXT)
    assert "upstream-secret-for-testing" not in json.dumps(report)
    assert c.patch("/v1/scans/" + row["id"], json={"notes": "Reviewed", "feedback": "helpful"}, headers=admin).json()["notes"] == "Reviewed"
    assert c.get("/v1/scans?q=Initial&status=completed", headers=admin).json()["total"] == 1
    assert c.get("/v1/reports/" + row["id"] + "?format=pdf", headers=admin).content.startswith(b"%PDF")
    assert "Initial report" in c.get("/v1/reports/" + row["id"] + "?format=csv", headers=admin).text
    assert c.get("/v1/reports/" + row["id"], headers=admin).json()["id"] == row["id"]


def test_idempotency_original_result_and_conflict(configured):
    c, admin, app, _, model, _ = configured
    headers = admin | {"Idempotency-Key": "sample-one"}
    a = c.post("/v1/scans", json={"text": TEXT}, headers=headers)
    b = c.post("/v1/scans", json={"text": TEXT}, headers=headers)
    assert a.json()["id"] == b.json()["id"]
    assert c.post("/v1/scans", json={"text": TEXT + " changed"}, headers=headers).status_code == 409
    # A default change must not retarget the original idempotent request.
    c.put("/v1/settings/default-model", json={"model_id": "open-pangram-llama"}, headers=admin)
    assert c.post("/v1/scans", json={"text": TEXT}, headers=headers).json()["id"] == a.json()["id"]
    assert c.get("/v1/scans", headers=admin).json()["total"] == 1


def test_batch_transaction_validation_progress(configured):
    c, admin, app, *_ = configured
    response = c.post("/v1/batches", json={"documents": [{"text": TEXT}, {"text": "too short"}]}, headers=admin)
    assert response.status_code == 422
    assert c.get("/v1/scans", headers=admin).json()["total"] == 0
    batch = c.post("/v1/batches", json={"documents": [{"text": TEXT}, {"text": TEXT}]}, headers=admin).json()
    complete(app)
    progress = c.get("/v1/batches/" + batch["id"], headers=admin).json()
    assert progress["counts"]["completed"] == 1 and not progress["finished"]
    complete(app)
    assert c.get("/v1/batches/" + batch["id"], headers=admin).json()["finished"]


def test_scoped_keys_revocation_and_audit(configured):
    c, admin, *_ = configured
    key = c.post("/v1/keys", json={"name": "Read only", "scopes": ["read"]}, headers=admin).json()
    reader = {"Authorization": "Bearer " + key["key"]}
    assert c.get("/v1/scans", headers=reader).status_code == 200
    assert c.post("/v1/scans", json={"text": TEXT}, headers=reader).status_code == 403
    assert c.get("/v1/keys", headers=reader).status_code == 403
    assert key["key"] not in c.get("/v1/keys", headers=admin).text
    c.delete("/v1/keys/" + key["id"], headers=admin)
    assert c.get("/v1/scans", headers=reader).status_code == 401
    assert c.get("/v1/audit", headers=admin).json()["items"][0]["action"] == "key.revoked"


def test_fail_retry_cancel_and_trash(configured):
    c, admin, app, provider, *_ = configured
    provider.error = ProviderError("provider_unavailable", "Unavailable")
    row = submit(c, admin)
    complete(app)
    failed = c.get("/v1/scans/" + row["id"], headers=admin).json()
    assert failed["status"] == "failed" and failed["result"] is None
    retry = c.post("/v1/scans/" + row["id"] + "/retry", headers=admin).json()
    assert retry["id"] != row["id"] and retry["status"] == "queued"
    assert c.post("/v1/scans/" + retry["id"] + "/cancel", headers=admin).json()["status"] == "cancelled"
    assert not complete(app)
    c.delete("/v1/scans/" + row["id"], headers=admin)
    assert c.get("/v1/scans/" + row["id"], headers=admin).status_code == 404
    assert c.get("/v1/scans?trash=true", headers=admin).json()["total"] == 1
    assert c.post("/v1/scans/" + row["id"] + "/restore", headers=admin).json()["status"] == "failed"


def test_persistence_across_restart(configured):
    c, admin, app, provider, *_ = configured
    row = submit(c, admin)
    replacement = create_app(app.state.settings, provider)
    with TestClient(replacement) as other:
        assert other.get("/v1/scans/" + row["id"], headers=admin).json()["status"] == "queued"
        assert complete(replacement)
    assert c.get("/v1/scans/" + row["id"], headers=admin).json()["status"] == "completed"


def test_exclusive_worker_claim_and_expired_lease(configured):
    c, admin, app, provider, *_ = configured
    row = submit(c, admin)
    second = ScanService(app.state.db, app.state.settings, provider)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda svc: svc.claim(), [app.state.service, second]))
    assert sum(r is not None for r in results) == 1
    assert not second.claim()
    app.state.db.execute("UPDATE scans SET lease_until=? WHERE id=?", (time.time()-1, row["id"]))
    assert second.claim()["id"] == row["id"]


def test_share_revocation_expiry_and_private_fields(configured):
    c, admin, app, *_ = configured
    row = submit(c, admin)
    complete(app)
    c.patch("/v1/scans/" + row["id"], json={"notes": "Private reviewer notes"}, headers=admin)
    share = c.post("/v1/scans/" + row["id"] + "/shares", json={}, headers=admin).json()
    response = c.get(share["path"])
    assert response.status_code == 200
    assert "Private reviewer notes" not in response.text and "endpoint" not in response.text
    assert response.headers["Cache-Control"] == "no-store"
    c.delete("/v1/shares/" + share["id"], headers=admin)
    assert c.get(share["path"]).status_code == 404
    share = c.post("/v1/scans/" + row["id"] + "/shares", json={}, headers=admin).json()
    app.state.db.execute("UPDATE shares SET expires_at=? WHERE id=?", (time.time()-1, share["id"]))
    assert c.get(share["path"]).status_code == 404
    share = c.post("/v1/scans/" + row["id"] + "/shares", json={}, headers=admin).json()
    c.delete("/v1/scans/" + row["id"], headers=admin)
    c.post("/v1/scans/" + row["id"] + "/restore", headers=admin)
    assert c.get(share["path"]).status_code == 404


def test_mixed_upload_batch_and_image_provider(configured):
    c, admin, app, _, _, _ = configured
    document, stream = Document(), io.BytesIO()
    document.add_paragraph(TEXT)
    document.save(stream)
    response = c.post("/v1/uploads", headers=admin, files=[("files", ("good.docx", stream.getvalue())), ("files", ("bad.exe", b"no"))])
    assert response.status_code == 202, response.text
    assert len(response.json()["items"]) == 1 and len(response.json()["errors"]) == 1
    image = Image.new("RGB", (512, 512), "white")
    image_data = io.BytesIO()
    image.save(image_data, format="PNG")
    assert c.post("/v1/images", files={"file": ("image.png", image_data.getvalue())}, headers=admin).status_code == 409
    model = c.post("/v1/models", json={"name": "Image test", "model_id": "image-model", "provider": "http", "task": "image", "endpoint": "https://classifier.example/image", "enabled": True}, headers=admin).json()
    response = c.post("/v1/images", data={"model_id": model["id"]}, files={"file": ("image.png", image_data.getvalue())}, headers=admin)
    assert response.status_code == 202, response.text
    complete(app)
    complete(app)
    result = c.get("/v1/scans/" + response.json()["id"], headers=admin).json()
    assert result["status"] == "completed" and result["kind"] == "image"
    assert "upload_path" not in result


def test_reference_corpus_scope(configured):
    c, admin, app, *_ = configured
    source = c.post("/v1/corpus", json={"title": "Reference", "text": TEXT}, headers=admin).json()
    row = submit(c, admin, check_plagiarism=True)
    complete(app)
    result = c.get("/v1/scans/" + row["id"], headers=admin).json()["result"]["plagiarism"]
    assert result["scope"] == "workspace_corpus_only"
    assert result["matched_word_fraction"] == 1
    assert result["matches"][0]["source_id"] == source["id"]
