import pytest
from fastapi.testclient import TestClient
from open_text_detector_backend.config import Settings
from open_text_detector_backend.main import create_app
from open_text_detector_backend.providers import normalize_prediction

TEXT = " ".join(f"word{i}" for i in range(70))


class TestProvider:
    __test__ = False
    def __init__(self):
        self.calls = []
        self.error = None

    def classify(self, model, text, image=None):
        self.calls.append((model, text, image))
        if self.error:
            raise self.error
        return normalize_prediction({"score": .65, "segments": [{"start": 0, "end": len(text), "score": .65}] if text else []}, text, model)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr("open_text_detector_backend.main.resolve_target", lambda *args, **kwargs: None)
    provider = TestProvider()
    settings = Settings(data_dir=tmp_path, worker_enabled=False, requests_per_minute=10000, _env_file=None)
    app = create_app(settings, providers=provider)
    with TestClient(app) as client:
        admin = {"Authorization": "Bearer " + (tmp_path / "admin.key").read_text()}
        yield client, admin, app, provider


@pytest.fixture
def configured(workspace):
    client, admin, app, provider = workspace
    payload = {"name": "Test HTTP classifier", "model_id": "test-v1", "provider": "http",
               "endpoint": "https://classifier.example/classify", "enabled": True,
               "api_key": "upstream-secret-for-testing"}
    response = client.post("/v1/models", json=payload, headers=admin)
    assert response.status_code == 201, response.text
    model = response.json()
    assert client.put("/v1/settings/default-model", json={"model_id": model["id"]}, headers=admin).status_code == 200
    return client, admin, app, provider, model, payload

