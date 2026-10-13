"""Transport integration: real HTTP sockets, no provider mocking or model weights."""
import asyncio
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from fastapi.testclient import TestClient
from open_text_detector_backend.config import Settings
from open_text_detector_backend.main import create_app
from open_text_detector_backend.network import request_bytes, NetworkError
from .conftest import TEXT
import pytest


@contextmanager
def classifier_server():
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append({"body": body, "authorization": self.headers.get("Authorization")})
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "http://127.0.0.1:1/private")
                self.end_headers()
                return
            payload = json.dumps({"score": .87, "segments": [{"start": 0, "end": len(body["text"]), "score": .87}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_real_http_classification_and_secret_forwarding(tmp_path):
    with classifier_server() as (endpoint, calls):
        app = create_app(Settings(data_dir=tmp_path, worker_enabled=False, allow_local_model_endpoints=True, _env_file=None))
        with TestClient(app) as c:
            admin = {"Authorization": "Bearer " + (tmp_path / "admin.key").read_text()}
            model = c.post("/v1/models", headers=admin, json={"name": "Local transport test", "provider": "http", "model_id": "test-classifier", "endpoint": endpoint + "/classify", "api_key": "fixture-secret", "enabled": True})
            assert model.status_code == 201, model.text
            scan = c.post("/v1/scans", headers=admin, json={"text": TEXT, "model_id": model.json()["id"]}).json()
            assert asyncio.run(app.state.service.process_one())
            result = c.get("/v1/scans/" + scan["id"], headers=admin).json()
            assert result["status"] == "completed", result
            assert result["result"]["score"] == .87 and result["result"]["label"] == "ai"
            assert calls[0]["authorization"] == "Bearer fixture-secret"
            assert calls[0]["body"]["model"] == "test-classifier"
            assert calls[0]["body"]["text"] == TEXT
            assert "fixture-secret" not in json.dumps(result)


def test_provider_redirect_does_not_forward_credentials():
    with classifier_server() as (endpoint, calls):
        with pytest.raises(NetworkError, match="redirects"):
            request_bytes(endpoint + "/redirect", payload={"text": TEXT}, token="fixture-secret", model_endpoint=True, allow_local=True)
        assert len(calls) == 1
