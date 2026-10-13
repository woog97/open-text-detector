import io
import math
import socket
import pytest
from reportlab.pdfgen import canvas
from open_text_detector_backend.network import resolve_target, NetworkError
from open_text_detector_backend.documents import extract_document, DocumentError
from open_text_detector_backend.providers import normalize_prediction, ProviderError
from open_text_detector_backend.reports import safe_csv
from open_text_detector_backend.config import Settings
from open_text_detector_backend.main import create_app
from fastapi.testclient import TestClient


@pytest.mark.parametrize("url", ["http://127.0.0.1", "http://[::1]", "http://169.254.169.254/latest/meta-data", "http://10.0.0.1", "http://192.168.1.1", "file:///etc/passwd", "https://user:password@example.com"])
def test_ssrf_blocked(url):
    with pytest.raises(NetworkError):
        resolve_target(url)


def test_local_model_requires_explicit_opt_in():
    with pytest.raises(NetworkError):
        resolve_target("http://127.0.0.1:9000/classify", model_endpoint=True)
    _, ip, port = resolve_target("http://127.0.0.1:9000/classify", model_endpoint=True, allow_local=True)
    assert ip == "127.0.0.1" and port == 9000


def test_dns_resolution_rejects_mixed_public_and_private(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("8.8.8.8", 443)), (2, 1, 6, "", ("127.0.0.1", 443))])
    with pytest.raises(NetworkError):
        resolve_target("https://example.com")


@pytest.mark.parametrize("prediction", [{"score": math.nan}, {"score": 1.2}, {"score": True}, {"score": .5, "segments": [{"start": 0, "end": 999, "score": .4}]}, {"score": .5, "segments": [{"start": 0, "end": 3, "score": .4}, {"start": 2, "end": 4, "score": .5}]}])
def test_invalid_model_output_rejected(prediction):
    with pytest.raises(ProviderError):
        normalize_prediction(prediction, "test text", {"lower_threshold": .2, "upper_threshold": .8})


def test_pdf_extraction_and_invalid_upload():
    stream = io.BytesIO()
    pdf = canvas.Canvas(stream)
    pdf.drawString(50, 700, "This is a readable PDF for classification.")
    pdf.save()
    assert "readable PDF" in extract_document("test.pdf", stream.getvalue())
    assert "hello" in extract_document("test.rtf", b"{\\rtf1\\ansi hello}")
    with pytest.raises(DocumentError):
        extract_document("test.pdf", b"not a pdf")
    with pytest.raises(DocumentError):
        extract_document("test.txt", b"x" * 100, 50)


def test_csv_formula_injection_escaped():
    assert safe_csv("=IMPORTDATA('bad')").startswith("'")
    assert safe_csv(" +SUM(A1:A2)").startswith("'")
    assert safe_csv("normal title") == "normal title"


def test_request_size_and_rate_limit(tmp_path):
    app = create_app(Settings(data_dir=tmp_path, worker_enabled=False, max_request_bytes=256, requests_per_minute=1, _env_file=None))
    with TestClient(app) as c:
        headers = {"Authorization": "Bearer " + (tmp_path / "admin.key").read_text()}
        assert c.post("/v1/scans", content=b"x" * 257, headers=headers).status_code == 413
        assert c.get("/v1/models", headers=headers).status_code == 200
        assert c.get("/v1/models", headers=headers).status_code == 429


def test_editlens_preprocessing_and_original_offsets():
    from open_text_detector_backend.providers.preprocess import preprocess
    original = "<think>ignored</think>Sure, here's an answer:\n\nHello WORLD!  👋\nNext line."
    cleaned, spans = preprocess(original)
    assert cleaned == "hello world! :waving_hand: next line."
    assert len(spans) == len(cleaned)
    assert original[spans[0][0]:spans[-1][1]] == "Hello WORLD!  👋\nNext line."
    emoji_index = cleaned.index(":waving_hand:")
    assert original[slice(*spans[emoji_index])] == "👋"
    assert preprocess("   \n \t ") == ("", [])
    assert preprocess("Title only")[0] == "title only"


def test_hugging_face_secret_loaded_without_exposure(tmp_path):
    env = tmp_path / '.env'
    env.write_text('HF_TOKEN=hf_fixture_only\n')
    settings = Settings(_env_file=env)
    assert settings.hf_token.get_secret_value() == 'hf_fixture_only'
    assert 'hf_fixture_only' not in repr(settings)
