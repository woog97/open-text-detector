from io import BytesIO
from reportlab.pdfgen import canvas
from open_text_detector_backend.pdf_reader import PDFReader


def test_reader_access_range_and_containment(workspace, tmp_path):
    client, auth, app, _ = workspace
    root = tmp_path / 'papers'
    root.mkdir()
    data = BytesIO()
    pdf = canvas.Canvas(data)
    pdf.drawString(40, 700, 'A real PDF for the local reader.')
    pdf.save()
    (root / 'paper.pdf').write_bytes(data.getvalue())
    (root / 'outside.pdf').symlink_to(tmp_path / 'admin.key')
    app.state.pdf_reader = PDFReader(app.state.service, root)
    assert client.get('/v1/pdf-reader').status_code == 401
    items = client.get('/v1/pdf-reader', headers=auth).json()['items']
    assert len(items) == 1
    key = items[0]['id']
    assert not items[0]['classified']
    assert client.get(f'/v1/pdf-reader/{key}', headers=auth).json()['reports'] == []
    response = client.get(f'/v1/pdf-reader/{key}/file', headers={**auth, 'Range': 'bytes=0-9'})
    assert response.status_code == 206
    assert response.content == data.getvalue()[:10]
    assert response.headers['content-type'] == 'application/pdf'
    assert response.headers['content-encoding'] == 'identity'
    assert client.get('/v1/pdf-reader/not-a-key/file', headers=auth).status_code == 404
    # Replacing a known file with a symlink must not expand its access.
    (root / 'paper.pdf').unlink()
    (root / 'paper.pdf').symlink_to(tmp_path / 'admin.key')
    assert client.get(f'/v1/pdf-reader/{key}/file', headers=auth).status_code == 404
