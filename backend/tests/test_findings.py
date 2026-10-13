import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from .conftest import TEXT
from open_text_detector_backend.findings import excerpt


def prepare(configured, tmp_path):
    c, admin, app, provider, model, payload = configured
    app.state.findings.root = tmp_path
    folder = tmp_path / 'iclr_2023'
    folder.mkdir()
    (folder / 'papers.jsonl').write_text(json.dumps({'id': 'paper', 'title': 'A paper', 'text': TEXT}) + '\n')
    return c, admin, app, provider, model, payload


def test_existing_report_reused(configured, tmp_path):
    c, admin, app, provider, model, _ = prepare(configured, tmp_path)
    scan = c.post('/v1/scans', headers=admin, json={'text': TEXT}).json()
    asyncio.run(app.state.service.process_one())
    assert c.get('/v1/findings').status_code == 401
    results = c.get('/v1/findings', headers=admin).json()
    assert results['total'] == 1 and results['items'][0]['model']['id'] == model['id']
    assert 'text' not in results['items'][0]
    assert 'secret' not in results['items'][0]['model']
    assert c.get('/v1/findings?model_id=missing', headers=admin).json()['total'] == 0
    response = c.post('/v1/finding-datasets/iclr_2023/0/compute', headers=admin, json={'model_id': model['id']}).json()
    assert response['reused'] and response['scan']['id'] == scan['id']
    assert len(provider.calls) == 1


def test_concurrent_reuse_model_changes_and_trash(configured, tmp_path):
    c, admin, app, provider, model, payload = prepare(configured, tmp_path)
    def compute():
        return c.post('/v1/finding-datasets/iclr_2023/0/compute', headers=admin, json={'model_id': model['id']})
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda _: compute(), range(4)))
    assert all(r.status_code == 202 for r in responses)
    ids = {r.json()['scan']['id'] for r in responses}
    assert len(ids) == 1
    asyncio.run(app.state.service.process_one())
    assert len(provider.calls) == 1
    payload['name'] = 'Renamed'
    c.put('/v1/models/' + model['id'], headers=admin, json=payload)
    assert compute().json()['scan']['id'] in ids
    payload['upper_threshold'] = .9
    c.put('/v1/models/' + model['id'], headers=admin, json=payload)
    second = compute().json()['scan']['id']
    assert second not in ids
    app.state.db.execute("UPDATE scans SET deleted_at='deleted' WHERE id=?", (second,))
    assert compute().status_code == 409


def test_catalogue_and_excerpt(configured, tmp_path):
    c, admin, app, *_ = prepare(configured, tmp_path)
    assert c.get('/v1/finding-datasets', headers=admin).json()['items'][0]['count'] == 1
    data = c.get('/v1/finding-datasets/iclr_2023?q=paper', headers=admin).json()
    assert data['total'] == 1 and 'offset' not in data['items'][0]
    assert c.get('/v1/finding-datasets/iclr_2023?offset=1', headers=admin).json()['items'] == []
    assert c.get('/v1/finding-datasets/unknown', headers=admin).status_code == 404
    text, shortened = excerpt({'text': 'word ' * 3000}, 'human_pg19')
    assert shortened and len(text.split()) == 2000
