"""Register the downloaded model and select it in the local workspace."""
from pathlib import Path
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'DarrenJiaImbue/editlens-qwen3-4b-merged-v3'


def main():
    token = (ROOT / 'backend/.data/admin.key').read_text().strip()
    def api(path, method='GET', data=None):
        request = urllib.request.Request('http://127.0.0.1:8000/v1/' + path,
                    data=json.dumps(data).encode() if data is not None else None,
                    method=method, headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    models = api('models')
    if isinstance(models, dict):
        models = models.get('models', models.get('items', []))
    existing = next((m for m in models if m['model_id'] == MODEL_ID), None)
    payload = {'name': 'EditLens · Qwen3 4B v3', 'provider': 'editlens', 'task': 'text',
               'model_id': MODEL_ID, 'enabled': True, 'lower_threshold': .2, 'upper_threshold': .8}
    if existing:
        model = api('models/' + existing['id'], 'PUT', payload)
    else:
        model = api('models', 'POST', payload)
    api('settings/default-model', 'PUT', {'model_id': model['id']})
    print(json.dumps({'default_model': model['name'], 'id': model['id'], 'enabled': model['enabled']}))


if __name__ == '__main__':
    main()
