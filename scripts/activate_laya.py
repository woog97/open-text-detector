"""Register Laya as an enabled alternative; preserve the current default unless requested."""
import argparse
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--default', action='store_true')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    args = parser.parse_args()
    from urllib.parse import urlsplit
    if urlsplit(args.base_url).hostname not in {'localhost', '127.0.0.1', '::1'}:
        raise ValueError('Owner credentials may only be sent to the local backend')
    token = (ROOT / 'backend/.data/admin.key').read_text().strip()
    def api(path, method='GET', data=None):
        request = urllib.request.Request(args.base_url + '/v1/' + path,
            data=json.dumps(data).encode() if data is not None else None, method=method,
            headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    payload = {'name': 'Laya · Experimental contextual phrases', 'provider': 'laya', 'task': 'text',
               'model_id': 'convaiinnovations/laya', 'enabled': True}
    existing = next((m for m in api('models')['items'] if m['model_id'] == payload['model_id']), None)
    model = api('models/' + existing['id'], 'PUT', payload) if existing else api('models', 'POST', payload)
    if args.default:
        api('settings/default-model', 'PUT', {'model_id': model['id']})
    print(json.dumps({'model': model['name'], 'id': model['id'], 'made_default': args.default}))


if __name__ == '__main__':
    main()
