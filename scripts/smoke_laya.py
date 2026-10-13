"""Submit a real local Laya scan and save its completed result for review."""
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
token = (ROOT / 'backend/.data/admin.key').read_text().strip()


def api(path, data=None):
    request = urllib.request.Request('http://127.0.0.1:8000/v1/' + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


model = next(m for m in api('models')['items'] if m['provider'] == 'laya')
text = ('I left the notes on the kitchen table, beside a cold cup of coffee. When I came back, '
        'the page had blown onto the floor. I remembered the experiment differently from my colleague. '
        'We checked the notebook and found that neither account was quite right.\n\n'
        'The results highlight the importance of a comprehensive evaluation methodology. '
        'By combining complementary approaches, the proposed framework addresses multiple challenges '
        'and offers a scalable foundation for future research. These findings warrant further investigation.')
scan = api('scans', {'text': text, 'title': 'Laya MVP verification', 'model_id': model['id']})
for _ in range(60):
    scan = api('scans/' + scan['id'])
    if scan['status'] in {'completed', 'failed'}:
        break
    time.sleep(.5)
if scan['status'] != 'completed':
    raise RuntimeError(scan.get('error') or scan['status'])
r = scan['result']
assert r['inference']['runtime'] == 'mlx'
assert r['localization']['granularity'] == 'phrase'
assert all(text[s['start']:s['end']] for s in r['segments'])
output = ROOT / 'research/benchmarks/laya/smoke-scan.json'
output.write_text(json.dumps(scan, indent=2))
print(json.dumps({'scan_id': scan['id'], 'score': r['score'], 'phrases': len(r['segments']),
                  'inference': r['inference'], 'result': str(output)}))
