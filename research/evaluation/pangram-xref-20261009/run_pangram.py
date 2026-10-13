"""Send each cut paper in papers.json to the Pangram API (model pangram-4) once. Task ids and results are cached
in OUT_DIR/pangram/<id>.json, so a rerun never pays twice. Never prints the key.
Usage: run_pangram.py OUT_DIR [N]   (N: only the first N papers)
"""
import json, sys, time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, build_opener, HTTPRedirectHandler
ROOT = Path(__file__).resolve().parents[3]
BASE, MODEL = 'https://text.external-api.pangram.com', 'pangram-4'

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *a): return None

key = next(l.split('=', 1)[1].strip().strip('"\'') for l in (ROOT / '.env.pangram').read_text().splitlines() if l.startswith('PANGRAM_API_KEY='))
opener = build_opener(NoRedirect())

def call(path, body=None):
    req = Request(BASE + path, data=json.dumps(body).encode() if body is not None else None,
                  headers={'x-api-key': key, 'Content-Type': 'application/json'})
    try:
        with opener.open(req, timeout=60) as r: return json.loads(r.read())
    except HTTPError as e:
        raise RuntimeError(f'HTTP {e.code}: {e.read()[:300].decode(errors="replace").replace(key, "[REDACTED]")}') from None
    except (URLError, TimeoutError) as e:
        raise RuntimeError(f'network error {type(e).__name__}') from None

out = Path(sys.argv[1]); cache = out / 'pangram'; cache.mkdir(exist_ok=True)
papers = json.loads((out / 'papers.json').read_text())['papers'][:int(sys.argv[2]) if len(sys.argv) > 2 else None]
for p in papers:
    f = cache / f"{p['id']}.json"; rec = json.loads(f.read_text()) if f.exists() else {}
    if rec.get('result', {}).get('stage') == 'STAGE_SUCCESS': print(p['bin'], 'cached'); continue
    if 'task_id' not in rec:
        rec = {'task_id': call('/task', {'text': p['text'], 'model': MODEL, 'public_dashboard_link': False})['task_id'],
               'words': p['words'], 'submitted': time.time()}
        f.write_text(json.dumps(rec))
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        r = call('/task/' + quote(rec['task_id'], safe=''))
        if r.get('stage') in ('STAGE_SUCCESS', 'STAGE_FAILED'): break
        time.sleep(3)
    else:
        print(p['bin'], 'poll timeout; task id kept, rerun to resume'); continue
    rec['result'] = r; f.write_text(json.dumps(rec))
    print(p['bin'], r.get('stage'), {k: v for k, v in r.items() if not isinstance(v, (list, dict))}, flush=True)
