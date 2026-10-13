"""Upload built Atlas detail objects through the token-gated atlas-upload worker.

Reads OUT/index/*.parquet from build_atlas_details.py and PUTs each content-addressed
object to <url>/<detail_key>, verifying the acknowledged size and MD5 ETag. Resumable:
uploaded keys are appended to OUT/uploaded.txt. Access comes from the ATLAS_UPLOAD_URL and
ATLAS_UPLOAD_TOKEN environment variables and is never printed.
Usage: upload_atlas_details.py OUT [--workers 32]
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

URL = os.environ['ATLAS_UPLOAD_URL']; TOKEN = os.environ['ATLAS_UPLOAD_TOKEN']


def put(key, path):
    raw = Path(path).read_bytes()
    for attempt in range(8):
        try:
            req = urllib.request.Request(f'{URL}/{key}', data=raw, method='PUT',
                                         headers={'Authorization': 'Bearer ' + TOKEN, 'User-Agent': 'open-text-detector-atlas-publisher/1.0'})
            with urllib.request.urlopen(req, timeout=180) as r: ack = json.loads(r.read())
            if ack['size'] != len(raw) or ack['etag'].strip('"') != hashlib.md5(raw).hexdigest(): raise ValueError('Upload acknowledgement mismatch')
            return key
        except Exception as e:
            if attempt == 7: raise RuntimeError(f'{key}: {type(e).__name__}') from None
            time.sleep(min(2 ** attempt, 30))


def main():
    out = Path(sys.argv[1]); workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 32
    import pyarrow.parquet as pq
    keys = set()
    for f in sorted((out / 'index').glob('index-*.parquet')):
        for r in pq.read_table(f, columns=['detail_key']).to_pylist():
            if r['detail_key']: keys.add(r['detail_key'])
    ledger = out / 'uploaded.txt'
    done = set(ledger.read_text().split()) if ledger.exists() else set()
    todo = sorted(keys - done); started = time.time(); n = 0; failures = []
    print(json.dumps({'objects': len(keys), 'already_uploaded': len(done & keys), 'to_upload': len(todo)}), flush=True)
    with ThreadPoolExecutor(workers) as pool, ledger.open('a') as log:
        futures = {pool.submit(put, k, out / 'objects' / k.rsplit('/', 1)[1]): k for k in todo}
        for fut in as_completed(futures):
            try: log.write(fut.result() + '\n'); log.flush(); n += 1
            except Exception as e: failures.append(str(e))
            if (n + len(failures)) % 1000 == 0:
                print(json.dumps({'uploaded': n, 'failed': len(failures), 'seconds': round(time.time() - started)}), flush=True)
    print(json.dumps({'state': 'complete' if not failures else 'partial', 'uploaded': n, 'failed': len(failures), 'failures': failures[:20],
                      'seconds': round(time.time() - started)}), flush=True)


if __name__ == '__main__':
    main()
