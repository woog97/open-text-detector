"""Stage a complete resumable MELD queue and verified outputs using local R2 auth."""
import concurrent.futures
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from open_text_detector_backend.result_codec import decode, encode
from upload_paper_pdfs import BASE, remote_objects, request

OUT = ROOT / 'research/benchmarks/meld-cuda'
STAGE = OUT / 'r2-batch'
STAGE.mkdir(exist_ok=True)
profile_path = ROOT / 'models/meld-v5/cuda-runtime-profile.json'
profile_sha = hashlib.sha256(profile_path.read_bytes()).hexdigest()
index = sqlite3.connect(ROOT / 'research/extractions/positioned/index.sqlite3')
index.row_factory = sqlite3.Row
rows = [dict(r) for r in index.execute('SELECT * FROM artifacts')]
db = sqlite3.connect(f'file:{ROOT}/backend/.data/workspace.sqlite3?mode=ro', uri=True)
completed_texts = {hashlib.sha256(t.encode()).hexdigest() for (t,) in db.execute("SELECT text FROM scans WHERE status='completed'")}
production = json.loads((OUT / 'production-summary.json').read_text())
completed_texts.update(x['text_sha256'] for x in production['items'])
completed_pdfs = {r['pdf_sha256'] for r in rows if r['text_sha256'] in completed_texts}
remote = remote_objects()
queue = []
uploads = {}


def extraction(row):
    key = f"extractions/positioned/{row['pdf_sha256']}/{row['text_sha256']}.pgf"
    uploads[key] = {'path': str(ROOT / row['path']), 'sha256': row['blob_sha256'], 'bytes': row['bytes']}
    return {'pdf_sha256': row['pdf_sha256'], 'text_sha256': row['text_sha256'],
            'r2_key': key, 'blob_sha256': row['blob_sha256'], 'bytes': row['bytes'],
            'mapping_status': row['mapping_status'], 'coverage': row['coverage']}


for r in sorted(rows, key=lambda r: (r['pdf_sha256'], r['text_sha256'])):
    if r['kind'] != 'canonical' or r['pdf_sha256'] in completed_pdfs:
        continue
    if f"papers/{r['pdf_sha256']}.pdf" not in remote:
        continue
    queue.append(extraction(r))

by_pair = {(r['pdf_sha256'], r['text_sha256']): r for r in rows}
result_items = []
for item in production['items']:
    result = decode((ROOT / item['path']).read_bytes())
    refs = []
    for ref in result['source_maps']:
        refs.append(extraction(by_pair[(ref['pdf_sha256'], ref['text_sha256'])]))
    result['source_maps'] = refs
    blob = encode(result, level=3)
    path = STAGE / (item['text_sha256'] + '.pgf')
    path.write_bytes(blob)
    assert decode(blob) == result
    key = f"classifications/meld-v5/{profile_sha}/{item['text_sha256']}.pgf"
    entry = {'path': str(path), 'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}
    uploads[key] = entry
    result_items.append({k: v for k, v in item.items() if k != 'path'} | {'r2_key': key, 'sha256': entry['sha256'], 'bytes': len(blob)})

print(json.dumps({'queued': len(queue), 'new_results': len(result_items), 'objects_to_verify': len(uploads), 'bytes': sum(x['bytes'] for x in uploads.values())}), flush=True)
(STAGE / 'upload-plan.json').write_text(json.dumps(uploads))
ledger = {}


def upload(pair):
    key, info = pair
    raw = Path(info['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != info['sha256']:
        raise ValueError('Local artifact checksum mismatch')
    md5 = hashlib.md5(raw).hexdigest()
    old = remote.get(key, {})
    if old.get('size') != len(raw) or old.get('etag', '').strip('"') != md5:
        request(BASE + '/' + key, raw, 'application/octet-stream')
    return key, {'sha256': info['sha256'], 'bytes': len(raw), 'md5': md5}


with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    for n, (key, info) in enumerate(pool.map(upload, uploads.items()), 1):
        ledger[key] = info
        if n % 100 == 0 or n == len(uploads):
            (STAGE / 'upload-ledger.json').write_text(json.dumps(ledger))
            print(f'{n}/{len(uploads)} objects staged', flush=True)
verified = remote_objects()
for key, info in ledger.items():
    obj = verified[key]
    assert obj['size'] == info['bytes'] and obj['etag'].strip('"') == info['md5'], key
queue_doc = {'format': 'meld-r2-queue-v1', 'model': 'meld-v5', 'profile_sha256': profile_sha,
             'created_at': time.time(), 'papers': queue, 'count': len(queue),
             'completed_pdf_count': len(completed_pdfs), 'result_prefix': f'classifications/meld-v5/{profile_sha}/'}
raw = json.dumps(queue_doc, separators=(',', ':')).encode()
queue_sha = hashlib.sha256(raw).hexdigest()
queue_key = f'queues/meld-v5/{queue_sha}.json'
request(BASE + '/' + queue_key, raw, 'application/json')
(STAGE / 'queue.json').write_bytes(raw)
results_doc = {'format': 'meld-r2-results-v1', 'profile_sha256': profile_sha, 'count': len(result_items), 'items': result_items}
raw_results = json.dumps(results_doc, separators=(',', ':')).encode()
results_key = f'classification-indexes/meld-v5/{hashlib.sha256(raw_results).hexdigest()}.json'
request(BASE + '/' + results_key, raw_results, 'application/json')
request(BASE + f'/runtime-profiles/meld-v5/{profile_sha}.json', profile_path.read_bytes(), 'application/json')
summary = {'queue_count': len(queue), 'new_results': len(result_items), 'queue_key': queue_key,
           'queue_sha256': queue_sha, 'results_index_key': results_key, 'objects_verified': len(ledger),
           'bytes_verified': sum(x['bytes'] for x in ledger.values()), 'profile_sha256': profile_sha}
(STAGE / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
