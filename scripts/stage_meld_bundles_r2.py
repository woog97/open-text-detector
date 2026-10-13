"""Bundle remaining queue artifacts to avoid per-object Cloudflare API throttling."""
import concurrent.futures
import hashlib
import io
import json
import sqlite3
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from open_text_detector_backend.result_codec import decode, encode
from upload_paper_pdfs import BASE, remote_objects, request

OUT = ROOT / 'research/benchmarks/meld-cuda'
STAGE = OUT / 'r2-batch'
BUNDLES = STAGE / 'bundles'
BUNDLES.mkdir(exist_ok=True)
plan = json.loads((STAGE / 'upload-plan.json').read_text())
remote = remote_objects()
locations = {}
bundles = []


def stage_group(entries):
    group = []
    size = 0
    def flush():
        nonlocal group, size
        if not group:
            return
        path = BUNDLES / f'bundle-{len(bundles):04d}.tar'
        with tarfile.open(path, 'w') as archive:
            for key, info, raw in group:
                member = hashlib.sha256(key.encode()).hexdigest() + '.pgf'
                meta = tarfile.TarInfo(member)
                meta.size = len(raw)
                archive.addfile(meta, io.BytesIO(raw))
        raw_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        bundle_key = f'meld-batches/objects/{raw_sha}.tar'
        bundles.append({'key': bundle_key, 'path': str(path), 'sha256': raw_sha, 'bytes': path.stat().st_size})
        for key, info, raw in group:
            locations[key] = {'bundle_key': bundle_key, 'bundle_sha256': raw_sha,
                              'member': hashlib.sha256(key.encode()).hexdigest()+'.pgf'}
        group = []
        size = 0
    for key, info in entries:
        raw = Path(info['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == info['sha256']
        old = remote.get(key, {})
        if old.get('size') == len(raw) and old.get('etag', '').strip('"') == hashlib.md5(raw).hexdigest():
            locations[key] = {'r2_key': key}
            continue
        if size + len(raw) > 32 * 1024 * 1024:
            flush()
        group.append((key, info, raw))
        size += len(raw)
    flush()


stage_group([(k,v) for k,v in plan.items() if k.startswith('extractions/')])
result_items = []
production = json.loads((OUT / 'production-summary.json').read_text())
profile = ROOT / 'models/meld-v5/cuda-runtime-profile.json'
profile_sha = hashlib.sha256(profile.read_bytes()).hexdigest()
result_entries = []
for item in production['items']:
    key = f"classifications/meld-v5/{profile_sha}/{item['text_sha256']}.pgf"
    obj = decode((ROOT / item['path']).read_bytes())
    refs = []
    for ref in obj['source_maps']:
        old_key = f"extractions/positioned/{ref['pdf_sha256']}/{ref['text_sha256']}.pgf"
        refs.append({'pdf_sha256':ref['pdf_sha256'], 'text_sha256':ref['text_sha256'],
                     'blob_sha256':ref['blob_sha256'], 'bytes':plan[old_key]['bytes'],
                     'mapping_status':ref['mapping_status'], 'coverage':ref['coverage'],
                     **locations[old_key]})
    obj['source_maps'] = refs
    raw = encode(obj, level=3)
    assert decode(raw) == obj
    path = STAGE / (item['text_sha256'] + '.pgf')
    path.write_bytes(raw)
    info = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    result_entries.append((key, info))
    result_items.append({k:v for k,v in item.items() if k!='path'} | {'logical_key':key, 'sha256':info['sha256'], 'bytes':len(raw)})
stage_group(result_entries)
print(json.dumps({'bundles':len(bundles),'upload_bytes':sum(x['bytes'] for x in bundles),'reused_individual_objects':sum('r2_key' in x for x in locations.values())}),flush=True)
(STAGE / 'bundle-plan.json').write_text(json.dumps(bundles,indent=2))
(STAGE / 'artifact-locations.json').write_text(json.dumps(locations))


def upload(info):
    raw = Path(info['path']).read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    old = remote.get(info['key'],{})
    if old.get('size')!=len(raw) or old.get('etag','').strip('"')!=md5:
        request(BASE+'/'+info['key'],raw,'application/x-tar')
    return info | {'md5':md5}


ledger=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for n, info in enumerate(pool.map(upload,bundles),1):
        ledger.append(info)
        (STAGE/'bundle-ledger.json').write_text(json.dumps(ledger))
        print(f'{n}/{len(bundles)} bundles uploaded',flush=True)
verified=remote_objects()
for info in ledger:
    obj=verified[info['key']]
    assert obj['size']==info['bytes'] and obj['etag'].strip('"')==info['md5']

index=sqlite3.connect(ROOT/'research/extractions/positioned/index.sqlite3');index.row_factory=sqlite3.Row
rows=[dict(r) for r in index.execute('SELECT * FROM artifacts')]
db=sqlite3.connect(f'file:{ROOT}/backend/.data/workspace.sqlite3?mode=ro',uri=True)
completed={hashlib.sha256(t.encode()).hexdigest() for (t,) in db.execute("SELECT text FROM scans WHERE status='completed'")}
completed.update(x['text_sha256'] for x in production['items'])
completed_pdfs={r['pdf_sha256'] for r in rows if r['text_sha256'] in completed}
queue=[]
for r in sorted(rows,key=lambda x:(x['pdf_sha256'],x['text_sha256'])):
    if r['kind']!='canonical' or r['pdf_sha256'] in completed_pdfs or f"papers/{r['pdf_sha256']}.pdf" not in verified:
        continue
    key=f"extractions/positioned/{r['pdf_sha256']}/{r['text_sha256']}.pgf"
    queue.append({'pdf_sha256':r['pdf_sha256'],'text_sha256':r['text_sha256'],
                  'blob_sha256':r['blob_sha256'],'bytes':r['bytes'],'mapping_status':r['mapping_status'],
                  'coverage':r['coverage'],**locations[key]})
for item in result_items:
    item.update(locations[item.pop('logical_key')])
queue_doc={'format':'meld-r2-queue-v1','model':'meld-v5','profile_sha256':profile_sha,
           'created_at':time.time(),'papers':queue,'count':len(queue),'completed_pdf_count':len(completed_pdfs)}
raw=json.dumps(queue_doc,separators=(',',':')).encode();queue_sha=hashlib.sha256(raw).hexdigest();queue_key=f'queues/meld-v5/{queue_sha}.json'
request(BASE+'/'+queue_key,raw,'application/json');(STAGE/'queue.json').write_bytes(raw)
results_doc={'format':'meld-r2-results-v1','profile_sha256':profile_sha,'count':len(result_items),'items':result_items}
raw_results=json.dumps(results_doc,separators=(',',':')).encode();results_key=f'classification-indexes/meld-v5/{hashlib.sha256(raw_results).hexdigest()}.json'
request(BASE+'/'+results_key,raw_results,'application/json');(STAGE/'results-index.json').write_bytes(raw_results)
request(BASE+f'/runtime-profiles/meld-v5/{profile_sha}.json',profile.read_bytes(),'application/json')
summary={'queue_count':len(queue),'new_results':len(result_items),'queue_key':queue_key,'queue_sha256':queue_sha,
         'results_index_key':results_key,'bundles_verified':len(ledger),'artifacts_available':len(locations),
         'new_bundle_bytes':sum(x['bytes'] for x in ledger),'profile_sha256':profile_sha}
(STAGE/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
