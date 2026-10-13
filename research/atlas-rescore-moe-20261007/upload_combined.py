"""Upload the combined (4B + MoE) Atlas detail objects to R2 in bundles (runs on the Space).

Same transport as scripts/upload_atlas_bundles.py: 40-object bundles POSTed to the atlas-upload Worker /bundle endpoint,
which writes each member to R2 (atlas-public/objects/<sha256>.json.gz, content-addressed, so nothing existing is
overwritten with different content). Every acknowledged size and MD5 is checked; acknowledged keys go to a ledger so
the upload resumes. Then 300 random objects are read back through the Worker and byte-compared.
Access: ATLAS_UPLOAD_URL / ATLAS_UPLOAD_TOKEN environment variables (never printed).
"""
import hashlib, json, os, random, struct, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUT = Path('/tmp/atlas-rescore/details-combined'); LEDGER = OUT / 'uploaded.txt'; STATUS = OUT / 'upload-status.json'
URL = os.environ['ATLAS_UPLOAD_URL']; TOKEN = os.environ['ATLAS_UPLOAD_TOKEN']


def status(**k): STATUS.write_text(json.dumps({**k, 'time': time.time()})); print(json.dumps(k), flush=True)


def req(path, data=None, method='GET'):
    r = urllib.request.Request(f'{URL}/{path}', data=data, method=method, headers={'Authorization': 'Bearer ' + TOKEN, 'User-Agent': 'open-text-detector-atlas-publisher/1.0'})
    with urllib.request.urlopen(r, timeout=300) as resp: return resp.read()


def send(group):
    bodies = [(OUT / 'objects' / k.rsplit('/', 1)[1]).read_bytes() for k in group]
    entries = []; off = 0
    for k, b in zip(group, bodies): entries.append({'key': k, 'offset': off, 'length': len(b)}); off += len(b)
    h = json.dumps(entries).encode(); payload = struct.pack('>I', len(h)) + h + b''.join(bodies)
    for attempt in range(6):
        try:
            acks = json.loads(req('bundle', payload, 'POST'))
            md5 = {k: hashlib.md5(b).hexdigest() for k, b in zip(group, bodies)}; size = {k: len(b) for k, b in zip(group, bodies)}
            if len(acks) != len(group) or any(a['size'] != size[a['key']] or a['etag'].strip('"') != md5[a['key']] for a in acks): raise ValueError('ack mismatch')
            return group
        except Exception as e:
            if attempt == 5: raise
            time.sleep(5 * (attempt + 1))


def main():
    import pyarrow.parquet as pq
    idx = pq.read_table(OUT / 'combined-index.parquet').to_pylist()
    bad = [r for r in idx if r.get('error') or not r.get('live_4b_identical')]
    if bad: raise SystemExit(f'{len(bad)} rows with errors or a 4B report that differs from the live one; not uploading')
    keys = sorted({r['detail_key'] for r in idx})
    done = set(LEDGER.read_text().split()) if LEDGER.exists() else set(); todo = [k for k in keys if k not in done]
    bundles = [todo[i:i + 40] for i in range(0, len(todo), 40)]
    status(state='uploading', objects=len(keys), already=len(keys) - len(todo), bundles=len(bundles)); t0 = time.time(); n = 0
    with ThreadPoolExecutor(8) as pool, LEDGER.open('a') as led:
        for fut in as_completed([pool.submit(send, g) for g in bundles]):
            g = fut.result(); led.write('\n'.join(g) + '\n'); led.flush(); n += 1
            if n % 50 == 0: status(state='uploading', bundles_done=n, of=len(bundles), seconds=round(time.time() - t0))
    done = set(LEDGER.read_text().split()); missing = [k for k in keys if k not in done]
    if missing: raise SystemExit(f'{len(missing)} objects not acknowledged')
    sample = random.Random(7).sample(keys, min(300, len(keys))); mism = 0
    for k in sample:
        if req(k) != (OUT / 'objects' / k.rsplit('/', 1)[1]).read_bytes(): mism += 1
    status(state='done', objects=len(keys), acknowledged=len(keys), readback_checked=len(sample), readback_mismatches=mism, seconds=round(time.time() - t0),
           bytes=sum((OUT / 'objects' / k.rsplit('/', 1)[1]).stat().st_size for k in keys))


if __name__ == '__main__':
    main()
