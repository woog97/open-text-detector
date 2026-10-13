"""Upload the Oct 10 Atlas detail objects to R2 in bundles (runs on the H200, where the staged tars are).

Same transport as research/atlas-rescore-moe-20261007/upload_combined.py: 40-object bundles POSTed to the atlas-upload
Worker /bundle endpoint, which writes each member to R2 (atlas-public/objects/<sha256>.json.gz; content-addressed, so
no live object is overwritten with different content). Members are read straight from build_oct10.py's
objects-NNN.tar files (no unpacking). Each acknowledged size and MD5 is checked against the index; acknowledged keys
go to a ledger so the upload resumes. Then 300 random objects are read back through the Worker and byte-compared.
Refuses to start if the index has any error row. Access: the Worker URL and token are read from stdin as two lines
(URL, then token), kept in memory only, never printed.
Usage: upload_oct10.py OUT_DIR [--threads 8]
"""
import hashlib, json, random, struct, sys, tarfile, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUT = Path(sys.argv[1]); THREADS = int(sys.argv[sys.argv.index('--threads') + 1]) if '--threads' in sys.argv else 8
URL = sys.stdin.readline().strip().rstrip('/'); TOKEN = sys.stdin.readline().strip()
LEDGER = OUT / 'uploaded.txt'; STATUS = OUT / 'upload-status.json'


def status(**k): STATUS.write_text(json.dumps({**k, 'time': time.time()})); print(json.dumps(k), flush=True)


def req(path, data=None, method='GET'):
    r = urllib.request.Request(f'{URL}/{path}', data=data, method=method, headers={'Authorization': 'Bearer ' + TOKEN, 'User-Agent': 'open-text-detector-atlas-publisher/1.0'})
    with urllib.request.urlopen(r, timeout=300) as resp: return resp.read()


def send(group):
    entries = []; off = 0
    for k, b in group: entries.append({'key': k, 'offset': off, 'length': len(b)}); off += len(b)
    h = json.dumps(entries).encode(); payload = struct.pack('>I', len(h)) + h + b''.join(b for _, b in group)
    md5 = {k: hashlib.md5(b).hexdigest() for k, b in group}; size = {k: len(b) for k, b in group}
    for attempt in range(6):
        try:
            acks = json.loads(req('bundle', payload, 'POST'))
            if len(acks) != len(group) or any(a['size'] != size[a['key']] or a['etag'].strip('"') != md5[a['key']] for a in acks): raise ValueError('ack mismatch')
            return [k for k, _ in group]
        except Exception:
            if attempt == 5: raise
            time.sleep(5 * (attempt + 1))


def main():
    import pyarrow.parquet as pq
    idx = pq.read_table(OUT / 'index.parquet').to_pylist()
    if any(r.get('error') for r in idx): raise SystemExit('index has error rows; not uploading')
    want = {r['detail_key']: r for r in idx}
    done = set(LEDGER.read_text().split()) if LEDGER.exists() else set()
    status(state='uploading', objects=len(want), already=len(done & set(want))); t0 = time.time(); n = 0; seen = set()
    with ThreadPoolExecutor(THREADS) as pool, LEDGER.open('a') as led:
        futs = []; group = []
        for tar in sorted(OUT.glob('objects-*.tar')):
            with tarfile.open(tar) as tf:
                for m in tf:
                    key = 'atlas-public/' + m.name
                    if key not in want or key in done or key in seen: continue
                    seen.add(key); b = tf.extractfile(m).read()
                    if len(b) != want[key]['bytes'] or hashlib.md5(b).hexdigest() != want[key]['md5']: raise SystemExit(f'tar member differs from index: {key}')
                    group.append((key, b))
                    if len(group) == 40: futs.append(pool.submit(send, group)); group = []
                    while len([f for f in futs if not f.done()]) > THREADS * 2: time.sleep(0.2)
                    for f in [f for f in futs if f.done()]:
                        led.write('\n'.join(f.result()) + '\n'); led.flush(); futs.remove(f); n += 1
                        if n % 100 == 0: status(state='uploading', bundles_done=n, seconds=round(time.time() - t0))
        if group: futs.append(pool.submit(send, group))
        for f in as_completed(futs): led.write('\n'.join(f.result()) + '\n'); led.flush(); n += 1
    done = set(LEDGER.read_text().split()); missing = [k for k in want if k not in done]
    if missing: raise SystemExit(f'{len(missing)} objects not acknowledged')
    sample = set(random.Random(7).sample(sorted(want), min(300, len(want)))); mism = 0; checked = 0
    for tar in sorted(OUT.glob('objects-*.tar')):
        with tarfile.open(tar) as tf:
            for m in tf:
                key = 'atlas-public/' + m.name
                if key in sample: checked += 1; mism += req(key) != tf.extractfile(m).read()
    status(state='done', objects=len(want), acknowledged=len(want), readback_checked=checked, readback_mismatches=mism, seconds=round(time.time() - t0))


if __name__ == '__main__':
    main()
