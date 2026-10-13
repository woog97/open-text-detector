"""Space-side inventory: every PDF in R2 (via the export Worker's /list) against every existing direct extraction
(datasets with pdf_sha256 + positions_path) in the bucket. Writes /data/workspace/corpus-extraction-20261007/inventory.json.
Env: PDF_EXPORT_URL, PDF_EXPORT_TOKEN."""
import glob, json, os, time, urllib.request
from collections import Counter, defaultdict
from pathlib import Path
import pyarrow.parquet as pq

O = Path('/data/workspace/corpus-extraction-20261007'); O.mkdir(parents=True, exist_ok=True)
keys, cursor = {}, ''
while True:
    req = urllib.request.Request(os.environ['PDF_EXPORT_URL'] + '/list' + (f'?cursor={cursor}' if cursor else ''),
                                 headers={'Authorization': 'Bearer ' + os.environ['PDF_EXPORT_TOKEN'], 'User-Agent': 'open-text-detector-inventory/1.0'})
    d = json.load(urllib.request.urlopen(req, timeout=120))
    for k in d['keys']: keys[k['key'][7:-4]] = k['size']
    cursor = d['cursor']
    if not cursor: break
have = defaultdict(list); ds_stats = {}
roots = sorted(set(glob.glob('/data/workspace/datasets/*')) | set(glob.glob('/data/workspace/*/datasets/*')) | {'/tmp/pangram-humancal/extraction', '/tmp/pangram-pre2023text/extraction'})
for root in roots:
    files = sorted(glob.glob(root + '/data/*.parquet'))
    if not files: continue
    try:
        names = pq.read_schema(files[0]).names
    except Exception as e:
        ds_stats[root] = {'error': repr(e)[:120]}; continue
    if 'pdf_sha256' not in names or 'positions_path' not in names: continue
    n = 0
    for f in files:
        for r in pq.read_table(f, columns=['pdf_sha256', 'positions_path']).to_pylist():
            if r['positions_path']: have[r['pdf_sha256']].append(root); n += 1
    shards = sorted(os.path.basename(p) for p in glob.glob(root + '/*.tar'))
    ds_stats[root] = {'rows_with_positions': n, 'tar_shards': len(shards), 'objects_dir': os.path.isdir(root + '/objects')}
in_r2 = set(keys)
out = {'r2_pdfs': len(in_r2), 'r2_gb': round(sum(keys.values()) / 1e9, 1), 'extracted_in_r2': len(in_r2 & set(have)),
       'missing_extraction': len(in_r2 - set(have)), 'extracted_not_in_r2': len(set(have) - in_r2), 'datasets': ds_stats,
       'missing_gb': round(sum(keys[s] for s in in_r2 - set(have)) / 1e9, 1), 'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
(O / 'inventory.json').write_text(json.dumps({**out, 'r2': keys, 'have': {s: v for s, v in have.items()}}))
print(json.dumps(out, indent=1))
