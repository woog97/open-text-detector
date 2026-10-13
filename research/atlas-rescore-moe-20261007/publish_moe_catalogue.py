"""Add the final MoE to the public Atlas catalogue as a second model on every ICLR 2027 paper (runs on the Mac).

For each catalogue item whose current detail_key is the live 4B object that build_combined.py rebuilt byte-identically,
the item gets: models += ['qwen36-35b-a3b'] (4B stays first), score_summaries['qwen36-35b-a3b'] = the MoE summary, and
detail_key -> the combined object (reports [4B, MoE]). The 4B score summary must equal the rebuilt one; it is kept as is.
Items that do not match are left untouched. The catalogue's models list gains the MoE entry with its calibration.
The write is conditional on the current ETag, a backup of the old catalogue is kept, and the result is read back.
Access: ~/.config/open-text-detector/atlas-upload-access.json (never printed).
Usage: publish_moe_catalogue.py COMBINED_INDEX.parquet CALIBRATION.json [--dry-run]
"""
import hashlib, json, sys, time, urllib.request
from pathlib import Path

ACCESS = json.loads((Path.home() / '.config/open-text-detector/atlas-upload-access.json').read_text())
MOE = 'qwen36-35b-a3b'


def http(method, key, raw=None, headers=None):
    req = urllib.request.Request(f"{ACCESS['url']}/{key}", data=raw, method=method,
                                 headers={'Authorization': 'Bearer ' + ACCESS['token'], 'User-Agent': 'open-text-detector-atlas-publisher/1.0', **(headers or {})})
    with urllib.request.urlopen(req, timeout=300) as r: return r.read(), r.headers.get('ETag')


def main():
    import pyarrow.parquet as pq
    index_path, cal_path = map(Path, sys.argv[1:3]); dry = '--dry-run' in sys.argv
    cal = json.loads(cal_path.read_text())
    rows = {r['id']: r for r in pq.read_table(index_path).to_pylist() if not r.get('error') and r.get('live_4b_identical')}
    raw, etag = http('GET', 'atlas-public/catalogue.json'); catalogue = json.loads(raw)
    stamp = time.strftime('%Y%m%dT%H%M%S'); backup = Path(f'/tmp/atlas-catalogue-backup-{stamp}.json'); backup.write_bytes(raw)
    counts = {'updated': 0, 'already': 0, 'no_index_row': 0, 'detail_key_differs': 0, 'summary_4b_differs': 0}; items = []
    for p in catalogue['items']:
        r = rows.get(p['id'])
        if r is None: counts['no_index_row'] += 1; items.append(p); continue
        if MOE in (p.get('models') or []) and p.get('detail_key') == r['detail_key']: counts['already'] += 1; items.append(p); continue
        if p.get('detail_key') != r['live_4b_key']: counts['detail_key_differs'] += 1; items.append(p); continue
        if (p.get('score_summaries') or {}).get('qwen35-4b') != json.loads(r['summary_4b_json']): counts['summary_4b_differs'] += 1; items.append(p); continue
        items.append({**p, 'models': [m for m in p.get('models', []) if m != MOE] + [MOE],
                      'score_summaries': {**p.get('score_summaries', {}), MOE: json.loads(r['summary_moe_json'])}, 'detail_key': r['detail_key']})
        counts['updated'] += 1
    one = next(t for t in cal['thresholds'] if t['target_fpr'] == 0.01)
    available = sum(1 for p in items if MOE in (p.get('models') or []))
    entry = {'id': MOE, 'name': cal['model']['name'], 'available': available, 'total': len(items), 'complete': False,
             'calibration': {'sentence_threshold_1pct_fpr': one['threshold'], 'heldout_fpr': one['heldout_fpr'], 'human_papers': cal['calibration_set']['papers'],
                             'page': '/?view=calibration'}}
    out_cat = {**catalogue, 'published_at': time.time(), 'items': items, 'models': [m for m in catalogue['models'] if m['id'] != MOE] + [entry]}
    out = json.dumps(out_cat, separators=(',', ':')).encode()
    print(json.dumps({'catalogue_items': len(items), **counts, 'moe_available': available, 'models': [m['id'] for m in out_cat['models']],
                      'bytes_before': len(raw), 'bytes_after': len(out), 'backup': str(backup), 'dry_run': dry}), flush=True)
    if dry: return
    ack, _ = http('PUT', 'atlas-public/catalogue.json', out, {'If-Match': etag, 'Content-Type': 'application/json'}); ack = json.loads(ack)
    if ack['size'] != len(out) or ack['etag'].strip('"') != hashlib.md5(out).hexdigest(): raise ValueError('Catalogue upload mismatch')
    if http('GET', 'atlas-public/catalogue.json')[0] != out: raise ValueError('Catalogue readback mismatch')
    print('catalogue published and verified; backup at', backup, flush=True)


if __name__ == '__main__':
    main()
