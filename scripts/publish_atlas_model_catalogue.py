"""Add a classified model's papers to the public Atlas catalogue (atlas-public/catalogue.json).

Inputs: the combined detail index from build_atlas_details.py (parquet: pdf_sha256, id, detail_key,
score_summary_json), the live reader listing (for title/filename/collection/bytes of papers whose
PDFs are already in R2) and the model's calibration JSON. Only papers present in the live listing
and not already published are added. The write is conditional on the catalogue's current ETag and
read back for verification. Access: ~/.config/open-text-detector/atlas-upload-access.json (never printed).
Usage: publish_atlas_model_catalogue.py INDEX.parquet LIVE_LISTING.json CALIBRATION.json [--dry-run]
"""
import hashlib, json, sys, time, urllib.request
from pathlib import Path

ACCESS = json.loads((Path.home() / '.config/open-text-detector/atlas-upload-access.json').read_text())


def http(method, key, raw=None, headers=None):
    req = urllib.request.Request(f"{ACCESS['url']}/{key}", data=raw, method=method,
                                 headers={'Authorization': 'Bearer ' + ACCESS['token'], 'User-Agent': 'open-text-detector-atlas-publisher/1.0', **(headers or {})})
    with urllib.request.urlopen(req, timeout=300) as r: return r.read(), r.headers.get('ETag')


def main():
    import pyarrow.parquet as pq
    index_path, live_path, calibration_path = map(Path, sys.argv[1:4]); dry = '--dry-run' in sys.argv
    cal = json.loads(calibration_path.read_text()); model = cal['model']
    index = [r for r in pq.read_table(index_path).to_pylist() if r.get('detail_key')]
    live = {i['id']: i for i in json.loads(live_path.read_text())['items']}
    raw, etag = http('GET', 'atlas-public/catalogue.json'); catalogue = json.loads(raw)
    stamp = time.strftime('%Y%m%dT%H%M%S')
    Path(f'/tmp/atlas-catalogue-backup-{stamp}.json').write_bytes(raw)
    published = {p['id']: p for p in catalogue['items']}
    added = []; upgraded = {}; skipped = {'not_in_live_listing': 0, 'already_published': 0}
    for r in index:
        meta = live.get(r['id'])
        if not meta: skipped['not_in_live_listing'] += 1; continue
        entry = {'id': r['id'], 'title': meta['title'], 'filename': meta['filename'], 'collection': meta['collection'], 'bytes': meta['bytes'],
                 'classified': True, 'models': [model['id']], 'score_summaries': {model['id']: json.loads(r['score_summary_json'])},
                 'pdf_key': f"papers/{r['pdf_sha256']}.pdf", 'detail_key': r['detail_key']}
        old = published.get(r['id'])
        if old is None: added.append(entry)
        # Unclassified placeholders (no models, no results) are replaced; papers with any model results are left alone.
        elif not old.get('models') and not old.get('classified'): upgraded[r['id']] = entry
        else: skipped['already_published'] += 1
    items = [upgraded.get(p['id'], p) for p in catalogue['items']] + added
    if len({p['id'] for p in items}) != len(items): raise ValueError('Duplicate catalogue ids')
    one_pct = next(t for t in cal['thresholds'] if t['target_fpr'] == 0.01)
    entry = {'id': model['id'], 'name': model['name'], 'available': len(added) + len(upgraded), 'total': len(items), 'complete': False,
             'calibration': {'sentence_threshold_1pct_fpr': one_pct['threshold'], 'heldout_fpr': one_pct['heldout_fpr'],
                             'human_papers': cal['calibration_set']['papers'], 'page': '/?view=calibration'}}
    catalogue = {**catalogue, 'published_at': time.time(), 'items': items,
                 'models': [m for m in catalogue['models'] if m['id'] != model['id']] + [entry]}
    out = json.dumps(catalogue, separators=(',', ':')).encode()
    print(json.dumps({'existing': len(published), 'added': len(added), 'upgraded_placeholders': len(upgraded), 'skipped': skipped, 'bytes': len(out), 'dry_run': dry}), flush=True)
    if dry: return
    ack, _ = http('PUT', 'atlas-public/catalogue.json', out, {'If-Match': etag, 'Content-Type': 'application/json'})
    ack = json.loads(ack)
    if ack['size'] != len(out) or ack['etag'].strip('"') != hashlib.md5(out).hexdigest(): raise ValueError('Catalogue upload mismatch')
    if http('GET', 'atlas-public/catalogue.json')[0] != out: raise ValueError('Catalogue readback mismatch')
    print('catalogue published and verified; backup at', f'/tmp/atlas-catalogue-backup-{stamp}.json', flush=True)


if __name__ == '__main__':
    main()
