"""Select 2019-2022 human papers from the legacy OpenReview downloads (PDFs in R2) for calibration.

Stratified by venue/year with a deterministic hash order. Papers already in the Atlas catalogue
(the 14,561-paper archive, which supplied the earlier 215 calibration papers) are excluded by
sha256 and by OpenReview id. Over-selects so that >=2,000 survive extraction and overlap removal.
Writes selection.json (ids, sha256, R2 keys; no paper text).
"""
import hashlib, json, sqlite3
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / 'research/data/openreview_legacy_download_queue/queue.sqlite3'
CATALOGUE = ROOT / 'research/exports/atlas-public/catalogue.json'
QUOTA = {'iclr/2019': 280, 'iclr/2020': 330, 'iclr/2021': 380, 'iclr/2022': 380,
         'neurips/2021': 380, 'neurips/2022': 380, 'corl/2021': 85, 'corl/2022': 85}  # 2,300


def order(pid):
    return hashlib.sha256(('human-cal-v1:' + pid).encode()).hexdigest()


def main():
    items = json.loads(CATALOGUE.read_text())['items']
    seen_sha = {i['pdf_key'].split('/')[-1].removesuffix('.pdf') for i in items if i.get('pdf_key')}
    seen_id = {i['filename'].removesuffix('.pdf') for i in items if i.get('filename')}
    rows = sqlite3.connect(QUEUE).execute(
        "select id, group_key, sha256, pages, bytes from papers where status='downloaded' and group_key in (%s)"
        % ','.join('?' * len(QUOTA)), list(QUOTA)).fetchall()
    pool = {}
    for pid, g, sha, pages, size in rows:
        if sha in seen_sha or pid in seen_id: continue
        pool.setdefault(g, {}).setdefault(sha, (pid, g, sha, pages, size))  # one row per PDF
    out = []
    for g, n in QUOTA.items():
        cands = sorted(pool[g].values(), key=lambda r: order(r[0]))
        out += [{'id': pid, 'venue': g.split('/')[0], 'year': int(g.split('/')[1]), 'sha256': sha,
                 'r2_key': f'papers/{sha}.pdf', 'pages': pages, 'bytes': size} for pid, g, sha, pages, size in cands[:n]]
    stats = {'pool': {g: len(v) for g, v in pool.items()}, 'selected': dict(Counter(f"{r['venue']}/{r['year']}" for r in out)),
             'total': len(out), 'gb': round(sum(r['bytes'] or 0 for r in out) / 1e9, 2)}
    (Path(__file__).parent / 'selection.json').write_text(json.dumps({'stats': stats, 'papers': out}, indent=0))
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
