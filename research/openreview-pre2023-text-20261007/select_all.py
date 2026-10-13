"""List every downloaded ICLR 2021/2022 and NeurIPS 2021/2022 PDF (legacy OpenReview queue, stored in R2) for full
text extraction, with flags so later samplers can exclude papers already used elsewhere. Writes selection.json
(ids, sha256, R2 keys, flags; no paper text). No paper is dropped here; filtering happens at sampling time.

Flags:
  calibration_reserved  in research/human-calibration-20261006/selection.json (human calibration set)
  in_atlas_catalogue    already in the 14,561-paper archive / Atlas catalogue (by sha256 or OpenReview id)
  paired_overlap        same PDF sha256 or normalized title as a paper in the paired data (paper-gap10000-v3-luna)
"""
import json, re, sqlite3
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / 'research/data/openreview_legacy_download_queue/queue.sqlite3'
CATALOGUE = ROOT / 'research/exports/atlas-public/catalogue.json'
CAL = ROOT / 'research/human-calibration-20261006/selection.json'
PAIRED = ROOT / 'research/data/paper-gap10000-v3-luna-20260930/papers.jsonl'
GROUPS = ['iclr/2021', 'iclr/2022', 'neurips/2021', 'neurips/2022']


def norm(t):
    return re.sub(r'[^a-z0-9]+', '', (t or '').lower())


def main():
    items = json.loads(CATALOGUE.read_text())['items']
    cat_sha = {i['pdf_key'].split('/')[-1].removesuffix('.pdf') for i in items if i.get('pdf_key')}
    cat_id = {i['filename'].removesuffix('.pdf') for i in items if i.get('filename')}
    cal = {p['sha256'] for p in json.loads(CAL.read_text())['papers']}
    paired = [json.loads(l) for l in open(PAIRED)]
    paired_sha = {p.get('pdf_sha256') for p in paired}; paired_titles = {norm(p['title']) for p in paired}
    rows = sqlite3.connect(QUEUE).execute(
        "select id, group_key, sha256, pages, bytes, file from papers where status='downloaded' and group_key in (%s)" % ','.join('?' * len(GROUPS)),
        GROUPS).fetchall()
    out, seen = [], set()
    for pid, g, sha, pages, size, f in sorted(rows):
        if sha in seen:  # one row per PDF
            continue
        seen.add(sha)
        out.append({'id': pid, 'venue': g.split('/')[0], 'year': int(g.split('/')[1]), 'sha256': sha, 'r2_key': f'papers/{sha}.pdf',
                    'pages': pages, 'bytes': size, 'calibration_reserved': sha in cal, 'in_atlas_catalogue': sha in cat_sha or pid in cat_id,
                    'paired_overlap_sha': sha in paired_sha})
    stats = {'papers': len(out), 'by_group': dict(Counter(f"{r['venue']}/{r['year']}" for r in out)),
             'calibration_reserved': sum(r['calibration_reserved'] for r in out), 'in_atlas_catalogue': sum(r['in_atlas_catalogue'] for r in out),
             'paired_overlap_sha': sum(r['paired_overlap_sha'] for r in out), 'gb': round(sum(r['bytes'] or 0 for r in out) / 1e9, 2),
             'note': 'paired title overlap is flagged after extraction (titles are not in the queue)'}
    (Path(__file__).parent / 'selection.json').write_text(json.dumps({'stats': stats, 'papers': out}, indent=0))
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
