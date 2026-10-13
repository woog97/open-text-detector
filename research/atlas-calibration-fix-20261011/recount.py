"""Atlas flag counts at the recalibrated cutoffs (Oct 11 2026): each counting mode at its 1% and 0.1% thresholds from
recalibrate.py, counting sentences at or above the threshold (as the site does since b263056).

Same counting code as research/atlas-bib-exclusion-20261008/fpr10.py atlas. As a check, it also counts each mode at the
thresholds of the live flag index (`old:<mode>`); merge() requires those to reproduce the live counts exactly (the old
index counted strictly above the threshold).
Usage: recount.py atlas WORK CATALOGUE.json RECAL.json LIVE_FLAGS.json OUT.json TAR [TAR ...]
       recount.py merge LIVE_FLAGS.json OUT.json PART.json [PART.json ...]
"""
import gzip, json, sys, tarfile
from pathlib import Path
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'atlas-bib-exclusion-20261008'))
from bib import MODES, included, section_of, sections  # noqa: E402


def atlas(work, catalogue, recal, live_flags, out_json, tars):
    cal = json.loads(Path(recal).read_text())['modes']
    thresholds = {}
    for m in MODES:
        t = {x['target_fpr']: x['threshold'] for x in cal[m]['thresholds']}
        thresholds[m], thresholds[m + '0.1'] = t[0.01], t[0.001]
    old = {'old:' + m: t for m, t in json.loads(Path(live_flags).read_text())['thresholds'].items()}
    base = lambda m: m.removeprefix('old:').removesuffix('10').removesuffix('0.1')
    by_pdf = {p['pdf_key'][7:-4]: p for p in json.loads(Path(catalogue).read_text())['items']}
    staged = {r['detail_key']: r['pdf_sha256'] for r in pq.read_table(Path(work) / 'atlas-details-index-moe.parquet', columns=['detail_key', 'pdf_sha256']).to_pylist()}
    by_key = {k: by_pdf[h] for k, h in staged.items() if h in by_pdf}
    out, missing = {}, 0
    for tar in tars:
        with tarfile.open(tar) as tf:
            for member in tf:
                if not member.isfile(): continue
                paper = by_key.get('atlas-public/' + member.name)
                if paper is None: missing += 1; continue
                d = json.loads(gzip.decompress(tf.extractfile(member).read()))
                r = next(x for x in d['reports'] if x['id'].startswith('qwen36-35b-a3b-')); t = r['text']; spans, tail = sections(t)
                counts = {m: [0, 0] for m in [*thresholds, *old]}; size = {'body': 0, 'bib': 0, 'app': 0}
                for s in r['result']['segments']:
                    if not any(c.isalnum() for c in t[s['start']:s['end']]): continue
                    sec = section_of(s['start'], spans, tail); size[sec] += 1
                    for m, th in thresholds.items():
                        if included(sec, base(m)): counts[m][1] += 1; counts[m][0] += s['score'] >= th
                    for m, th in old.items():
                        if included(sec, base(m)): counts[m][1] += 1; counts[m][0] += s['score'] > th
                out[paper['id']] = {'modes': counts, 'sections': [size['body'], size['bib'], size['app']]}
        print(f'{Path(tar).name}: {len(out):,} papers so far', flush=True)
    Path(out_json).write_text(json.dumps({'thresholds': thresholds, 'papers': out, 'objects_not_in_catalogue': missing}))
    print(f'done: {len(out):,} papers; objects not in catalogue {missing}', flush=True)


def merge(live_flags, out_json, parts):
    live = json.loads(Path(live_flags).read_text())
    papers, thresholds = {}, None
    for f in parts:
        d = json.loads(Path(f).read_text()); papers.update(d['papers']); thresholds = d['thresholds']
    missing = set(live['papers']) - set(papers)
    diff = [k for k, v in live['papers'].items() if k in papers and ({m: papers[k]['modes']['old:' + m] for m in v['modes']}, papers[k]['sections']) != (v['modes'], v['sections'])]
    print(f'papers {len(papers):,} (live {len(live["papers"]):,}); missing {len(missing)}; old-threshold differences {len(diff)}')
    assert not missing and not diff, 'recount does not reproduce the live counts at the old thresholds'
    for p in papers.values(): p['modes'] = {m: c for m, c in p['modes'].items() if m in thresholds}
    new = {**live, 'thresholds': thresholds, 'papers': papers,
           'modes': {m: d for m, d in live['modes'].items() if m in thresholds},
           'fprs': {m: f for m, f in live['fprs'].items() if m in thresholds},
           'rule': 'sentences at or above each mode\'s 1%- or 0.1%-FPR threshold, recalibrated on 2,400 human papers '
                   '(Oct 11, 2026; research/atlas-calibration-fix-20261011/recount.py)'}
    Path(out_json).write_text(json.dumps(new))
    changed = {m: sum(papers[k]['modes'][m] != live['papers'][k]['modes'][m] for k in live['papers']) for m in thresholds}
    print('papers whose counts changed per mode', changed)
    print('thresholds', json.dumps(thresholds))


if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    if cmd == 'atlas': atlas(a[0], a[1], a[2], a[3], a[4], a[5:])
    else: merge(a[0], a[1], a[2:])
