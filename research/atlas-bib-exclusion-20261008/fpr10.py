"""10% and 0.1% false-positive variants of the four counting modes (Alice, Oct 9 2026): same code as recompute.py.

calibrate WORK OUT.json   recompute.py calibrate with targets 0.1%, 1%, 5% and 10% (the published 1% thresholds must
                          come out unchanged; bootstrap intervals of the other targets shift because the shared RNG
                          is consumed in a different order, so only the new 10% entries are used).
atlas WORK CATALOGUE.json CALIBRATION.json OUT.json TAR [TAR ...]
                          recompute.py atlas, counting each mode at its 1%, 10% and 0.1% thresholds: modes 'all',
                          'nobib', 'noapp', 'body' (1%, as published), 'all10' ... (10%) and 'all0.1' ... (0.1%).
"""
import gzip, json, sys, tarfile
from pathlib import Path
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'atlas-rescore-moe-20261007'))
import cal_analyze as ca  # noqa: E402
import recompute as rc  # noqa: E402
from bib import MODES, included, section_of, sections  # noqa: E402


def calibrate(work, out_json):
    ca.TARGETS = [0.001, 0.01, 0.05, 0.1]
    rc.calibrate(work, out_json)


def atlas(work, catalogue, calibration, out_json, tars):
    cal = json.loads(Path(calibration).read_text())['modes']
    thresholds = {}
    for m in MODES:
        t = {x['target_fpr']: x['threshold'] for x in cal[m]['thresholds']}
        thresholds[m], thresholds[m + '10'], thresholds[m + '0.1'] = t[0.01], t[0.1], t[0.001]
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
                counts = {m: [0, 0] for m in thresholds}; size = {'body': 0, 'bib': 0, 'app': 0}
                for s in r['result']['segments']:
                    if not any(c.isalnum() for c in t[s['start']:s['end']]): continue
                    sec = section_of(s['start'], spans, tail); size[sec] += 1
                    for m, th in thresholds.items():
                        if included(sec, m.removesuffix('10').removesuffix('0.1')): counts[m][1] += 1; counts[m][0] += s['score'] > th
                out[paper['id']] = {'modes': counts, 'sections': [size['body'], size['bib'], size['app']]}
        print(f'{Path(tar).name}: {len(out):,} papers so far', flush=True)
    Path(out_json).write_text(json.dumps({'thresholds': thresholds, 'papers': out, 'objects_not_in_catalogue': missing}))
    print(f'done: {len(out):,} papers; objects not in catalogue {missing}', flush=True)


if __name__ == '__main__':
    mode, *args = sys.argv[1:]
    if mode == 'calibrate': calibrate(*args)
    else: atlas(args[0], args[1], args[2], args[3], args[4:])
