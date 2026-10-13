"""0.01%-FPR flag counts ('nobib0.01', 'body0.01') for the Oct 10 Atlas flag index (Oct 12 2026; nothing is uploaded).

count   OUT_DIR CALIB_DIR PART_DIR   (on the H200)
        Reads build_oct10.py's staged detail objects (OUT_DIR/objects-NNN.tar, OUT_DIR/index.parquet maps detail_key -> id)
        and recounts every paper's report exactly as build_oct10.count does (sentences with a letter or digit, the mode's
        sections by bib.sections/section_of/included, flagged when the stored 5-decimal score >= the mode's threshold), for
        all eight existing modes and the two new ones. New thresholds: the target_fpr 0.0001 rows of
        CALIB_DIR/qwen36-35b-a3b-oct10-{nobib,body}.json (ca.threshold_at on the 2,400 human papers).
        Writes PART_DIR/counts-NNN.json: {id: {'modes': {...}, 'sections': [body, bib, app]}}.
merge   FLAGS.json PART_DIR CALIB_DIR OUT_FLAGS.json
        Checks that the recounted existing modes (all eight) and section sizes equal FLAGS.json (the staged/live index) for
        every paper, then writes OUT_FLAGS.json = FLAGS.json with 'nobib0.01' and 'body0.01' appended to each paper's modes
        and to the thresholds, modes and fprs maps (existing values unchanged).
"""
import gzip, json, re, sys, tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent / 'atlas-bib-exclusion-20261008',):
    sys.path.insert(0, str(p))
NEW = ('nobib0.01', 'body0.01')
DESC = {'nobib0.01': 'bibliography excluded at 0.01% FPR', 'body0.01': 'main text only (bibliography and appendices excluded) at 0.01% FPR'}
base = lambda m: re.sub(r'0\.0?1$', '', m)


def new_thresholds(calib_dir):
    out = {}
    for m in NEW:
        d = json.loads((Path(calib_dir) / f'qwen36-35b-a3b-oct10-{base(m)}.json').read_text())
        out[m] = next(t['threshold'] for t in d['thresholds'] if t['target_fpr'] == 0.0001)
    return out


def count_tar(job):
    from bib import included, section_of, sections
    tar, key2id, th, part = job; res = {}
    with tarfile.open(tar) as tf:
        for m in tf:
            if not m.isfile(): continue
            d = json.loads(gzip.decompress(tf.extractfile(m).read())); r = d['reports'][0]; text = r['text']
            spans, tail = sections(text); c = {k: [0, 0] for k in th}; size = {'body': 0, 'bib': 0, 'app': 0}
            for s in r['result']['segments']:
                if not any(ch.isalnum() for ch in text[s['start']:s['end']]): continue
                sec = section_of(s['start'], spans, tail); size[sec] += 1
                for k, t in th.items():
                    if included(sec, base(k)): c[k][1] += 1; c[k][0] += s['score'] >= t
            res[key2id['atlas-public/' + m.name]] = {'modes': c, 'sections': [size['body'], size['bib'], size['app']]}
    Path(part, Path(tar).stem.replace('objects', 'counts') + '.json').write_text(json.dumps(res))
    return tar, len(res)


def count(out_dir, calib_dir, part):
    import pyarrow.parquet as pq
    Path(part).mkdir(parents=True, exist_ok=True)
    key2id = {r['detail_key']: r['id'] for r in pq.read_table(Path(out_dir) / 'index.parquet', columns=['detail_key', 'id']).to_pylist()}
    flags = json.loads((Path(out_dir) / 'flags-oct10.json').read_text()); th = {**flags['thresholds'], **new_thresholds(calib_dir)}
    jobs = [(str(t), key2id, th, part) for t in sorted(Path(out_dir).glob('objects-*.tar'))]
    with ProcessPoolExecutor(len(jobs)) as pool:
        for t, n in pool.map(count_tar, jobs): print(t, n, flush=True)


def merge(flags_path, part, calib_dir, out):
    fl = json.loads(Path(flags_path).read_text()); th = new_thresholds(calib_dir); got = {}
    for f in sorted(Path(part).glob('counts-*.json')): got.update(json.loads(f.read_text()))
    assert set(got) == set(fl['papers']), (len(got), len(fl['papers']))
    bad = [i for i, v in fl['papers'].items() if v['sections'] != got[i]['sections'] or any(v['modes'][m] != got[i]['modes'][m] for m in v['modes'])]
    print(json.dumps({'papers': len(got), 'existing_modes_checked': list(next(iter(fl['papers'].values()))['modes']), 'mismatches': len(bad), 'examples': bad[:5]}))
    if bad: sys.exit('recount does not reproduce the flag index; nothing written')
    for i, v in fl['papers'].items():
        for m in NEW: v['modes'][m] = got[i]['modes'][m]
    for m in NEW:
        fl['thresholds'][m] = th[m]; fl['modes'][m] = DESC[m]; fl['fprs'][m] = 0.0001
    fl['rule'] += "; the 0.01% modes (nobib0.01, body0.01) use the target_fpr 0.0001 thresholds of the Calibration files (research/atlas-oct10-20261012/flags_p01_oct10.py)"
    Path(out).parent.mkdir(parents=True, exist_ok=True); Path(out).write_text(json.dumps(fl)); print('wrote', out, th)


if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    count(*a) if cmd == 'count' else merge(*a)
