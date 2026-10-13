"""Oct 10 Atlas detail objects and per-paper flag counts (runs on the H200; nothing is uploaded).

Switches the Paper Atlas from the Oct 7 MoE (qwen36-35b-a3b) to the Oct 10 final-moe (abl-noseg-nomix), model id
qwen36-35b-a3b-oct10. Inputs:
  - OCT7: the Oct 7 staged MoE-only detail objects (bucket workspace/atlas-rescore-moe-20261007/staging/objects-*.tar
    and atlas-details-index-moe.parquet). They supply each paper's clean text, text hash and position map, which do not
    depend on the model, so nothing has to be re-extracted.
  - SENT: Oct 10 sentence scores, research/human-calibration-20261006/sentence_scores_space.py `papers` on the Oct 10
    token shards (atlas-rescore/out-final-moe-20261011), same sentence rule as the Oct 7 publication.
  - TEXT: the two clean-text datasets (id, pdf_sha256, text_sha256), to map clean ids to PDFs.
  - RECAL: recal-final-moe.json (research/atlas-calibration-fix-20261011/recalibrate.py on the Oct 10 calibration
    scores; 2,400 papers). Non-"/before" blocks; 1% and 0.1% targets per counting mode.
  - LIVE_FLAGS: the live indexes/flags/current.json (Oct 7 counts), used as a check: the Oct 7 report in each staged
    object is recounted at the live thresholds and must reproduce the live counts exactly.
For every paper the new object is {id, version, version_verified, reports: [Oct 10 report], position_maps} (Oct 10
only; the reader picks the report whose id starts with the model id). Report id: 'qwen36-35b-a3b-oct10-' + the first
16 hex digits of text_sha256. Segment labels and the score summary's `flagged` use the 'all' 1%-FPR threshold with
'>' (same meaning as the Oct 7 summary). Flag counts follow research/atlas-calibration-fix-20261011/recount.py: per
mode, sentences with a letter or digit in the mode's sections, flagged when score >= the mode's threshold.
Every staged object is checked: report text hash, sentence spans identical to the Oct 7 report's.
Outputs in OUT: objects-NNN.tar (one per input tar, members objects/<sha256>.json.gz), index-NNN.parquet
(pdf_sha256, id, detail_key, bytes, summary_json, flags_json, sections_json, checks), and after `merge`,
index.parquet, flags-oct10.json (indexes/flags/current.json format) and checks.json.
Usage: build_oct10.py build OCT7 SENT OUT RECAL LIVE_FLAGS TEXT [TEXT ...]
       build_oct10.py merge OUT RECAL LIVE_FLAGS
"""
import gzip, hashlib, io, json, sys, tarfile, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'atlas-bib-exclusion-20261008'))
from bib import MODES, included, section_of, sections  # noqa: E402

MODEL_ID = 'qwen36-35b-a3b-oct10'
MODEL_NAME = 'Qwen3.6 35B-A3B MoE · Oct 10'
NOTICE = ('Final model (Qwen3.6-35B-A3B MoE, trained Oct 10, 2026). Scores are raw model probabilities. '
          '“Elevated evidence” marks sentences above the threshold that flags 1% of sentences in human-written papers from '
          '2022 or earlier (2,400 papers). Flags measure resemblance to AI-generated text and are not conclusions about '
          'authorship. See the Calibration page.')
ALL8 = [m + s for m in MODES for s in ('', '0.1')]
HIDDEN = ['all', 'all0.1', 'noapp', 'noapp0.1']  # 0.1% cutoff with the bibliography counted misses its target (0.97)
base = lambda m: m.removesuffix('0.1')


def thresholds(recal):
    cal = json.loads(Path(recal).read_text())['modes']; out = {}
    for m in MODES:
        t = {x['target_fpr']: x['threshold'] for x in cal[m]['thresholds']}
        out[m], out[m + '0.1'] = t[0.01], t[0.001]
    return out


def count(text, segs, th, strict=False):
    """{mode: [flagged, counted]} and [body, bib, app] sentence counts for one report's segments."""
    spans, tail = sections(text); c = {m: [0, 0] for m in th}; size = {'body': 0, 'bib': 0, 'app': 0}
    for s in segs:
        if not any(ch.isalnum() for ch in text[s['start']:s['end']]): continue
        sec = section_of(s['start'], spans, tail); size[sec] += 1
        for m, t in th.items():
            if included(sec, base(m)): c[m][1] += 1; c[m][0] += (s['score'] > t) if strict else (s['score'] >= t)
    return c, [size['body'], size['bib'], size['app']]


def gz(d): return gzip.compress(json.dumps(d, separators=(',', ':'), ensure_ascii=False).encode(), compresslevel=6, mtime=0)


def build_tar(job):
    k, tar_path, out_dir, sent_dir, recal, live_flags, clean_map, staged = job
    import pyarrow as pa, pyarrow.parquet as pq
    th = thresholds(recal); label_t = th['all']
    live = json.loads(Path(live_flags).read_text()); old_th = live['thresholds']; live_p = live['papers']
    # Oct 10 sentence scores for the papers of this tar only.
    sent = {}
    for f in sorted(Path(sent_dir).glob('sentences-*.npz')):
        z = np.load(f)
        for key in z.files:
            pid, kind = key.rsplit('/', 1)
            if pid in clean_map['want']: sent.setdefault(pid, {})[kind] = z[key]
    rows = []; out_tar = Path(out_dir) / f'objects-{k:03}.tar'; tmp = out_tar.with_suffix('.tmp')
    with tarfile.open(tar_path) as tf, tarfile.open(tmp, 'w') as ot:
        for m in tf:
            if not m.isfile(): continue
            key = 'atlas-public/' + m.name; pdf = staged.get(key)
            row = {'pdf_sha256': pdf, 'id': pdf[:24] if pdf else None, 'oct7_staged_key': key}
            try:
                if pdf is None: raise ValueError('object not in staged index')
                d = json.loads(gzip.decompress(tf.extractfile(m).read()))
                assert len(d['reports']) == 1 and d['reports'][0]['id'].startswith('qwen36-35b-a3b-'), 'unexpected Oct 7 object'
                r7 = d['reports'][0]; text = r7['text']; tsha = r7['text_sha256']
                if hashlib.sha256(text.encode()).hexdigest() != tsha: raise ValueError('text hash mismatch')
                cid = clean_map['by_pdf_text'].get(f'{pdf}:{tsha}')
                if cid is None: raise ValueError('no clean row with this pdf and text hash')
                s = sent.get(cid)
                if s is None: raise ValueError('no Oct 10 sentence scores')
                old_spans = [(x['start'], x['end']) for x in r7['result']['segments']]
                new_spans = list(zip(s['starts'].tolist(), s['ends'].tolist()))
                if old_spans != new_spans: raise ValueError(f'sentence spans differ ({len(old_spans)} vs {len(new_spans)})')
                segs = [{'start': int(a), 'end': int(e), 'score': round(float(v), 5), 'label': 'ai_evidence' if float(v) > label_t else 'below_threshold'}
                        for a, e, v in zip(s['starts'], s['ends'], s['scores'].astype(np.float32))]
                rep = {'id': f'{MODEL_ID}-{tsha[:16]}', 'text': text, 'text_sha256': tsha, 'model': {'name': MODEL_NAME},
                       'result': {'segments': segs, 'notice': NOTICE}}
                hist = [0] * 21; excl = 0; flagged = 0
                for x in segs:
                    if not any(c.isalnum() for c in text[x['start']:x['end']]): excl += 1; continue
                    hist[int(np.ceil(x['score'] * 20))] += 1; flagged += x['label'] == 'ai_evidence'
                summary = {'histogram': hist, 'total': sum(hist), 'excluded': excl, 'policy': 2, 'flagged': flagged,
                           'flag_threshold': label_t, 'flag_target_fpr': 0.01}
                flags, secs = count(text, segs, th)
                # Check: the Oct 7 report at the live thresholds must reproduce the live counts (live counted with >=).
                old_flags, old_secs = count(text, r7['result']['segments'], old_th)
                lp = live_p.get(pdf[:24])
                row['oct7_live_match'] = lp is not None and lp['modes'] == old_flags and lp['sections'] == old_secs
                row['oct7_flags_json'] = json.dumps(old_flags)
                raw = gz({'id': d['id'], 'version': d['version'], 'version_verified': d['version_verified'], 'reports': [rep],
                          'position_maps': d['position_maps']})
                digest = hashlib.sha256(raw).hexdigest(); info = tarfile.TarInfo('objects/' + digest + '.json.gz'); info.size = len(raw)
                ot.addfile(info, io.BytesIO(raw))
                row.update(detail_key=f'atlas-public/objects/{digest}.json.gz', bytes=len(raw), md5=hashlib.md5(raw).hexdigest(),
                           summary_json=json.dumps(summary), flags_json=json.dumps(flags), sections_json=json.dumps(secs), sentences=len(segs))
            except Exception as e:
                row['error'] = f'{type(e).__name__}: {e}'
            rows.append(row)
    tmp.replace(out_tar)
    pq.write_table(pa.Table.from_pylist(rows), Path(out_dir) / f'index-{k:03}.parquet')
    return k, len(rows), sum('error' in r for r in rows)


def build(oct7, sent_dir, out, recal, live_flags, texts):
    import pyarrow.parquet as pq
    out = Path(out); out.mkdir(parents=True, exist_ok=True); oct7 = Path(oct7)
    by_pdf_text = {}
    for d in texts:
        for f in sorted((Path(d) / 'data').glob('train-*.parquet')):
            for r in pq.read_table(f, columns=['id', 'pdf_sha256', 'text_sha256']).to_pylist():
                by_pdf_text[f"{r['pdf_sha256']}:{r['text_sha256']}"] = r['id']
    staged = {r['detail_key']: r['pdf_sha256'] for r in pq.read_table(oct7 / 'atlas-details-index-moe.parquet', columns=['detail_key', 'pdf_sha256']).to_pylist()}
    cmap = {'by_pdf_text': by_pdf_text, 'want': set(by_pdf_text.values())}
    tars = sorted(oct7.glob('objects-*.tar')); t0 = time.time()
    jobs = [(int(t.stem.split('-')[1]), str(t), str(out), sent_dir, recal, live_flags, cmap, staged) for t in tars
            if not (out / f"index-{int(t.stem.split('-')[1]):03}.parquet").exists()]
    with ProcessPoolExecutor(len(jobs) or 1) as pool:
        for k, n, err in pool.map(build_tar, jobs):
            print(json.dumps({'tar': k, 'objects': n, 'errors': err, 'seconds': round(time.time() - t0)}), flush=True)


def merge(out, recal, live_flags):
    import pyarrow as pa, pyarrow.parquet as pq
    out = Path(out); rows = [r for f in sorted(out.glob('index-[0-9]*.parquet')) for r in pq.read_table(f).to_pylist()]
    pq.write_table(pa.Table.from_pylist(rows), out / 'index.parquet')
    live = json.loads(Path(live_flags).read_text()); th = thresholds(recal)
    ok = [r for r in rows if not r.get('error')]
    papers = {r['id']: {'modes': json.loads(r['flags_json']), 'sections': json.loads(r['sections_json'])} for r in ok}
    flags = {'model': MODEL_ID, 'format': live.get('format', 2), 'thresholds': th, 'target_fpr': 0.01, 'modes': live['modes'],
             'fprs': {m: (0.001 if m.endswith('0.1') else 0.01) for m in ALL8}, 'hidden_modes': HIDDEN,
             'rule': "sentences at or above each mode's 1%- or 0.1%-FPR threshold for the Oct 10 final MoE (qwen36-35b-a3b-oct10), "
                     "calibrated on 2,400 human papers (research/atlas-calibration-fix-20261011/recalibrate.py); the all/noapp modes "
                     "are computed but not shown (their 0.1% cutoff misses its target) (research/atlas-oct10-20261012/build_oct10.py)",
             'papers': papers}
    (out / 'flags-oct10.json').write_text(json.dumps(flags))
    checks = {'objects': len(rows), 'errors': len(rows) - len(ok), 'error_examples': [r['error'] for r in rows if r.get('error')][:10],
              'papers': len(papers), 'duplicate_ids': len(ok) - len(papers), 'distinct_detail_keys': len({r['detail_key'] for r in ok}),
              'oct7_live_match': sum(bool(r['oct7_live_match']) for r in ok),
              'live_papers_missing': sorted(set(live['papers']) - set(papers)), 'bytes': sum(r['bytes'] for r in ok), 'thresholds': th}
    (out / 'checks.json').write_text(json.dumps(checks, indent=1)); print(json.dumps({k: v for k, v in checks.items() if k != 'live_papers_missing'}), 'missing', len(checks['live_papers_missing']))


if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    if cmd == 'build': build(a[0], a[1], a[2], a[3], a[4], a[5:])
    else: merge(*a)
