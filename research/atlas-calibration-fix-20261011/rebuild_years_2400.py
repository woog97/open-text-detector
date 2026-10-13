"""MoE Calibration page year table at the published thresholds (2,400 human papers), Oct 12 2026.

The stored `years` rows were built by build_calibration_modes.py (Oct 8) at each mode's 1% threshold from all 2,403
papers. This recomputes every row (human groups, year diagnostic 2023-2026, ICLR 2027 submissions) at the published
1% threshold (`years`) and 0.1% threshold (new key `years_p1`) of app/lib/model-calibration/qwen36-35b-a3b-final*.json,
with the same rules as build_moe_calibration.row: per-paper flagged share = share of the mode's sentences scoring
strictly above the threshold; over_human_p99 = share of a group's papers above the 99th percentile of the human papers'
flagged share at the same threshold (human papers = the 2,400); document_score_median = median document probability.

count WORK CATALOGUE.json RECAL.json OLD_CAL.json OUT.json TAR [TAR ...]
    ICLR 2027: per paper and mode, sentences above the old 1% threshold (OLD_CAL.json = calibration-modes.json, as
    recompute.py atlas), above the published 1% and 0.1% thresholds (RECAL.json = recalibrate.py output), and the
    total. Same counting code as recompute.py atlas (bib.py sections; sentences without letters or digits skipped).
rows WORK WORK_CAL COUNTS.json [COUNTS.json ...] --json-dir DIR [--write]
    WORK: build_calibration_modes.py work directory (out-yd/, yd-text/, out-space/data, out-h200/data,
    atlas-details-index-moe.parquet, modes-*.json); WORK_CAL: recompute.py work dir for the human papers.
    Before writing, every stored row must be reproduced at the old threshold from all 2,403 human papers, and the
    ICLR 2027 old-threshold counts must equal modes-*.json. Otherwise nothing is written.
"""
import glob, gzip, json, sys, tarfile
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'atlas-bib-exclusion-20261008')); sys.path.insert(0, str(HERE.parent / 'atlas-rescore-moe-20261007'))
from bib import MODES, included, section_of, sections  # noqa: E402

FILES = {'all': 'qwen36-35b-a3b-final.json', 'nobib': 'qwen36-35b-a3b-final-nobib.json',
         'noapp': 'qwen36-35b-a3b-final-noapp.json', 'body': 'qwen36-35b-a3b-final-body.json'}


def count(work, catalogue, recal, old_cal, out_json, tars):
    new = {m: {x['target_fpr']: x['threshold'] for x in json.loads(Path(recal).read_text())['modes'][m]['thresholds']} for m in MODES}
    old = {m: next(x['threshold'] for x in json.loads(Path(old_cal).read_text())['modes'][m]['thresholds'] if x['target_fpr'] == 0.01) for m in MODES}
    th = {m: (old[m], new[m][0.01], new[m][0.001]) for m in MODES}
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
                c = {m: [0, 0, 0, 0] for m in MODES}
                for s in r['result']['segments']:
                    if not any(ch.isalnum() for ch in t[s['start']:s['end']]): continue
                    sec = section_of(s['start'], spans, tail)
                    for m in MODES:
                        if included(sec, m):
                            v = c[m]; v[3] += 1; v[0] += s['score'] > th[m][0]; v[1] += s['score'] > th[m][1]; v[2] += s['score'] > th[m][2]
                out[paper['id']] = c
        print(f'{Path(tar).name}: {len(out):,} papers so far', flush=True)
    Path(out_json).write_text(json.dumps({'thresholds': th, 'columns': ['old_1pct', 'new_1pct', 'new_0.1pct', 'total'], 'papers': out, 'objects_not_in_catalogue': missing}))
    print(f'done: {len(out):,} papers; objects not in catalogue {missing}', flush=True)


def rows(work, work_cal, counts_files, json_dir, write=False):
    import build_moe_calibration as bm
    import cal_analyze as ca
    import recompute as rc
    from recalibrate import EXCLUDE
    work = Path(work)
    by_mode, _, _ = rc.human_scores_by_mode(work_cal)
    _, cdocs = ca.load_tokens(Path(work_cal) / 'out-cal')
    meta = {p['id']: p for p in json.loads((Path(work_cal) / 'selection.json').read_text())['papers']}
    old_cal = json.loads((work / 'calibration-modes.json').read_text())['modes']
    # year diagnostic: per paper sentence scores under each mode
    ytok, ydocs = ca.load_tokens(work / 'out-yd'); ytext, ymeta = {}, {}
    for f in sorted((work / 'yd-text' / 'data').glob('*.parquet')):
        cols = [c for c in ('id', 'text', 'conference', 'year', 'source') if c in pq.read_schema(f).names]
        for r in pq.read_table(f, columns=cols).to_pylist(): ytext[r['id']] = r.pop('text'); ymeta[r['id']] = r
    yd = {}
    for pid, z in ytok.items():
        if pid not in ytext: continue
        s, starts = rc.sentence_scores_spans(ytext[pid], z['offsets'], z['probs']); spans, tail = sections(ytext[pid])
        sec = [section_of(a, spans, tail) for a in starts]
        yd[pid] = {m: (s[np.array([included(x, m) for x in sec], bool)] if len(sec) else s) for m in MODES}
    # ICLR 2027: counts, plus document probabilities in the order build_moe_calibration used
    counts = {}
    for f in counts_files: counts.update(json.loads(Path(f).read_text())['papers'])
    modes_old = {}
    for f in sorted(glob.glob(str(work / 'modes-[0-9].json'))): modes_old.update(json.load(open(f))['papers'])
    idx = pq.read_table(work / 'atlas-details-index-moe.parquet', columns=['pdf_sha256', 'score_summary_json']).to_pylist()
    atl_docs = {}
    for o in ('out-space', 'out-h200'):
        for f in sorted((work / o / 'data').glob('summary-*.parquet')):
            for r in pq.read_table(f, columns=['pdf_sha256', 'document_prob']).to_pylist(): atl_docs[r['pdf_sha256']] = r['document_prob']
    iclr_docs = [atl_docs.get(r['pdf_sha256']) for r in idx if r['score_summary_json']]
    iclr_docs = [d for d in iclr_docs[:len(modes_old)] if d is not None]
    ok = mismatch_counts = 0
    for pid, c in modes_old.items():
        for m in MODES:
            if pid not in counts or [counts[pid][m][0], counts[pid][m][3]] != c['modes'][m]: mismatch_counts += 1
    print(f'ICLR 2027 papers: modes-*.json {len(modes_old):,}, recount {len(counts):,}, old-threshold count mismatches {mismatch_counts}', flush=True)
    ok = mismatch_counts == 0 and set(counts) == set(modes_old)
    share = lambda s, t: float((s > t).mean()) if len(s) else 0.0
    out = {}
    for mode, fname in FILES.items():
        path = Path(json_dir) / fname; data = json.loads(path.read_text())
        pub = {t['target_fpr']: t['threshold'] for t in data['thresholds']}
        t_old = next(x['threshold'] for x in old_cal[mode]['thresholds'] if x['target_fpr'] == 0.01)
        cal_new = {k: v for k, v in by_mode[mode].items() if k not in EXCLUDE}
        t_new = {f: ca.threshold_at(np.concatenate(list(cal_new.values())), f) for f in (0.01, 0.001)}
        assert all(bm.r5(t_new[f]) == pub[f] for f in t_new), (mode, t_new, pub)

        def build(cal, thr, col):
            hs = {k: share(s, thr) for k, s in cal.items()}; h99 = float(np.quantile(list(hs.values()), .99)); res = []
            grp = defaultdict(list)
            for k, sh in hs.items():
                pid = k[4:] if k.startswith('old:') else k
                g = ('archive', 2022) if k.startswith('old:') else (meta[pid]['venue'], meta[pid]['year'])
                grp[g].append({'share': sh, 'doc': cdocs.get(pid, {}).get('document_prob')})
            for (v, y), ps in grp.items():
                res.append(bm.row(f"{bm.VENUE.get(v, v)} {y}" if v != 'archive' else 'ARCHIVE', 'human', y, ps, h99))
            ygrp = defaultdict(list)
            for pid, sc in yd.items():
                mt = ymeta[pid]; ygrp[(mt['source'], mt['conference'], mt['year'])].append({'share': share(sc[mode], thr), 'doc': ydocs.get(pid, {}).get('document_prob')})
            for (src, venue, year), ps in sorted(ygrp.items()):
                s = {'archive': 'archive', 'openreview-2026-10': 'OpenReview download'}.get(src, src)
                res.append(bm.row(f'{bm.VENUE.get(venue, venue)} {year}, {s}', 'iclr' if venue == 'iclr' else 'other', int(year), ps, h99))
            ps = [{'share': counts[p][mode][col] / max(counts[p][mode][3], 1), 'doc': None} for p in modes_old]
            ps[:len(iclr_docs)] = [{**p, 'doc': d} for p, d in zip(ps, iclr_docs)]
            res.append(bm.row('ICLR 2027 submissions', 'iclr', 2027, ps, h99))
            return {r['label']: r for r in res}, h99

        stored = data['years']; arch = next(r['label'] for r in stored if r['kind'] == 'human' and r['year'] == 2022 and 'rchive' in r['label'])
        relabel = lambda d: {(arch if l == 'ARCHIVE' else l): ({**r, 'label': arch} if l == 'ARCHIVE' else r) for l, r in d.items()}
        old_rows, _ = build(by_mode[mode], t_old, 0); old_rows = relabel(old_rows)
        # stored archive row was already rebuilt on 212 papers by rebuild_2400.py; check it against that rule
        arch_212, _ = build(cal_new, t_old, 0); arch_212 = relabel(arch_212)[arch]
        bad = [r['label'] for r in stored if (arch_212 if r['label'] == arch else old_rows.get(r['label'])) != r]
        ok &= not bad
        print(f'{mode:6s} stored rows reproduce at the old threshold {t_old:.6g}: {"OK" if not bad else "FAIL " + str(bad)}', flush=True)
        if bad:
            for l in bad[:5]: print('   stored', json.dumps(next(r for r in stored if r['label'] == l), ensure_ascii=False), '\n   got   ', json.dumps(arch_212 if l == arch else old_rows.get(l), ensure_ascii=False))
        new1, h1 = build(cal_new, t_new[0.01], 1); new01, h01 = build(cal_new, t_new[0.001], 2)
        new1, new01 = relabel(new1), relabel(new01)
        assert set(new1) == {r['label'] for r in stored}, set(new1) ^ {r['label'] for r in stored}
        data['years'] = [new1[r['label']] for r in stored]; data['years_p1'] = [new01[r['label']] for r in stored]
        print(f'{mode:6s} human p99 of flagged share: 1% {h1:.5f}, 0.1% {h01:.5f}')
        for r in stored:
            a, b = new1[r['label']], new01[r['label']]
            print(f"   {r['label']:34s} n {a['papers']:6,}  median {r['flagged_share_median']} -> {a['flagged_share_median']} (0.1%: {b['flagged_share_median']})  "
                  f"over_p99 {r['over_human_p99']} -> {a['over_human_p99']} (0.1%: {b['over_human_p99']})", flush=True)
        out[path] = data
    if not ok: sys.exit('not reproduced; nothing written')
    if write:
        for path, data in out.items(): path.write_text(json.dumps(data, separators=(',', ':'), ensure_ascii=False))
        print('wrote', len(out), 'files')


if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    if cmd == 'count': count(a[0], a[1], a[2], a[3], a[4], a[5:])
    else:
        i = a.index('--json-dir'); rows(a[0], a[1], a[2:i], a[i + 1], '--write' in a)
