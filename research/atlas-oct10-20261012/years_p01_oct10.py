"""`years_p01` rows (0.01% FPR) for the Oct 10 Calibration files qwen36-35b-a3b-oct10-{nobib,body}.json (Oct 12 2026).

Same rows, groups and definitions as build_calibration_oct10.py's `years` / `years_p1` (its years() code, copied),
at each mode's target_fpr 0.0001 threshold:
- human groups: per-paper share of the mode's sentences scoring strictly above the threshold, on the 2,400 human papers
  of WORK_NEW (recalibrate.EXCLUDE dropped); over_human_p99 against those papers at the same threshold;
- 2023-2026 groups: OUT_YD (year-diagnostic-v1 rescored with Oct 10) with YD_TEXT, same rule;
- ICLR 2027: flag counts from FLAGS (flags_p01_oct10.py output; sentences at or above the threshold, as the site counts),
  mode key m + '0.1' / m + '0.01'; document_score_median from ICLR_DOCS (pdf_sha256 -> Oct 10 document probability).
Checks before writing: the 0.0001 threshold recomputed with ca.threshold_at on the 2,400 papers equals the stored one,
the flag file's thresholds equal the stored ones, and years_p1 recomputed with this code equals the stored years_p1
exactly. With --write, inserts 'years_p01' right after 'years_p1' (other keys unchanged; indent=1 as score_map_oct10.py writes them).
Usage: years_p01_oct10.py WORK_NEW OUT_YD YD_TEXT FLAGS.json ATLAS_INDEX.parquet ICLR_DOCS.json JSON_DIR [--write]
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent; RESEARCH = HERE.parent
MAIN_RESEARCH = RESEARCH
for p in (RESEARCH / 'atlas-calibration-fix-20261011', MAIN_RESEARCH / 'atlas-bib-exclusion-20261008', RESEARCH / 'atlas-rescore-moe-20261007'):
    sys.path.insert(0, str(p))
import build_moe_calibration as bm  # noqa: E402
import cal_analyze as ca  # noqa: E402
import recompute as rc  # noqa: E402
from bib import MODES, included, section_of, sections  # noqa: E402
from recalibrate import EXCLUDE  # noqa: E402

ARCHIVE = 'Archive papers, 2020–2022'
TARGET = ('nobib', 'body')
KEY = {0.001: '0.1', 0.0001: '0.01'}


def main(work_new, out_yd, yd_text, flags_path, atlas_index, iclr_docs, json_dir, write=False):
    work_new, json_dir = Path(work_new), Path(json_dir); ok = True
    by_new, _, _ = rc.human_scores_by_mode(work_new); cal = {m: {k: v for k, v in by_new[m].items() if k not in EXCLUDE} for m in TARGET}
    assert all(len(cal[m]) == 2400 for m in TARGET)
    meta = {p['id']: p for p in json.loads((work_new / 'selection.json').read_text())['papers']}
    _, cdocs = ca.load_tokens(work_new / 'out-cal')
    files = {m: json_dir / f'qwen36-35b-a3b-oct10-{m}.json' for m in TARGET}
    data = {m: json.loads(files[m].read_text()) for m in TARGET}
    thr = {m: {t['target_fpr']: t['threshold'] for t in data[m]['thresholds']} for m in TARGET}
    flags = json.loads(Path(flags_path).read_text())
    for m in TARGET:
        got = ca.threshold_at(np.concatenate(list(cal[m].values())), 0.0001); same = got == thr[m][0.0001]; ok &= same
        print(f'{m:6s} 0.01% threshold stored {thr[m][0.0001]!r} recomputed {got!r} {"OK" if same else "FAIL"}')
        for f, s in KEY.items(): assert flags['thresholds'][m + s] == thr[m][f], (m, f)
    ytok, ydocs = ca.load_tokens(Path(out_yd)); ytext, ymeta = {}, {}
    for f in sorted((Path(yd_text) / 'data').glob('*.parquet')):
        for r in pq.read_table(f, columns=['id', 'text', 'conference', 'year', 'source']).to_pylist(): ytext[r['id']] = r.pop('text'); ymeta[r['id']] = r
    assert set(ytok) == set(ytext), (len(ytok), len(ytext))
    yd = {}
    for pid, z in ytok.items():
        s, starts = rc.sentence_scores_spans(ytext[pid], z['offsets'], z['probs']); spans, tail = sections(ytext[pid])
        sec = [section_of(a, spans, tail) for a in starts]
        yd[pid] = {m: (s[np.array([included(x, m) for x in sec], bool)] if len(sec) else s) for m in TARGET}
    id2pdf = {r['id']: r['pdf_sha256'] for r in pq.read_table(atlas_index, columns=['id', 'pdf_sha256']).to_pylist()}
    idocs = json.loads(Path(iclr_docs).read_text())
    share = lambda s, t: float((s > t).mean()) if len(s) else 0.0

    def years(m, f):  # build_calibration_oct10.years, with the flag key for 0.1% / 0.01%
        t = thr[m][f]; key = m + KEY[f]
        hs = {k: share(s, t) for k, s in cal[m].items()}; h99 = float(np.quantile(list(hs.values()), .99)); res = {}
        grp = defaultdict(list)
        for k, sh in hs.items():
            pid = k[4:] if k.startswith('old:') else k
            g = (ARCHIVE, 2022) if k.startswith('old:') else (f"{bm.VENUE.get(meta[pid]['venue'], meta[pid]['venue'])} {meta[pid]['year']}", meta[pid]['year'])
            grp[g].append({'share': sh, 'doc': cdocs.get(pid, {}).get('document_prob')})
        for (label, y), ps in grp.items(): res[label] = bm.row(label, 'human', y, ps, h99)
        ygrp = defaultdict(list)
        for pid, sc in yd.items():
            mt = ymeta[pid]; ygrp[(mt['source'], mt['conference'], mt['year'])].append({'share': share(sc[m], t), 'doc': ydocs.get(pid, {}).get('document_prob')})
        for (src, venue, year), ps in sorted(ygrp.items()):
            s = {'archive': 'archive', 'openreview-2026-10': 'OpenReview download'}.get(src, src); label = f'{bm.VENUE.get(venue, venue)} {year}, {s}'
            res[label] = bm.row(label, 'iclr' if venue == 'iclr' else 'other', int(year), ps, h99)
        ps = [{'share': v['modes'][key][0] / max(v['modes'][key][1], 1), 'doc': idocs.get(id2pdf[i])} for i, v in flags['papers'].items()]
        res['ICLR 2027 submissions'] = bm.row('ICLR 2027 submissions', 'iclr', 2027, ps, h99)
        return res, h99, float(np.mean(np.array(list(hs.values())) > h99)), hs

    out = {}
    for m in TARGET:
        d = data[m]; labels = [r['label'] for r in d['years']]
        r1, _, _, _ = years(m, 0.001); rebuilt = [r1[l] for l in labels]; same = rebuilt == d['years_p1']; ok &= same
        print(f'{m:6s} years_p1 recomputed == stored: {same}')
        r01, h99, over, hs = years(m, 0.0001); assert set(r01) == set(labels)
        hv = np.array(list(hs.values()))
        print(f'{m:6s} 0.01%: human p99 {h99:.5f} (above {over:.4f}); all 2,400 human papers: median {np.median(hv)} mean {hv.mean():.6f} share with a flag {(hv > 0).mean():.4f}')
        for l in labels:
            a = r01[l]; print(f"   {l:32s} n {a['papers']:6,}  median {a['flagged_share_median']} p25-p75 {a['flagged_share_p25']}-{a['flagged_share_p75']} over_p99 {a['over_human_p99']}")
        new = {}
        for k, v in d.items():
            if k == 'years_p01': continue
            new[k] = v
            if k == 'years_p1': new['years_p01'] = [r01[l] for l in labels]
        out[files[m]] = new
    if not ok: sys.exit('checks failed; nothing written')
    if write:
        for p, x in out.items(): p.write_text(json.dumps(x, indent=1, ensure_ascii=False) + '\n')  # format of score_map_oct10.py
        print('wrote', *[p.name for p in out])


if __name__ == '__main__':
    main(*sys.argv[1:8], write='--write' in sys.argv[8:])
