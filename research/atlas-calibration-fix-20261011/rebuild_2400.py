"""MoE Calibration page curves on the same 2,400 human papers as its thresholds (Oct 12 2026).

recalibrate.py set the page's thresholds without the three archive papers in its EXCLUDE dict (2,400 papers), but the
curves and the document threshold still used all 2,403. This rebuilds, in app/lib/model-calibration/
qwen36-35b-a3b-final*.json (all four counting modes), with the same code as the stored values:
- sentence_curves: 'One-sentence edits', 'Two-sentence edits', 'Paragraph edits' (extend_roc_grid.py: suite edit
  sentences against recompute.human_scores_by_mode(WORK)), 'AI sections' (sections_roc.py curves: t21 section rows,
  oct7 scores) and 'AI sections, new final-moe' (negatives = final-moe's own human scores, WORK_NEW). Grid NEW_GRID.
  tpr_at_1pct at the published 1% threshold of the page (final-moe curve: final-moe's own 2,400-paper 1% threshold).
- document_curves and document_threshold (build_moe_calibration.py doc_roc / doc_block; the shared RNG is consumed
  mode by mode in bib.MODES order, as build_calibration_modes.py did).
- the 'Oct 5 set (archive, ≤2022)' year row: same flagged-share rule and the same 1% threshold as the other year rows
  (the pre-Oct 11 2,403-paper threshold, which every year row and over_human_p99 still use).
- text: document_threshold.note, caveats[0] (paper count) and the caveat quoting the mode's 1% threshold.

Before anything is written, every stored value must be reproduced from all 2,403 papers: n_ai, n_human, auc (5 dp),
tpr within 1e-5 on every grid point, tpr_at_1pct, the document threshold block and the nine human year rows.
Usage: rebuild_2400.py WORK WORK_NEW SUITE_SENTENCES.json.gz OUT_SUITE POS_OCT7.json POS_NEW.json JSON_DIR [--write]
WORK / WORK_NEW as for recompute.py (out-cal/, cal-text/, old-text/, overlap.json, selection.json) for the oct7 and the
new final-moe scores; OUT_SUITE holds the suite's data/summary-*.parquet; POS_* are sections_roc.py pos outputs.
"""
import gzip, json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'atlas-bib-exclusion-20261008')); sys.path.insert(0, str(HERE.parent / 'atlas-rescore-moe-20261007'))
import build_moe_calibration as bm  # noqa: E402
import cal_analyze as ca  # noqa: E402
import extend_roc_grid as eg  # noqa: E402
import recompute as rc  # noqa: E402
from bib import MODES  # noqa: E402
from recalibrate import EXCLUDE  # noqa: E402

OLD_DOC_GRID = bm.FPR_GRID
EDITS = {'One-sentence edits': 'papers_one', 'Two-sentence edits': 'papers_two', 'Paragraph edits': 'papers_para'}
ARCHIVE = 'Oct 5 set (archive, ≤2022)'


def sent_curve(name, pos, neg, thr):
    bm.FPR_GRID = eg.NEW_GRID
    try: c = bm.curve(name, bm.roc(pos, neg, thr))
    finally: bm.FPR_GRID = OLD_DOC_GRID
    c['fpr'] = [eg.fpr_value(v) for v in eg.NEW_GRID]; return c


def same_curve(a, b):
    bad = [k for k in ('n_ai', 'n_human', 'auc', 'tpr_at_1pct', 'fpr') if a[k] != b[k]]
    if len(a['tpr']) != len(b['tpr']) or max(abs(x - y) for x, y in zip(a['tpr'], b['tpr'])) > 1e-5: bad.append('tpr')
    return bad


def t1(cal):
    return ca.threshold_at(np.concatenate(list(cal.values())), 0.01)


def year_rows(cal, docs, old_ids, meta, thr):
    """Human year rows as build_moe_calibration.main: per-paper share of sentences above thr; returns {label: row}."""
    grp = defaultdict(list)
    for k, s in cal.items():
        pid = k[4:] if k.startswith('old:') else k
        label = ARCHIVE if k.startswith('old:') else f"{bm.VENUE.get(meta[pid]['venue'], meta[pid]['venue'])} {meta[pid]['year']}"
        year = 2022 if k.startswith('old:') else meta[pid]['year']
        grp[(label, year)].append({'share': float((s > thr).mean()) if len(s) else 0.0, 'doc': docs.get(pid, {}).get('document_prob')})
    return {label: bm.row(label, 'human', year, ps, None) for (label, year), ps in grp.items()}


def main(work, work_new, suite_path, out_suite, pos_oct7, pos_new, json_dir, write=False):
    work = Path(work)
    by_mode, _, _ = rc.human_scores_by_mode(work); by_new, _, _ = rc.human_scores_by_mode(work_new)
    assert set(EXCLUDE) <= set(by_mode['all']) and set(EXCLUDE) <= set(by_new['all'])
    drop = lambda cal: {k: v for k, v in cal.items() if k not in EXCLUDE}
    suite = json.load(gzip.open(suite_path, 'rt')); pos = defaultdict(list)
    for r in suite:
        if 'sentences' not in r: continue
        for k in bm.sets_for(r): pos[k] += [s for s, g in r['sentences'] if g == 1]
    sec = lambda p: np.array((lambda P: P['section_draft'] + P['section_rewrite'])(json.loads(Path(p).read_text())['by_kind']))
    sec_oct7, sec_new = sec(pos_oct7), sec(pos_new)
    # Document scores: human papers in by_mode order (= build_moe_calibration's `human` order), suite manuscripts.
    _, cdocs = ca.load_tokens(work / 'out-cal'); _, sdocs = ca.load_tokens(out_suite)
    man = [r['id'] for r in suite if r.get('profile') == 'manuscripts']
    old_ids = {k[4:] for k in by_mode['all'] if k.startswith('old:')}
    meta = {p['id']: p for p in json.loads((work / 'selection.json').read_text())['papers']}

    def doc_part(keys):
        ids = [k[4:] if k.startswith('old:') else k for k in keys]
        hd = {c: np.array([cdocs[p][c] for p in ids if cdocs.get(p, {}).get(c) is not None], float) for c in ('document_prob', 'token_prob_mean')}
        curves = []
        for c, name in (('document_prob', 'Document head'), ('token_prob_mean', 'Mean token score')):
            p = np.array([sdocs[i][c] for i in man if i in sdocs and sdocs[i].get(c) is not None], float)
            curves.append(bm.curve(name, bm.roc(p, hd[c], ca.threshold_at(hd[c], 0.01))))
        bm.rng = np.random.default_rng(20261007)
        return curves, {m: bm.doc_block(hd['document_prob']) for m in MODES}  # one draw sequence across modes, in MODES order

    docs_old, dblock_old = doc_part(list(by_mode['all'])); docs_new, dblock_new = doc_part(list(drop(by_mode['all'])))
    ok = True; out = {}; report = {}
    for mode, fname in eg.FILES.items():
        path = Path(json_dir) / fname; data = json.loads(path.read_text()); stored = {c['name']: c for c in data['sentence_curves']}
        cal_old, cal_new_set = by_mode[mode], drop(by_mode[mode]); m_old, m_new = by_new[mode], drop(by_new[mode])
        thr_old, thr = t1(cal_old), t1(cal_new_set); thr_moe = t1(m_new)
        pub = next(t['threshold'] for t in data['thresholds'] if t['target_fpr'] == 0.01)
        assert bm.r5(thr) == pub, (mode, thr, pub)
        neg_old, neg = np.concatenate(list(cal_old.values())), np.concatenate(list(cal_new_set.values()))
        nm_old, nm = np.concatenate(list(m_old.values())), np.concatenate(list(m_new.values()))
        # name -> (positives, negatives 2,403, negatives 2,400, tpr_at_1pct threshold as stored, as published now)
        spec = {**{n: (pos[k], neg_old, neg, thr_old, thr) for n, k in EDITS.items()},
                'AI sections': (sec_oct7, neg_old, neg, thr, thr), 'AI sections, new final-moe': (sec_new, nm_old, nm, thr_moe, thr_moe)}
        assert list(spec) == list(stored), (list(spec), list(stored))
        new_curves = []
        for name, (p, n_old, n_new, th_old, th_new) in spec.items():
            old = sent_curve(name, p, n_old, th_old); bad = same_curve(old, stored[name])
            new = sent_curve(name, p, n_new, th_new)
            extra = {k: v for k, v in stored[name].items() if k not in new}; new.update(extra)
            new = {k: new[k] for k in stored[name]}  # same key order
            ok &= not bad; new_curves.append(new)
            g_old, g_new = dict(zip(old['fpr'], old['tpr'])), dict(zip(new['fpr'], new['tpr']))
            report[(mode, name)] = (old, new, g_old, g_new)
            print(f"{mode:6s} {name:27s} reproduce {'OK' if not bad else 'FAIL ' + str(bad)}  n_human {old['n_human']:,} -> {new['n_human']:,} "
                  f"(-{old['n_human'] - new['n_human']:,})  auc {old['auc']} -> {new['auc']}  tpr@0.1% {g_old[0.001]} -> {g_new[0.001]}  "
                  f"tpr@1% {g_old[0.01]} -> {g_new[0.01]}  tpr_at_1pct {old['tpr_at_1pct']} -> {new['tpr_at_1pct']}", flush=True)
        # sentence-count drop must equal the three papers' sentences in this mode
        for nm_, a, b, full in (('oct7', neg_old, neg, cal_old), ('final-moe', nm_old, nm, m_old)):
            want = sum(len(full[k]) for k in EXCLUDE); assert len(a) - len(b) == want, (mode, nm_, len(a) - len(b), want)
        for o, s in zip(docs_old, data['document_curves']):
            bad = [k for k in ('name', 'n_ai', 'n_human', 'auc', 'tpr_at_1pct', 'fpr', 'tpr') if o[k] != s[k]]
            ok &= not bad; print(f"{mode:6s} {o['name']:27s} reproduce {'OK' if not bad else 'FAIL ' + str(bad)}", flush=True)
        db_old = dict(dblock_old[mode]); bad = db_old != data['document_threshold']
        ok &= not bad; print(f"{mode:6s} document_threshold reproduce {'OK' if not bad else 'FAIL ' + json.dumps(db_old)}", flush=True)
        # year rows: the stored rows use the 2,403-paper 1% threshold
        docs_by_id = cdocs
        rows_old = year_rows(cal_old, docs_by_id, old_ids, meta, thr_old)
        stored_h = {r['label']: r for r in data['years'] if r['kind'] == 'human'}
        bad = [l for l in stored_h if rows_old.get(l) != stored_h[l]]
        ok &= not bad; print(f"{mode:6s} human year rows reproduce {'OK' if not bad else 'FAIL ' + str(bad)}", flush=True)
        arch_new = year_rows(cal_new_set, docs_by_id, old_ids, meta, thr_old)[ARCHIVE]
        arch_pub = year_rows(cal_new_set, docs_by_id, old_ids, meta, thr)[ARCHIVE]
        h99 = lambda cal, th: float(np.quantile([float((s > th).mean()) if len(s) else 0.0 for s in cal.values()], .99))
        print(f"{mode:6s} archive row old {json.dumps(stored_h[ARCHIVE], ensure_ascii=False)}\n{'':7s}new (same threshold as other rows) {json.dumps(arch_new, ensure_ascii=False)}\n"
              f"{'':7s}at the published threshold {json.dumps(arch_pub, ensure_ascii=False)}\n{'':7s}h99 2,403 {h99(cal_old, thr_old):.5f} 2,400 same thr {h99(cal_new_set, thr_old):.5f} "
              f"2,400 published thr {h99(cal_new_set, thr):.5f}", flush=True)
        # assemble
        data['sentence_curves'] = new_curves
        data['document_curves'] = [{k: c[k] for k in s} for c, s in zip(docs_new, data['document_curves'])]
        data['document_threshold'] = dblock_new[mode]
        data['years'] = [arch_new if r['label'] == ARCHIVE else r for r in data['years']]
        data['caveats'][0] = data['caveats'][0].replace('2,403 human papers', '2,400 human papers')
        if mode != 'all':  # 'all' keeps "about 0.006" (still right); the others quote 4 decimals of the old threshold
            data['caveats'] = [re.sub(r'probability about 0\.\d+', f'probability about {thr:.4f}', c) for c in data['caveats']]
        print(f"{mode:6s} document_threshold new {json.dumps(dblock_new[mode])}", flush=True)
        out[path] = data
    if not ok: sys.exit('not reproduced; nothing written')
    if write:
        for path, data in out.items(): path.write_text(json.dumps(data, separators=(',', ':'), ensure_ascii=False))
        print('wrote', len(out), 'files')


if __name__ == '__main__':
    main(*sys.argv[1:8], write='--write' in sys.argv[8:])
