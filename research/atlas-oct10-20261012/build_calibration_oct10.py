"""Calibration page data for the Oct 10 final MoE only (qwen36-35b-a3b-oct10-{all,nobib,noapp,body}.json), Oct 12 2026.

Model: final-moe = abl-noseg-nomix, single checkpoint stage2-epoch0 (score_atlas_fast.py, the scorer of the Oct 10
Atlas). Same schema and rules as the Oct 7 files (app/lib/model-calibration/qwen36-35b-a3b-final*.json), every number
from Oct 10 scores:
- thresholds: recal-final-moe.json (recalibrate.py on the Oct 10 human scores, 2,400 papers), non-'/before' blocks,
  targets 0.1% and 1% (as the Oct 7 files), values copied at full precision; 'stable' by build_moe_calibration's rule.
  Checked: ca.threshold_at on WORK_NEW's 2,400 papers per mode reproduces them exactly.
- sentence_curves: the '<set>, new final-moe' curves already in the Oct 7 files (edits_roc.py), renamed to '<set>'
  without other_model. Checked first: rebuilt from POS_EDITS / POS_SEC (sections minus sec-ai-text-matches.json) against
  the Oct 10 human scores of the mode (2,400 papers) and the Oct 10 1% threshold, they must equal the stored curves.
- document_curves / document_threshold: build_moe_calibration.doc_roc / doc_block on Oct 10 document probabilities of the
  2,400 human papers (WORK_NEW/out-cal) and the suite's 120 AI manuscripts (OUT_MAN, scored Oct 12 on the H200); the
  shared RNG (seed 20261007) is consumed mode by mode in bib.MODES order, as before.
- years (1%) / years_p1 (0.1%): rebuild_years_2400 rules at the mode's published thresholds: per-paper share of the
  mode's sentences scoring strictly above the threshold; over_human_p99 against the 2,400 human papers at the same
  threshold; document_score_median = median Oct 10 document probability. Human groups from WORK_NEW; 2023-2026 groups
  from OUT_YD (year-diagnostic-v1 rescored with Oct 10 on Oct 12) with YD_TEXT; ICLR 2027 from the Oct 10 Atlas flag
  counts (flags-oct10.json: sentences at or above the threshold, as the site counts) and Oct 10 document
  probabilities (ICLR_DOCS: pdf_sha256 -> document_prob from atlas-rescore/out-final-moe-20261011).
Usage: build_calibration_oct10.py WORK_NEW RECAL.json POS_EDITS.json POS_SEC.json OUT_MAN OUT_YD YD_TEXT FLAGS_OCT10.json
       ATLAS_INDEX.parquet ICLR_DOCS.json JSON_DIR [--write]
"""
import json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent; RESEARCH = HERE.parent
MAIN_RESEARCH = RESEARCH  # atlas-bib-exclusion-20261008 is tracked since Oct 12
for p in (RESEARCH / 'atlas-calibration-fix-20261011', MAIN_RESEARCH / 'atlas-bib-exclusion-20261008', RESEARCH / 'atlas-rescore-moe-20261007'):
    sys.path.insert(0, str(p))
import build_moe_calibration as bm  # noqa: E402
import cal_analyze as ca  # noqa: E402
import extend_roc_grid as eg  # noqa: E402
import rebuild_2400 as rb  # noqa: E402
import recompute as rc  # noqa: E402
from bib import MODES, included, section_of, sections  # noqa: E402
from recalibrate import EXCLUDE  # noqa: E402

OLD = {'all': 'qwen36-35b-a3b-final.json', 'nobib': 'qwen36-35b-a3b-final-nobib.json', 'noapp': 'qwen36-35b-a3b-final-noapp.json', 'body': 'qwen36-35b-a3b-final-body.json'}
NEW = {m: f'qwen36-35b-a3b-oct10-{m}.json' for m in OLD}
SIZES = {'one': 'One-sentence edits', 'two': 'Two-sentence edits', 'para': 'Paragraph edits'}
SUFFIX = ', new final-moe'
ARCHIVE = 'Archive papers, 2020–2022'
SHOWN = (0.001, 0.01)

DETAIL = ('LoRA classifier on Qwen3.6-35B-A3B (mixture of experts), trained on October 10, 2026 in one pass over the same 72,000 '
          'training passages as the October 7 run: small edits from Claude, GPT-6 Luna, GPT-6.1 Sol and DeepSeek, paragraph edits, AI '
          'sections and human paper text. Trained with token, sentence and document losses only (no segment or mixed-label loss); '
          'one checkpoint, taken at the end of the pass. Papers are scored in 510-token windows that do not overlap.')
DESCRIPTION = ("2,400 human papers (human by project rule: 2022 or earlier). 2,188 are ICLR 2019–2022, NeurIPS 2021–2022 and CoRL "
               "2021–2022 papers sampled by venue and year from the site's PDF collection, after removing those that share text with "
               "the model’s training data; 212 are the 2020–2022 archive papers used before October 7. Three archive papers labelled "
               "TMLR 2022 were removed on October 11 because they cite 2023–2025 work.")
CURVES_NOTE = ('Checked for leakage: 1,196 edited rows were removed because their paper is in the training data or they share a '
               'sequence of 10 or more words with it; 5,049 rows remain. This model was chosen among the October 10 training runs '
               'partly on its scores on these held-out edit sets, so these curves may be slightly optimistic.')


def stable(t): return t['heldout_ci95_paper_bootstrap'][0] <= t['target_fpr'] <= t['heldout_ci95_paper_bootstrap'][1] and t['heldout_ci95_paper_bootstrap'][1] <= 2 * t['target_fpr']


def main(work_new, recal, pos_edits, pos_sec, out_man, out_yd, yd_text, flags_path, atlas_index, iclr_docs, json_dir, write=False):
    work_new, json_dir = Path(work_new), Path(json_dir); ok = True
    recal = json.loads(Path(recal).read_text())['modes']
    by_new, _, _ = rc.human_scores_by_mode(work_new); cal = {m: {k: v for k, v in by_new[m].items() if k not in EXCLUDE} for m in MODES}
    assert all(len(cal[m]) == 2400 for m in MODES)
    meta = {p['id']: p for p in json.loads((work_new / 'selection.json').read_text())['papers']}
    _, cdocs = ca.load_tokens(work_new / 'out-cal')
    # --- thresholds: recal values must be reproduced exactly from the Oct 10 human scores
    thr = {}
    for m in MODES:
        neg = np.concatenate(list(cal[m].values())); blk = recal[m]
        assert blk['n_papers'] == 2400 and blk['n_sentences'] == len(neg), (m, blk['n_papers'], blk['n_sentences'], len(neg))
        for t in blk['thresholds']:
            got = ca.threshold_at(neg, t['target_fpr']); same = got == t['threshold']; ok &= same
            print(f"{m:6s} threshold {t['target_fpr']:<6} recal {t['threshold']!r} recomputed {got!r} {'OK' if same else 'FAIL'}")
        thr[m] = {t['target_fpr']: t['threshold'] for t in blk['thresholds']}
    # --- sentence curves: stored 'new final-moe' curves, checked against a rebuild from the Oct 10 scores
    PE = json.loads(Path(pos_edits).read_text())['by_kind']
    drop_sec = set(json.loads((RESEARCH / 'calibration-edits-20261012' / 'sec-ai-text-matches.json').read_text()))
    srows = json.loads(Path(pos_sec).read_text())['rows']; assert drop_sec <= set(srows)
    sec_pos = np.array([x for i, r in srows.items() if i not in drop_sec for x, g in r if g == 1])
    pos = {**{n: np.array(PE[k]) for k, n in SIZES.items()}, 'AI sections': sec_pos}
    # --- document scores
    hum_ids = [k[4:] if k.startswith('old:') else k for k in cal['all']]
    hd = {c: np.array([cdocs[p][c] for p in hum_ids if cdocs.get(p, {}).get(c) is not None], float) for c in ('document_prob', 'token_prob_mean')}
    _, mdocs = ca.load_tokens(Path(out_man)); assert len(mdocs) == 120, len(mdocs)
    doc_curves = []
    for c, name in (('document_prob', 'Document head'), ('token_prob_mean', 'Mean token score')):
        p = np.array([mdocs[i][c] for i in sorted(mdocs) if mdocs[i].get(c) is not None], float)
        doc_curves.append(bm.curve(name, bm.roc(p, hd[c], ca.threshold_at(hd[c], 0.01))))
    bm.rng = np.random.default_rng(20261007); dblock = {m: bm.doc_block(hd['document_prob']) for m in MODES}
    # --- year diagnostic 2023-2026: Oct 10 tokens, per paper and mode
    ytok, ydocs = ca.load_tokens(Path(out_yd)); ytext, ymeta = {}, {}
    for f in sorted((Path(yd_text) / 'data').glob('*.parquet')):
        for r in pq.read_table(f, columns=['id', 'text', 'conference', 'year', 'source']).to_pylist(): ytext[r['id']] = r.pop('text'); ymeta[r['id']] = r
    assert set(ytok) == set(ytext), (len(ytok), len(ytext))
    yd = {}
    for pid, z in ytok.items():
        s, starts = rc.sentence_scores_spans(ytext[pid], z['offsets'], z['probs']); spans, tail = sections(ytext[pid])
        sec = [section_of(a, spans, tail) for a in starts]
        yd[pid] = {m: (s[np.array([included(x, m) for x in sec], bool)] if len(sec) else s) for m in MODES}
    # --- ICLR 2027: Atlas flags (>=), Oct 10 document probabilities
    flags = json.loads(Path(flags_path).read_text())
    id2pdf = {r['id']: r['pdf_sha256'] for r in pq.read_table(atlas_index, columns=['id', 'pdf_sha256']).to_pylist()}
    idocs = json.loads(Path(iclr_docs).read_text())
    for m in MODES:
        assert flags['thresholds'][m] == thr[m][0.01] and flags['thresholds'][m + '0.1'] == thr[m][0.001], m
    share = lambda s, t: float((s > t).mean()) if len(s) else 0.0

    def years(m, f):
        t = thr[m][f]; key = m if f == 0.01 else m + '0.1'
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
        over = float(np.mean(np.array(list(hs.values())) > h99))
        return res, h99, over

    out = {}
    for m, fname in OLD.items():
        old = json.loads((json_dir / fname).read_text()); stored = {c['name']: c for c in old['sentence_curves']}
        neg = np.concatenate(list(cal[m].values())); t1 = thr[m][0.01]
        curves = []
        for name in (*SIZES.values(), 'AI sections'):
            s = stored[name + SUFFIX]; rebuilt = rb.sent_curve(name + SUFFIX, pos[name], neg, t1)
            bad = rb.same_curve(rebuilt, s); bad += [] if s['name'] == rebuilt['name'] else ['name']; ok &= not bad
            print(f"{m:6s} {name:20s} stored vs rebuilt from Oct 10 scores: {'OK' if not bad else 'FAIL ' + str(bad)}  auc {s['auc']} n_ai {s['n_ai']} n_human {s['n_human']} tpr_at_1pct {s['tpr_at_1pct']}")
            curves.append({'name': name, **{k: s[k] for k in ('auc', 'n_ai', 'n_human', 'tpr_at_1pct', 'fpr', 'tpr')}})
        rows1, h1, o1 = years(m, 0.01); rows01, h01, o01 = years(m, 0.001)
        labels = [r['label'] for r in old['years']]
        assert set(rows1) == set(labels) == set(rows01), set(rows1) ^ set(labels)
        print(f"{m:6s} human p99 of flagged share 1% {h1:.5f} (human papers above it {o1:.4f}); 0.1% {h01:.5f} (above {o01:.4f})")
        for l in labels:
            a, b, o = rows1[l], rows01[l], next(r for r in old['years'] if r['label'] == l)
            print(f"   {l:32s} n {a['papers']:6,} (Oct 7 {o['papers']:6,})  1%: median {a['flagged_share_median']} over_p99 {a['over_human_p99']}  0.1%: median {b['flagged_share_median']} "
                  f"p25-p75 {b['flagged_share_p25']}-{b['flagged_share_p75']} over_p99 {b['over_human_p99']}  doc {a['document_score_median']}")
        tb = {t['target_fpr']: t for t in recal[m]['thresholds']}
        n_top = int((hd['document_prob'] > ca.threshold_at(hd['document_prob'], 0.01)).sum()) + 1
        db = dict(dblock[m]); db['note'] = f'The 1% threshold rests on the top {n_top} of the 2,400 human papers.'
        assert db['note'].replace('the 2,400', '2400') == dblock[m]['note'], (db['note'], dblock[m]['note'])
        p01 = thr[m][0.001]; p1 = t1
        data = {
            'model': {'id': 'qwen36-35b-a3b-oct10', 'name': 'Qwen3.6 35B-A3B MoE · Oct 10', 'detail': DETAIL},
            'calibration_set': {'papers': 2400, 'sentences': recal[m]['n_sentences'], 'description': DESCRIPTION, 'method': old['calibration_set']['method']},
            'sentence_score': old['sentence_score'],
            'thresholds': [{'target_fpr': f, 'threshold': tb[f]['threshold'], 'threshold_ci95': tb[f]['threshold_ci95'], 'heldout_fpr': tb[f]['heldout_fpr'],
                            'heldout_ci95': tb[f]['heldout_ci95_paper_bootstrap'], 'heldout_sentences': tb[f]['heldout_sentences'], 'heldout_papers': tb[f]['heldout_papers'],
                            'stable': stable(tb[f])} for f in SHOWN],
            'sentence_curves': curves,
            'document_curves': doc_curves,
            'document_threshold': db,
            'years': [rows1[l] for l in labels],
            'years_p1': [rows01[l] for l in labels],
            'caveats': [old['caveats'][0], old['caveats'][1],
                        (f'The thresholds are low raw probabilities (about {p01:.3f} at 0.1% and {p1:.4f} at 1%): this model gives most human sentences '
                         'scores near zero, so moderately scored sentences are flagged. Very short or badly extracted texts can therefore show high flagged shares.'
                         if p01 < 0.5 else
                         f'With bibliography sentences counted, the 0.1% threshold is high (raw probability about {p01:.3f}) because some human '
                         f'bibliography sentences score near 1; the 1% threshold is low (about {p1:.4f}). Very short or badly extracted texts can show high flagged shares.'),
                        old['caveats'][3]],
            'counting': old['counting'],
            'curves_note': CURVES_NOTE,
        }
        assert [k for k in data if k != 'candidates'] and set(old) - set(data) == {'candidates'}, set(old) ^ set(data)
        out[json_dir / NEW[m]] = data
        for t in data['thresholds']: print(f"{m:6s} threshold {t['target_fpr']}: {t['threshold']!r} heldout {t['heldout_fpr']:.6f} {t['heldout_ci95']} stable {t['stable']}")
        print(f"{m:6s} document: {[(c['name'], c['auc'], c['tpr_at_1pct'], c['n_ai'], c['n_human']) for c in doc_curves]} block {json.dumps(db)}")
    if not ok: sys.exit('checks failed; nothing written')
    if write:
        for path, data in out.items(): path.write_text(json.dumps(data, separators=(',', ':'), ensure_ascii=False))
        print('wrote', *[p.name for p in out])


if __name__ == '__main__':
    main(*sys.argv[1:12], write='--write' in sys.argv[12:])
