"""Calibration page edit curves from the three held-out small-edit sets (Oct 12 2026).

The page's 'One-sentence edits', 'Two-sentence edits' and 'Paragraph edits' curves used the evaluation suite's 108 GPT-6
Luna edits per size (111 / 228 / 1,273 AI sentences). This replaces them with the edited rows of heldout-writers,
t21-heldout-v2 and wide1-heldout (build_edit_rows.py: rows sharing a paper or a 10-word sequence with either model's
training data removed), each row scored in one pass like the AI sections (job.sh on the H200), and adds the same
curves for the new final-moe. Sentence gold, negatives, grid and thresholds follow rebuild_2400.py: the 2,400 human
papers of each counting mode, tpr_at_1pct at the published 1% threshold (final-moe: its own 2,400-paper threshold).

pos TOKENS_DIR EDIT_ROWS.jsonl.gz OUT.json
    AI sentences per size, and per row (with set and writer), exactly as sections_roc.py pos: regex sentences, a
    sentence kept only when all its non-whitespace tokens share one label, score = mean token AI probability.
curves POS_OCT7.json POS_NEW.json SEC_OCT7.json SEC_NEW.json WORK WORK_NEW JSON_DIR [--write]
    Before writing, the stored 'AI sections' curves must be reproduced from SEC_* with these negatives and thresholds,
    which checks that the new curves use the same human side. The AI-sections curves are then rebuilt without the
    section rows in sec-ai-text-matches.json (177 of 497: the AI-written section shares a 10-word sequence with the
    training data; Alice's choice, Oct 12; rows that share only human context are kept). Then writes the edit curves and their final-moe
    versions into the four qwen36-35b-a3b-final*.json files and a per-writer breakdown to OUT next to POS_OCT7.
WORK / WORK_NEW: recompute.py work folders (out-cal/, cal-text/, old-text/, overlap.json, selection.json).
"""
import gzip, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; RESEARCH = HERE.parent
MAIN_RESEARCH = RESEARCH  # atlas-bib-exclusion-20261008 is tracked since Oct 12
for p in (RESEARCH / 'atlas-calibration-fix-20261011', MAIN_RESEARCH / 'atlas-bib-exclusion-20261008', RESEARCH / 'atlas-rescore-moe-20261007',
          RESEARCH / 'evaluation/space-suite-v1/package', RESEARCH / 'human-calibration-20261006'):
    sys.path.insert(0, str(p))

SIZES = {'one': 'One-sentence edits', 'two': 'Two-sentence edits', 'para': 'Paragraph edits'}
NEW = ', new final-moe'


def pos(tokens_dir, rows_path, out):
    import cal_analyze as ca
    from common import labels_from_regions
    from sentence_scores_space import sentence_spans
    rows = {r['id']: r for r in map(json.loads, gzip.open(rows_path, 'rt'))}
    tok = ca.load_tokens(tokens_dir)[0]; assert set(tok) == set(rows), (len(tok), len(rows))
    res = defaultdict(list); per_row = {}
    for rid, d in tok.items():
        r = rows[rid]; text = r['text']; off = np.asarray(d['offsets']).reshape(-1, 2); prob = d['probs'].astype(np.float64)
        ys = np.asarray(labels_from_regions(text, off, r['regions'])); valid = np.array([bool(text[a:b].strip()) for a, b in off])
        sent = []
        for a, b in sentence_spans(text):
            use = valid & (off[:, 1] > a) & (off[:, 0] < b)
            if not use.any(): continue
            labs = set(ys[use].tolist())
            if len(labs) == 1 and next(iter(labs)) in (0, 1): sent.append([round(float(prob[use].mean()), 5), int(next(iter(labs)))])
        per_row[rid] = {'set': r['set'], 'writer': r['writer'], 'size': r['condition'], 'sentences': sent}
        res[r['condition']] += [s for s, g in sent if g == 1]
    Path(out).write_text(json.dumps({'by_kind': dict(res), 'rows': per_row}))
    print({k: (len(v), round(float(np.median(v)), 4)) for k, v in res.items()},
          'rows without an AI sentence', sum(1 for v in per_row.values() if not any(g == 1 for _, g in v['sentences'])))


def curves(pos_oct7, pos_new, sec_oct7, sec_new, work, work_new, json_dir, write=False):
    import build_moe_calibration as bm
    import extend_roc_grid as eg
    import recompute as rc
    import rebuild_2400 as rb
    from recalibrate import EXCLUDE
    P = {m: json.loads(Path(p).read_text()) for m, p in (('oct7', pos_oct7), ('new', pos_new))}
    sec = lambda p: np.array((lambda S: S['section_draft'] + S['section_rewrite'])(json.loads(Path(p).read_text())['by_kind']))
    sec_o, sec_n = sec(sec_oct7), sec(sec_new)
    drop_sec = set(json.loads((HERE / 'sec-ai-text-matches.json').read_text()))
    def sec_kept(p):
        rows = json.loads(Path(p).read_text())['rows']; assert drop_sec <= set(rows), 'section ids not found'
        return np.array([x for i, r in rows.items() if i not in drop_sec for x, g in r if g == 1]), len(rows) - len(drop_sec)
    (sec_ko, n_rows), (sec_kn, _) = sec_kept(sec_oct7), sec_kept(sec_new)
    by_mode, _, _ = rc.human_scores_by_mode(Path(work)); by_new, _, _ = rc.human_scores_by_mode(Path(work_new))
    drop = lambda cal: {k: v for k, v in cal.items() if k not in EXCLUDE}
    ok = True; out = {}; breakdown = {}
    for mode, fname in eg.FILES.items():
        path = Path(json_dir) / fname; data = json.loads(path.read_text()); stored = {c['name']: c for c in data['sentence_curves']}
        cal, cal_new = drop(by_mode[mode]), drop(by_new[mode])
        thr, thr_moe = rb.t1(cal), rb.t1(cal_new)
        assert bm.r5(thr) == next(t['threshold'] for t in data['thresholds'] if t['target_fpr'] == 0.01), mode
        neg, nm = np.concatenate(list(cal.values())), np.concatenate(list(cal_new.values()))
        for name, p, n, th in (('AI sections', sec_o, neg, thr), ('AI sections' + NEW, sec_n, nm, thr_moe)):
            bad = rb.same_curve(rb.sent_curve(name, p, n, th), stored[name]); ok &= not bad
            print(f"{mode:6s} {name:27s} reproduce {'OK' if not bad else 'FAIL ' + str(bad)}", flush=True)
        new_curves = []
        for size, name in SIZES.items():
            new_curves.append(rb.sent_curve(name, np.array(P['oct7']['by_kind'][size]), neg, thr))
            c = rb.sent_curve(name + NEW, np.array(P['new']['by_kind'][size]), nm, thr_moe); c['other_model'] = True; new_curves.append(c)
        sk = rb.sent_curve('AI sections', sec_ko, neg, thr), rb.sent_curve('AI sections' + NEW, sec_kn, nm, thr_moe); sk[1]['other_model'] = True
        new_curves += list(sk)
        data['sentence_curves'] = new_curves
        for c in new_curves:
            g = dict(zip(c['fpr'], c['tpr']))
            print(f"{mode:6s} {c['name']:38s} n_ai {c['n_ai']:6,} n_human {c['n_human']:,} auc {c['auc']} tpr@0.1% {g[0.001]} tpr_at_1pct {c['tpr_at_1pct']}", flush=True)
        if mode == 'all':  # recall at the published 1% threshold per set and writer, for the write-up
            for m, th in (('oct7', thr), ('new', thr_moe)):
                cell = defaultdict(list)
                for r in P[m]['rows'].values():
                    s = [x for x, g in r['sentences'] if g == 1]
                    cell[(r['size'], r['set'], r['writer'])] += s; cell[(r['size'], 'all', r['writer'])] += s
                breakdown[m] = {' | '.join(k): {'n': len(v), 'tpr_at_1pct': round(float(np.mean(np.array(v) > th)), 4)} for k, v in sorted(cell.items()) if v}
        out[path] = data
    (Path(pos_oct7).parent / 'edit-breakdown.json').write_text(json.dumps(breakdown, indent=1))
    if not ok: sys.exit('stored AI-sections curves not reproduced; nothing written')
    if write:
        for path, data in out.items(): path.write_text(json.dumps(data, separators=(',', ':'), ensure_ascii=False))
        print('wrote', len(out), 'files;', n_rows, 'section rows kept')


if __name__ == '__main__':
    if sys.argv[1] == 'pos': pos(*sys.argv[2:5])
    else: curves(*sys.argv[2:9], write='--write' in sys.argv[9:])
