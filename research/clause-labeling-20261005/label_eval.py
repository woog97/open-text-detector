r"""Score soft n-gram labeling against known per-character provenance.

For each T unit y (from a given splitter):
  L(y) = mean of character 5-gram and word-bigram containment of y in S (lowercased,
         whitespace-normalised); 1.0 means every n-gram of y occurs in S.
  E(y) = max embedding cosine to an S unit or adjacent S-unit pair (from space_units.py).
  Human if L >= tH; AI-Assisted if L >= tA or E >= tE; else AI-Generated (report §3.5, Alg. 1).
Thresholds are tuned per splitter on the dev half (by paper) to maximise character-level
macro-F1, then frozen and scored on the test half. Two truth conventions are scored:
  ours    polish = AI-Assisted
  report  polish = Human ("exact or near-exact lexical match" counts as Human in §3.5)

Tokens for word n-grams are \w+ runs and single punctuation marks, so a unit that starts
mid-token (e.g. "; see") still matches the source.

Usage: label_eval.py PAIRS.jsonl UNITS.jsonl OUT.json
"""
import json, re, sys
from collections import defaultdict

import numpy as np

H, AA, AG = 0, 1, 2
NAMES = ['human', 'assisted', 'generated']


def norm(t):
    return ' '.join(t.lower().split())


TOK = re.compile(r"\w+|[^\w\s]")


def grams(t):
    t = norm(t); w = TOK.findall(t)
    return ({t[i:i + 5] for i in range(max(1, len(t) - 4))}, {tuple(w[i:i + 2]) for i in range(len(w) - 1)}, set(w))


def lexical(y, s_grams):
    # Units shorter than the n-gram size fall back to shorter grams (whole string, unigrams).
    c, w, u = grams(y)
    char = len(c & s_grams[0]) / len(c) if norm(y) and len(norm(y)) >= 5 else float(norm(y) in norm_cache(s_grams))
    word = len(w & s_grams[1]) / len(w) if w else len(u & s_grams[2]) / max(1, len(u))
    return 0.5 * (char + word)


_NORM = {}


def norm_cache(s_grams):
    return _NORM[id(s_grams)]


def char_truth(p, convention):
    lab = np.zeros(len(p['target']), dtype=np.int8); op = np.full(len(p['target']), 'gap', dtype=object)
    for t in p['truth']:
        v = {'human': H, 'assisted': AA, 'generated': AG}[t['label']]
        if convention == 'report' and t['op'] == 'polish':
            v = H
        lab[t['start']:t['end']] = v; op[t['start']:t['end']] = t['op']
    return lab, op


def predict(units, L, E, th):
    tH, tA, tE = th
    return [H if l >= tH else AA if (l >= tA or e >= tE) else AG for l, e in zip(L, E)]


def paint(n, units, labels):
    out = np.full(n, H, dtype=np.int8)  # whitespace between units counts as Human
    for (a, b), v in zip(units, labels):
        out[a:b] = v
    return out


def macro_f1(y, p):
    f = []
    for c in (H, AA, AG):
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c))
        f.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def mask_nonspace(t):
    return np.array([not ch.isspace() for ch in t])


def main():
    pairs = {p['passage_id']: p for p in map(json.loads, open(sys.argv[1])) if p['state'] == 'ok'}
    rows = defaultdict(dict)
    for r in map(json.loads, open(sys.argv[2])):
        rows[r['passage_id']][r['splitter']] = r
    splitters = ['luna', 'spacy', 'sentence', 'paragraph']
    common = [pid for pid in pairs if all(s in rows[pid] for s in splitters)]
    # Precompute L for every unit.
    feats = {}
    for pid in common:
        p = pairs[pid]; sg = grams(p['source']); _NORM[id(sg)] = norm(p['source'])
        for s in splitters:
            r = rows[pid][s]
            feats[pid, s] = (r['target_units'], [lexical(p['target'][a:b], sg) for a, b in r['target_units']], r['E'])
    grid = [(tH, tA, tE) for tH in np.arange(0.70, 1.001, 0.02) for tA in np.arange(0.0, 0.70, 0.05)
            for tE in np.arange(0.50, 0.96, 0.025) if tA < tH]
    result = {'n_pairs': len(common), 'splitters': {}}
    for conv in ('ours', 'report'):
        truth = {pid: char_truth(pairs[pid], conv) for pid in common}
        nonspace = {pid: mask_nonspace(pairs[pid]['target']) for pid in common}
        def stack(ids, s):
            L, E, C = [], [], []
            for pid in ids:
                u, l, e = feats[pid, s]; lab = truth[pid][0]; m = nonspace[pid]
                for (a, b), li, ei in zip(u, l, e):
                    L.append(li); E.append(ei); C.append(np.bincount(lab[a:b][m[a:b]], minlength=3))
            return np.array(L), np.array(E), np.array(C, dtype=float)
        def score_arr(arr, th):
            L, E, C = arr; tH, tA, tE = th
            pr = np.where(L >= tH, H, np.where((L >= tA) | (E >= tE), AA, AG))
            conf = np.stack([C[pr == c].sum(0) for c in (H, AA, AG)])  # rows predicted, cols true
            f = [0 if conf[c, c] == 0 else 2 * conf[c, c] / (conf[c].sum() + conf[:, c].sum()) for c in (H, AA, AG)]
            return float(np.mean(f))
        def score(ids, s, th):
            return score_arr(stack(ids, s), th)
        dev = [pid for pid in common if pairs[pid]['split'] == 'dev']
        test = [pid for pid in common if pairs[pid]['split'] == 'test']
        for s in splitters:
            dev_arr = stack(dev, s)
            best = max(grid, key=lambda th: score_arr(dev_arr, th))
            rep = {'thresholds': [round(float(x), 3) for x in best], 'dev_macro_f1': round(score(dev, s, best), 4),
                   'test_macro_f1': round(score(test, s, best), 4)}
            # Detailed test diagnostics.
            per_op = defaultdict(lambda: np.zeros(3)); fai_err, ctrl_fp, edited_hit = [], [], defaultdict(list)
            for pid in test:
                p = pairs[pid]; u, L, E = feats[pid, s]
                pr = paint(len(p['target']), u, predict(u, L, E, best)); lab, op = truth[pid]; m = nonspace[pid]
                for o in ('keep', 'polish', 'paraphrase', 'insert', 'clause_paraphrase', 'clause_append'):
                    sel = m & (op == o)
                    if sel.any():
                        per_op[o] += np.bincount(pr[sel], minlength=3)
                n = m.sum()
                f_pred = (0.5 * np.sum(pr[m] == AA) + np.sum(pr[m] == AG)) / n
                f_true = (0.5 * np.sum(lab[m] == AA) + np.sum(lab[m] == AG)) / n
                fai_err.append(abs(f_pred - f_true))
                if p['budget'] == 'control':
                    ctrl_fp.append(bool(np.any(pr[m] != H)))
                for t in p['truth']:
                    if t['op'] in ('polish', 'paraphrase', 'insert', 'clause_paraphrase', 'clause_append'):
                        sel = m[t['start']:t['end']]
                        edited_hit[t['op']].append(float(np.mean(pr[t['start']:t['end']][sel] != H)) > 0.5)
            rep['per_op_char_share'] = {o: dict(zip(NAMES, (v / v.sum()).round(3).tolist())) for o, v in per_op.items()}
            rep['edited_sentence_flagged'] = {o: round(float(np.mean(v)), 3) for o, v in edited_hit.items()}
            rep['control_paragraph_any_flag'] = round(float(np.mean(ctrl_fp)), 3) if ctrl_fp else None
            rep['f_ai_mae'] = round(float(np.mean(fai_err)), 4)
            rep['units_per_target'] = round(float(np.mean([len(feats[pid, s][0]) for pid in test])), 2)
            result['splitters'].setdefault(s, {})[conv] = rep
            print(conv, s, rep['thresholds'], 'dev', rep['dev_macro_f1'], 'test', rep['test_macro_f1'], 'fAI MAE', rep['f_ai_mae'],
                  'ctrl', rep['control_paragraph_any_flag'], 'flagged', rep['edited_sentence_flagged'], flush=True)
    result['n_test'] = len(test); result['n_dev'] = len(dev)
    json.dump(result, open(sys.argv[3], 'w'), indent=2)


if __name__ == '__main__':
    main()
