import re, numpy as np, pandas as pd
SENT = re.compile(r'\S.*?(?:[.!?](?=\s|$)|$)', re.S)
def sent_spans_scores(text, off, prob):
    off = np.asarray(off).reshape(-1, 2); prob = np.asarray(prob, np.float64)
    valid = np.array([bool(text[a:b].strip()) for a, b in off], np.float64)
    cs = np.r_[0, np.cumsum(prob * valid)]; cv = np.r_[0, np.cumsum(valid)]; res = []
    for m in SENT.finditer(text):
        if not m.group().strip(): continue
        a, b = m.start(), m.end()
        lo = np.searchsorted(off[:, 1], a, side='right'); hi = np.searchsorted(off[:, 0], b, side='left'); n = cv[hi] - cv[lo]
        if hi > lo and n > 0: res.append(((a, b), (cs[hi] - cs[lo]) / n))
    return res
def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float); r = pd.Series(np.r_[pos, neg]).rank().to_numpy()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
