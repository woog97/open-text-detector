"""Build one- and two-sentence AI edits by splicing existing Luna paragraph rewrites into their human originals.

No new generation: for each paper paragraph we already have the human original and Luna's full rewrite in identical
surrounding context. Sentences are aligned in order by word overlap; a confidently matched, genuinely reworded AI
sentence (or two adjacent ones) replaces its human counterpart, and only those characters are labeled AI.

Usage: build_splices.py POOL_PAPERS_JSONL_GZ OUT_JSONL_GZ [--limit N]
"""
import argparse, ast, gzip, json, random, re
from collections import defaultdict

SENT = re.compile(r'\S.*?(?:[.!?](?=\s|$)|$)', re.S)
WORD = re.compile(r"[a-z0-9]+")


def sentences(text):
    return [(m.start(), m.end()) for m in SENT.finditer(text) if m.group().strip()]


def words(s):
    return WORD.findall(s.lower())


def sim(a, b):
    """Overlap of word unigrams and bigrams (Jaccard averaged): content match, robust to light rewording."""
    wa, wb = words(a), words(b)
    if not wa or not wb:
        return 0.
    u = len(set(wa) & set(wb)) / len(set(wa) | set(wb))
    ba, bb = set(zip(wa, wa[1:])), set(zip(wb, wb[1:]))
    bi = len(ba & bb) / len(ba | bb) if ba | bb else 0.
    return (u + bi) / 2


def align(H, A):
    """Order-preserving 1:1 alignment maximizing total similarity (gaps allowed, no reward)."""
    n, m = len(H), len(A)
    S = [[sim(h, a) for a in A] for h in H]
    dp = [[0.] * (m + 1) for _ in range(n + 1)]; bk = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            opts = [(dp[i - 1][j - 1] + S[i - 1][j - 1], 'm'), (dp[i - 1][j], 'u'), (dp[i][j - 1], 'l')]
            dp[i][j], bk[i][j] = max(opts)
    pairs = []; i, j = n, m
    while i and j:
        if bk[i][j] == 'm':
            pairs.append((i - 1, j - 1, S[i - 1][j - 1])); i -= 1; j -= 1
        elif bk[i][j] == 'u':
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def good(h, a, s):
    lh, la = len(words(h)), len(words(a))
    return 0.25 <= s <= 0.85 and lh >= 6 and la >= 6 and 0.6 <= la / lh <= 1.6


def regions_of(row):
    r = row['regions']
    return ast.literal_eval(r) if isinstance(r, str) else r


def build(rows, rng, per_pair=2):
    groups = defaultdict(dict)
    for r in rows:
        g, kind = r['id'].rsplit('/', 1); groups[g][kind] = r
    out = []
    for g, d in sorted(groups.items()):
        if 'human_original' not in d or 'paragraph_generate' not in d:
            continue
        hu, ai = d['human_original'], d['paragraph_generate']
        hs, he = int(hu['target_start']), int(hu['target_end']); as_, ae = int(ai['target_start']), int(ai['target_end'])
        H_text, A_text = hu['text'][hs:he], ai['text'][as_:ae]
        if hu['text'][:hs] != ai['text'][:as_]:
            continue  # contexts must match exactly
        Hs, As = sentences(H_text), sentences(A_text)
        Hstr = [H_text[a:b] for a, b in Hs]; Astr = [A_text[a:b] for a, b in As]
        pairs = [(i, j, s) for i, j, s in align(Hstr, Astr) if good(Hstr[i], Astr[j], s)]
        cands = [[p] for p in pairs] + [[p, q] for p, q in zip(pairs, pairs[1:]) if q[0] == p[0] + 1 and q[1] == p[1] + 1]
        rng.shuffle(cands)
        for chosen in cands[:per_pair]:
            i0, i1 = chosen[0][0], chosen[-1][0]; j0, j1 = chosen[0][1], chosen[-1][1]
            a_start, a_end = Hs[i0][0], Hs[i1][1]; ai_span = A_text[As[j0][0]:As[j1][1]]
            new_target = H_text[:a_start] + ai_span + H_text[a_end:]
            text = hu['text'][:hs] + new_target + hu['text'][he:]
            s0 = hs + a_start; s1 = s0 + len(ai_span)
            out.append({'id': f"{g}/splice-{'one' if len(chosen) == 1 else 'two'}-{i0}", 'paper_id': hu['paper_id'], 'group': hu['group'],
                        'dataset': 'papers_splice', 'text': text, 'target_start': hs, 'target_end': hs + len(new_target),
                        'regions': [{'start': 0, 'end': s0, 'label': 0}, {'start': s0, 'end': s1, 'label': 1}, {'start': s1, 'end': len(text), 'label': 0}],
                        'edit_size': 'one' if len(chosen) == 1 else 'two', 'similarity': [round(c[2], 3) for c in chosen],
                        'replaced_human': H_text[a_start:a_end], 'inserted_ai': ai_span})
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('pool'); ap.add_argument('out'); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args(); rng = random.Random(20261005)
    rows = [json.loads(l) for l in gzip.open(a.pool, 'rt')]
    out = build(rows, rng)
    if a.limit:
        rng.shuffle(out); out = out[:a.limit]
    with gzip.open(a.out, 'wt') as f:
        for r in out:
            f.write(json.dumps(r) + '\n')
    from collections import Counter
    print(len(out), 'splices', Counter(r['edit_size'] for r in out), 'from', len({r['group'] for r in out}), 'paragraph pairs')
