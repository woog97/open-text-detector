"""Which word n-grams are disproportionately common in AI text vs matched human text?

Two matched comparisons (topic held constant):
  general: Luna mirrors vs the exact human passages they were generated from (pool-mirrors / pool-human)
  papers:  Luna paragraph rewrites vs the original human paragraphs (pool-papers target spans)
Tokens: lowercase words plus individual punctuation marks; n = 1..4.
Score: log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi & Quinn 2008) -> z-score.
Writes ngrams.json for the dashboard (top AI- and human-leaning n-grams per n, with example contexts).
"""
import gzip, json, math, re, sys, random
from collections import Counter, defaultdict
from pathlib import Path

TOK = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?|[^\sa-z0-9]", re.I)
POOLS = Path(sys.argv[1]); OUT = Path(__file__).resolve().parent / 'ngrams.json'
N, TOP, MIN_TOTAL, ALPHA0 = 4, 1500, 15, 2000.0


def toks(t):
    return [x.lower() for x in TOK.findall(t)]


def load():
    hum = {json.loads(l)['id']: json.loads(l)['text'] for l in gzip.open(POOLS / 'pool-human.jsonl.gz', 'rt')}
    mir = [json.loads(l) for l in gzip.open(POOLS / 'pool-mirrors.jsonl.gz', 'rt')]
    g_ai, g_h = [], []
    for m in mir:
        src = m['id'].split('/', 1)[1]
        if src in hum:
            g_ai.append(m['text']); g_h.append(hum[src])
    pap = [json.loads(l) for l in gzip.open(POOLS / 'pool-papers.jsonl.gz', 'rt')]
    by = defaultdict(dict)
    for r in pap:
        g, kind = r['id'].rsplit('/', 1); by[g][kind] = r['text'][int(r['target_start']):int(r['target_end'])]
    p_ai = [d['paragraph_generate'] for d in by.values() if 'paragraph_generate' in d and 'human_original' in d]
    p_h = [d['human_original'] for d in by.values() if 'paragraph_generate' in d and 'human_original' in d]
    return {'general': (g_ai, g_h), 'papers': (p_ai, p_h)}


def count(docs):
    c = [Counter() for _ in range(N + 1)]; df = [Counter() for _ in range(N + 1)]; total = [0] * (N + 1)
    for d in docs:
        t = toks(d)
        for n in range(1, N + 1):
            grams = [' '.join(t[i:i + n]) for i in range(len(t) - n + 1)]
            c[n].update(grams); df[n].update(set(grams)); total[n] += len(grams)
    return c, df, total


def contexts(docs, gram, k=2, width=70):
    out = []; pat = re.compile(r'(?<![a-z0-9])' + r'\s*'.join(re.escape(x) for x in gram.split(' ')) + r'(?![a-z0-9])', re.I)
    for d in docs:
        m = pat.search(d)
        if m:
            a, b = max(0, m.start() - width), min(len(d), m.end() + width)
            out.append([('…' if a else '') + d[a:m.start()], d[m.start():m.end()], d[m.end():b] + ('…' if b < len(d) else '')])
            if len(out) >= k:
                break
    return out


def compare(ai_docs, h_docs):
    ca, da, ta = count(ai_docs); ch, dh, th = count(h_docs); res = {}
    rng = random.Random(1); ai_s = ai_docs[:]; h_s = h_docs[:]; rng.shuffle(ai_s); rng.shuffle(h_s)
    for n in range(1, N + 1):
        rows = []; tot = ta[n] + th[n]
        for g in set(ca[n]) | set(ch[n]):
            ya, yh = ca[n][g], ch[n][g]
            if ya + yh < MIN_TOTAL:
                continue
            aw = ALPHA0 * (ya + yh) / tot
            d = math.log((ya + aw) / (ta[n] + ALPHA0 - ya - aw)) - math.log((yh + aw) / (th[n] + ALPHA0 - yh - aw))
            z = d / math.sqrt(1 / (ya + aw) + 1 / (yh + aw))
            rows.append({'g': g, 'n': n, 'ai': ya, 'h': yh, 'ai_pm': round(1e6 * ya / ta[n], 2), 'h_pm': round(1e6 * yh / th[n], 2),
                         'ai_doc': round(100 * da[n][g] / len(ai_docs), 2), 'h_doc': round(100 * dh[n][g] / len(h_docs), 2),
                         'ratio': round(((ya + .5) / ta[n]) / ((yh + .5) / th[n]), 3), 'z': round(z, 2)})
        rows.sort(key=lambda r: -r['z']); keep = rows[:TOP] + rows[-TOP:]
        for r in keep[:150] + keep[-150:]:
            r['ex_ai'] = contexts(ai_s, r['g']); r['ex_h'] = contexts(h_s, r['g'])
        res[n] = {'rows': keep, 'tokens_ai': ta[n], 'tokens_h': th[n], 'candidates': len(rows)}
        print(n, 'grams:', len(rows), 'candidates; top AI:', [r['g'] for r in rows[:8]], flush=True)
    return res, len(ai_docs), len(h_docs)


if __name__ == '__main__':
    data = load(); out = {}
    for name, (ai, h) in data.items():
        print('==', name, len(ai), 'AI docs vs', len(h), 'human docs', flush=True)
        res, na, nh = compare(ai, h)
        out[name] = {'docs_ai': na, 'docs_h': nh, 'by_n': res}
    OUT.write_text(json.dumps(out, separators=(',', ':'))); print('wrote', OUT, OUT.stat().st_size // 1024, 'KB')
