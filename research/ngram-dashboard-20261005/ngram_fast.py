"""Fast n-gram statistics over all human and AI training-source text, with per-group breakdowns.

Each document is assigned one group (e.g. human:wikipedia:seed, ai:mirror:wikipedia, ai:manuscript:gpt-6-luna,
gradtex:human). Workers tokenize in parallel, hash n-grams (n = 1..4) to 64-bit ints and count with numpy; counts and
document frequencies are kept per group, so any comparison is a sum of groups. Strings are recovered only for
n-grams that end up displayed.

Usage: ngram_fast.py POOLS_DIR [--limit-per-pool N]
"""
import argparse, gzip, hashlib, json, math, os, re, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

TOK = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?|[^\sa-z0-9]", re.I)
N = 4; MULT = np.uint64(0x9E3779B97F4A7C15)
HERE = Path(__file__).resolve().parent


def tok_ids(text, cache):
    out = []
    for t in TOK.findall(text.lower()):
        v = cache.get(t)
        if v is None:
            v = int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), 'little'); cache[t] = v
        out.append(v)
    return np.asarray(out, dtype=np.uint64)


def grams(ids, n):
    if len(ids) < n:
        return np.empty(0, np.uint64)
    h = ids[:len(ids) - n + 1].copy()
    with np.errstate(over='ignore'):
        for k in range(1, n):
            h = h * MULT + ids[k:len(ids) - n + 1 + k]
        h = h * MULT + np.uint64(n)  # keep n-gram orders apart
    return h


def worker(job):
    """Count one shard of documents that all belong to one group."""
    group, texts = job; cache = {}; res = {}
    per = {n: [] for n in range(1, N + 1)}; per_df = {n: [] for n in range(1, N + 1)}; ntok = {n: 0 for n in range(1, N + 1)}
    for t in texts:
        ids = tok_ids(t, cache)
        for n in range(1, N + 1):
            g = grams(ids, n); per[n].append(g); per_df[n].append(np.unique(g)); ntok[n] += len(g)
    for n in range(1, N + 1):
        allg = np.concatenate(per[n]) if per[n] else np.empty(0, np.uint64)
        k, c = np.unique(allg, return_counts=True)
        dk, dc = np.unique(np.concatenate(per_df[n]) if per_df[n] else np.empty(0, np.uint64), return_counts=True)
        res[n] = (k, c.astype(np.int64), dk, dc.astype(np.int64), ntok[n])
    return group, len(texts), res


def merge(parts):
    """Sum (keys, counts) arrays from several shards."""
    k = np.concatenate([p[0] for p in parts]); c = np.concatenate([p[1] for p in parts])
    o = np.argsort(k, kind='stable'); k, c = k[o], c[o]
    starts = np.r_[0, np.nonzero(k[1:] != k[:-1])[0] + 1]
    return k[starts], np.add.reduceat(c, starts)


def load(pools, limit):
    def rd(name):
        rows = [json.loads(l) for l in gzip.open(pools / f'pool-{name}.jsonl.gz', 'rt')]
        return rows[:limit] if limit else rows
    human = rd('human'); mirrors = rd('mirrors'); papers = rd('papers'); full = rd('fullpapers'); grad = rd('gradtex')
    src = {r['id']: r['source'] for r in human}; seeded = {m['id'].split('/', 1)[1] for m in mirrors}
    docs = defaultdict(list)
    for r in human:
        docs[f"human:{r['source']}:{'seed' if r['id'] in seeded else 'noseed'}"].append(r['text'])
    for m in mirrors:
        docs[f"ai:mirror:{src.get(m['id'].split('/', 1)[1], 'unknown')}"].append(m['text'])
    for r in papers:
        kind = r['id'].rsplit('/', 1)[1]; t = r['text'][int(r['target_start']):int(r['target_end'])]
        docs['human:paper_original' if kind == 'human_original' else 'ai:paper_rewrite'].append(t)
    for r in full:
        docs[f"ai:manuscript:{r['model']}"].append(r['text'])
    for r in grad:
        docs['gradtex:human' if str(r['document_label']) == '0' else 'gradtex:ai_involved'].append(r['text'])
    return docs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('pools'); ap.add_argument('--limit-per-pool', type=int, default=0); a = ap.parse_args()
    docs = load(Path(a.pools), a.limit_per_pool)
    jobs = []
    for g, ts in docs.items():
        for i in range(0, len(ts), 1500):
            jobs.append((g, ts[i:i + 1500]))
    acc = defaultdict(lambda: {n: [] for n in range(1, N + 1)}); ndocs = defaultdict(int)
    with ProcessPoolExecutor(max_workers=min(11, os.cpu_count())) as ex:
        for g, nd, res in ex.map(worker, jobs, chunksize=1):
            ndocs[g] += nd
            for n, v in res.items():
                acc[g][n].append(v)
    groups = {}
    for g, by_n in acc.items():
        groups[g] = {}
        for n, parts in by_n.items():
            k, c = merge([(p[0], p[1]) for p in parts]); dk, dc = merge([(p[2], p[3]) for p in parts])
            groups[g][n] = {'k': k, 'c': c, 'dk': dk, 'dc': dc, 'tok': sum(p[4] for p in parts)}
    del acc
    report = analyze(groups, ndocs)
    strings, examples = recover(docs, report['selected'], report['example_hashes'])
    finalize(report, strings, examples, ndocs)


CATEGORY = {**{s: 'creative' for s in ['gutenberg', 'writingprompts', 'scp', 'wikisource']},
            **{s: 'scientific' for s in ['pmc', 'arxiv', 'pes2o', 'acl']},
            **{s: 'reference' for s in ['wikipedia', 'pressbooks', 'libretexts', 'wikibooks']},
            **{s: 'reviews' for s in ['amazon2018', 'imdb', 'opinrank']},
            **{s: 'social' for s in ['stack_nontech', 'stack_tech', 'ubuntu_irc', 'wiki_talk']},
            **{s: 'web' for s in ['cccc', 'eff', 'foodista', 'wikivoyage']},
            **{s: 'news' for s in ['globalvoices', 'voa', 'wikinews', 'scidev', 'oanc_slate']},
            **{s: 'essays' for s in ['asap2', 'persuade']},
            **{s: 'professional' for s in ['govreport', 'fed', 'worldbank', 'oanc_icic']}}
SCI = ['pmc', 'arxiv', 'pes2o', 'acl']
COMPARISONS = {
    'overall': ('All AI text vs all human text', lambda g: g.startswith('ai:'), lambda g: g.startswith('human:')),
    'matched_general': ('Mirrors vs the human passages they came from', lambda g: g.startswith('ai:mirror:'), lambda g: g.startswith('human:') and g.endswith(':seed')),
    'matched_papers': ('Paper rewrites vs original paragraphs', lambda g: g == 'ai:paper_rewrite', lambda g: g == 'human:paper_original'),
    'manuscripts': ('AI manuscripts vs human scientific writing', lambda g: g.startswith('ai:manuscript:'),
                    lambda g: g == 'human:paper_original' or (g.startswith('human:') and g.split(':')[1] in SCI)),
    'gradtex': ('GRADTEX: AI-involved vs human documents', lambda g: g == 'gradtex:ai_involved', lambda g: g == 'gradtex:human'),
}
TOP, MIN_TOTAL, ALPHA0, NEX = 500, 15, 2000.0, 100


def breakdown_group(g):
    p = g.split(':')
    if p[0] == 'human':
        return 'Human · papers (originals)' if p[1] == 'paper_original' else f'Human · {CATEGORY.get(p[1], "other")}'
    if p[0] == 'ai':
        if p[1] == 'mirror':
            return f'AI mirror · {CATEGORY.get(p[2], "other")}'
        return 'AI · paper rewrites' if p[1] == 'paper_rewrite' else f'AI manuscript · {p[2]}'
    return 'GRADTEX · ' + ('AI-involved' if p[1] == 'ai_involved' else 'human')


def lookup(keys, vals, q):
    i = np.searchsorted(keys, q); i = np.minimum(i, len(keys) - 1)
    return np.where((len(keys) > 0) & (keys[i] == q), vals[i], 0) if len(keys) else np.zeros(len(q), np.int64)


def analyze(groups, ndocs):
    res = {'comparisons': {}, 'selected': set(), 'example_hashes': {}}
    for name, (title, is_ai, is_h) in COMPARISONS.items():
        ga = [g for g in groups if is_ai(g)]; gh = [g for g in groups if is_h(g)]
        if not ga or not gh:
            continue
        comp = {'title': title, 'docs_ai': sum(ndocs[g] for g in ga), 'docs_h': sum(ndocs[g] for g in gh), 'by_n': {}}
        for n in range(1, N + 1):
            ka, ca = merge([(groups[g][n]['k'], groups[g][n]['c']) for g in ga]); kh, ch = merge([(groups[g][n]['k'], groups[g][n]['c']) for g in gh])
            dka, dca = merge([(groups[g][n]['dk'], groups[g][n]['dc']) for g in ga]); dkh, dch = merge([(groups[g][n]['dk'], groups[g][n]['dc']) for g in gh])
            ta = sum(groups[g][n]['tok'] for g in ga); th = sum(groups[g][n]['tok'] for g in gh)
            keys, inv = np.unique(np.concatenate([ka, kh]), return_inverse=True)
            ya = np.zeros(len(keys), np.int64); yh = np.zeros(len(keys), np.int64)
            ya[inv[:len(ka)]] = ca; yh[inv[len(ka):]] = ch
            m = ya + yh >= MIN_TOTAL; keys, ya, yh = keys[m], ya[m].astype(float), yh[m].astype(float)
            aw = ALPHA0 * (ya + yh) / (ta + th)
            d = np.log((ya + aw) / (ta + ALPHA0 - ya - aw)) - np.log((yh + aw) / (th + ALPHA0 - yh - aw))
            z = d / np.sqrt(1 / (ya + aw) + 1 / (yh + aw))
            o = np.argsort(-z); pick = np.r_[o[:TOP], o[-TOP:]]
            sel = keys[pick]
            rows = {'k': sel, 'ai': ya[pick].astype(int), 'h': yh[pick].astype(int), 'z': np.round(z[pick], 2),
                    'ai_doc': np.round(100 * lookup(dka, dca, sel) / comp['docs_ai'], 2), 'h_doc': np.round(100 * lookup(dkh, dch, sel) / comp['docs_h'], 2),
                    'ai_pm': np.round(1e6 * ya[pick] / ta, 2), 'h_pm': np.round(1e6 * yh[pick] / th, 2),
                    'ratio': np.round(((ya[pick] + .5) / ta) / ((yh[pick] + .5) / th), 3)}
            comp['by_n'][n] = {'rows': rows, 'tokens_ai': ta, 'tokens_h': th, 'candidates': int(m.sum())}
            res['selected'].update(int(x) for x in sel)
            for x in np.r_[keys[o[:NEX]], keys[o[-NEX:]]]:
                res['example_hashes'][int(x)] = (ga, gh)
            print(name, n, 'candidates', int(m.sum()), flush=True)
        res['comparisons'][name] = comp
    sel = np.array(sorted(res['selected']), dtype=np.uint64)
    bnames = sorted({breakdown_group(g) for g in groups})
    rates = {}
    for b in bnames:
        mem = [g for g in groups if breakdown_group(g) == b]
        r = np.zeros(len(sel))
        for n in range(1, N + 1):
            k, c = merge([(groups[g][n]['k'], groups[g][n]['c']) for g in mem]); tok = sum(groups[g][n]['tok'] for g in mem)
            r += 1e6 * lookup(k, c, sel) / max(tok, 1)  # each hash belongs to exactly one n
        rates[b] = np.round(r, 2)
    res['breakdown'] = {'groups': bnames, 'docs': {b: sum(ndocs[g] for g in groups if breakdown_group(g) == b) for b in bnames}, 'sel': sel, 'rates': rates}
    return res


def recover_worker(job):
    group, texts, want = job; want = np.asarray(want, dtype=np.uint64); cache = {}; strings = {}; ex = {}
    for t in texts:
        low = t.lower(); spans = [(m.start(), m.end()) for m in TOK.finditer(low)]
        ids = tok_ids(t, cache)
        for n in range(1, N + 1):
            g = grams(ids, n)
            if not len(g):
                continue
            hit = np.nonzero(np.isin(g, want))[0]
            for i in hit[:400]:
                h = int(g[i])
                if h not in strings:
                    strings[h] = ' '.join(low[a:b] for a, b in spans[i:i + n])
                lst = ex.setdefault(h, [])
                if len(lst) < 2:
                    a, b = spans[i][0], spans[i + n - 1][1]; s0, s1 = max(0, a - 70), min(len(t), b + 70)
                    lst.append([('…' if s0 else '') + t[s0:a], t[a:b], t[b:s1] + ('…' if s1 < len(t) else '')])
    return group, strings, ex


def recover(docs, selected, example_hashes):
    want = sorted(selected); jobs = []
    for g, ts in docs.items():
        for i in range(0, len(ts), 1500):
            jobs.append((g, ts[i:i + 1500], want))
    strings = {}; by_group = defaultdict(dict)
    with ProcessPoolExecutor(max_workers=min(11, os.cpu_count())) as ex:
        for g, s, e in ex.map(recover_worker, jobs, chunksize=1):
            strings.update(s)
            for h, lst in e.items():
                cur = by_group[g].setdefault(h, [])
                cur.extend(lst[:2 - len(cur)])
    return strings, by_group


def finalize(report, strings, ex_by_group, ndocs):
    out = {'comparisons': {}, 'breakdown': {'groups': report['breakdown']['groups'], 'docs': report['breakdown']['docs']}}
    sel = report['breakdown']['sel']; idx = {int(h): i for i, h in enumerate(sel)}
    rates = report['breakdown']['rates']; bd = {}
    for name, comp in report['comparisons'].items():
        oc = {'title': comp['title'], 'docs_ai': comp['docs_ai'], 'docs_h': comp['docs_h'], 'by_n': {}}
        for n, blk in comp['by_n'].items():
            R = blk['rows']; rows = []
            for j, h in enumerate(R['k']):
                h = int(h); s = strings.get(h)
                if s is None:
                    continue
                r = {'g': s, 'ai': int(R['ai'][j]), 'h': int(R['h'][j]), 'z': float(R['z'][j]), 'ai_doc': float(R['ai_doc'][j]), 'h_doc': float(R['h_doc'][j]),
                     'ai_pm': float(R['ai_pm'][j]), 'h_pm': float(R['h_pm'][j]), 'ratio': float(R['ratio'][j]), 'id': h % 10**12}
                if h in report['example_hashes']:
                    ga, gh = report['example_hashes'][h]
                    r['ex_ai'] = [e for g in ga for e in ex_by_group.get(g, {}).get(h, [])][:2]
                    r['ex_h'] = [e for g in gh for e in ex_by_group.get(g, {}).get(h, [])][:2]
                rows.append(r)
                bd[str(h % 10**12)] = [float(rates[b][idx[h]]) for b in report['breakdown']['groups']]
            oc['by_n'][n] = {'rows': rows, 'tokens_ai': blk['tokens_ai'], 'tokens_h': blk['tokens_h'], 'candidates': blk['candidates']}
        out['comparisons'][name] = oc
    out['breakdown']['rates'] = bd
    (HERE / 'ngrams-all.json').write_text(json.dumps(out, separators=(',', ':')))
    print('wrote ngrams-all.json', (HERE / 'ngrams-all.json').stat().st_size // 1024, 'KB', flush=True)


if __name__ == '__main__':
    main()
