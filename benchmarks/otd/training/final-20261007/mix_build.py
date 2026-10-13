"""Space-side final single-pass mix: 72,000 stage-2 rows (T2.1 PLAN x 3, DeepSeek added) written as three epoch files of
24,000 with no window used twice anywhere, unless a whole group runs out (reported as repeats). Stage 1, selection and
calibration windows are copied unchanged from T2.1, so train_sweep.py consumes the result as it did T2.1 (3 stage-2 passes
over 3 different files = one pass over 72k distinct windows).

Screening per row (as build_t21.py): 510-token crop (Qwen tokenizer), never_train_hit over every held-out list (the Mac's
NEVER_TRAIN_FILES + wide1-heldout-never-train.json), evaluation/selection/calibration paper ids, shared 10+-word
sentences, reserved manuscript families. Never trained: Fable, GPT-6 Sol, GLM, MiMo, any held-out split.

Sources: mix-inputs.tgz from mix_pools.py (Claude/GPT/Luna edits, sections, paragraph edits, old hosts, Claude full
papers), wide1 API outputs (DeepSeek d/e, Luna e; split=train only), the T2 source pools (paired papers, mirrors,
fullpapers, generic human, GRADTEX) and the wide clean-v2 pool (pool-v1, else pool-early) for new human paper text and hosts.

Human sources (--human-source):
  balanced (default)  human_paper_orig: the older paired human windows first, clean v2 for the rest; human_host: clean v2
                      share = clean-v2 share of the hosted AI rows in the mix, older PMC/peS2o/arXiv hosts for the rest.
                      Keeps text format from predicting the label.
  clean               both keys from clean v2 only.
Writes OUT/prepared-v2 (manifest + files) and OUT/mix-report.json.
"""
import argparse, collections, glob, gzip, hashlib, io, json, math, os, random, re, shutil, tarfile, time, traceback, zlib
from collections import Counter, defaultdict
from pathlib import Path

PLAN = {  # key: rows over the whole single pass (T2.1 per-epoch rows x 3; DeepSeek takes 1/3 of Luna small edits and 999 Luna paragraph rows)
    'edit_claude': 13320, 'edit_luna': 6720, 'edit_deepseek': 3360, 'edit_gpt': 1800,
    'human_paper_orig': 9000, 'human_section': 3000, 'human_para_e': 3000, 'human_host': 3000,
    'para_luna': 4221, 'para_e': 3480, 'para_deepseek': 999,
    'sections_ai': 4200,
    'mirrors': 4200, 'fullpapers': 1500, 'claude_fp': 1500,
    'human': 5100, 'gradtex': 3600}
GROUP = {'small edits': ['edit_claude', 'edit_luna', 'edit_deepseek', 'edit_gpt'],
         'human paper text': ['human_paper_orig', 'human_section', 'human_para_e', 'human_host'],
         'paragraph edits': ['para_luna', 'para_e', 'para_deepseek'], 'sections': ['sections_ai'], 'fully AI': ['mirrors', 'fullpapers', 'claude_fp'],
         'generic human': ['human'], 'GRADTEX': ['gradtex']}
assert sum(PLAN.values()) == 72000 and all(v % 3 == 0 for v in PLAN.values())
AI_KEYS = ['edit_claude', 'edit_luna', 'edit_deepseek', 'edit_gpt', 'para_luna', 'para_e', 'para_deepseek', 'sections_ai', 'mirrors', 'fullpapers', 'claude_fp']
EXCLUDED_GEN = re.compile(r'fable|gpt-6-sol|glm|mimo|space-bunny', re.I)
SENT = re.compile(r'\S.*?(?:[.!?](?=\s|$)|$)', re.S)
G = {}  # worker globals (inherited through fork)
log = None


def say(**k):
    log.write(json.dumps({'t': time.strftime('%H:%M:%S'), **k}) + '\n'); log.flush()


def sents(t): return {' '.join(w) for m in SENT.finditer(t) for w in [re.findall(r'[a-z0-9]+', m.group().lower())] if len(w) >= 10}


def norm_hash(t): return hashlib.sha256(re.sub(r'\W+', '', t.lower()).encode()).hexdigest()


def never_train_hit(row):
    nt = G['nt']
    ids = {str(row.get(k, '')).replace('paper:', '').replace('human:', '').split('/')[0] for k in ('paper_id', 'group', 'group_id', 'seed_id', 'item_id')}
    ids |= {x for x in re.split(r'[/]', str(row.get('id', ''))) if x.startswith(('claude-fp-', 'claude-sec-'))} | {str(row.get('id', '')).replace('fullpaper-', '')}
    if ids & nt['papers']: return True
    text = row.get('text', '')
    if any(norm_hash(p) in nt['hashes'] for p in text.split('\n\n') if len(re.sub(r'\W+', '', p)) >= 100): return True
    n = re.sub(r'\W+', '', text.lower())
    return any(hashlib.sha256(n[i:i + 60].encode()).hexdigest()[:12] in nt['shingles'] for i in range(max(0, len(n) - 59)))


def crop510(r, limit=500, max_tokens=510):
    """Rows longer than the window: keep a <=limit-token window around the AI span (or target), snapped to whitespace."""
    tok = G['tok']; t = r['text']; off = tok(t, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    if len(off) <= max_tokens: return r
    rng = random.Random(r['id'])
    ai = [g for g in r['regions'] if g['label'] == 1]
    a0 = min(g['start'] for g in ai) if ai else int(r.get('target_start', 0)); a1 = max(g['end'] for g in ai) if ai else int(r.get('target_end', len(t)))
    if not ai and a1 - a0 >= len(t) * .9: a0 = a1 = rng.randrange(len(t))
    i0 = next((i for i, (a, b) in enumerate(off) if b > a0), 0); i1 = next((i for i, (a, b) in enumerate(off) if b >= a1), len(off) - 1)
    span = i1 - i0 + 1
    st = i0 if span >= limit else max(0, min(len(off) - limit, i0 - rng.randrange(limit - span + 1)))
    cs, ce = off[st][0], off[min(len(off), st + limit) - 1][1]
    while cs > 0 and cs < len(t) and not t[cs - 1].isspace(): cs += 1
    while ce < len(t) and ce > cs and not t[ce - 1].isspace() and not t[ce].isspace(): ce -= 1
    txt = t[cs:ce]
    rg = [{'start': max(g['start'], cs) - cs, 'end': min(g['end'], ce) - cs, 'label': g['label']} for g in r['regions'] if min(g['end'], ce) > max(g['start'], cs)]
    return {**r, 'id': f"{r['id']}@{cs}", 'text': txt, 'regions': rg, 'source_start': cs, 'source_end': ce,
            'target_start': max(0, min(len(txt), int(r.get('target_start', 0)) - cs)), 'target_end': max(0, min(len(txt), int(r.get('target_end', len(t))) - cs))}


def screen_one(r):
    try:
        if EXCLUDED_GEN.search(str(r.get('generator'))): return None, 'excluded_writer'
        doc = r.get('supervision') == 'document_only'
        if not doc:
            orig, r, limit = r, crop510(r), 500
            for _ in range(6):  # every backbone tokenizer must fit 510 (EmbeddingGemma's 262k vocab splits some text finer than Qwen's)
                worst = max(len(t(r['text'], add_special_tokens=False)['input_ids']) for t in G['extra_toks']) if G['extra_toks'] else 0
                if worst <= 510: break
                limit = int(limit * 500 / worst) - 2; r = crop510(orig, limit=limit, max_tokens=limit)
            else: return None, 'over_510_tokens_other_tokenizer'
        k = len(G['tok'](r['text'], add_special_tokens=False)['input_ids'])
        if not doc and not 0 < k <= 510: return None, 'over_510_tokens'
        if doc and k == 0: return None, 'empty'
        if never_train_hit({k2: v for k2, v in r.items() if k2 != 'regions'} if doc else r): return None, 'never_train'
        ids = {str(r.get('paper_id', '')).replace('paper:', ''), r['id'].split('/')[0], str(r.get('group', '')).replace('paper:', '').replace('human:', '')}
        if ids & G['keys']: return None, 'eval_paper_id'
        if ids & G['blocked']: return None, 'reserved_family'
        if sents(r['text']) & G['held']: return None, 'eval_shared_sentence'
        return r, None
    except Exception as e:
        return None, 'error:' + type(e).__name__


def screen(rows, cat, rejects, pool):
    out = []
    for r, why in pool.imap(screen_one, rows, chunksize=64):
        if why: rejects[cat][why] += 1
        else: out.append(r)
    return out


def rr(rows, key=lambda r: r.get('group') or r['paper_id'], seed=0):
    """Round-robin over papers/groups (seeded order): every group gives one row before any gives a second."""
    by = defaultdict(list)
    for r in rows: by[key(r)].append(r)
    order = sorted(by); random.Random(seed).shuffle(order); out = []; i = 0
    while True:
        layer = [by[g][i] for g in order if len(by[g]) > i]
        if not layer: return out
        out += layer; i += 1


def windows(doc, tok, size=500, cap=None, rng=None):
    """Non-overlapping <=size-token windows over a long document, from a random token offset; label/regions cropped."""
    t = doc['text']; off = tok(t, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']; n = len(off)
    if not n: return []
    regions = doc.get('regions') or [{'start': 0, 'end': len(t), 'label': doc['label']}]
    shift = rng.randrange(min(size, max(1, n - size + 1))) if n > size else 0; out = []
    for st in range(shift, n, size):
        en = min(n, st + size)
        if en - st < 100 and out: break
        cs, ce = off[st][0], off[en - 1][1]
        while cs > 0 and cs < len(t) and not t[cs - 1].isspace(): cs += 1
        txt = t[cs:ce]
        if len(txt.strip()) < 200: continue
        rg = [{'start': max(g['start'], cs) - cs, 'end': min(g['end'], ce) - cs, 'label': g['label']} for g in regions if min(g['end'], ce) > max(g['start'], cs)]
        out.append({'id': f"{doc['id']}@{cs}", 'paper_id': doc['paper_id'], 'group': doc.get('group'), 'kind': doc.get('kind') or ('novel_human' if doc.get('label') == 0 else 'generated'),
                    'text': txt, 'source_start': cs, 'source_end': ce, 'regions': rg, 'target_start': 0, 'target_end': len(txt), 'generator': doc.get('model') or doc.get('generator')})
        if cap and len(out) >= cap: break
    return out


def full_text_windows(pool_dir, nt, tok, need, corpus='/data/workspace/datasets/corpus-clean-text-v2'):
    """Human paper windows from the WHOLE clean v2 text of every kept wide-pool paper (pool papers.parquet with no exclusion),
    reference lists cut off, all <=500-token windows per paper, papers taken round-robin over venue/year strata. For scaled-up
    mixes whose human_paper_orig need exceeds what host paragraphs can give (mixB full, Oct 8)."""
    import glob, pyarrow.parquet as pq
    kept = {}
    for p in pq.read_table(pool_dir / 'papers.parquet').to_pylist():
        if not p.get('excluded') and p['pdf_sha256'] not in nt['papers']: kept[p['pdf_sha256']] = f"{p.get('venue')}/{p.get('year')}"
    texts = {}
    for f in sorted(glob.glob(corpus + '/data/*.parquet')):
        for r in pq.read_table(f, columns=['pdf_sha256', 'text']).to_pylist():
            if r['pdf_sha256'] in kept and r['text']: texts[r['pdf_sha256']] = r['text']
    REF = re.compile(r'^\s*(\d+\.?\s*)?(references|bibliography|reference list)\s*$', re.I | re.M)
    strat = defaultdict(list); papers = sorted(texts); random.Random(9).shuffle(papers)
    for p in papers: strat[kept[p]].append(p)
    order = [p for _, _, p in sorted(((i + .5) / len(v), g, p) for g, v in strat.items() for i, p in enumerate(v))]
    rng = random.Random(10); out = []
    for p in order:
        t = texts[p]; m = list(REF.finditer(t))
        if m and m[-1].start() > len(t) * .4: t = t[:m[-1].start()]
        for w in windows({'id': f'cleanfull:{p[:16]}', 'paper_id': p, 'group': 'human:' + p, 'kind': 'human_paper_window', 'text': t, 'label': 0}, tok, rng=rng):
            out.append({**w, 'text_format': 'clean_v2', 'generator': None})
        if len(out) >= need: break
    say(event='full_text_windows', papers_available=len(texts), windows=len(out))
    return out


def clean_pool(S, pool_dir, cand, nt, need_win):
    """Clean-v2 human windows (consecutive host paragraphs, <=5 per paper) and hosts (<=2 per paper, different papers),
    from wide-pool papers never used by generation, held-out lists or the older pools."""
    import pyarrow.parquet as pq
    used_papers = set()
    for f in [S / 'early/exclude-papers.json', S / 'wide1/exclude.json']:
        if f.exists(): used_papers |= set(json.loads(f.read_text()))
    if (S / 'wide1/hosts-sample.jsonl').exists(): used_papers |= {json.loads(l)['record_id'] for l in open(S / 'wide1/hosts-sample.jsonl')}
    for v in cand.values():
        used_papers |= {r['paper_id'] for r in v if str(r.get('host_id') or '').startswith('or:') or r.get('text_format') == 'clean_v2'}
    byp = defaultdict(list)
    for h in pq.read_table(pool_dir / 'hosts.parquet').to_pylist():
        if h['record_id'] not in used_papers and h['record_id'] not in nt['papers']: byp[h['record_id']].append(h)
    strat = defaultdict(list); papers = sorted(byp); random.Random(7).shuffle(papers)
    for p in papers: strat[f"{byp[p][0].get('venue')}/{byp[p][0].get('year')}"].append(p)
    order = [p for _, _, p in sorted(((i + .5) / len(v), g, p) for g, v in strat.items() for i, p in enumerate(v))]
    cw = []; i = 0
    while i < len(order) and len(cw) < need_win * 1.25:
        p = order[i]; i += 1
        hs = sorted(byp[p], key=lambda h: int(h['host_id'].rsplit('/p', 1)[1])); cur = []; last = None; mine = []
        for h in hs:
            idx = int(h['host_id'].rsplit('/p', 1)[1])
            if cur and last is not None and idx - last > 2:  # don't join paragraphs far apart
                if sum(len(x) for x in cur) >= 600: mine.append('\n\n'.join(cur))
                cur = []
            cur.append(h['target']); last = idx
            if sum(len(x) for x in cur) >= 1700: mine.append('\n\n'.join(cur)); cur = []
        if cur and sum(len(x) for x in cur) >= 600: mine.append('\n\n'.join(cur))
        for k, txt in enumerate(mine[:5]):
            cw.append({'id': f'cleanv2:{p[:16]}/w{k}', 'paper_id': p, 'group': 'human:' + p, 'kind': 'human_paper_window', 'text': txt, 'source_start': 0,
                       'source_end': len(txt), 'regions': [{'start': 0, 'end': len(txt), 'label': 0}], 'target_start': 0, 'target_end': len(txt),
                       'generator': None, 'text_format': 'clean_v2'})
    if len(cw) < need_win * 1.25:
        # Not enough unused papers: also take windows from training papers already used for generation (never held-out
        # ones), skipping each used host paragraph and its neighbours (+-1), at most 2 windows per paper.
        used_hosts = set()
        for f in [S / 'early/hosts-sample.jsonl', S / 'wide1/hosts-sample.jsonl']:
            if f.exists(): used_hosts |= {json.loads(l)['host_id'] for l in open(f)}
        for v in cand.values(): used_hosts |= {r['host_id'] for r in v if str(r.get('host_id') or '').startswith('or:')}
        near = defaultdict(set)
        for hid in used_hosts:
            pp, idx = hid.rsplit('/p', 1); near[pp] |= {int(idx) - 1, int(idx), int(idx) + 1}
        byu = defaultdict(list)
        for h in pq.read_table(pool_dir / 'hosts.parquet').to_pylist():
            if h['record_id'] in used_papers and h['record_id'] not in nt['papers']:
                pp, idx = h['host_id'].rsplit('/p', 1)
                if int(idx) not in near[pp]: byu[h['record_id']].append(h)
        upapers = sorted(byu); random.Random(8).shuffle(upapers); n_used = 0
        for p in upapers:
            if len(cw) >= need_win * 1.25: break
            hs = sorted(byu[p], key=lambda h: int(h['host_id'].rsplit('/p', 1)[1])); cur = []; last = None; mine = []
            for h in hs:
                idx = int(h['host_id'].rsplit('/p', 1)[1])
                if cur and last is not None and idx - last > 1:  # used hosts leave gaps; never join across them
                    if sum(len(x) for x in cur) >= 600: mine.append('\n\n'.join(cur))
                    cur = []
                cur.append(h['target']); last = idx
                if sum(len(x) for x in cur) >= 1700: mine.append('\n\n'.join(cur)); cur = []
            if cur and sum(len(x) for x in cur) >= 600: mine.append('\n\n'.join(cur))
            for k, txt in enumerate(mine[:2]):
                cw.append({'id': f'cleanv2u:{p[:16]}/w{k}', 'paper_id': p, 'group': 'human:' + p, 'kind': 'human_paper_window', 'text': txt, 'source_start': 0,
                           'source_end': len(txt), 'regions': [{'start': 0, 'end': len(txt), 'label': 0}], 'target_start': 0, 'target_end': len(txt),
                           'generator': None, 'text_format': 'clean_v2'})
            n_used += bool(mine)
        say(event='clean_windows_from_used_papers', papers=n_used, windows=len(cw))
    ch = []
    for p in order[i:]:
        for h in random.Random(p).sample(byp[p], min(2, len(byp[p]))):
            t = '\n\n'.join(x for x in (h['prev'], h['target'], h['next']) if x)
            ch.append({'id': 'host:' + h['host_id'], 'paper_id': p, 'group': 'human:' + p, 'kind': 'human_host', 'text': t, 'source_start': 0, 'source_end': len(t),
                       'regions': [{'start': 0, 'end': len(t), 'label': 0}], 'target_start': 0, 'target_end': len(t), 'generator': None, 'text_format': 'clean_v2',
                       'host_id': h['host_id']})
    return cw, ch


def main():
    global log
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--pool', default=None, help='wide clean-v2 pool dir (default pool-v1, else pool-early)')
    ap.add_argument('--sampling', default='/data/workspace/corpus-sampling-20261007')
    ap.add_argument('--base', default='/data/workspace/pangram-t2-20261006/prepared/qwen35-4b-t21/prepared-v2')
    ap.add_argument('--tok', default='/data/workspace/backbone-launch-20261003/assets/qwen35-4b')
    ap.add_argument('--sources', default='/data/workspace/backbone-launch-20261003/source-pools')
    ap.add_argument('--reservation', default='/data/workspace/backbone-launch-20261003/reservation.json')
    ap.add_argument('--extra-tok', default='/data/workspace/backbone-launch-20261003/assets/qwen36-35b-a3b,/data/workspace/backbone-launch-20261003/assets/gemma4-12b',
                    help='comma list: every other backbone tokenizer the windows must fit (EmbeddingGemma 2 is added automatically when found)')
    ap.add_argument('--human-source', choices=['balanced', 'clean'], default='balanced'); ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--plan', default=None, help='JSON file {key: rows} replacing PLAN (same keys; multiples of 3; any total, e.g. 72,000)')
    ap.add_argument('--doc-window-cap', type=int, default=2, help='windows per mirror / generic-human document (0 = all)')
    ap.add_argument('--human-windows', choices=['hosts', 'full'], default='hosts',
                    help='clean-v2 human paper windows: hosts = joined host paragraphs, <=5 per unused paper (default); '
                         'full = all windows of each kept paper\'s whole text (for scaled-up mixes)')
    a = ap.parse_args()
    if a.plan:
        new = json.loads(Path(a.plan).read_text()); assert set(new) == set(PLAN), set(new) ^ set(PLAN)
        assert all(v % 3 == 0 for v in new.values()), 'plan rows must be multiples of 3 (three disjoint epoch files)'
        PLAN.clear(); PLAN.update(new)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); log = open(out / 'mix-build.log', 'a'); t0 = time.time()
    try:
        say(event='start', args=vars(a))
        S = Path(a.sampling); pool_dir = Path(a.pool) if a.pool else next(p for p in (S / 'pool-v1', S / 'pool-early') if (p / 'stats.json').exists())
        work = Path('/tmp/pangram-final-mix/inputs'); shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
        with tarfile.open(a.inputs) as t: t.extractall(work)
        nt_files = sorted((work / 'never-train').glob('*.json')) + [f for f in [S / 'wide1/wide1-heldout-never-train.json'] if f.exists()]
        nt = {'papers': set(), 'hashes': set(), 'shingles': set()}
        for f in nt_files:
            d = json.loads(f.read_text()); nt['papers'] |= set(d['paper_ids']); nt['hashes'] |= set(d['paragraph_sha256']); nt['shingles'] |= set(d.get('paragraph_shingle_sha256_12') or [])
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tok, local_files_only=True)
        keys, held = set(), set()
        evp = next(p for p in (Path('/tmp/pangram-t2-20261006/sweeps/sweep-eval-rows.jsonl.gz'), Path('/data/workspace/pangram-t2-20261006/sweeps/sweep-eval-rows.jsonl.gz')) if p.exists())
        for r in (json.loads(l) for l in gzip.open(evp, 'rt')):
            keys.add(r['id'].split('/')[0]); keys.add(str(r.get('group_id') or '').replace('paper:', '')); held |= sents(r['text'])
        base = Path(a.base)
        for name in ['selection-windows', 'calibration-windows']:
            for l in gzip.open(base / f'{name}.jsonl.gz', 'rt'):
                r = json.loads(l); keys.add(str(r.get('paper_id', '')).replace('paper:', '')); keys.add(r['id'].split('/')[0]); held |= sents(r['text'])
        keys.discard('')
        blocked = set(json.loads(Path(a.reservation).read_text())['exclude_from_training']['source_paper_ids'])
        extra = [x for x in a.extra_tok.split(',') if x] + sorted({str(Path(f).parent) for pat in ('/data/workspace/backbone-launch-20261003/assets/*embeddinggemma*/tokenizer.json',
                                                                   '/data/workspace/final-20261007/assets/*/tokenizer.json', '/tmp/pangram-*/assets/*embeddinggemma*/tokenizer.json')
                                                                   for f in glob.glob(pat) if 'gemma' in f.lower()})
        extra_toks = [AutoTokenizer.from_pretrained(x, local_files_only=True) for x in extra]
        say(event='tokenizers', primary=a.tok, extra=extra)
        G.update(tok=tok, extra_toks=extra_toks, nt=nt, keys=keys, held=held, blocked=blocked)
        say(event='loaded', never_train_files=[f.name for f in nt_files], nt_papers=len(nt['papers']), eval_keys=len(keys), pool=str(pool_dir))

        # ---- candidates
        cand = defaultdict(list)
        for l in gzip.open(work / 'pools.jsonl.gz', 'rt'):
            r = json.loads(l); cand[r.pop('pool')].append(r)
        cand['human_host_old'] = cand.pop('human_host', [])
        api = Counter()
        for w, job in [('deepseek', 'd'), ('deepseek', 'e'), ('luna', 'e')]:
            f = S / f'wide1/out-{w}-{job}/accepted.jsonl'
            if not f.exists(): api[f'{w}-{job}:missing'] += 1; continue
            for r in map(json.loads, open(f)):
                if r.get('split', 'train') != 'train': api[f'{w}-{job}:heldout'] += 1; continue
                fmt = 'clean_v2' if str(r.get('host_id', '')).startswith('or:') else f"host_{r.get('source')}"
                base_r = {'id': r['id'], 'paper_id': r['paper_id'], 'group': r['group'], 'text': r['text'], 'source_start': 0, 'source_end': len(r['text']),
                          'regions': r['regions'], 'target_start': r['target_start'], 'target_end': r['target_end'], 'generator': r.get('generator'),
                          'text_format': fmt, 'host_id': r.get('host_id')}
                if r['dataset'] == 'hsm_llm_edit': cand['edit_deepseek'].append({**base_r, 'kind': 'llm_edit'})
                elif r['dataset'] == 'hsm_paragraph_edit': cand['para_deepseek' if w == 'deepseek' else 'para_luna'].append({**base_r, 'kind': 'paragraph_edit'})
                elif r['dataset'] == 'hsm_paragraph_edit_human': cand['human_para_e'].append({**base_r, 'kind': 'human_matched'})
                api[f'{w}-{job}:{r["dataset"]}'] += 1
        say(event='api_rows', **api)
        src = Path(a.sources); rng = random.Random(20261007)
        def pool_rows(ds):
            return [json.loads(l) for l in gzip.open(src / f'pool-{ds}.jsonl.gz', 'rt')]
        for r in pool_rows('papers'):
            sub = r['id'].rsplit('/', 1)[-1]
            row = {**r, 'kind': 'paired', 'text_format': 'paired_pdf_v1', 'generator': r.get('generator') or ('openai/gpt-6-luna' if sub != 'human_original' else None)}
            if 'regions' not in row: row.update(regions=[{'start': 0, 'end': len(r['text']), 'label': r.get('label', 0)}])
            cand['human_paper_old' if sub == 'human_original' else 'para_luna' if sub == 'paragraph_generate' else 'papers_other'].append(row)
        dcap = a.doc_window_cap or None  # 0 = every window of each document (scaled-up mixes)
        for ds, key, cap, fmt in [('mirrors', 'mirrors', dcap, 'generated_doc'), ('fullpapers', 'fullpapers', None, 'generated_markdown'), ('human', 'human', dcap, 'generic_human')]:
            docs = [d for d in pool_rows(ds) if d['paper_id'] not in blocked and d.get('group') not in {'paper:' + p for p in blocked}]
            rng.shuffle(docs); need = PLAN[key] * 2 if cap or key != 'fullpapers' else 10 ** 9; got = []
            for d in docs:
                for w in windows(d, tok, cap=cap, rng=rng): got.append({**w, 'text_format': fmt})
                if len(got) >= need: break
            cand[key] = got
        gt = pool_rows('gradtex'); rng.shuffle(gt)
        cand['gradtex'] = [{**r, 'text_format': 'gradtex', 'kind': 'gradtex'} for r in gt[:PLAN['gradtex'] * 2]]

        say(event='candidates', sizes={k: len(v) for k, v in cand.items()}, seconds=round(time.time() - t0))

        # ---- screen
        import multiprocessing as mp
        rejects = defaultdict(Counter)
        with mp.get_context('fork').Pool(a.workers) as pool:
            for k in [k for k in list(cand) if k != 'papers_other']:
                cand[k] = screen(cand[k], k, rejects, pool)
                say(event='screened', key=k, ok=len(cand[k]), rejects=dict(rejects[k]), seconds=round(time.time() - t0))
            cand.pop('papers_other', None)
            # wide clean-v2 pool, sized to what the human keys still need after the older sources were screened
            n_old = len(cand['human_paper_old']) if a.human_source == 'balanced' else 0
            need_win = max(0, PLAN['human_paper_orig'] - n_old) + PLAN['human_section'] + PLAN['human_para_e']  # matched-human shortfalls are filled from here too
            cw, ch = clean_pool(S, pool_dir, cand, nt, need_win)
            if a.human_windows == 'full':
                cw = full_text_windows(pool_dir, nt, tok, int(need_win * 1.15))
            cand['human_paper_clean'] = screen(cw, 'human_paper_clean', rejects, pool)
            cand['human_host_clean'] = screen(ch, 'human_host_clean', rejects, pool)
            say(event='screened_clean', windows=len(cand['human_paper_clean']), hosts=len(cand['human_host_clean']), rejects={k: dict(rejects[k]) for k in ('human_paper_clean', 'human_host_clean')})
        # matched human rows whose AI partner failed screening stay usable (they are plain human text); dedupe by id
        for k in cand: cand[k] = list({r['id']: r for r in cand[k]}.values())

        # ---- allocate AI keys (group fill before repeats)
        picked = {}; notes = defaultdict(dict)
        def take(key, rows, n):
            o = rr(rows, seed=zlib.crc32(key.encode()) % 1000)
            return o[:n]
        def alloc_group(keys_, sources):
            need = {k: PLAN[k] for k in keys_}; sup = {k: len(sources[k]) for k in keys_}
            got = {k: take(k, sources[k], min(need[k], sup[k])) for k in keys_}
            short = sum(max(0, need[k] - sup[k]) for k in keys_)
            surplus = {k: sup[k] - len(got[k]) for k in keys_ if sup[k] > len(got[k])}
            while short > 0 and surplus:
                tot = sum(surplus.values()); moved = 0
                for k in list(surplus):
                    add = min(surplus[k], max(1, round(short * surplus[k] / tot)), short - moved)
                    if add <= 0: continue
                    extra = rr(sources[k], seed=zlib.crc32(k.encode()) % 1000)[len(got[k]):len(got[k]) + add]
                    got[k] += extra; surplus[k] -= len(extra); moved += len(extra); notes[k]['filled_for_group'] = notes[k].get('filled_for_group', 0) + len(extra)
                    if surplus[k] <= 0: surplus.pop(k)
                short -= moved
                if moved == 0: break
            if short > 0:  # repeat within each short key, cycling its own distinct rows
                for k in keys_:
                    gap = need[k] - len(got[k]) if sup[k] < need[k] else 0
                    if gap > 0 and got[k] and short > 0:
                        add = min(gap, short); base_rows = list(got[k])
                        got[k] += [{**base_rows[i % len(base_rows)], 'id': base_rows[i % len(base_rows)]['id'] + f'#rep{i // len(base_rows) + 1}'} for i in range(add)]
                        notes[k]['repeated_rows'] = add; short -= add
                for k in keys_:  # a key with no rows at all (e.g. an API writer that has not run): repeat the others in its group
                    if short <= 0: break
                    base_rows = [r for r in got[k] if '#rep' not in r['id']]
                    if base_rows:
                        add = min(short, len(base_rows)); off = notes[k].get('repeated_rows', 0)
                        got[k] += [{**base_rows[(off + i) % len(base_rows)], 'id': base_rows[(off + i) % len(base_rows)]['id'] + f'#rep{(off + i) // len(base_rows) + 1}'} for i in range(add)]
                        notes[k]['repeated_rows'] = off + add; short -= add
            for k in keys_: picked[k] = got[k]
            return short
        unfilled = {}
        for g, ks in GROUP.items():
            if g == 'human paper text': continue
            unfilled[g] = alloc_group(ks, cand)

        # ---- human paper text: format-balanced sources, then group fill
        hosted_ai = [r for k in ('edit_claude', 'edit_deepseek', 'edit_gpt', 'para_e', 'para_deepseek', 'para_luna') for r in picked[k] if r['text_format'] != 'paired_pdf_v1']
        clean_share = sum(r['text_format'] == 'clean_v2' for r in hosted_ai) / max(1, len(hosted_ai))
        hsrc = {}
        if a.human_source == 'clean':
            hsrc['human_paper_orig'] = rr(cand['human_paper_clean'], seed=11)
            hsrc['human_host'] = rr(cand['human_host_clean'], seed=12)
        else:
            old = rr(cand['human_paper_old'], seed=13)[:PLAN['human_paper_orig']]
            hsrc['human_paper_orig'] = old + rr(cand['human_paper_clean'], seed=11)
            n_clean = round(PLAN['human_host'] * clean_share)
            oh = rr(cand['human_host_old'], seed=14); chh = rr(cand['human_host_clean'], seed=12)
            n_old = PLAN['human_host'] - n_clean
            if len(oh) < n_old: n_clean += n_old - len(oh); n_old = len(oh)
            hsrc['human_host'] = oh[:n_old] + chh[:n_clean] + chh[n_clean:] + oh[n_old:]
        hsrc['human_section'] = cand['human_section']; hsrc['human_para_e'] = cand['human_para_e']
        hp = GROUP['human paper text']
        for k in ('human_paper_orig', 'human_host'):  # already ordered; don't re-round-robin (it would undo the source split)
            hsrc[k] = list({r['id']: r for r in hsrc[k]}.values())
        def take_h(key, rows, n): return rows[:n] if key in ('human_paper_orig', 'human_host') else rr(rows, seed=zlib.crc32(key.encode()) % 1000)[:n]
        got = {k: take_h(k, hsrc[k], PLAN[k]) for k in hp}
        short = {k: PLAN[k] - len(got[k]) for k in hp if len(got[k]) < PLAN[k]}
        for k, gap in short.items():  # fill matched-human shortfalls with paper windows / hosts not yet used
            for fk in ('human_paper_orig', 'human_host'):
                if gap <= 0: break
                usedids = {r['id'] for r in got[fk]}; spare = [r for r in hsrc[fk] if r['id'] not in usedids][:gap]
                got[fk] += spare; notes[fk]['filled_for_group'] = notes[fk].get('filled_for_group', 0) + len(spare); gap -= len(spare)
            if gap > 0: unfilled['human paper text'] = unfilled.get('human paper text', 0) + gap
        for k in hp: picked[k] = got[k]

        # ---- epochs: every third row of each key's round-robin order -> epoch e (distinct rows per epoch)
        total = sum(len(v) for v in picked.values()); assert total <= sum(PLAN.values())
        out_dir = out / 'prepared-v2'
        if out_dir.exists(): shutil.rmtree(out_dir)
        shutil.copytree(base, out_dir); man = json.loads((base / 'manifest.json').read_text())
        rng_e = random.Random(20261007); report = {}
        for e in range(3):
            key = f'stage2-epoch{e}'; rows_out = []
            for cat in PLAN:
                rs = picked.get(cat, [])[e::3]
                for i, r in enumerate(rs):
                    rows_out.append({**r, 'dataset': cat, 'draw_id': f'{key}-{cat}-{i}'})
            b = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows_out).encode()
            (out_dir / f'{key}.jsonl.gz').write_bytes(gzip.compress(b, mtime=0))
            man['files'][key] = {**man['files'][key], 'rows': len(rows_out), 'sha256': hashlib.sha256(b).hexdigest()}
            man['counts'][key] = dict(Counter(r['dataset'] for r in rows_out))
        man['policy'] = ('Final single pass (Oct 7): 3 stage-2 files of distinct windows (T2.1 PLAN x3 + DeepSeek), clean v2 for new human paper text; '
                         'stage 1, selection and calibration copied from T2.1.')
        (out_dir / 'manifest.json').write_text(json.dumps(man, indent=1))
        # ---- report
        ids_all = Counter(r['id'] for v in picked.values() for r in v)
        for cat in PLAN:
            v = picked.get(cat, [])
            report[cat] = {'need': PLAN[cat], 'rows': len(v), 'distinct': len({r['id'].split('#rep')[0] for r in v}), 'supply_after_screening': len(hsrc.get(cat, cand.get(cat, []))),
                           'by_format': dict(Counter(r['text_format'] for r in v)), 'by_generator': dict(Counter(str(r.get('generator')) for r in v).most_common(8)), **notes.get(cat, {})}
        fmt = defaultdict(lambda: defaultdict(Counter))
        for g, ks in GROUP.items():
            for k in ks:
                for r in picked.get(k, []):
                    lab = 'AI' if (k in AI_KEYS or (r.get('supervision') == 'document_only' and r.get('document_label') == 1)) else 'human'
                    fmt[g][r['text_format']][lab] += 1
        paper_like = defaultdict(Counter)  # paper text only (everything except generic human, mirrors, GRADTEX, generated full papers)
        for k in ('edit_claude', 'edit_luna', 'edit_deepseek', 'edit_gpt', 'para_luna', 'para_e', 'para_deepseek', 'sections_ai', 'human_paper_orig', 'human_section', 'human_para_e', 'human_host'):
            for r in picked.get(k, []): paper_like[r['text_format']]['AI' if k in AI_KEYS else 'human'] += 1
        res = {'total_rows': total, 'per_key': report, 'unfilled_by_group': unfilled, 'duplicate_ids_across_keys': sum(1 for c in ids_all.values() if c > 1),
               'rejects': {k: dict(v) for k, v in rejects.items()}, 'api_rows': dict(api), 'human_source': a.human_source,
               'hosted_ai_clean_v2_share': round(clean_share, 3), 'format_check_by_group': {g: {f: dict(c) for f, c in v.items()} for g, v in fmt.items()},
               'format_check_paper_text': {f: {**dict(c), 'ai_share': round(c['AI'] / max(1, c['AI'] + c['human']), 3)} for f, c in sorted(paper_like.items())},
               'pool': str(pool_dir), 'never_train_files': [f.name for f in nt_files], 'manifest_sha256': hashlib.sha256((out_dir / 'manifest.json').read_bytes()).hexdigest(),
               'seconds': round(time.time() - t0)}
        (out / 'mix-report.json').write_text(json.dumps(res, indent=1)); shutil.rmtree(work, ignore_errors=True)
        say(event='done', total_rows=total, seconds=res['seconds'])
    except Exception as e:
        say(event='failed', error=repr(e), tb=traceback.format_exc()[-2500:]); raise


if __name__ == '__main__':
    main()
