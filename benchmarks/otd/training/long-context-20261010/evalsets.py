"""Long-context (<=1,000-token) copies of three held-out evaluation sets, with IDENTICAL added context for each
'mixed' row and its matching 'human_controls' row (Oct 10). See SPEC.md for the per-row contract this follows.

Sets (eval-inputs/ in $SCRATCH/longctx/screen):
  heldout-score-rows.jsonl.gz          build_llm_edits.prepare_heldout candidates '<paper>/hNNN' (candidates-heldout.jsonl):
                                       sweep-eval paired PDF papers (paper-gap10000-v3 or paper-eval-workflows dirs) or
                                       archive_pre2023_unused papers. Mixed '<cand>/<writer>-<edit>', control '<cand>/original'.
  t21-heldout-score-rows-v2.jsonl.gz   research/t21-heldout-20261006/build_t21_split.py: small edits (paired 'q', archive 'p',
                                       hosted pmc/pes2o/arxiv 'p'), hosted paragraph edits ('<host>#para/original'),
                                       sections ('claude-sec-*#section/original' <-> 'claude-sec-*/<kind>').
  wide1-heldout-rows.jsonl.gz          research/corpus-sampling-20261007 (clean v2 hosts 'or:<sha16>/pNNN'): small edits
                                       '<host>/<w>-<edit>' <-> '<host>/<w>-<edit>/orig'; paragraph edits '<host>/<w>-draft|rewrite'
                                       <-> '<host>/human'.

Paper renderings (each the function that produced the rows' own text; all joined with '\\n\\n'):
  paired PDF   paper_gap250_sources.paragraphs (as paired.py:_pdf_paragraphs, with g.OUT set to the paper's dataset dir)
  archive      build_llm_edits.paragraphs on research/exports/paper-text-hf batch-001.parquet text (as paired.py build())
  hosted HSM   build_hosted.split_paragraphs on the human-source-mix record text (build_hosted.candidates, host index pNN)
  sections     segments.jsonl.gz / segments-wide1.jsonl.gz front + section paragraphs (as paired.py build(), all variants)
  clean v2     build_pool.clean_paragraphs on corpus-clean-text-v2 text (as cleanv2.py _render 'para', host index pNNN)

Method per pair (control C, mixed rows M):
  1. Locate C in the paper rendering D: verbatim (unique; host paragraph index breaks ties), else through anchors: C's text
     before the edit and after it (edit bounds from the mixed row's AI span) located verbatim, with the text between equal
     to C's middle up to whitespace (insert_one controls carry a double space where the sentence was removed).
  2. Context = D before / after that location, token-balanced, budget = 1000 - max(tokens of every row of the pair), snapped
     with paired._snap_before/_snap_after (paragraph start/end within 300 chars, else word boundary). The SAME
     (before, after) strings are added to every row of the pair; each extended row is checked <= 1000 tokens.
  3. Added text is human paper text (papers <= 2022): label 0. Rows keep id/order/slot/condition/writer; text = before +
     text + after; regions and target offsets shifted by len(before). If any row of a pair cannot be extended, none is.

  uv run --with transformers --with tokenizers --with pyarrow --with beautifulsoup4 --with lxml python evalsets.py build|run
"""
import collections, copy, glob, gzip, json, os, re, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paired  # noqa: E402  (snapping helpers; not modified)

ROOT = Path(__file__).resolve().parents[4]
R = ROOT / 'research'
SCRATCH = Path(os.environ.get('SCRATCH') or '/SCRATCH-not-set')  # Claude Code scratch folder holding the intermediate files (outside the repo); set SCRATCH to it
LC = SCRATCH / 'longctx'
IN = LC / 'screen/eval-inputs'
OUT = LC / 'eval-1020'
CACHE = OUT / 'cache'
SETS = ['heldout-score-rows.jsonl.gz', 't21-heldout-score-rows-v2.jsonl.gz', 'wide1-heldout-rows.jsonl.gz']
PAIRED_DIRS = [R / 'data/paper-gap10000-v3-luna-20260930', R / 'data/paper-eval-workflows-luna-20260930']
ARCHIVE = R / 'exports/paper-text-hf/data/batch-001.parquet'
HSM = Path(os.environ.get('HSM_DATA') or '/HSM_DATA-not-set')  # hsm/data in the Oct 6 session's scratch folder; set HSM_DATA
SEGMENTS = [R / 'claude-sections-20261006/segments.jsonl.gz', R / 'claude-sections-20261006/segments-wide1.jsonl.gz']
CANDS = R / 'llm-sentence-edits-20261005/candidates-heldout.jsonl'
CORPUS = OUT / 'tmp-corpus'
SEP = '\n\n'
MAX_TOKENS = 1000
for d in ('research/llm-sentence-edits-20261005', 'research/claude-hosted-edits-20261006', 'research/claude-sections-20261006',
          'research/corpus-sampling-20261007', 'benchmarks/otd'):
    sys.path.insert(0, str(ROOT / d))


# ---------------------------------------------------------------- pairing
def pair_key(r, setname):
    i = r['id']
    if setname.startswith('wide1'):
        if i.endswith('/orig'): return i[:-5]
        if r['slot'] == 'human_controls' or r['condition'] in ('para_draft', 'para_rewrite'): return i.rsplit('/', 1)[0] + '#para'
        return i
    k = i.rsplit('/', 1)[0]
    return k if setname.startswith('heldout') else (k.split('#')[0] + ('#para' if '#para' in k else ''))


def t21_key(r):
    """t21 pairs: small edit '<host>/<w>' <-> '<host>/original'; para '<host>/<w>' (paragraph_*) <-> '<host>#para/original';
    section 'claude-sec-N/<kind>' <-> 'claude-sec-N#section/original'."""
    i = r['id']
    if r['slot'] == 'human_controls':
        return i.rsplit('/', 1)[0].replace('#section', '')
    base = i.rsplit('/', 1)[0]
    if r['condition'].startswith('paragraph_'): return base + '#para'
    return base


def groups(rows, setname):
    g = collections.defaultdict(list)
    for n, r in enumerate(rows):
        g[t21_key(r) if setname.startswith('t21') else pair_key(r, setname)].append(n)
    return g


# ---------------------------------------------------------------- paper renderings
def doc_specs(r, setname, cands):
    """[(doc key, host paragraph index or None)] for a row."""
    i = r['id']
    if setname.startswith('wide1'):
        m = re.match(r'or:[0-9a-f]{16}/p(\d+)', i)
        return [('cv2:' + r['paper_id'], int(m.group(1)) if m else None)]
    if setname.startswith('heldout'):
        cid = i.rsplit('/', 1)[0]; c = cands.get(cid)
        if c is None: return []
        h = int(re.search(r'/h(\d+)$', cid).group(1))
        return [('pdf:' + c['paper_id'] if c['source'] == 'sweep_eval_paper' else 'arch:' + c['paper_id'], h)]
    # t21
    if i.startswith('claude-sec-'):
        return [('sec:' + r['paper_id'], None)]
    m = re.match(r'(pmc|pes2o|arxiv|acl):[0-9a-f]{16}/p(\d+)', i)
    if m: return [('hsm:' + r['paper_id'], int(m.group(2)))]
    m = re.match(r'([^/]+)/q(\d+)', i)
    if m: return [('pdf:' + r['paper_id'], int(m.group(2)))]
    m = re.match(r'[0-9a-f]{16}/p(\d+)', i)
    if m: return [('arch:' + r['paper_id'], int(m.group(1)))]
    return []


_PAPERS = {}


def _pdf(pid):
    import paper_gap50 as g, paper_gap250_sources as srcmod
    d, paper = _PAPERS[pid]
    g.OUT = d
    try: return pid, [x['text'] for x in srcmod.paragraphs(paper)]
    except Exception as e: return pid, 'error:' + type(e).__name__


def build():
    """Cache every needed paper rendering: CACHE/docs.jsonl.gz {key, paragraphs} (sections: key#variant)."""
    import multiprocessing as mp
    import pyarrow as pa, pyarrow.parquet as pq
    CACHE.mkdir(parents=True, exist_ok=True)
    cands = {json.loads(l)['cand_id']: json.loads(l) for l in open(CANDS)}
    need = collections.defaultdict(set)
    for s in SETS:
        for l in gzip.open(IN / s, 'rt'):
            r = json.loads(l)
            for k, _ in doc_specs(r, s, cands):
                kind, pid = k.split(':', 1); need[kind].add(pid)
    print({k: len(v) for k, v in need.items()}, flush=True)
    docs = {}
    # sections
    variants = collections.Counter()
    for f in SEGMENTS:
        for l in gzip.open(f, 'rt'):
            s = json.loads(l)
            if s['paper_id'] in need['sec']:
                ps = list(s.get('front') or []) + [x for sec in s['sections'] for x in sec['paragraphs']]
                k = 'sec:' + s['paper_id']
                if any(docs.get(k + ('' if i == 0 else f'#{i}')) == ps for i in range(variants[k])): continue
                docs[k + ('' if variants[k] == 0 else f'#{variants[k]}')] = ps; variants[k] += 1
    # archive
    from build_llm_edits import paragraphs as ble_paragraphs
    for b in pq.ParquetFile(ARCHIVE).iter_batches(columns=['paper_id', 'text'], batch_size=2000):
        for r in b.to_pylist():
            if r['paper_id'] in need['arch']: docs['arch:' + r['paper_id']] = ble_paragraphs(r['text'])
    # hosted human-source-mix records
    from build_hosted import split_paragraphs
    for f in sorted(HSM.glob('*.parquet')):
        for r in pq.read_table(f, columns=['record_id', 'text']).to_pylist():
            if r['record_id'] in need['hsm']: docs['hsm:' + r['record_id']] = split_paragraphs(r['text'])
    # clean v2 corpus (downloaded shards, deleted after the build)
    if need['cv2']:
        from build_pool import clean_paragraphs
        vs = pa.array(sorted(need['cv2']))
        for f in sorted(glob.glob(str(CORPUS / '*.parquet'))):
            tb = pq.read_table(f, columns=['pdf_sha256', 'text'])
            for r in tb.filter(pa.compute.is_in(tb['pdf_sha256'], value_set=vs)).to_pylist():
                docs['cv2:' + r['pdf_sha256']] = clean_paragraphs(r['text'])
    # paired PDF papers
    for d in PAIRED_DIRS:
        for l in open(d / 'papers.jsonl'):
            p = json.loads(l)
            if p['paper_id'] in need['pdf'] and p['paper_id'] not in _PAPERS and (d / 'sources' / f"{p['paper_id']}.bbox.html").exists():
                _PAPERS[p['paper_id']] = (d, p)
    want = sorted(_PAPERS)
    with mp.get_context('fork').Pool(10) as pool:
        for i, (pid, ps) in enumerate(pool.imap_unordered(_pdf, want, chunksize=4)):
            if isinstance(ps, list): docs['pdf:' + pid] = ps
            if i % 200 == 0: print('pdf', i, len(want), flush=True)
    with gzip.open(CACHE / 'docs.jsonl.gz', 'wt') as f:
        for k, v in docs.items(): f.write(json.dumps({'key': k, 'paragraphs': v}, ensure_ascii=False) + '\n')
    have = collections.Counter(k.split(':')[0] for k in docs if '#' not in k)
    print('docs', dict(have), 'needed', {k: len(v) for k, v in need.items()}, flush=True)


# ---------------------------------------------------------------- locating a pair
def _norm(s): return re.sub(r'\s+', ' ', s).strip()


def edit_bounds(C, mixed):
    """(a, e): the control's part that the mixed rows replaced. C[:a] / C[e:] are shared with every mixed row."""
    a, e = len(C), 0
    for m in mixed:
        T = m['text']; ai = [g for g in m['regions'] if g['label'] != 0]
        if not ai: continue
        s, b = min(g['start'] for g in ai), max(g['end'] for g in ai)
        a = min(a, s); e = max(e, len(C) - (len(T) - b))
    return (a, e) if a <= e else (a, a)


def locate(C, D, ps, host, mixed):
    """-> ((start, end) of the control's place in D, how) or (None, reason)."""
    starts, pos = [], 0
    for p in ps: starts.append(pos); pos += len(p) + len(SEP)

    def at_host(o, n):
        if host is None or host >= len(ps): return True
        return o <= starts[host] + len(ps[host]) and o + n >= starts[host]
    occ = paired._occ(D, C)
    if len(occ) > 1: occ = [o for o in occ if at_host(o, len(C))]
    if len(occ) == 1: return (occ[0], occ[0] + len(C)), 'verbatim'
    if len(occ) > 1: return None, 'control_ambiguous'
    a, e = edit_bounds(C, mixed)
    pre, suf = C[:a].rstrip(), C[e:].lstrip()
    mid = _norm(C[len(pre):len(C) - len(suf)]); cands = []
    if len(pre) + len(suf) < 40: return None, 'control_not_in_paper'
    if not suf or not pre:  # the control is one anchor plus whitespace only (an insertion at its very end / start)
        if mid: return None, 'control_not_in_paper'
        anc = pre or suf; cands = [(o, o + len(anc)) for o in paired._occ(D, anc)]
    for o in (paired._occ(D, pre) if pre and suf else []):
        p = o + len(pre); q = D.find(suf, p)
        while 0 <= q <= p + len(mid) + 200:  # the suffix occurrence right after the (whitespace-equal) middle
            if _norm(D[p:q]) == mid: cands.append((o, q + len(suf))); break
            q = D.find(suf, q + 1)
    if len(cands) > 1: cands = [c for c in cands if at_host(c[0], c[1] - c[0])]
    if len(cands) == 1: return cands[0], 'anchored'
    return None, 'control_ambiguous' if cands else 'control_not_in_paper'


# ---------------------------------------------------------------- context
_C = {}


def docs():
    if 'docs' not in _C:
        _C['docs'] = {}
        for l in gzip.open(CACHE / 'docs.jsonl.gz', 'rt'):
            d = json.loads(l); _C['docs'][d['key']] = d['paragraphs']
    return _C['docs']


def ntok(tok, t): return len(tok(t, add_special_tokens=False)['input_ids'])


def context(E, r0, r1, texts, tok, max_tokens=MAX_TOKENS):
    """(before, after) taken from E around [r0, r1), the same for every text in texts, each result <= max_tokens.
    Token-balanced and snapped exactly as paired.extend (paired.py: CTX_CHARS window, _snap_before/_snap_after)."""
    n_row = max(ntok(tok, t) for t in texts)
    budget = max_tokens - n_row
    if budget <= 0: return None, 'row_at_max_tokens'
    b0 = max(0, r0 - paired.CTX_CHARS); bef = E[b0:r0]; aft = E[r1:min(len(E), r1 + paired.CTX_CHARS)]
    ob = tok(bef, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping'] if bef else []
    oa = tok(aft, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping'] if aft else []
    nb, na = len(ob), len(oa)
    if nb + na == 0: return None, 'no_context_in_paper'
    tb = min(nb, max(budget // 2, budget - na)); ta = min(na, budget - tb)
    for _ in range(40):
        x = b0 + ob[nb - tb][0] if tb > 0 else r0
        y = r1 + oa[ta - 1][1] if ta > 0 else r1
        x = paired._snap_before(E, x, r0) if x < r0 else r0
        y = paired._snap_after(E, y, r1) if y > r1 else r1
        before, after = E[x:r0], E[r1:y]
        n = max(ntok(tok, before + t + after) for t in texts)
        if n <= max_tokens: break
        over = n - max_tokens + 2
        if tb >= ta: tb = max(0, tb - over)
        else: ta = max(0, ta - over)
    else:
        return None, 'could_not_fit'
    if not before and not after: return None, 'no_context_in_paper'
    return (before, after), None


def apply(row, before, after, src):
    t = row['text']; L = len(before)
    out = copy.deepcopy(row)
    out['text'] = before + t + after
    assert out['text'][L:L + len(t)] == t
    rg = [{'start': g['start'] + L, 'end': g['end'] + L, 'label': g['label']} for g in row['regions']]
    if before: rg.insert(0, {'start': 0, 'end': L, 'label': 0})
    if after: rg.append({'start': L + len(t), 'end': len(out['text']), 'label': 0})
    out['regions'] = rg
    for k in ('target_start', 'target_end'):
        if row.get(k) is not None: out[k] = int(row[k]) + L
    out.update(context_extended=True, ctx_before_chars=L, ctx_after_chars=len(after), ctx_source=src)
    return out


def extend_pair(rows, setname, cands, tok):
    """rows: one control + its mixed rows. -> (new rows, None) or (None, reason)."""
    ctl = [r for r in rows if r['slot'] == 'human_controls']; mixed = [r for r in rows if r['slot'] == 'mixed']
    if len(ctl) != 1 or not mixed: return None, 'unpaired'
    C = ctl[0]['text']
    a, e = edit_bounds(C, mixed)
    for m in mixed:  # the mixed row's human edges must be the control's (outer whitespace of an inserted sentence excepted)
        T = m['text']; ai = [g for g in m['regions'] if g['label'] != 0]
        s, b = min(g['start'] for g in ai), max(g['end'] for g in ai)
        if not (C.startswith(T[:s].rstrip()) and C.endswith(T[b:].lstrip())): return None, 'pair_edges_differ'
    specs = doc_specs(ctl[0], setname, cands)
    if not specs: return None, 'no_paper_source'
    key, host = specs[0]
    D_ = docs(); why = 'paper_text_missing'
    for v in range(10):
        ps = D_.get(key + ('' if v == 0 else f'#{v}'))
        if ps is None: break
        D = SEP.join(ps)
        loc, how = locate(C, D, ps, host, mixed)
        if loc is None: why = how if why == 'paper_text_missing' else why; continue
        ctx, w = context(D, loc[0], loc[1], [r['text'] for r in rows], tok)
        if ctx is None: return None, w
        src = key.split(':')[0] + ':' + how
        return [apply(r, ctx[0], ctx[1], src) for r in rows], None
    return None, why


# ---------------------------------------------------------------- run + validation
def q(v, p):
    if not v: return None
    v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))]


def run():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(LC / 'tokenizer'))
    cands = {json.loads(l)['cand_id']: json.loads(l) for l in open(CANDS)}
    report = {'max_tokens': MAX_TOKENS, 'tokenizer': 'Qwen/Qwen3.6-35B-A3B@995ad96e (add_special_tokens=False)', 'sets': {}}
    for s in SETS:
        rows = [json.loads(l) for l in gzip.open(IN / s, 'rt')]
        new = list(rows); reasons = {}; pairs_ok = 0
        G = groups(rows, s)
        for k, idx in G.items():
            out, why = extend_pair([rows[i] for i in idx], s, cands, tok)
            if out is None:
                for i in idx: reasons[i] = why
                continue
            pairs_ok += 1
            for i, o in zip(idx, out): new[i] = o
        # validation
        checks = collections.Counter()
        for i, (r, n) in enumerate(zip(rows, new)):
            assert n['id'] == r['id'] and all(n.get(f) == r.get(f) for f in ('slot', 'condition', 'writer', 'split', 'label'))
            if not n.get('context_extended'): continue
            L = n['ctx_before_chars']; t = n['text']
            checks['verbatim_ok'] += t[L:L + len(r['text'])] == r['text']
            cov = sorted((g['start'], g['end']) for g in n['regions'])
            checks['regions_valid'] += all(0 <= x < y <= len(t) for x, y in cov) and cov[0][0] == 0 and cov[-1][1] == len(t) \
                and all(cov[j][1] == cov[j + 1][0] for j in range(len(cov) - 1))
            ai_old = [r['text'][g['start']:g['end']] for g in r['regions'] if g['label'] == 1]
            ai_new = [t[g['start']:g['end']] for g in n['regions'] if g['label'] == 1]
            checks['ai_chars_unchanged'] += ai_old == ai_new
            checks['added_label_0'] += all(g['label'] == 0 for g in n['regions'] if g['end'] <= L or g['start'] >= L + len(r['text']))
            checks['tokens_ok'] += ntok(tok, t) <= MAX_TOKENS
            checks['rows_checked'] += 1
        for k, idx in G.items():
            ext = [new[i] for i in idx if new[i].get('context_extended')]
            if not ext: continue
            assert len(ext) == len(idx)
            ctxs = {(n['text'][:n['ctx_before_chars']], n['text'][len(n['text']) - n['ctx_after_chars']:] if n['ctx_after_chars'] else '') for n in ext}
            checks['pairs_identical_context'] += len(ctxs) == 1
            checks['pairs_checked'] += 1
        assert all(v == checks['rows_checked'] for k, v in checks.items() if k not in ('pairs_identical_context', 'pairs_checked', 'rows_checked'))
        assert checks['pairs_identical_context'] == checks['pairs_checked']
        # per condition stats
        per = {}
        bycond = collections.defaultdict(list)
        for i, r in enumerate(rows): bycond[r['condition']].append(i)
        tokb = [ntok(tok, r['text']) for r in rows]; toka = [ntok(tok, n['text']) for n in new]
        for c, idx in sorted(bycond.items()):
            ex = [i for i in idx if new[i].get('context_extended')]
            per[c] = {'rows': len(idx), 'extended': len(ex),
                      'tokens_before_median': q([tokb[i] for i in idx], .5), 'tokens_before_p90': q([tokb[i] for i in idx], .9),
                      'tokens_after_median': q([toka[i] for i in idx], .5), 'tokens_after_p90': q([toka[i] for i in idx], .9),
                      'extended_tokens_after_median': q([toka[i] for i in ex], .5), 'extended_tokens_after_p10': q([toka[i] for i in ex], .1),
                      'failures': dict(collections.Counter(reasons[i] for i in idx if i in reasons)),
                      'ctx_source': dict(collections.Counter(new[i]['ctx_source'] for i in ex))}
            if rows[idx[0]]['slot'] == 'mixed' or c != 'original':
                per[c]['pairs_extended'] = len([i for i in ex if rows[i]['slot'] == 'mixed'])
        if s.startswith('t21'):
            bysrc = collections.defaultdict(list)
            for i, r in enumerate(rows): bysrc[(r.get('source') or '?') + ('/section' if r['id'].startswith('claude-sec') else '')].append(i)
            src_stats = {k: {'rows': len(v), 'extended': sum(1 for i in v if new[i].get('context_extended')),
                             'failures': dict(collections.Counter(reasons[i] for i in v if i in reasons))} for k, v in sorted(bysrc.items())}
        else:
            src_stats = None
        report['sets'][s] = {'rows': len(rows), 'extended': sum(1 for n in new if n.get('context_extended')), 'pairs': len(G),
                             'pairs_extended': pairs_ok, 'failures': dict(collections.Counter(reasons.values())), 'checks': dict(checks),
                             'tokens_all': {'before_median': q(tokb, .5), 'before_p90': q(tokb, .9), 'after_median': q(toka, .5), 'after_p90': q(toka, .9),
                                            'after_max': max(toka)},
                             'per_condition': per, **({'per_source': src_stats} if src_stats else {})}
        with gzip.open(OUT / s, 'wt') as f:
            for n in new: f.write(json.dumps(n, ensure_ascii=False) + '\n')
        print(s, json.dumps({k: v for k, v in report['sets'][s].items() if k not in ('per_condition', 'per_source')}), flush=True)
    (OUT / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    {'build': build, 'run': run}[sys.argv[1]]()
