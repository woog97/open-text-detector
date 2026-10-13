"""Long-context extension for the `paired` family of the Oct 7 mix (see SPEC.md in this folder).

Family: every mix row with text_format paired_pdf_v1 or archive_v1:
  edit_luna (splices + Luna llm edits), edit_claude (paired part), para_luna, human_paper_orig (paired part),
  sections_ai / human_section (paired_pdf_v1 and archive_v1).

Method (exact, no approximation):
  1. Each mix row is a crop of a pool row (mix_pools.py / mix_build.py crop510 adds '@<cs>' to the id and sets
     source_start). The pool row is loaded from the local source file and row.text == pool.text[source_start:...]
     is asserted.
  2. The pool row's paper is rebuilt as a paragraph list with the SAME function that produced the pool row's context,
     joined with '\n\n' (the separator every one of these builders uses):
       paired PDF papers   benchmarks/otd/paper_gap250_sources.py:6 paragraphs(paper) (g.OUT = paper-gap10000-v3)
                           -- used by paper_scale10000_acquire.snapshot.py:86 (paired passages, hence splices and
                           paragraph_generate) and build_llm_edits.py prepare_v1 (the 'q' llm-edit candidates).
       archive papers      research/llm-sentence-edits-20261005/build_llm_edits.py:39 paragraphs(raw) on the
                           research/exports/paper-text-hf batch-001.parquet text ('p' llm-edit candidates).
       sections            research/claude-sections-20261006/segments.jsonl.gz (output of build_sections.py
                           segment_paired:77 / segment_archive:114): front + every section's paragraphs, the document
                           that build_sections.py excerpt():228 cuts the section pool rows from.
  3. The pool row's leading human part (before its first AI region) and trailing human part (after its last AI
     region) are located verbatim in that document; a fully human pool row is located whole. The AI-edited document is
     E = D[:start] + pool.text + D[end:]; the mix row sits at a known offset in E. Ambiguous or missing matches -> None.
     Where the pool row records the replaced human text (splices, llm edits), D between the anchors must equal it.
  4. Context is added on both sides, token-balanced, up to max_tokens, snapped to word / paragraph starts.
     Added text keeps E's labels (paper text 0; any cropped-off part of the pool row keeps the pool row's labels).
  5. Added context is screened like mix_build.py screen_one: build_llm_edits.never_train_hit and 10+-word sentences
     shared with the sweep-eval rows. A hit returns None.

  python paired.py build     # cache: pool rows + paper paragraph lists for every family row (multiprocess, ~5 min)
  python paired.py run       # extend all family rows -> $SCRATCH/longctx/paired/{extended-ids.jsonl.gz,report.json}
Run with: uv run --with beautifulsoup4 --with lxml --with httpx --with requests --with pypdf --with pyarrow
          --with transformers --with tokenizers python paired.py ...
"""
import copy, gzip, json, os, re, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCRATCH = Path(os.environ.get('SCRATCH') or '/SCRATCH-not-set')  # Claude Code scratch folder holding the intermediate files (outside the repo); set SCRATCH to it
OUT = SCRATCH / 'longctx/paired'
CACHE = OUT / 'cache'
TOKDIR = SCRATCH / 'longctx/tokenizer'
INPUTS = [SCRATCH / f'longctx/input/stage2-epoch{e}.jsonl.gz' for e in range(3)]
FORMATS = ('paired_pdf_v1', 'archive_v1')
R = ROOT / 'research'
PAIRED = R / 'data/paper-gap10000-v3-luna-20260930'
ARCHIVE = R / 'exports/paper-text-hf/data/batch-001.parquet'
POOL_FILES = [R / 'splice-edits-20261005/splices-v1.jsonl.gz', R / 'llm-sentence-edits-20261005/llm-edits-v1.jsonl.gz',
              R / 'llm-sentence-edits-20261005/llm-edits-pilot.jsonl.gz', R / 'llm-sentence-edits-20261005/llm-edits-claude-v1.jsonl.gz',
              R / 'claude-sections-20261006/claude-sections-v1-train.jsonl.gz']
PASSAGES = ROOT / 'benchmarks/otd/exports/ai-paper-provenance-v3-10000/data/passages/train-00000-of-00001.parquet'
SEGMENTS = [R / 'claude-sections-20261006/segments.jsonl.gz', R / 'claude-sections-20261006/segments-wide1.jsonl.gz']
EVAL_ROWS = ROOT / 'benchmarks/otd/training/overnight-sweep-20261004/sweep-eval-rows.jsonl.gz'
SEP = '\n\n'
CTX_CHARS = 9000      # characters of E examined on each side (>= 1,000 tokens of any text)
SNAP_PARA = 300       # prefer a paragraph boundary when one lies within this many characters of the token cut
MAX_GAP = 40000       # sanity bound on the replaced original span between the two anchors
SENT = re.compile(r'\S.*?(?:[.!?](?=\s|$)|$)', re.S)  # mix_build.py SENT / sents()


def _regs(x):
    import ast
    x = ast.literal_eval(x) if isinstance(x, str) else x
    return [{'start': int(g['start']), 'end': int(g['end']), 'label': int(g['label'])} for g in x]


def pool_id(row_id):
    return re.sub(r'@\d+$', '', re.sub(r'#rep\d+$', '', row_id))


def doc_key(pool_row, row):
    """Which paragraph rebuild produced this pool row's context."""
    pid = str(row['paper_id']).replace('paper:', '')
    rid = pool_row['id']
    if rid.startswith('claude-sec-'):
        return 'sec:' + pid
    if re.search(r'/p\d{3,}/', rid):
        return 'arch:' + pid          # build_llm_edits archive_pre2023 candidates (Poppler raw text)
    return 'pdf:' + pid               # paired PDF papers: passages, splices, 'q' llm-edit candidates


def ctx_source(key):
    return {'pdf': 'paired_pdf:paper_gap250_sources.paragraphs', 'arch': 'archive:build_llm_edits.paragraphs',
            'sec': 'sections:build_sections.segments'}[key.split(':')[0]]


# ---------------------------------------------------------------- cache build

def family_rows():
    for f in INPUTS:
        for l in gzip.open(f, 'rt'):
            r = json.loads(l)
            if r['text_format'] in FORMATS:
                yield r


def _pdf_paragraphs(pid):
    sys.path.insert(0, str(ROOT / 'benchmarks/otd'))
    import paper_gap50 as g, paper_gap250_sources as srcmod
    g.OUT = PAIRED
    paper = _PAPERS[pid]
    try:
        return pid, [x['text'] for x in srcmod.paragraphs(paper)]
    except Exception as e:
        return pid, 'error:' + type(e).__name__


_PAPERS = {}


def build():
    import multiprocessing as mp
    import pyarrow.parquet as pq
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = list(family_rows())
    need = {pool_id(r['id']) for r in rows}
    need |= {i.split('/')[0] + '/human' for i in need if i.startswith('claude-sec-')}  # matched human section (partner)
    pools = {}
    for f in POOL_FILES:
        for l in gzip.open(f, 'rt'):
            r = json.loads(l)
            if r['id'] in need:
                pools[r['id']] = {'id': r['id'], 'text': r['text'], 'regions': _regs(r['regions']), 'replaced_human': r.get('replaced_human'),
                                  'target_start': int(r['target_start']), 'target_end': int(r['target_end'])}
    for r in pq.read_table(PASSAGES, columns=['id', 'text', 'regions']).to_pylist():
        if r['id'] in need:
            pools[r['id']] = {'id': r['id'], 'text': r['text'], 'regions': _regs(r['regions']), 'replaced_human': None}
    missing = {i for i in need - set(pools) if not i.endswith('/human')}
    print('pool rows', len(pools), 'missing', len(missing))
    keys = Counter()
    for r in rows:
        p = pools.get(pool_id(r['id']))
        if p: keys[doc_key(p, r)] += 1
    docs = {}; variants = Counter()
    # sections: segments (front + every section's paragraphs, in document order)
    want_sec = {k[4:] for k in keys if k.startswith('sec:')}
    for f in SEGMENTS:
        for l in gzip.open(f, 'rt'):
            s = json.loads(l)
            if s['paper_id'] in want_sec:  # a paper can be segmented in both files (different runs): keep every variant
                ps = list(s.get('front') or []) + [x for sec in s['sections'] for x in sec['paragraphs']]
                k = 'sec:' + s['paper_id']
                if any(docs.get(k + ('' if i == 0 else f'#{i}')) == ps for i in range(variants[k])): continue
                docs[k + ('' if variants[k] == 0 else f'#{variants[k]}')] = ps; variants[k] += 1
    # archive papers: build_llm_edits.paragraphs on the archive parquet text
    want_arch = {k[5:] for k in keys if k.startswith('arch:')}
    if want_arch:
        sys.path.insert(0, str(R / 'llm-sentence-edits-20261005'))
        from build_llm_edits import paragraphs as ble_paragraphs
        pf = pq.ParquetFile(ARCHIVE)
        for b in pf.iter_batches(columns=['paper_id', 'text'], batch_size=2000):
            for r in b.to_pylist():
                if r['paper_id'] in want_arch:
                    docs['arch:' + r['paper_id']] = ble_paragraphs(r['text'])
    # paired PDF papers: paper_gap250_sources.paragraphs (Poppler bbox), in parallel
    want_pdf = sorted(k[4:] for k in keys if k.startswith('pdf:'))
    for l in open(PAIRED / 'papers.jsonl'):
        p = json.loads(l); _PAPERS[p['paper_id']] = p
    want_pdf = [p for p in want_pdf if p in _PAPERS]
    with mp.get_context('fork').Pool(10) as pool:
        for i, (pid, ps) in enumerate(pool.imap_unordered(_pdf_paragraphs, want_pdf, chunksize=4)):
            if isinstance(ps, list): docs['pdf:' + pid] = ps
            if i % 200 == 0: print('pdf', i, len(want_pdf), flush=True)
    with gzip.open(CACHE / 'pools.jsonl.gz', 'wt') as f:
        for p in pools.values(): f.write(json.dumps(p, ensure_ascii=False) + '\n')
    with gzip.open(CACHE / 'docs.jsonl.gz', 'wt') as f:
        for k, v in docs.items(): f.write(json.dumps({'key': k, 'paragraphs': v}, ensure_ascii=False) + '\n')
    print('section variants', Counter(variants.values()), 'docs', Counter(k.split(':')[0] for k in docs), 'needed', Counter(k.split(':')[0] for k in keys))


# ---------------------------------------------------------------- extension

_C = {}


def _load():
    if _C: return _C
    _C['pools'] = {json.loads(l)['id']: json.loads(l) for l in gzip.open(CACHE / 'pools.jsonl.gz', 'rt')}
    _C['docs'] = {}
    for l in gzip.open(CACHE / 'docs.jsonl.gz', 'rt'):
        d = json.loads(l); _C['docs'][d['key']] = SEP.join(d['paragraphs'])
    held = set()
    if EVAL_ROWS.exists():
        for l in gzip.open(EVAL_ROWS, 'rt'): held |= _sents(json.loads(l)['text'])
    _C['held'] = held
    sys.path.insert(0, str(R / 'llm-sentence-edits-20261005'))
    from build_llm_edits import never_train_hit
    _C['nt'] = never_train_hit
    return _C


def _sents(t):
    return {' '.join(w) for m in SENT.finditer(t) for w in [re.findall(r'[a-z0-9]+', m.group().lower())] if len(w) >= 10}


def _occ(hay, needle):
    out, i = [], hay.find(needle)
    while i >= 0:
        out.append(i); i = hay.find(needle, i + 1)
    return out


def _embed(D, P, regs, o, tail):
    """E = D[:o] + P + D[tail:], with E's regions (paper text 0, pool row its own labels)."""
    E = D[:o] + P + D[tail:]
    er = ([{'start': 0, 'end': o, 'label': 0}] if o else []) + [{'start': g['start'] + o, 'end': g['end'] + o, 'label': g['label']} for g in regs]
    if tail < len(D): er.append({'start': o + len(P), 'end': len(E), 'label': 0})
    assert E[o:o + len(P)] == P and E[:o] == D[:o] and E[o + len(P):] == D[tail:]
    return E, er, o


def locate(P, regs, D, replaced=None, partner=None):
    """(E, E_regions, pool_offset_in_E) or (None, reason).

    Fully human pool row: located whole (exactly one occurrence). Otherwise its human text before the first AI region
    and after the last one are located verbatim (outer whitespace excepted: an inserted sentence brings its own joining
    space); the text between the anchors in the paper must be the recorded replaced human text when there is one.
    Sections whose AI part touches the pool row's edge are placed through the matched human row (same context
    paragraphs, original section in place of the AI one), located whole."""
    ai = [g for g in regs if g['label'] != 0]
    if not ai:
        o = _occ(D, P)
        if len(o) != 1: return None, 'human_not_found' if not o else 'human_ambiguous'
        return (D, [{'start': 0, 'end': len(D), 'label': 0}], o[0]), None
    a, b = min(g['start'] for g in ai), max(g['end'] for g in ai)
    pre, suf = P[:a].rstrip(), P[b:].lstrip()
    if not pre or not suf:
        if partner is None: return None, 'ai_at_pool_edge'
        H, hs, he = partner['text'], partner['target_start'], partner['target_end']
        if P[:a] != H[:hs] or P[b:] != H[he:]: return None, 'partner_context_differs'
        o = _occ(D, H)
        if len(o) != 1: return None, 'partner_not_found' if not o else 'partner_ambiguous'
        return _embed(D, P, regs, o[0], o[0] + len(H)), None
    cands = []
    for o in _occ(D, pre):
        p = o + len(pre); q = D.find(suf, p)
        if q >= 0 and q - p <= MAX_GAP: cands.append((o, p, q))
    if not cands: return None, 'anchor_not_found'
    if len(cands) > 1: return None, 'anchor_ambiguous'
    o, p, q = cands[0]
    if replaced is not None and D[p:q].strip() != replaced.strip(): return None, 'replaced_text_mismatch'
    return _embed(D, P, regs, o, q + len(suf)), None


def _snap_before(E, x, lo_limit):
    """Move a start offset forward to a word start, preferring a paragraph start within SNAP_PARA chars."""
    if x <= 0: return 0
    k = E.find(SEP, max(0, x - 2), min(lo_limit, x + SNAP_PARA))
    if k >= 0: return k + len(SEP)
    while x < lo_limit and not E[x - 1].isspace(): x += 1
    while x < lo_limit and E[x].isspace(): x += 1
    return x


def _snap_after(E, y, hi_limit):
    """Move an end offset back to a word end, preferring a paragraph end within SNAP_PARA chars."""
    if y >= len(E): return len(E)
    k = E.rfind(SEP, max(hi_limit, y - SNAP_PARA), y + 2)
    if k >= hi_limit: return k
    while y > hi_limit and not E[y].isspace() and not E[y - 1].isspace(): y -= 1
    while y > hi_limit and E[y - 1].isspace(): y -= 1
    return y


def extend(row, tok, max_tokens=1000, _why=None):
    """New row with real surrounding paper context (SPEC.md contract) or None. _why: optional list for the None reason."""
    def none(reason):
        if _why is not None: _why.append(reason)
        return None
    if row.get('text_format') not in FORMATS: return none('not_family')
    C = _load()
    P = C['pools'].get(pool_id(row['id']))
    if P is None: return none('pool_row_missing')
    key = doc_key(P, row)
    if key not in C['docs']: return none('paper_text_missing')
    ss = int(row.get('source_start') or 0); text = row['text']
    if P['text'][ss:ss + len(text)] != text: return none('row_not_in_pool_row')
    partner = C['pools'].get(P['id'].split('/')[0] + '/human') if P['id'].startswith('claude-sec-') else None
    replaced = P.get('replaced_human')
    if replaced is None and partner is not None and any(g['label'] for g in P['regions']):
        replaced = partner['text'][partner['target_start']:partner['target_end']]  # the original human section
    loc = why = None
    for i in range(10):  # section papers may have more than one segmentation variant; the first that matches is used
        D = C['docs'].get(key + ('' if i == 0 else f'#{i}'))
        if D is None: break
        loc, w = locate(P['text'], P['regions'], D, replaced, partner)
        if loc is not None: break
        why = why or w
    if loc is None: return none(why)
    E, ER, o = loc
    r0 = o + ss; r1 = r0 + len(text)
    assert E[r0:r1] == text
    n_row = len(tok(text, add_special_tokens=False)['input_ids'])
    budget = max_tokens - n_row
    if budget <= 0: return none('row_at_max_tokens')
    b0 = max(0, r0 - CTX_CHARS); bef = E[b0:r0]; aft = E[r1:min(len(E), r1 + CTX_CHARS)]
    ob = tok(bef, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping'] if bef else []
    oa = tok(aft, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping'] if aft else []
    nb, na = len(ob), len(oa)
    if nb + na == 0: return none('no_context_in_paper')
    tb = min(nb, max(budget // 2, budget - na)); ta = min(na, budget - tb)
    for _ in range(30):
        x = b0 + ob[nb - tb][0] if tb > 0 else r0
        y = r1 + oa[ta - 1][1] if ta > 0 else r1
        x = _snap_before(E, x, r0) if x < r0 else r0
        y = _snap_after(E, y, r1) if y > r1 else r1
        new = E[x:y]
        n = len(tok(new, add_special_tokens=False)['input_ids'])
        if n <= max_tokens: break
        over = n - max_tokens + 2
        if tb >= ta: tb = max(0, tb - over)
        else: ta = max(0, ta - over)
    else:
        return none('could_not_fit')
    if x == r0 and y == r1: return none('no_context_in_paper')
    added_before, added_after = E[x:r0], E[r1:y]
    ctx = (added_before + '\n\n' + added_after).strip()
    if C['nt']({'paper_id': row.get('paper_id'), 'group': row.get('group'), 'id': row['id'], 'text': added_before + text + added_after}):
        return none('context_never_train_hit')
    if _sents(ctx) & C['held']: return none('context_eval_shared_sentence')
    shift = len(added_before)
    out = copy.deepcopy(row)
    out['text'] = added_before + text + added_after
    assert out['text'][shift:shift + len(text)] == text
    rg = [{**g, 'start': g['start'] + shift, 'end': g['end'] + shift} for g in row['regions']]
    for s, e in ((x, r0), (r1, y)):
        for g in ER:
            a, b = max(g['start'], s), min(g['end'], e)
            if b > a: rg.append({'start': a - x, 'end': b - x, 'label': g['label']})
    out['regions'] = sorted(rg, key=lambda g: (g['start'], g['end']))
    for k in ('target_start', 'target_end'):
        if row.get(k) is not None: out[k] = int(row[k]) + shift
    out.update(context_extended=True, ctx_before_chars=len(added_before), ctx_after_chars=len(added_after),
               ctx_source=ctx_source(key))
    return out


# ---------------------------------------------------------------- run + validation

def run():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOKDIR))
    stats = defaultdict(lambda: {'attempted': 0, 'extended': 0, 'none': Counter(), 'tok_before': [], 'tok_after': [], 'tok_before_all': []})
    checks = Counter(); examples = []
    with gzip.open(OUT / 'extended-ids.jsonl.gz', 'wt') as f:
        for r in family_rows():
            k = f"{r['dataset']}|{r['text_format']}|{r['kind']}"; st = stats[k]; st['attempted'] += 1
            n0 = len(tok(r['text'], add_special_tokens=False)['input_ids']); st['tok_before_all'].append(n0)
            why = []
            out = extend(r, tok, 1000, why)
            if out is None:
                st['none'][why[0] if why else '?'] += 1; continue
            st['extended'] += 1; st['tok_before'].append(n0)
            n1 = len(tok(out['text'], add_special_tokens=False)['input_ids']); st['tok_after'].append(n1)
            sh = out['ctx_before_chars']; t = out['text']
            checks['verbatim_ok'] += t[sh:sh + len(r['text'])] == r['text']
            checks['tokens_ok'] += n1 <= 1000
            checks['regions_valid'] += all(0 <= g['start'] < g['end'] <= len(t) for g in out['regions'])
            cov = sorted((g['start'], g['end']) for g in out['regions'])
            checks['regions_cover_no_overlap'] += cov[0][0] == 0 and cov[-1][1] == len(t) and all(a[1] == b[0] for a, b in zip(cov, cov[1:]))
            checks['ai_chars_unchanged'] += all(t[g['start'] + sh:g['end'] + sh] == r['text'][g['start']:g['end']] for g in r['regions'] if g['label'] == 1)
            checks['rows_checked'] += 1
            if len(examples) < 5 and (not examples or examples[-1]['key'] != k):
                examples.append({'key': k, 'id': r['id'], 'before_join': t[max(0, sh - 160):sh] + ' ⟦ROW⟧ ' + t[sh:sh + 120],
                                 'after_join': t[sh + len(r['text']) - 120:sh + len(r['text'])] + ' ⟦/ROW⟧ ' + t[sh + len(r['text']):sh + len(r['text']) + 160],
                                 'added_regions': [g for g in out['regions'] if g['end'] <= sh or g['start'] >= sh + len(r['text'])]})
            f.write(json.dumps(out, ensure_ascii=False) + '\n')

    def q(v, p):
        if not v: return None
        v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))]
    rep = {'module': 'paired', 'formats': list(FORMATS), 'checks': dict(checks), 'per_key': {}, 'examples': examples}
    tot = Counter()
    for k, st in sorted(stats.items()):
        rep['per_key'][k] = {'attempted': st['attempted'], 'extended': st['extended'], 'none': dict(st['none']),
                             'extended_rate': round(st['extended'] / st['attempted'], 4),
                             'tokens_before_median_all': q(st['tok_before_all'], .5), 'tokens_before_p90_all': q(st['tok_before_all'], .9),
                             'tokens_before_median': q(st['tok_before'], .5), 'tokens_before_p90': q(st['tok_before'], .9),
                             'tokens_after_median': q(st['tok_after'], .5), 'tokens_after_p90': q(st['tok_after'], .9),
                             'tokens_after_p10': q(st['tok_after'], .1)}
        tot['attempted'] += st['attempted']; tot['extended'] += st['extended']
    rep['total'] = dict(tot)
    (OUT / 'report.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: {x: v[x] for x in ('attempted', 'extended', 'none', 'tokens_before_median', 'tokens_after_median', 'tokens_after_p90')}
                      for k, v in rep['per_key'].items()}, indent=1))
    print('checks', dict(checks), 'total', dict(tot))
    for e in examples: print('-----', e['key'], e['id'], '\n', repr(e['before_join']), '\n', repr(e['after_join']))


if __name__ == '__main__':
    {'build': build, 'run': run}[sys.argv[1]]()
