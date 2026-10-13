"""Long-context extension, `pools` family: mix rows that are crops of a longer text that already exists in a pool row or a
generation file. See SPEC.md in this folder for the contract.

Keys and sources (the row id encodes the source document and the character offsets of every crop applied to it):
  mirrors      generated_doc       pool-mirrors.jsonl.gz     id '<pool id>@<cs>[@<cs2>]'      whole doc label (1)
  human        generic_human       pool-human.jsonl.gz       id '<pool id>@<cs>[@<cs2>]'      whole doc label (0)
  fullpapers   generated_markdown  pool-fullpapers.jsonl.gz  id '<pool id>@<cs>[@<cs2>]'      whole doc label (1)
  claude_fp    generated_markdown  research/claude-fullpapers-20261006/claude-fullpapers-v1-train.jsonl.gz
                                   id '<paper id>/w<a>[@<cs>...]'  the document's own regions (all 1)
  sections_ai, human_section (paired_pdf_v1, clean_v2, archive_v1)
                                   research/claude-sections-20261006/claude-sections-v1-train.jsonl.gz
                                   id '<section row id>@<cs>[@<cs2>]'  the section row's own regions (human context 0, AI section 1)
  gradtex      gradtex             whole documents (document labels only): extend() returns None.

The pool files are the T2 source pools that final-20261007/mix_build.py reads (--sources default
/data/workspace/backbone-launch-20261003/source-pools, mix_build.py:258, pool_rows() mix_build.py:324-325), i.e. bucket
workspace/backbone-launch-20261003/source-pools/pool-{mirrors,human,fullpapers}.jsonl.gz.

How each id was produced (reproduced here only to locate the row text; the text itself is checked verbatim):
  windows()          final-20261007/mix_build.py:128-145   id f"{doc['id']}@{cs}", source_start=cs into the pool doc text
  crop510()          final-20261007/mix_build.py:65-82     appends f"@{cs}" with cs relative to the text it cropped (screen_one,
                     mix_build.py:85-95, re-crops rows that exceed 510 tokens under any backbone tokenizer)
  claude_fp windows  final-20261007/mix_pools.py:87-100    id f"{r['id']}/w{a}", window starts at char a of the full paper
  section rows       final-20261007/mix_pools.py:62-68     whole section rows (source_start=0), then crop510 adds '@cs'
So the row text starts at (w-offset, if any) + sum of all '@' offsets in the source document text.

Usage:
  uv run --with transformers --with tokenizers python pools.py            # run on all family rows, write outputs
"""
import collections, gzip, json, os, re, statistics, sys
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
SCRATCH = Path(os.environ.get('SCRATCH') or '/SCRATCH-not-set')  # Claude Code scratch folder holding the intermediate files (outside the repo); set SCRATCH to it
LC = SCRATCH / 'longctx'
OUT = LC / 'pools'
POOL_DIR = OUT  # bucket pool files downloaded here

POOL_KEYS = {'mirrors': 'mirrors', 'human': 'human', 'fullpapers': 'fullpapers'}
SECTION_KEYS = ('sections_ai', 'human_section')
FAMILY_KEYS = ('mirrors', 'human', 'fullpapers', 'claude_fp', 'sections_ai', 'human_section', 'gradtex')
CLAUDE_FP = REPO / 'research/claude-fullpapers-20261006/claude-fullpapers-v1-train.jsonl.gz'
SECTIONS = REPO / 'research/claude-sections-20261006/claude-sections-v1-train.jsonl.gz'
SNAP_FRAC = 0.4  # give up at most this share of a side's added characters to end it at a paragraph (newline) boundary
SLICE = 12000    # characters of source tokenised on each side of the row (grown when a side needs more)


# ---------------------------------------------------------------- sources (cached)
def _regs(x, n, label):
    if not x: return [{'start': 0, 'end': n, 'label': int(label)}]
    return [{'start': int(g['start']), 'end': int(g['end']), 'label': int(g['label'])} for g in x]


@lru_cache(maxsize=None)
def _pool(name):
    """{doc id: (text, regions)} for a T2 source pool (mix_build.py windows() uses doc regions or the doc label)."""
    out = {}
    with gzip.open(POOL_DIR / f'pool-{name}.jsonl.gz', 'rt') as f:
        for line in f:
            d = json.loads(line)
            out[d['id']] = (d['text'], _regs(d.get('regions'), len(d['text']), d['label']))
    return out


@lru_cache(maxsize=None)
def _file(path):
    out = {}
    with gzip.open(path, 'rt') as f:
        for line in f:
            d = json.loads(line)
            out[d['id']] = (d['text'], _regs(d.get('regions'), len(d['text']), d.get('label', 2)))
    return out


def _locate(row):
    """-> (ctx_source, (text, regions), char offset of the row text in that text) or (None, reason, None)."""
    key = row.get('dataset'); rid = row['id'].split('#rep')[0]
    parts = rid.split('@'); base = parts[0]
    try: offs = [int(x) for x in parts[1:]]
    except ValueError: return None, 'unparsable_id', None
    if offs and int(row.get('source_start', -1)) != offs[-1]: return None, 'source_start_disagrees_with_id', None
    if key in POOL_KEYS:
        if not offs: return None, 'no_offset_in_id', None
        src = f'pool-{POOL_KEYS[key]}'; doc = _pool(POOL_KEYS[key]).get(base); start = sum(offs)
    elif key == 'claude_fp':
        m = re.fullmatch(r'(.+)/w(\d+)', base)
        if not m: return None, 'unparsable_id', None
        src = 'claude-fullpapers-v1-train'; doc = _file(str(CLAUDE_FP)).get(m.group(1)); start = int(m.group(2)) + sum(offs)
    elif key in SECTION_KEYS:
        src = 'claude-sections-v1-train'; doc = _file(str(SECTIONS)).get(base); start = sum(offs)
    else:
        return None, 'not_in_family', None
    if doc is None: return None, 'source_doc_not_found', None
    return src, doc, start


# ---------------------------------------------------------------- helpers
def _offsets(tok, text):
    return tok(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']


def _ntok(tok, text):
    return len(tok(text, add_special_tokens=False)['input_ids'])


def _clip(regions, a, b, shift):
    """regions clipped to [a, b) and moved by shift (new = old - a + shift)."""
    return [{'start': max(g['start'], a) - a + shift, 'end': min(g['end'], b) - a + shift, 'label': g['label']}
            for g in regions if min(g['end'], b) > max(g['start'], a)]


def _labels(regions, n):
    lab = [None] * n
    for g in regions:
        for i in range(max(0, g['start']), min(n, g['end'])): lab[i] = g['label']
    return lab


def _merge(regions):
    out = []
    for g in sorted(regions, key=lambda g: g['start']):
        if out and out[-1]['label'] == g['label'] and out[-1]['end'] == g['start']: out[-1]['end'] = g['end']
        else: out.append(dict(g))
    return out


def _snap_before(S, cs, o):
    """Move cs forward (never backward) so the added prefix starts at a paragraph/line start, else a word start."""
    if cs <= 0 or cs >= o: return max(cs, 0)
    limit = cs + int((o - cs) * SNAP_FRAC)
    nl = S.find('\n', cs, o)
    if nl != -1 and nl < limit:
        p = nl
        while p < o and S[p] == '\n': p += 1
        return p
    p = cs
    while p < o and not S[p - 1].isspace(): p += 1
    while p < o and S[p].isspace() and S[p] != '\n': p += 1
    return p


def _snap_after(S, ce, e):
    """Move ce backward (never forward) so the added suffix ends at a paragraph/line end, else a word end."""
    if ce >= len(S) or ce <= e: return min(ce, len(S))
    limit = ce - int((ce - e) * SNAP_FRAC)
    nl = S.rfind('\n', e, ce)
    if nl != -1 and nl >= limit:
        p = nl
        while p > e and S[p - 1] == '\n': p -= 1
        return p
    p = ce
    while p > e and not (S[p].isspace() or S[p - 1].isspace()): p -= 1
    while p > e and S[p - 1].isspace(): p -= 1
    return p


# ---------------------------------------------------------------- contract
def extend_with_reason(row, tok, max_tokens=1000):
    """-> (new row, None) or (None, reason)."""
    if row.get('dataset') == 'gradtex': return None, 'gradtex_whole_document'
    src, doc, o = _locate(row)
    if src is None: return None, doc
    S, sregs = doc; t = row['text']; e = o + len(t)
    if S[o:e] != t: return None, 'row_text_not_verbatim_in_source'
    # the row's own labels must equal the source labels over the same characters
    if _labels(row['regions'], len(t)) != _labels(_clip(sregs, o, e, 0), len(t)): return None, 'row_regions_differ_from_source'
    if o == 0 and e == len(S): return None, 'row_is_whole_source'
    n_row = _ntok(tok, t)
    if n_row >= max_tokens: return None, 'row_already_at_max'
    budget = max_tokens - n_row
    for attempt in range(8):
        L = SLICE
        while True:
            a0, b0 = max(0, o - L), min(len(S), e + L)
            off = [(a + a0, b + a0) for a, b in _offsets(tok, S[a0:b0])]
            pre = [x for x in off if x[1] <= o]; post = [x for x in off if x[0] >= e]
            if a0 > 0: pre = pre[1:]  # the first/last token of a cut slice may be partial
            if b0 < len(S): post = post[:-1]
            need_more = (a0 > 0 and len(pre) < budget) or (b0 < len(S) and len(post) < budget)
            if not need_more or L > 200000: break
            L *= 2
        kb = min(len(pre), budget // 2); ka = min(len(post), budget - kb); kb = min(len(pre), budget - ka)
        cs = pre[-kb][0] if kb else o; ce = post[ka - 1][1] if ka else e
        if kb and cs == pre[0][0] and a0 == 0: cs = 0
        cs = _snap_before(S, cs, o) if cs > 0 else 0
        ce = _snap_after(S, ce, e) if ce < len(S) else len(S)
        if cs == o and ce == e: return None, 'no_context_after_snapping'
        new_text = S[cs:ce]; n_new = _ntok(tok, new_text)
        if n_new <= max_tokens: break
        budget -= (n_new - max_tokens) + 2
        if budget <= 0: return None, 'cannot_fit_max_tokens'
    else:
        return None, 'cannot_fit_max_tokens'
    before = S[cs:o]; after = S[e:ce]; d = len(before)
    assert new_text[d:d + len(t)] == t and new_text == before + t + after
    regions = _merge(_clip(sregs, cs, o, 0) + [{**g, 'start': g['start'] + d, 'end': g['end'] + d} for g in row['regions']]
                     + _clip(sregs, e, ce, d + len(t)))
    new = dict(row)
    new.update(text=new_text, regions=regions, context_extended=True, ctx_before_chars=len(before), ctx_after_chars=len(after),
               ctx_source=src, ctx_n_tokens=n_new)
    for k in ('target_start', 'target_end'):
        if k in row and row[k] is not None: new[k] = int(row[k]) + d
    new['source_start'], new['source_end'] = cs, ce  # now absolute offsets in the ctx_source document text
    return new, None


def extend(row, tok, max_tokens=1000):
    return extend_with_reason(row, tok, max_tokens)[0]


# ---------------------------------------------------------------- validation run
def _validate(old, new):
    t = old['text']; d = new['ctx_before_chars']; n = len(new['text'])
    assert new['text'][d:d + len(t)] == t
    assert all(0 <= g['start'] < g['end'] <= n for g in new['regions'])
    lab_new = _labels(new['regions'], n)
    assert None not in lab_new, 'uncovered characters'
    lab_old = _labels(old['regions'], len(t))
    assert lab_new[d:d + len(t)] == lab_old, 'row labels changed'
    ai_old = ''.join(c for c, l in zip(t, lab_old) if l == 1); ai_new_core = ''.join(c for c, l in zip(new['text'][d:d + len(t)], lab_new[d:d + len(t)]) if l == 1)
    assert ai_old == ai_new_core


def _q(xs, p):
    if not xs: return None
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))]


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(LC / 'tokenizer'))
    rows = []
    for e in range(3):
        with gzip.open(LC / f'input/stage2-epoch{e}.jsonl.gz', 'rt') as f:
            rows += [r for r in map(json.loads, f) if r['dataset'] in FAMILY_KEYS]
    stats = collections.defaultdict(lambda: {'attempted': 0, 'extended': 0, 'none': collections.Counter(), 'before': [], 'after': [],
                                             'added_before_tokens_share': []})
    examples = []; OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT / 'extended-ids.jsonl.gz', 'wt') as g:
        for i, r in enumerate(rows):
            k = (r['dataset'], r.get('text_format')); s = stats[k]; s['attempted'] += 1
            nb = _ntok(tok, r['text']); s['before'].append(nb)
            new, why = extend_with_reason(r, tok)
            if new is None:
                s['none'][why] += 1; s['after'].append(nb); continue
            _validate(r, new)
            s['extended'] += 1; s['after'].append(new['ctx_n_tokens'])
            g.write(json.dumps(new, ensure_ascii=False) + '\n')
            if len(examples) < 5 and r['dataset'] not in {x[0] for x in examples} and new['ctx_before_chars'] and new['ctx_after_chars']:
                examples.append((r['dataset'], new))
            if i % 2000 == 0: print(f'{i}/{len(rows)}', file=sys.stderr, flush=True)
    rep = {}
    for k, s in sorted(stats.items()):
        rep[f'{k[0]}|{k[1]}'] = {'attempted': s['attempted'], 'extended': s['extended'], 'none': dict(s['none']),
                                 'tokens_before': {'median': statistics.median(s['before']), 'p90': _q(s['before'], .9), 'max': max(s['before'])},
                                 'tokens_after': {'median': statistics.median(s['after']), 'p90': _q(s['after'], .9), 'max': max(s['after'])}}
    tot = {'attempted': sum(v['attempted'] for v in rep.values()), 'extended': sum(v['extended'] for v in rep.values())}
    (OUT / 'report.json').write_text(json.dumps({'total': tot, 'per_key_format': rep, 'max_tokens': 1000,
                                                 'tokenizer': 'Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0'}, indent=1))
    print(json.dumps({'total': tot, 'per_key_format': rep}, indent=1))
    for key, new in examples:
        d = new['ctx_before_chars']; n = len(new['text']) - new['ctx_after_chars']; T = new['text']
        print(f'\n=== {key} {new["id"]} tokens={new["ctx_n_tokens"]} before={d} after={new["ctx_after_chars"]} src={new["ctx_source"]}')
        print('  ...' + repr(T[max(0, d - 150):d]) + ' ||ROW|| ' + repr(T[d:d + 150]) + '...')
        print('  ...' + repr(T[n - 150:n]) + ' ||END ROW|| ' + repr(T[n:n + 150]) + '...')


if __name__ == '__main__':
    main()
