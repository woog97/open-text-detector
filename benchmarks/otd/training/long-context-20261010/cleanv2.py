"""Long-context extension for the clean-v2 family (every mix row with text_format == 'clean_v2').

Keys: edit_claude, edit_deepseek, human_paper_orig (clean-v2 part), human_para_e, para_deepseek, para_e (paragraph rendering)
and sections_ai, human_section (section rendering). See SPEC.md in this folder for the contract.

Source: bucket workspace/datasets/corpus-clean-text-v2 (whole-paper clean v2 text, joined on pdf_sha256 == row['paper_id']).
Two renderings of a paper, each reproducing how the row's own text was built:
  paragraph rendering  '\\n\\n'.join(clean_paragraphs(text))  -- research/corpus-sampling-20261007/build_pool.py:43 (clean_paragraphs);
      host ids or:<sha16>/pNNN index this list (build_pool.py:63-68); hosted rows are prev+target+next joined by '\\n\\n'
      (build_pool.py:68, mix_build.py:244); human_paper_orig windows join host targets by '\\n\\n' (mix_build.py:201-246).
  section rendering    '\\n\\n'.join(front + every section's paragraphs) from build_sections.segment_archive
      (research/claude-sections-20261006/build_sections.py:114) with build_llm_edits.paragraphs = clean_paragraphs and the
      top_sections_v2 fallback, exactly as build_pool.one does (build_pool.py:57-72). Section rows are excerpt() output
      (build_sections.py:226-245: <=400 words of section paragraphs each side, no headings) joined by '\\n\\n' (build_sections.py:457).
The row's human edge text (leading/trailing label-0 run) is located verbatim in the rendering; context is the rendering's text
before/after it. Rows whose edges cannot be located exactly (or only ambiguously) return None. Added context is human paper text
(papers are <=2022), label 0.

CLI:  python cleanv2.py run   -> $SCRATCH/longctx/cleanv2/{papers.parquet, extended-ids.jsonl.gz, report.json}
"""
import gzip, json, os, re, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _d in ('research/corpus-sampling-20261007', 'research/llm-sentence-edits-20261005', 'research/claude-hosted-edits-20261006',
           'research/claude-sections-20261006'):
    sys.path.insert(0, str(REPO / _d))
SCRATCH = Path(os.environ.get('SCRATCH') or '/SCRATCH-not-set')  # Claude Code scratch folder holding the intermediate files (outside the repo); set SCRATCH to it
OUT = SCRATCH / 'longctx/cleanv2'
PAPERS_PARQUET = OUT / 'papers.parquet'
SECTION_KEYS = {'sections_ai', 'human_section'}
SOURCE_NAME = {'para': 'corpus-clean-text-v2:clean_paragraphs', 'sec': 'corpus-clean-text-v2:segment_archive'}

_papers = None
_render_cache = {}
_last_reason = None  # reason for the most recent None (for reporting; not part of the contract)


def _load_papers():
    global _papers
    if _papers is None:
        import pyarrow.parquet as pq
        _papers = {r['pdf_sha256']: r for r in pq.read_table(PAPERS_PARQUET, columns=['pdf_sha256', 'year', 'text']).to_pylist()}
    return _papers


def _render(sha, mode):
    """Paragraph list + joined text + paragraph start offsets for one paper in one rendering (cached)."""
    key = (sha, mode)
    if key in _render_cache: return _render_cache[key]
    if len(_render_cache) > 4000: _render_cache.clear()
    r = _load_papers().get(sha); out = None
    if r and r['text']:
        from build_pool import clean_paragraphs
        if mode == 'para':
            ps = clean_paragraphs(r['text'])
        else:
            import build_sections as bs, build_llm_edits as ble
            ble.paragraphs = clean_paragraphs  # as build_pool.one does (segment_archive imports it at call time)
            seg = bs.segment_archive(r['text'])
            if seg is None:
                orig = bs.top_sections; bs.top_sections = bs.top_sections_v2
                try: seg = bs.segment_archive(r['text'])
                finally: bs.top_sections = orig
            ps = (list(seg[0]) + [x for sec in seg[1] for x in sec['paragraphs']]) if seg else None
        if ps:
            starts, pos = [], 0
            for p in ps: starts.append(pos); pos += len(p) + 2
            out = (ps, '\n\n'.join(ps), starts)
    _render_cache[key] = out
    return out


def _find_all(hay, needle):
    out, i = [], hay.find(needle)
    while i >= 0: out.append(i); i = hay.find(needle, i + 1)
    return out


def _encode(tok, text):
    if hasattr(tok, 'encode_batch') and not hasattr(tok, 'convert_tokens_to_ids'):  # tokenizers.Tokenizer
        return tok.encode(text, add_special_tokens=False).offsets
    return tok(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']


def ntok(tok, text):
    return len(_encode(tok, text))


def _human_runs(row):
    """(lead_end, trail_start): extent of the leading/trailing label-0 runs (None when the edge is not human)."""
    t = row['text']; rg = sorted(row.get('regions') or [], key=lambda g: g['start'])
    if not rg or rg[0]['start'] != 0 or rg[-1]['end'] != len(t): return None, None
    lead = trail = None
    if rg[0]['label'] == 0:
        e = rg[0]['end']
        for g in rg[1:]:
            if g['label'] == 0 and g['start'] == e: e = g['end']
            else: break
        lead = e
    if rg[-1]['label'] == 0:
        s = rg[-1]['start']
        for g in reversed(rg[:-1]):
            if g['label'] == 0 and g['end'] == s: s = g['start']
            else: break
        trail = s
    return lead, trail


def _locate(row, mode):
    """Return (D, a, b, how) with D[:a] = text before the row, D[b:] = text after it; a/b None if that side is unavailable."""
    t = row['text']; R = _render(row['paper_id'], mode)
    if R is None: return None, 'no_paper_text' if row['paper_id'] not in _load_papers() else 'unsegmentable'
    ps, D, starts = R
    lead, trail = _human_runs(row)
    if lead is None and trail is None: return None, 'no_human_edge'
    host = None
    m = re.search(r'/p(\d+)$', str(row.get('host_id') or ''))
    if m and mode == 'para':
        i = int(m.group(1))
        if i >= len(ps): return None, 'host_index_out_of_range'
        host = (starts[i], starts[i] + len(ps[i]))
    if lead == len(t):  # all-human row: the whole text should sit contiguously in D
        occ = _find_all(D, t)
        if host: occ = [o for o in occ if o <= host[1] and o + len(t) >= host[0]]
        if len(occ) == 1: return (D, occ[0], occ[0] + len(t), 'contiguous'), None
        if len(occ) > 1: return None, 'ambiguous_match'
        # human_paper_orig windows may skip 1-2 non-host paragraphs between joined hosts (mix_build.py:207): anchor the first
        # and last paragraph and require every row paragraph to be a D paragraph, in order, in between.
        rp = t.split('\n\n')
        if len(rp) < 2: return None, 'row_not_in_paper'
        idx = {}
        for k, p in enumerate(ps): idx.setdefault(p, []).append(k)
        firsts = [k for k in idx.get(rp[0], [])]
        cands = []
        for f in firsts:
            k, ok = f, True
            for p in rp[1:]:
                nxt = [j for j in idx.get(p, []) if j > k]
                if not nxt or nxt[0] - k > 3: ok = False; break
                k = nxt[0]
            if ok: cands.append((f, k))
        if len(cands) != 1: return None, 'row_not_in_paper' if not cands else 'ambiguous_match'
        f, k = cands[0]
        return (D, starts[f], starts[k] + len(ps[k]), 'gapped'), None
    a = b = None; lead_end = trail_start = None
    if lead is not None:
        anc = t[:lead].rstrip()
        if not anc: lead = None
        else:
            occ = _find_all(D, anc)
            if host: occ = [o for o in occ if host[0] - 2 <= o + len(anc) <= host[1]]
            if len(occ) > 1: return None, 'ambiguous_match'
            if not occ: return None, 'lead_not_in_paper'
            a = occ[0]; lead_end = a + len(anc)
    if trail is not None:
        s = trail
        while s < len(t) and t[s].isspace(): s += 1
        anc = t[s:]
        if anc:
            occ = _find_all(D, anc)
            if host: occ = [o for o in occ if host[0] <= o <= host[1] + 2]
            if lead_end is not None: occ = [o for o in occ if o >= lead_end]
            if len(occ) > 1 and lead_end is not None: occ = occ[:1]  # nearest after the lead anchor
            if len(occ) > 1: return None, 'ambiguous_match'
            if not occ: return None, 'trail_not_in_paper'
            trail_start = occ[0]; b = occ[0] + len(anc)
    if lead_end is not None and trail_start is not None and not host:
        # without a host index, the replaced original between the anchors must be one plausible span (< 40k chars)
        if trail_start - lead_end > 40000: return None, 'anchors_too_far_apart'
    return (D, a, b, 'anchored'), None


def _host_ok(p):
    import build_hosted as bh
    return bool(bh.host_ok(p))


def extend(row, tok, max_tokens=1000):
    """See SPEC.md. Returns a new row with real clean-v2 context on both sides, or None."""
    global _last_reason
    _last_reason = None
    def none(reason):
        global _last_reason
        _last_reason = reason; return None
    if row.get('text_format') != 'clean_v2': return none('not_clean_v2')
    pap = _load_papers().get(row.get('paper_id'))
    if pap is None: return none('no_paper_text')
    if not pap.get('year') or int(pap['year']) > 2022: return none('paper_not_pre2023')
    mode = 'sec' if row.get('dataset') in SECTION_KEYS else 'para'
    loc, why = _locate(row, mode)
    if loc is None: return none(why)
    D, a, b, how = loc
    t = row['text']; T0 = ntok(tok, t); budget = max_tokens - T0
    if budget < 16: return none('row_already_at_max')
    B = D[:a] if a else ''; A = D[b:] if b is not None and b < len(D) else ''
    if not B.strip() and not A.strip(): return none('no_context_in_paper')
    win = budget * 12 + 200
    Bt = B[-win:]; Bo = _encode(tok, Bt) if Bt.strip() else []
    At = A[:win]; Ao = _encode(tok, At) if At.strip() else []
    nb, na = len(Bo), len(Ao)
    tb = min(nb, budget // 2); ta = min(na, budget - tb); tb = min(nb, budget - ta)

    def cut(tb, ta):
        before = after = ''
        if tb > 0:
            c = Bo[nb - tb][0]
            while c < len(Bt) and c > 0 and not Bt[c - 1].isspace(): c += 1  # snap forward to a word start
            if nb - tb == 0 and len(Bt) == len(B): c = 0  # whole remaining paper prefix
            before = Bt[c:].lstrip()
            if not before.strip(): before = ''
        if ta > 0:
            d = Ao[ta - 1][1]
            if not (ta == na and len(At) == len(A)):
                while d > 0 and d < len(At) and not At[d].isspace(): d -= 1  # snap back to a word end
            after = At[:d].rstrip()
            if not after.strip(): after = ''
        return before, after

    for _ in range(50):
        before, after = cut(tb, ta)
        new = before + t + after
        n = ntok(tok, new)
        if n <= max_tokens: break
        ex = n - max_tokens + 2
        if tb >= ta: tb = max(0, tb - ex)
        else: ta = max(0, ta - ex)
    else:
        return none('could_not_fit')
    if not before and not after: return none('no_context_in_paper')
    from build_llm_edits import never_train_hit
    for ctx in (before, after):
        if ctx and never_train_hit({'paper_id': '', 'id': '', 'text': ctx}): return none('context_never_train_hit')
    L = len(before)
    assert new[L:L + len(t)] == t
    out = dict(row)
    out['text'] = new
    rg = [{'start': g['start'] + L, 'end': g['end'] + L, 'label': g['label']} for g in row.get('regions') or []]
    if before: rg.insert(0, {'start': 0, 'end': L, 'label': 0})
    if after: rg.append({'start': L + len(t), 'end': len(new), 'label': 0})
    out['regions'] = rg
    for k in ('target_start', 'target_end'):
        if row.get(k) is not None: out[k] = row[k] + L
    out.update(context_extended=True, ctx_before_chars=L, ctx_after_chars=len(after), ctx_source=SOURCE_NAME[mode] + ':' + how)
    return out


# ---------------------------------------------------------------- runner (validation + outputs)

def _shard_subset(rows, corpus_dir):
    """Keep only the papers our rows need, from the 80 corpus shards, in one small parquet."""
    import glob, pyarrow as pa, pyarrow.parquet as pq
    need = {r['paper_id'] for r in rows}; keep = []
    for f in sorted(glob.glob(str(corpus_dir / '*.parquet'))):
        tb = pq.read_table(f, columns=['pdf_sha256', 'year', 'text'])
        m = pa.compute.is_in(tb['pdf_sha256'], value_set=pa.array(sorted(need)))
        keep.append(tb.filter(m))
    tb = pa.concat_tables(keep); pq.write_table(tb, PAPERS_PARQUET, compression='zstd')
    return tb.num_rows, len(need)


def _added_paragraph_stats(row, new, mode):
    """host_ok over the rendering paragraphs that the added context touches (char-weighted)."""
    R = _render(row['paper_id'], mode); ps = R[0]; acc = Counter()
    for ctx in (new['text'][:new['ctx_before_chars']], new['text'][len(new['text']) - new['ctx_after_chars']:] if new['ctx_after_chars'] else ''):
        for frag in (x for x in ctx.split('\n\n') if x.strip()):
            full = next((p for p in ps if frag in p), None)  # the paragraph this fragment comes from
            ok = _host_ok(full) if full is not None else None
            acc['chars'] += len(frag); acc['paras'] += 1
            if ok is False: acc['rej_chars'] += len(frag); acc['rej_paras'] += 1
    return acc


def run():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(SCRATCH / 'longctx/tokenizer'))
    rows = []
    for e in range(3):
        for l in gzip.open(SCRATCH / f'longctx/input/stage2-epoch{e}.jsonl.gz', 'rt'):
            r = json.loads(l)
            if r.get('text_format') == 'clean_v2': rows.append(r)
    if not PAPERS_PARQUET.exists():
        print('subset', _shard_subset(rows, OUT / 'corpus/data'), flush=True)
    st = defaultdict(lambda: {'attempted': 0, 'extended': 0, 'none': Counter(), 'how': Counter(), 'tok_before': [], 'tok_after': [],
                              'tok_before_extended': [], 'tok_after_extended': [], 'ctx_tok_before': [], 'ctx_tok_after': []})
    rej = defaultdict(Counter); checks = Counter(); examples = []
    with gzip.open(OUT / 'extended-ids.jsonl.gz', 'wt') as f:
        for n, r in enumerate(rows):
            k = r['dataset']; s = st[k]; s['attempted'] += 1
            t0 = ntok(tok, r['text']); s['tok_before'].append(t0)
            new = extend(r, tok)
            if new is None:
                s['none'][_last_reason] += 1; s['tok_after'].append(t0); continue
            s['extended'] += 1; s['how'][new['ctx_source']] += 1; s['tok_before_extended'].append(t0)
            t1 = ntok(tok, new['text']); s['tok_after'].append(t1); s['tok_after_extended'].append(t1)
            s['ctx_tok_before'].append(ntok(tok, new['text'][:new['ctx_before_chars']]))
            s['ctx_tok_after'].append(ntok(tok, new['text'][len(new['text']) - new['ctx_after_chars']:]) if new['ctx_after_chars'] else 0)
            L = new['ctx_before_chars']
            checks['verbatim_ok'] += new['text'][L:L + len(r['text'])] == r['text']
            checks['len_ok'] += t1 <= 1000
            rg = new['regions']; cover = sorted((g['start'], g['end']) for g in rg)
            checks['regions_ok'] += all(0 <= a < b <= len(new['text']) for a, b in cover) and cover[0][0] == 0 and cover[-1][1] == len(new['text']) \
                and all(cover[i][1] == cover[i + 1][0] for i in range(len(cover) - 1))
            ai_old = ''.join(r['text'][g['start']:g['end']] for g in r['regions'] if g['label'] == 1)
            ai_new = ''.join(new['text'][g['start']:g['end']] for g in rg if g['label'] == 1)
            checks['ai_chars_unchanged'] += ai_old == ai_new
            grp = 'ai_rows' if any(g['label'] == 1 for g in r['regions']) else 'human_rows'
            mode = 'sec' if k in SECTION_KEYS else 'para'
            rej[f'{grp}/{mode}'] += _added_paragraph_stats(r, new, mode); rej[grp] += _added_paragraph_stats(r, new, mode)
            f.write(json.dumps(new, ensure_ascii=False) + '\n')
            if len(examples) < 5 and k not in {e_['key'] for e_ in examples}:
                T = new['text']; e1 = L + len(r['text'])
                examples.append({'key': k, 'id': r['id'], 'join_before': (T[max(0, L - 120):L] + ' ||| ' + T[L:L + 120]) if L else None,
                                 'join_after': (T[e1 - 120:e1] + ' ||| ' + T[e1:e1 + 120]) if new['ctx_after_chars'] else None})
            if n % 2000 == 0: print(n, flush=True)
    q = lambda v, p: (sorted(v)[min(len(v) - 1, int(p * len(v)))] if v else None)
    report = {'family': 'cleanv2', 'text_format': 'clean_v2', 'rows': len(rows), 'checks': dict(checks), 'per_key': {}, 'host_ok_rejected_share': {}, 'examples': examples}
    tot = Counter()
    for k, s in sorted(st.items()):
        report['per_key'][k] = {'attempted': s['attempted'], 'extended': s['extended'], 'rate': round(s['extended'] / s['attempted'], 4),
                                'none_reasons': dict(s['none']), 'how': dict(s['how']),
                                'tokens_before_all': {'median': statistics.median(s['tok_before']), 'p90': q(s['tok_before'], .9)},
                                'tokens_before_extended': {'median': statistics.median(s['tok_before_extended']) if s['tok_before_extended'] else None, 'p90': q(s['tok_before_extended'], .9)},
                                'tokens_after_extended': {'median': statistics.median(s['tok_after_extended']) if s['tok_after_extended'] else None, 'p90': q(s['tok_after_extended'], .9)},
                                'added_tokens_before_median': statistics.median(s['ctx_tok_before']) if s['ctx_tok_before'] else None,
                                'added_tokens_after_median': statistics.median(s['ctx_tok_after']) if s['ctx_tok_after'] else None,
                                'both_sides_extended': sum(1 for x, y in zip(s['ctx_tok_before'], s['ctx_tok_after']) if x and y),
                                'tokens_after_all': {'median': statistics.median(s['tok_after']), 'p90': q(s['tok_after'], .9)}}
        tot['attempted'] += s['attempted']; tot['extended'] += s['extended']
    report['total'] = dict(tot)
    for g, c in sorted(rej.items()):
        report['host_ok_rejected_share'][g] = {'char_share': round(c['rej_chars'] / max(1, c['chars']), 4), 'para_share': round(c['rej_paras'] / max(1, c['paras']), 4),
                                               'added_chars': c['chars'], 'added_paragraph_fragments': c['paras']}
    (OUT / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != 'examples'}, indent=1))
    for e_ in examples: print(json.dumps(e_, ensure_ascii=False, indent=1))


if __name__ == '__main__' and sys.argv[1:] == ['run']:
    run()
