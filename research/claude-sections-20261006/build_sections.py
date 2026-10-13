"""Claude-written full sections inside human papers (dated 2022 or earlier), labeled exactly as AI spans.

Two item kinds, 50/50 per writer:
  draft   - the writer sees title, abstract, the neighbouring sections and the target section's heading, role,
            subsection headings and length, NOT its text, and writes a replacement section.
  rewrite - the writer sees the original human section and rewrites/polishes it, keeping its content.
Section roles (target mix): introduction 25%, related work 25%, discussion/conclusion 20%, method 15%,
experiments/results 15%. 15% of items are held out for evaluation (heldout-never-train.json, split=heldout); held-out
items come only from papers that are in no training data (paired-data validation/test papers outside the sweep-eval,
selection, calibration and space-suite-v1 sets, and archive papers never used anywhere).

Document representation matches the paired training data: body paragraphs from the Poppler extraction joined by blank
lines, without headings. The AI section replaces the human paragraphs of exactly one top-level section; regions label
that span 1 and everything else 0. Ingest also writes the unmodified human document as a matched negative.

Usage:
  build_sections.py segment ARCHIVE_PARQUET            # -> segments.jsonl.gz (all candidate papers, gitignored)
  build_sections.py prepare EXCLUDE_IDS_JSON SCRATCH   # items, batches, INSTRUCTIONS.md inputs, never-train list
  build_sections.py ingest SCRATCH [--writer W]        # works on partial outputs
"""
import argparse, os, gzip, hashlib, json, random, re, statistics, sys, time
from collections import Counter, defaultdict
from pathlib import Path
SCRATCH = Path(os.environ.get('SCRATCH') or '/SCRATCH-not-set')  # Claude Code scratch folder of the Oct 6 session that made the intermediate files (outside the repo); set SCRATCH to it

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAIRED = ROOT / 'research/data/paper-gap10000-v3-luna-20260930'
SUITE = ROOT / 'research/evaluation/space-suite-v1/package'
EDITS = ROOT / 'research/llm-sentence-edits-20261005'
sys.path.insert(0, str(EDITS))
QUOTA = {'sonnet': 500, 'opus': 500, 'haiku': 200}
CYCLE = ['sonnet', 'opus', 'haiku', 'sonnet', 'opus']
MODEL = {'sonnet': 'claude-sonnet-5-5', 'opus': 'claude-opus-5-5', 'haiku': 'claude-haiku-4-5'}
ROLE_MIX = {'introduction': .25, 'related_work': .25, 'discussion_conclusion': .20, 'method': .15, 'experiments_results': .15}
HELDOUT_FRAC = .15
ROLE_RE = [('introduction', r'^introduction\b'),
           ('related_work', r'related work|background|prior work|previous work|literature|preliminar'),
           ('discussion_conclusion', r'conclu|discussion|summary|future work|limitations|outlook'),
           ('experiments_results', r'experiment|evaluation|results|empirical|benchmark|case stud'),
           ('method', r'method|approach|model|framework|algorithm|formulation|architecture|proposed|technique|our |learning|inference|training')]
MIN_WORDS, MAX_WORDS = 180, 1800
MIN_WORDS_ROLE = {'discussion_conclusion': 100}  # conclusions are often a single paragraph


def role_of(title):
    t = title.lower().strip()
    for r, pat in ROLE_RE:
        if re.search(pat, t):
            return r
    return None


def words(t):
    return len(t.split())


# ---------- segmentation ----------

def top_sections(heads):
    """heads: ordered (pos, num_str, title). Keep the first consistent run 1, 2, 3, ... of top-level headings."""
    out, want = [], 1
    for pos, num, title in heads:
        if '.' in num.strip('.'):
            continue
        n = int(num.strip('.'))
        if re.search(r'under review|published (as|in)|reviewed on|conference paper|proceedings|preprint', title, re.I):
            continue  # running page headers following a page number
        if n == want and re.match(r'[A-Z]', title) and words(title) <= 10 and sum(c.isalpha() or c.isspace() for c in title) / len(title) > .8:
            out.append((pos, n, title)); want += 1
    return out


def sub_heads(heads):
    return [(pos, num, title) for pos, num, title in heads if '.' in num.strip('.') and re.match(r'[A-Z]', title) and words(title) <= 12]


def segment_paired(p, d):
    import paper_gap50 as g, paper_gap250_sources as srcmod
    from bs4 import BeautifulSoup
    g.OUT = d
    paras = srcmod.paragraphs(p)
    xml = BeautifulSoup((d / 'sources' / f"{p['paper_id']}.bbox.html").read_text(), 'xml')
    heads, prev = [], None
    for pn, page in enumerate(xml.find_all('page'), 1):
        for bn, block in enumerate(page.find_all('block')):
            lines = block.find_all('line')
            if not lines:
                continue
            t = g.normalize([' '.join(w.text for w in l.find_all('word')) for l in lines])
            m = re.match(r'^(\d{1,2}(?:\.\d{1,2})*)\.?\s+(\S.{1,80})$', t)
            if m and len(lines) <= 2 and not re.search(r'[.!?]$', t):
                heads.append(((pn, bn), m.group(1), m.group(2).strip()))
            elif prev and re.fullmatch(r'\d{1,2}(?:\.\d{1,2})*\.?', prev[1]) and len(lines) <= 2 and words(t) <= 12 and not re.search(r'[.!?]$', t):
                heads.append((prev[0], prev[1].strip('.'), t))
            prev = ((pn, bn), t)
    tops = top_sections(heads)
    if len(tops) < 4 or not re.match(r'introduction', tops[0][2], re.I):
        return None
    subs = sub_heads(heads)
    secs = []
    for i, (pos, n, title) in enumerate(tops):
        end = tops[i + 1][0] if i + 1 < len(tops) else (10 ** 6, 0)
        ps = [x['text'] for x in paras if pos < (x['page'], x['block']) < end]
        secs.append({'num': n, 'title': title, 'paragraphs': ps,
                     'subsections': [f'{num} {t}' for q, num, t in subs if pos < q < end and num.split('.')[0] == str(n)]})
    front = [x['text'] for x in paras if (x['page'], x['block']) < tops[0][0]]
    return front, secs


def prose_ok(t):
    return t[:1].isupper() and sum(c.isalpha() for c in t) / max(1, len(t)) > .7 and '@' not in t


def segment_archive(text):
    from build_llm_edits import paragraphs
    items = paragraphs(text)
    heads = []
    caps = lambda x: re.sub(r'\b([A-Z]) (?=[A-Z]{2,})', r'\1', x)  # ICLR small caps: "I NTRODUCTION" -> "INTRODUCTION"
    for i, t in enumerate(items):
        t2 = caps(t); m = re.match(r'^(\d{1,2}(?:\.\d{1,2})*)\.?\s+([A-Z][^.!?]{1,80})$', t2)
        if not m and re.fullmatch(r'\d{1,2}(?:\.\d{1,2})*\.?', t.strip()) and i + 1 < len(items):
            nxt = caps(items[i + 1])  # number and title extracted as separate fragments
            if words(nxt) <= 12 and re.match(r'[A-Z]', nxt) and not re.search(r'[.!?]$', nxt):
                m = re.match(r'^(\S+)\s+(.*)$', t.strip().rstrip('.') + ' ' + nxt)
                heads.append((i + 1, m.group(1), nxt.title() if nxt.isupper() else nxt)); continue
        if m and words(t2) <= 12:
            title = m.group(2).strip()
            heads.append((i, m.group(1), title.title() if title.isupper() else title))
    tops = top_sections(heads)
    if len(tops) < 4 or not re.match(r'introduction', tops[0][2], re.I):
        return None
    head_idx = {h[0] for h in heads}
    subs = sub_heads(heads)
    secs = []
    for k, (pos, n, title) in enumerate(tops):
        end = tops[k + 1][0] if k + 1 < len(tops) else len(items)
        ps = [items[j] for j in range(pos + 1, end) if j not in head_idx and words(items[j]) >= 12 and prose_ok(items[j])]
        secs.append({'num': n, 'title': title, 'paragraphs': ps,
                     'subsections': [f'{num} {t}' for q, num, t in subs if pos < q < end and num.split('.')[0] == str(n)]})
    front = [items[j] for j in range(tops[0][0]) if j not in head_idx and words(items[j]) >= 12]
    return front, secs


def segment(parquet):
    import pyarrow.parquet as pq
    sys.path.insert(0, str(ROOT / 'benchmarks/otd'))
    out, stats = [], Counter()
    ab = {}
    for l in open(PAIRED / 'passages.jsonl'):
        x = json.loads(l); ab.setdefault(x['paper_id'], x['abstract'])
    cache = HERE / 'segments.jsonl.gz'
    cached = [json.loads(l) for l in gzip.open(cache, 'rt')] if cache.exists() else []
    paired_done = [x for x in cached if x['source'] == 'paired']
    out.extend(paired_done)
    for p in ([] if paired_done else map(json.loads, open(PAIRED / 'papers.jsonl'))):
        try:
            r = segment_paired(p, PAIRED)
        except Exception:
            r = None
        stats[('paired', 'ok' if r else 'unsegmented')] += 1
        if r:
            out.append({'paper_id': p['paper_id'], 'source': 'paired', 'paired_split': p['split'], 'title': p['title'],
                        'abstract': ab.get(p['paper_id'], ''), 'conference': p['conference'].lower(), 'year': p['year'],
                        'front': r[0], 'sections': r[1]})
    t = pq.read_table(parquet, columns=['paper_id', 'forum_id', 'title', 'conference', 'year', 'text'])
    sys.path.insert(0, str(ROOT / 'research/claude-fullpapers-20261006'))
    from build_fullpapers import archive_abstract
    for r in t.to_pylist():
        if r['year'] > 2022:
            continue
        s = segment_archive(r['text'])
        stats[('archive', 'ok' if s else 'unsegmented')] += 1
        if s:
            out.append({'paper_id': r['paper_id'], 'forum_id': r['forum_id'], 'source': 'archive', 'paired_split': None,
                        'title': re.sub(r'\s+', ' ', r['title']).strip(), 'abstract': archive_abstract(r['text']) or '',
                        'conference': r['conference'], 'year': r['year'], 'front': s[0], 'sections': s[1]})
    with gzip.open(HERE / 'segments.jsonl.gz', 'wt') as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + '\n')
    print({f'{a}/{b}': v for (a, b), v in stats.items()}, len(out), 'segmented papers')



# ---------- prepare ----------

CTX_WORDS = 400          # human context kept on each side of the target section in the training excerpt
FULL_NEIGHBOUR = 1200    # neighbouring sections up to this length are shown in full, longer ones as an outline
C12 = ['sonnet', 'opus', 'sonnet', 'opus', 'haiku', 'sonnet', 'opus', 'sonnet', 'opus', 'haiku', 'sonnet', 'opus']  # 5:5:2


def sec_words(sec):
    return sum(words(x) for x in sec['paragraphs'])


def outline(sec):
    firsts = []
    for x in sec['paragraphs']:
        m = re.match(r'(.+?[.!?])(\s|$)', x)
        firsts.append((m.group(1) if m else x[:200]) + ' [...]')
    return firsts


def eligible_sections(p):
    out = []
    for i, sec in enumerate(p['sections']):
        r = role_of(sec['title']); w = sec_words(sec)
        if r and MIN_WORDS_ROLE.get(r, MIN_WORDS) <= w <= MAX_WORDS and sec['paragraphs'] and (len(sec['paragraphs']) >= 2 or r == 'discussion_conclusion'):
            from build_llm_edits import mathy
            if not mathy(' '.join(sec['paragraphs'])):
                out.append((i, r))
    return out


def used_archive_ids():
    ids = set()
    for f in ('llm-edits-pilot.jsonl.gz', 'llm-edits-v1.jsonl.gz', 'heldout-eval-v1.jsonl.gz'):
        if (EDITS / f).exists():
            ids |= {json.loads(l)['paper_id'] for l in gzip.open(EDITS / f, 'rt')}
    for f in ('candidates-claude-v1.jsonl', 'candidates-heldout.jsonl'):
        if (EDITS / f).exists():
            ids |= {json.loads(l)['paper_id'] for l in open(EDITS / f)}
    fp = ROOT / 'research/claude-fullpapers-20261006/seeds.jsonl'
    if fp.exists():
        ids |= {json.loads(l)['source_paper_id'] for l in open(fp)}
    return ids


def excerpt(p, si, replacement=None):
    """Paragraph list around section si: up to CTX_WORDS of human paragraphs before/after; returns (paras, t0, t1)."""
    allp = [(k, x) for k, sec in enumerate(p['sections']) for x in sec['paragraphs']]
    before = [x for k, x in allp if k < si]; after = [x for k, x in allp if k > si]
    if not before:
        before = list(p.get('front', []))
    pre, n = [], 0
    for x in reversed(before):
        if n >= CTX_WORDS:
            break
        pre.insert(0, x); n += words(x)
    post, n = [], 0
    for x in after:
        if n >= CTX_WORDS:
            break
        post.append(x); n += words(x)
    mid = replacement if replacement is not None else p['sections'][si]['paragraphs']
    return pre + mid + post, len(pre), len(pre) + len(mid)


def prepare(exclude_json, scratch, overlap_json=None):
    from build_llm_edits import never_train_hit
    rng = random.Random(20261006)
    segs = [json.loads(l) for l in gzip.open(HERE / 'segments.jsonl.gz', 'rt')]
    ex = set(json.loads(Path(exclude_json).read_text())['paper_ids'])
    suite = set()
    for f in SUITE.glob('*.jsonl.gz'):
        for l in gzip.open(f, 'rt'):
            x = json.loads(l)
            for k in ('paper_id', 'group_id', 'forum_id'):
                if x.get(k) not in (None, 'None', ''):
                    suite.add(str(x[k]).split('/')[0].split(':')[-1])
    used = used_archive_ids(); stats = Counter(); pools = {'train': [], 'heldout': []}
    for p in segs:
        ids = {p['paper_id'], str(p.get('forum_id'))}
        if ids & ex:
            stats['excluded_eval_selection_calibration'] += 1; continue
        if ids & suite:
            stats['excluded_space_suite'] += 1; continue
        if p['source'] == 'paired' and p['paired_split'] != 'train':
            split = 'heldout'
        elif p['source'] == 'archive' and p['paper_id'] not in used:
            split = 'heldout'
        else:
            split = 'train'
        if split == 'train' and never_train_hit({'paper_id': p['paper_id'], 'text': '\n\n'.join(x for sec in p['sections'] for x in sec['paragraphs'])}):
            stats['excluded_never_train'] += 1; continue
        for si, r in eligible_sections(p):
            pools[split].append({'paper': p, 'si': si, 'role': r, 'cell': (p['conference'], p['year'])})
    if overlap_json is None:  # pass 1: snippets of held-out candidate sections for the training-window check
        sn = {f"{c['paper']['paper_id']}#{c['si']}": [y for x in c['paper']['sections'][c['si']]['paragraphs'][:3]
              for n in [re.sub(r'\W+', '', x.lower())] for y in (n[:60], n[len(n) // 2:len(n) // 2 + 60]) if len(y) == 60] for c in pools['heldout']}
        (Path(scratch) / 'heldout-section-snippets.json').write_text(json.dumps(sn))
        print(dict(stats), {k: len(v) for k, v in pools.items()}, 'snippets written'); return
    bad = set(json.loads(Path(overlap_json).read_text()))
    pools['heldout'] = [c for c in pools['heldout'] if f"{c['paper']['paper_id']}#{c['si']}" not in bad]
    stats['heldout_sections_in_training_text'] = len(bad)
    total = sum(QUOTA.values())
    cap = {'train': 2, 'heldout': 3}  # items per paper; the held-out pool is small, so it may use up to 3 sections per paper
    avail = {sp: Counter(c['role'] for c in pools[sp]) for sp in pools}
    n_split = {'heldout': min(round(HELDOUT_FRAC * total), sum(avail['heldout'].values()))}; n_split['train'] = total - n_split['heldout']

    def role_needs(n, av):
        need = {r: round(f * n) for r, f in ROLE_MIX.items()}
        need[max(need, key=need.get)] += n - sum(need.values())
        for _ in range(10):  # roles short of candidates give their deficit to the others, proportionally to the mix
            short = {r: need[r] - av[r] for r in need if need[r] > av[r]}
            if not short:
                break
            for r in short:
                need[r] = av[r]
            spare = [r for r in need if need[r] < av[r]]; d = sum(short.values()); wsum = sum(ROLE_MIX[r] for r in spare)
            for r in spare:
                need[r] += round(d * ROLE_MIX[r] / wsum)
            need[spare[0]] += n - sum(need.values()) if spare else 0
        return need
    chosen = []; stats['role_targets'] = {}
    for split in ('heldout', 'train'):
        if split == 'train':
            n_split['train'] = total - sum(c['split'] == 'heldout' for c in chosen)
        per_paper = Counter(); needs = role_needs(n_split[split], avail[split]); stats['role_targets'][split] = needs
        for role, frac in ROLE_MIX.items():
            need = needs[role]
            cand = [c for c in pools[split] if c['role'] == role]; rng.shuffle(cand)
            bycell = defaultdict(list)
            for c in cand:
                bycell[c['cell']].append(c)
            alloc = Counter(); picked = []
            while len(picked) < need:
                open_ = [cl for cl in bycell if bycell[cl]]
                if not open_:
                    stats.setdefault('shortfalls', {})[f'{split}/{role}'] = need - len(picked); break
                lo = min(alloc[cl] for cl in open_)
                for cl in sorted([cl for cl in open_ if alloc[cl] == lo], key=lambda _: rng.random()):
                    if len(picked) >= need:
                        break
                    xs = bycell[cl]; xs.sort(key=lambda c: per_paper[c['paper']['paper_id']])  # spread over papers
                    c = xs.pop(0)
                    if per_paper[c['paper']['paper_id']] >= cap[split] or any(q['paper'] is c['paper'] and q['si'] == c['si'] for q in picked):
                        continue
                    picked.append(c); alloc[cl] += 1; per_paper[c['paper']['paper_id']] += 1
            for k, c in enumerate(picked):
                chosen.append({**c, 'split': split, 'kind': 'draft' if k % 2 == 0 else 'rewrite'})
    # writers: 5:5:2 cycle, separately for draft and rewrite items, each list interleaving held-out and train items
    for kind in ('draft', 'rewrite'):
        xs = [c for c in chosen if c['kind'] == kind]
        tr = [c for c in xs if c['split'] == 'train']; ho = [c for c in xs if c['split'] == 'heldout']
        rng.shuffle(tr); rng.shuffle(ho)
        order = sorted(tr + ho, key=lambda c: (tr.index(c) / len(tr)) if c['split'] == 'train' else (ho.index(c) / len(ho)))
        for i, c in enumerate(order):
            c['writer'] = C12[i % len(C12)]
    rng.shuffle(chosen)
    items = []
    for i, c in enumerate(sorted(chosen, key=lambda c: (c['writer'], rng.random()))):
        p, si = c['paper'], c['si']; sec = p['sections'][si]
        items.append({'item_id': f'claude-sec-{i + 1:04d}', 'paper_id': p['paper_id'], 'source': p['source'], 'si': si, 'role': c['role'],
                      'kind': c['kind'], 'writer': c['writer'], 'split': c['split'], 'conference': p['conference'], 'year': p['year'],
                      'heading': f"{sec['num']} {sec['title']}", 'target_words': sec_words(sec), 'target_paragraphs': len(sec['paragraphs'])})
    with open(HERE / 'items.jsonl', 'w') as f:
        for it in items:
            f.write(json.dumps(it) + '\n')
    bypid = {p['paper_id']: p for p in segs}
    scratch = Path(scratch); bd = scratch / 'batches'; bd.mkdir(parents=True, exist_ok=True)
    for w in QUOTA:
        xs = [it for it in items if it['writer'] == w]
        for b in range(0, len(xs), 8):
            with open(bd / f'{w}-{b // 8 + 1:03d}.jsonl', 'w') as f:
                for it in xs[b:b + 8]:
                    f.write(json.dumps(batch_record(it, bypid[it['paper_id']]), ensure_ascii=False) + '\n')
    hold = [it for it in items if it['split'] == 'heldout']
    paras = [x for it in hold for x in excerpt(bypid[it['paper_id']], it['si'])[0]] + [bypid[it['paper_id']]['abstract'] for it in hold]
    write_never_train(sorted({it['paper_id'] for it in hold} | {it['item_id'] for it in hold}), paras)
    cells = defaultdict(Counter)
    for it in items:
        cells[f"{it['conference']} {it['year']}"][it['split']] += 1
    out = {**stats, 'items': len(items), 'papers': len({it['paper_id'] for it in items}),
           'by_writer_kind': {w: dict(Counter(it['kind'] for it in items if it['writer'] == w)) for w in QUOTA},
           'by_writer_split': {w: dict(Counter(it['split'] for it in items if it['writer'] == w)) for w in QUOTA},
           'by_role_split': {r: dict(Counter(it['split'] for it in items if it['role'] == r)) for r in ROLE_MIX},
           'by_source_split': {s_: dict(Counter(it['split'] for it in items if it['source'] == s_)) for s_ in ('paired', 'archive')},
           'cells': {k: dict(v) for k, v in sorted(cells.items())},
           'target_words_median': statistics.median(it['target_words'] for it in items),
           'batch_files': {w: -(-sum(it['writer'] == w for it in items) // 8) for w in QUOTA}}
    (HERE / 'prepare-stats.json').write_text(json.dumps(out, indent=1)); print(json.dumps({k: v for k, v in out.items() if k != 'cells'}, indent=1))


def batch_record(it, p):
    si = it['si']; sec = p['sections'][si]
    def neighbour(k):
        if k < 0 or k >= len(p['sections']):
            return None
        s_ = p['sections'][k]
        full = sec_words(s_) <= FULL_NEIGHBOUR
        return {'heading': f"{s_['num']} {s_['title']}", 'form': 'full_text' if full else 'outline_first_sentences',
                'paragraphs': s_['paragraphs'] if full else outline(s_)}
    rec = {'item_id': it['item_id'], 'kind': it['kind'], 'role': it['role'], 'paper_title': p['title'], 'abstract': p['abstract'],
           'conference': p['conference'], 'year': p['year'],
           'paper_outline': [f"{s_['num']} {s_['title']} ({sec_words(s_)} words)" for s_ in p['sections']],
           'section_heading': f"{sec['num']} {sec['title']}", 'subsection_headings': sec['subsections'],
           'target_words': it['target_words'], 'target_paragraphs': it['target_paragraphs'],
           'previous_section': neighbour(si - 1), 'next_section': neighbour(si + 1)}
    if it['kind'] == 'rewrite':
        rec['original_section_paragraphs'] = sec['paragraphs']
    return rec


NEVER_TRAIN = HERE / 'heldout-never-train.json'


def write_never_train(ids, paras):
    old = json.loads(NEVER_TRAIN.read_text()) if NEVER_TRAIN.exists() else {'paper_ids': [], 'paragraph_sha256': [], 'paragraph_shingle_sha256_12': []}
    nh = lambda t: hashlib.sha256(re.sub(r'\W+', '', t.lower()).encode()).hexdigest()
    hs = {nh(x) for x in paras if x.strip()}
    sh = {hashlib.sha256(n[i:i + 60].encode()).hexdigest()[:12] for x in paras for n in [re.sub(r'\W+', '', x.lower())] for i in range(0, len(n) - 59, 10)}
    NEVER_TRAIN.write_text(json.dumps({
        'purpose': 'Held-out Claude section-replacement evaluation items (claude-sections-v1 split=heldout) and their source papers. Never train on these.',
        'updated_pdt': time.strftime('%Y-%m-%d %H:%M', time.localtime()),
        'check': 'build_llm_edits.never_train_hit(row) loads this file together with the other held-out lists.',
        'normalization': "re.sub(r'\\W+', '', text.lower())",
        'paper_ids': sorted(set(old['paper_ids']) | set(ids)), 'paragraph_sha256': sorted(set(old['paragraph_sha256']) | hs),
        'paragraph_shingle_sha256_12': sorted(set(old['paragraph_shingle_sha256_12']) | sh)}, separators=(',', ':')))


# ---------- ingest ----------

def ingest(scratch, writer=None):
    from build_llm_edits import never_train_hit, sim
    scratch = Path(scratch)
    items = {json.loads(l)['item_id']: json.loads(l) for f in ('items.jsonl', 'items-gap.jsonl', 'items-scale.jsonl', 'items-wide1.jsonl') if (HERE / f).exists() for l in open(HERE / f)}
    bypid = {}; want = {it['paper_id'] for it in items.values()}
    for seg in ('segments.jsonl.gz', 'segments-wide1.jsonl.gz'):
        if not (HERE / seg).exists(): continue
        for l in gzip.open(HERE / seg, 'rt'):
            x = json.loads(l)
            if x['paper_id'] in want:
                bypid[x['paper_id']] = x
    got = {}; rejects = Counter()
    for f in sorted((scratch / 'outputs').glob('*/*.jsonl')):
        for l in open(f):
            l = l.strip()
            if not l:
                continue
            try:
                x = json.loads(l)
            except ValueError:
                rejects['bad_json_line'] += 1; continue
            if x.get('item_id') in items and x['item_id'] not in got:
                got[x['item_id']] = x.get('section_text', '')
    ai_rows, human_rows, held_ai, seen_neg = [], [], {}, set()
    for iid, txt in got.items():
        it = items[iid]
        if writer and it['writer'] != writer:
            continue
        p = bypid[it['paper_id']]; orig = p['sections'][it['si']]['paragraphs']
        paras = [re.sub(r'\s+', ' ', x).strip() for x in re.split(r'\n\s*\n', txt or '') if x.strip()]
        w = sum(words(x) for x in paras)
        if not paras:
            rejects['empty'] += 1; continue
        if any(x.startswith('#') or '\\(' in x or '\\[' in x or '$' in x or '**' in x for x in paras):
            rejects['markdown_or_latex'] += 1; continue
        if not .5 <= w / max(1, it['target_words']) <= 1.8:
            rejects['length_out_of_range'] += 1; continue
        if it['kind'] == 'rewrite':
            s_ = sim(' '.join(orig), ' '.join(paras))
            if s_ > .9:
                rejects['rewrite_too_close'] += 1; continue
            if s_ < .12:
                rejects['rewrite_lost_content'] += 1; continue
        ex_ai, a0, a1 = excerpt(p, it['si'], paras); ex_h, h0, h1 = excerpt(p, it['si'])
        text = '\n\n'.join(ex_ai); s0 = len('\n\n'.join(ex_ai[:a0])) + (2 if a0 else 0); s1 = s0 + len('\n\n'.join(ex_ai[a0:a1]))
        regions = [r for r in ({'start': 0, 'end': s0, 'label': 0}, {'start': s0, 'end': s1, 'label': 1}, {'start': s1, 'end': len(text), 'label': 0}) if r['end'] > r['start']]
        base = {'paper_id': it['paper_id'], 'group': 'paper:' + it['paper_id'], 'item_id': iid, 'section_role': it['role'],
                'section_heading': it['heading'], 'item_kind': it['kind'], 'split': it['split'], 'source': it['source'],
                'conference': it['conference'], 'year': it['year']}
        row = {'id': f'{iid}/{it["kind"]}', **base, 'dataset': 'papers_section', 'kind': 'generated_section', 'text': text,
               'regions': regions, 'target_start': s0, 'target_end': s1, 'generator': MODEL[it['writer']], 'ai_words': w,
               'human_section_words': it['target_words']}
        if it['split'] == 'train' and never_train_hit(row):
            rejects['never_train_hit'] += 1; continue
        ai_rows.append(row)
        if it['split'] == 'heldout':
            held_ai[iid] = paras
        htext = '\n\n'.join(ex_h)
        hs0 = len('\n\n'.join(ex_h[:h0])) + (2 if h0 else 0)
        human_rows.append({'id': f'{iid}/human', **base, 'dataset': 'papers_section_human', 'kind': 'human_matched', 'text': htext,
                           'regions': [{'start': 0, 'end': len(htext), 'label': 0}], 'target_start': hs0,
                           'target_end': hs0 + len('\n\n'.join(ex_h[h0:h1])), 'matched_item_id': iid, 'generator': None})
    for split in ('train', 'heldout'):
        path = HERE / f'claude-sections-v1-{split}.jsonl.gz'
        old = [json.loads(l) for l in gzip.open(path, 'rt')] if path.exists() and writer else []
        mine = {iid for iid, it in items.items() if it['writer'] == writer} if writer else set()
        old = [r for r in old if r['item_id'] not in mine]
        new = [r for r in ai_rows + human_rows if r['split'] == split]
        with gzip.open(path, 'wt') as f:
            for r in old + new:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    if held_ai:
        write_never_train(list(held_ai), [x for v in held_ai.values() for x in v])
    rep = {'writer': writer or 'all', 'returned': len(got), 'accepted_ai_rows': len(ai_rows), 'matched_human_rows': len(human_rows),
           'heldout_ai': len(held_ai), 'rejects': dict(rejects), 'by_kind': dict(Counter(r['item_kind'] for r in ai_rows)),
           'by_role': dict(Counter(r['section_role'] for r in ai_rows))}
    print(json.dumps(rep, indent=1))


GAP_QUOTA = {'sonnet': 400, 'opus': 400, 'haiku': 150}
C19 = ['sonnet', 'opus', 'haiku', 'sonnet', 'opus', 'sonnet', 'opus', 'haiku', 'sonnet', 'opus', 'sonnet', 'opus', 'haiku',
       'sonnet', 'opus', 'sonnet', 'opus', 'sonnet', 'opus']  # 8:8:3


def prepare_gap(exclude_json, scratch, n_total=950):
    """950 NEW training items from sections not used in Job C; held-out/never-train papers excluded."""
    from build_llm_edits import never_train_hit
    rng = random.Random(20261007)
    segs = [json.loads(l) for l in gzip.open(HERE / 'segments.jsonl.gz', 'rt')]
    ex = set(json.loads(Path(exclude_json).read_text())['paper_ids'])
    suite = set()
    for f in SUITE.glob('*.jsonl.gz'):
        for l in gzip.open(f, 'rt'):
            x = json.loads(l)
            for k in ('paper_id', 'group_id', 'forum_id'):
                if x.get(k) not in (None, 'None', ''):
                    suite.add(str(x[k]).split('/')[0].split(':')[-1])
    used_items = [json.loads(l) for l in open(HERE / 'items.jsonl')]
    used = {(it['paper_id'], it['si']) for it in used_items}; per_paper = Counter(it['paper_id'] for it in used_items)
    held_papers = {it['paper_id'] for it in used_items if it['split'] == 'heldout'}
    used_arch = used_archive_ids(); stats = Counter(); pool = []
    for p in segs:
        ids = {p['paper_id'], str(p.get('forum_id'))}
        if ids & ex or ids & suite or p['paper_id'] in held_papers:
            stats['excluded_eval_suite_or_heldout_paper'] += 1; continue
        if (p['source'] == 'paired' and p['paired_split'] != 'train') or (p['source'] == 'archive' and p['paper_id'] not in used_arch):
            stats['excluded_reserved_for_heldout'] += 1; continue  # keep the held-out-eligible pool untouched
        if never_train_hit({'paper_id': p['paper_id'], 'text': '\n\n'.join(x for sec in p['sections'] for x in sec['paragraphs'])}):
            stats['excluded_never_train'] += 1; continue
        for si, r in eligible_sections(p):
            if (p['paper_id'], si) in used:
                continue
            pool.append({'paper': p, 'si': si, 'role': r, 'cell': (p['conference'], p['year'])})
    avail = Counter(c['role'] for c in pool); stats['available_by_role'] = dict(avail)
    needs = {r: round(f * n_total) for r, f in ROLE_MIX.items()}
    needs[max(needs, key=needs.get)] += n_total - sum(needs.values())
    for _ in range(10):
        short = {r: needs[r] - avail[r] for r in needs if needs[r] > avail[r]}
        if not short:
            break
        for r in short:
            needs[r] = avail[r]
        spare = [r for r in needs if needs[r] < avail[r]]; d = sum(short.values()); wsum = sum(ROLE_MIX[r] for r in spare)
        for r in spare:
            needs[r] += round(d * ROLE_MIX[r] / wsum)
        if spare:
            needs[spare[0]] += n_total - sum(needs.values())
    stats['role_targets'] = needs
    chosen = []
    for role in ROLE_MIX:
        cand = [c for c in pool if c['role'] == role]; rng.shuffle(cand)
        bycell = defaultdict(list)
        for c in cand:
            bycell[c['cell']].append(c)
        alloc = Counter(); picked = []
        while len(picked) < needs[role]:
            open_ = [cl for cl in bycell if bycell[cl]]
            if not open_:
                stats.setdefault('shortfalls', {})[role] = needs[role] - len(picked); break
            lo = min(alloc[cl] for cl in open_)
            for cl in sorted([cl for cl in open_ if alloc[cl] == lo], key=lambda _: rng.random()):
                if len(picked) >= needs[role]:
                    break
                xs = bycell[cl]; xs.sort(key=lambda c: per_paper[c['paper']['paper_id']]); c = xs.pop(0)
                if per_paper[c['paper']['paper_id']] >= 3:
                    continue
                picked.append(c); alloc[cl] += 1; per_paper[c['paper']['paper_id']] += 1
        for k, c in enumerate(picked):
            chosen.append({**c, 'split': 'train', 'kind': 'draft' if k % 2 == 0 else 'rewrite'})
    # fill any shortfall (per-paper caps) from the roles with spare candidates, in mix order
    taken = {(c['paper']['paper_id'], c['si']) for c in chosen}
    spare = sorted([c for c in pool if (c['paper']['paper_id'], c['si']) not in taken and per_paper[c['paper']['paper_id']] < 3],
                   key=lambda c: (-ROLE_MIX[c['role']], rng.random()))
    k = 0
    while len(chosen) < n_total and spare:
        c = spare.pop(0)
        if per_paper[c['paper']['paper_id']] >= 3:
            continue
        per_paper[c['paper']['paper_id']] += 1; chosen.append({**c, 'split': 'train', 'kind': 'draft' if k % 2 == 0 else 'rewrite'}); k += 1
    stats['filled_after_shortfall'] = k
    for kind in ('draft', 'rewrite'):
        xs = [c for c in chosen if c['kind'] == kind]; rng.shuffle(xs)
        for i, c in enumerate(xs):
            c['writer'] = C19[i % len(C19)]
    items = []
    for i, c in enumerate(sorted(chosen, key=lambda c: (c['writer'], rng.random()))):
        p, si = c['paper'], c['si']; sec = p['sections'][si]
        items.append({'item_id': f'claude-sec-gap-{i + 1:04d}', 'paper_id': p['paper_id'], 'source': p['source'], 'si': si, 'role': c['role'],
                      'kind': c['kind'], 'writer': c['writer'], 'split': 'train', 'conference': p['conference'], 'year': p['year'],
                      'heading': f"{sec['num']} {sec['title']}", 'target_words': sec_words(sec), 'target_paragraphs': len(sec['paragraphs'])})
    with open(HERE / 'items-gap.jsonl', 'w') as f:
        for it in items:
            f.write(json.dumps(it) + '\n')
    bypid = {p['paper_id']: p for p in segs}
    bd = Path(scratch) / 'batches-gap'; bd.mkdir(parents=True, exist_ok=True)
    for w in GAP_QUOTA:
        xs = [it for it in items if it['writer'] == w]
        for b in range(0, len(xs), 8):
            with open(bd / f'gap-{w}-{b // 8 + 1:03d}.jsonl', 'w') as f:
                for it in xs[b:b + 8]:
                    f.write(json.dumps(batch_record(it, bypid[it['paper_id']]), ensure_ascii=False) + '\n')
    cells = Counter(f"{it['conference']} {it['year']}" for it in items)
    out = {**stats, 'items': len(items), 'papers': len({it['paper_id'] for it in items}),
           'by_writer_kind': {w: dict(Counter(it['kind'] for it in items if it['writer'] == w)) for w in GAP_QUOTA},
           'by_role': dict(Counter(it['role'] for it in items)), 'by_source': dict(Counter(it['source'] for it in items)),
           'cells': dict(sorted(cells.items())), 'target_words_median': statistics.median(it['target_words'] for it in items),
           'batch_files': {w: -(-sum(it['writer'] == w for it in items) // 8) for w in GAP_QUOTA}}
    (HERE / 'prepare-gap-stats.json').write_text(json.dumps(out, indent=1)); print(json.dumps({k: v for k, v in out.items() if k != 'cells'}, indent=1))


# ---------- scale-up (Oct 6 afternoon): more segmented papers, wider roles, cap 3 per paper with distinct roles ----------

def top_sections_v2(heads):
    """Like top_sections, but the run must start at '1 Introduction', and for each number n it takes the LAST
    candidate before the first candidate numbered n+1 (figure/table labels such as '2 Adversarial Doc' usually
    precede the real heading in reading order)."""
    ok = lambda t: (re.match(r'[A-Z]', t) and words(t) <= 10 and sum(c.isalpha() or c.isspace() for c in t) / len(t) > .8
                    and not re.search(r'under review|published (as|in)|reviewed on|conference paper|proceedings|preprint', t, re.I))
    cand = [(k, pos, int(num.strip('.')), title) for k, (pos, num, title) in enumerate(heads) if '.' not in num.strip('.') and ok(title)]
    start = next((c for c in cand if c[2] == 1 and re.match(r'introduction', c[3], re.I)), None)
    if not start:
        return []
    out = [(start[1], 1, start[3])]; k0 = start[0]; n = 2
    while True:
        cur = [c for c in cand if c[0] > k0 and c[2] == n]
        if not cur:
            break
        nxt = [c for c in cand if c[0] > cur[0][0] and c[2] == n + 1]
        pick = [c for c in cur if not nxt or c[0] < nxt[0][0]][-1]
        out.append((pick[1], n, pick[3])); k0 = pick[0]; n += 1
    return out


def segment_more(parquet):
    """Re-segment papers that the first pass could not segment, with top_sections_v2; append to segments.jsonl.gz.
    Existing entries are never changed (items reference their section indices)."""
    global top_sections
    sys.path.insert(0, str(ROOT / 'benchmarks/otd'))
    old = [json.loads(l) for l in gzip.open(HERE / 'segments.jsonl.gz', 'rt')]
    have = {x['paper_id'] for x in old}; orig = top_sections; top_sections = top_sections_v2; new = []; stats = Counter()
    try:
        ab = {}
        for l in open(PAIRED / 'passages.jsonl'):
            x = json.loads(l); ab.setdefault(x['paper_id'], x['abstract'])
        for p in map(json.loads, open(PAIRED / 'papers.jsonl')):
            if p['paper_id'] in have:
                continue
            try:
                r = segment_paired(p, PAIRED)
            except Exception:
                r = None
            stats[('paired', bool(r))] += 1
            if r:
                new.append({'paper_id': p['paper_id'], 'source': 'paired', 'paired_split': p['split'], 'title': p['title'],
                            'abstract': ab.get(p['paper_id'], ''), 'conference': p['conference'].lower(), 'year': p['year'],
                            'front': r[0], 'sections': r[1], 'segmenter': 'v2'})
        import pyarrow.parquet as pq
        sys.path.insert(0, str(ROOT / 'research/claude-fullpapers-20261006'))
        from build_fullpapers import archive_abstract
        for r in pq.read_table(parquet, columns=['paper_id', 'forum_id', 'title', 'conference', 'year', 'text']).to_pylist():
            if r['year'] > 2022 or r['paper_id'] in have:
                continue
            sg = segment_archive(r['text']); stats[('archive', bool(sg))] += 1
            if sg:
                new.append({'paper_id': r['paper_id'], 'forum_id': r['forum_id'], 'source': 'archive', 'paired_split': None,
                            'title': re.sub(r'\s+', ' ', r['title']).strip(), 'abstract': archive_abstract(r['text']) or '',
                            'conference': r['conference'], 'year': r['year'], 'front': sg[0], 'sections': sg[1], 'segmenter': 'v2'})
    finally:
        top_sections = orig
    with gzip.open(HERE / 'segments.jsonl.gz', 'wt') as f:
        for x in old + new:
            f.write(json.dumps(x, ensure_ascii=False) + '\n')
    print({f'{a}/{"ok" if b else "fail"}': v for (a, b), v in stats.items()}, 'added', len(new), 'total', len(old) + len(new))


EXTRA_ROLES = [('method', r'analysis|theor|problem (setup|setting|statement|definition|formulation)|notation|setup|setting|optimization|main result|bound|convergence|extension|overview'),
               ('experiments_results', r'data|simulation|application|implementation|example|ablation|stud(y|ies)')]


def role_wide(title):
    r = role_of(title)
    if r:
        return r
    t = title.lower()
    if re.search(r'acknowledg|reference|appendix|broader impact|ethic|funding|supplementar', t):
        return None
    for rr, pat in EXTRA_ROLES:
        if re.search(pat, t):
            return rr
    return None


SCALE_QUOTA = {'sonnet': 8, 'opus': 8, 'haiku': 3}  # same 8:8:3 split as the gap batches


def prepare_scale(exclude_json, scratch, n_total=2470):
    from build_llm_edits import never_train_hit, mathy
    rng = random.Random(20261009)
    segs = [json.loads(l) for l in gzip.open(HERE / 'segments.jsonl.gz', 'rt')]
    ex = set(json.loads(Path(exclude_json).read_text())['paper_ids'])
    suite = set()
    for f in SUITE.glob('*.jsonl.gz'):
        for l in gzip.open(f, 'rt'):
            x = json.loads(l)
            for k in ('paper_id', 'group_id', 'forum_id'):
                if x.get(k) not in (None, 'None', ''):
                    suite.add(str(x[k]).split('/')[0].split(':')[-1])
    prev = [json.loads(l) for f in ('items.jsonl', 'items-gap.jsonl') for l in open(HERE / f)]
    used = {(it['paper_id'], it['si']) for it in prev}; per_paper = Counter(it['paper_id'] for it in prev)
    roles_used = defaultdict(set)
    for it in prev:
        roles_used[it['paper_id']].add(it['role'])
    held_papers = {it['paper_id'] for it in prev if it['split'] == 'heldout'}
    used_arch = used_archive_ids(); stats = Counter(); pool = []
    for p in segs:
        ids = {p['paper_id'], str(p.get('forum_id'))}
        if ids & ex or ids & suite or p['paper_id'] in held_papers:
            stats['excluded_eval_suite_or_heldout_paper'] += 1; continue
        if (p['source'] == 'paired' and p['paired_split'] != 'train') or (p['source'] == 'archive' and p['paper_id'] not in used_arch):
            stats['excluded_reserved_for_heldout'] += 1; continue
        if never_train_hit({'paper_id': p['paper_id'], 'text': '\n\n'.join(x for sec in p['sections'] for x in sec['paragraphs'])}):
            stats['excluded_never_train'] += 1; continue
        for si, sec in enumerate(p['sections']):
            r = role_wide(sec['title']); w = sec_words(sec)
            if (p['paper_id'], si) in used or not r or not sec['paragraphs'] or not 120 <= w <= 2600 or mathy(' '.join(sec['paragraphs'])):
                continue
            pool.append({'paper': p, 'si': si, 'role': r, 'cell': (p['conference'], p['year'])})
    avail = Counter(c['role'] for c in pool); stats['available_by_role'] = dict(avail)
    # role-weighted greedy with cap 3 per paper (all jobs) and distinct roles per paper
    needs = {r: round(f * n_total) for r, f in ROLE_MIX.items()}
    chosen = []; taken = set()

    def can(c):
        pid = c['paper']['paper_id']
        return per_paper[pid] < 3 and c['role'] not in roles_used[pid] and (pid, c['si']) not in taken

    def take(c):
        pid = c['paper']['paper_id']; per_paper[pid] += 1; roles_used[pid].add(c['role']); taken.add((pid, c['si'])); chosen.append(c)
    for role in sorted(ROLE_MIX, key=lambda r: avail[r]):  # scarce roles first so they are not crowded out by the cap
        cand = [c for c in pool if c['role'] == role]; rng.shuffle(cand)
        bycell = defaultdict(list)
        for c in cand:
            bycell[c['cell']].append(c)
        got = 0
        while got < needs[role]:
            progressed = False
            for cl in sorted(bycell, key=lambda _: rng.random()):
                while bycell[cl] and not can(bycell[cl][0]):
                    bycell[cl].pop(0)
                if bycell[cl] and got < needs[role]:
                    take(bycell[cl].pop(0)); got += 1; progressed = True
            if not progressed:
                break
    spare = [c for c in pool if can(c)]; rng.shuffle(spare); fill = 0
    for c in spare:
        if len(chosen) >= n_total:
            break
        if can(c):
            take(c); fill += 1
    stats['filled_from_other_roles'] = fill
    rng.shuffle(chosen)
    for k, c in enumerate(chosen):
        c['kind'] = 'draft' if k % 2 == 0 else 'rewrite'
    cyc = [w for w, n in SCALE_QUOTA.items() for _ in range(n)]
    for kind in ('draft', 'rewrite'):
        xs = [c for c in chosen if c['kind'] == kind]
        for i, c in enumerate(xs):
            c['writer'] = cyc[(i * 7) % len(cyc)]
    items = []
    for i, c in enumerate(sorted(chosen, key=lambda c: (c['writer'], rng.random()))):
        p, si = c['paper'], c['si']; sec = p['sections'][si]
        items.append({'item_id': f'claude-sec-scale-{i + 1:04d}', 'paper_id': p['paper_id'], 'source': p['source'], 'si': si, 'role': c['role'],
                      'kind': c['kind'], 'writer': c['writer'], 'split': 'train', 'conference': p['conference'], 'year': p['year'],
                      'heading': f"{sec['num']} {sec['title']}", 'target_words': sec_words(sec), 'target_paragraphs': len(sec['paragraphs'])})
    with open(HERE / 'items-scale.jsonl', 'w') as f:
        for it in items:
            f.write(json.dumps(it) + '\n')
    bypid = {p['paper_id']: p for p in segs}
    bd = Path(scratch) / 'batches-scale'; bd.mkdir(parents=True, exist_ok=True)
    for w in SCALE_QUOTA:
        xs = [it for it in items if it['writer'] == w]
        for b in range(0, len(xs), 8):
            with open(bd / f'scale-{w}-{b // 8 + 1:03d}.jsonl', 'w') as f:
                for it in xs[b:b + 8]:
                    f.write(json.dumps(batch_record(it, bypid[it['paper_id']]), ensure_ascii=False) + '\n')
    out = {**stats, 'items': len(items), 'target_assigned': n_total, 'papers': len({it['paper_id'] for it in items}),
           'by_writer_kind': {w: dict(Counter(it['kind'] for it in items if it['writer'] == w)) for w in SCALE_QUOTA},
           'by_role': dict(Counter(it['role'] for it in items)), 'role_mix_realized': {r: round(v / max(1, len(items)), 3) for r, v in Counter(it['role'] for it in items).items()},
           'by_source': dict(Counter(it['source'] for it in items)), 'from_v2_segmenter': sum(1 for it in items if bypid[it['paper_id']].get('segmenter') == 'v2'),
           'cells': dict(sorted(Counter(f"{it['conference']} {it['year']}" for it in items).items())),
           'target_words_median': statistics.median(it['target_words'] for it in items) if items else None,
           'batch_files': {w: -(-sum(it['writer'] == w for it in items) // 8) for w in SCALE_QUOTA}}
    (HERE / 'prepare-scale-stats.json').write_text(json.dumps(out, indent=1)); print(json.dumps({k: v for k, v in out.items() if k != 'cells'}, indent=1))


MODEL['fable'] = 'claude-fable-5-1'
FABLE_SCR = (SCRATCH / 'claude-fable')


def carve_fable(scratch, per_writer):
    path = HERE / 'items-scale.jsonl'; items = [json.loads(l) for l in open(path)]
    if any(it['writer'] == 'fable' for it in items):
        raise SystemExit('already carved')
    for w in ('sonnet', 'opus'):
        for it in [it for it in items if it['writer'] == w][-per_writer:]:
            it['writer'] = 'fable'; it['carved_from'] = w
    with open(path, 'w') as f:
        for it in items:
            f.write(json.dumps(it) + '\n')
    segs = {}
    need = {it['paper_id'] for it in items}
    for l in gzip.open(HERE / 'segments.jsonl.gz', 'rt'):
        x = json.loads(l)
        if x['paper_id'] in need:
            segs[x['paper_id']] = x
    bd = Path(scratch) / 'batches-scale'
    for f in bd.glob('scale-*.jsonl'):
        f.unlink()
    for w in ('sonnet', 'opus', 'haiku', 'fable'):
        xs = [it for it in items if it['writer'] == w]
        out_dir, pref = (FABLE_SCR / 'scale', 'fable-sec') if w == 'fable' else (bd, f'scale-{w}')
        out_dir.mkdir(parents=True, exist_ok=True)
        for b in range(0, len(xs), 8):
            with open(out_dir / f'{pref}-{b // 8 + 1:03d}.jsonl', 'w') as f:
                for it in xs[b:b + 8]:
                    f.write(json.dumps(batch_record(it, segs[it['paper_id']]), ensure_ascii=False) + '\n')
    print(json.dumps({'by_writer': dict(Counter(it['writer'] for it in items)), 'fable_kinds': dict(Counter(it['kind'] for it in items if it['writer'] == 'fable')),
                      'fable_roles': dict(Counter(it['role'] for it in items if it['writer'] == 'fable'))}))

# ---------- Wide pool (Oct 7): sections sampled from the corpus clean-v2 text (research/corpus-sampling-20261007) ----------
def prepare_wide(sample_jsonl, scratch, writers='opus,sonnet'):
    """sample.py sections output -> items-wide1.jsonl, segments-wide1.jsonl.gz and batches of 8 (scratch/batches-wide1).
    One section per paper (the sampler caps it); writers alternate; kinds alternate draft/rewrite per writer. All train."""
    ws = writers.split(','); n = Counter(); items = []; segs = {}
    for k, x in enumerate(map(json.loads, open(sample_jsonl))):
        si, role = x['target_section']; w = ws[k % len(ws)]; sec = x['sections'][si]
        seg = {kk: v for kk, v in x.items() if kk != 'target_section'}; segs[x['paper_id']] = seg
        items.append({'item_id': f'claude-sec-wide1-{k + 1:05d}', 'paper_id': x['paper_id'], 'source': 'corpus', 'si': si, 'role': role,
                      'kind': 'draft' if n[w] % 2 == 0 else 'rewrite', 'writer': w, 'split': 'train', 'conference': x['conference'], 'year': x['year'],
                      'heading': f"{sec['num']} {sec['title']}", 'target_words': sec_words(sec), 'target_paragraphs': len(sec['paragraphs'])}); n[w] += 1
    with open(HERE / 'items-wide1.jsonl', 'w') as f:
        for it in items: f.write(json.dumps(it, ensure_ascii=False) + '\n')
    with gzip.open(HERE / 'segments-wide1.jsonl.gz', 'wt') as f:
        for seg in segs.values(): f.write(json.dumps(seg, ensure_ascii=False) + '\n')
    bd = Path(scratch) / 'batches-wide1'; bd.mkdir(parents=True, exist_ok=True)
    for w in ws:
        xs = [it for it in items if it['writer'] == w]
        for b in range(0, len(xs), 8):
            with open(bd / f'wide1-{w}-{b // 8 + 1:03d}.jsonl', 'w') as f:
                for it in xs[b:b + 8]: f.write(json.dumps(batch_record(it, segs[it['paper_id']]), ensure_ascii=False) + '\n')
    out = {'items': len(items), 'papers': len(segs), 'by_writer': dict(Counter(it['writer'] for it in items)),
           'by_role': dict(Counter(it['role'] for it in items)), 'by_kind': dict(Counter(it['kind'] for it in items)),
           'batches': {w: -(-sum(it['writer'] == w for it in items) // 8) for w in ws}}
    (HERE / 'prepare-wide1-stats.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    p0 = sub.add_parser('segment'); p0.add_argument('parquet')
    p1 = sub.add_parser('prepare'); p1.add_argument('exclude'); p1.add_argument('scratch'); p1.add_argument('--overlap', default=None)
    p2 = sub.add_parser('ingest'); p2.add_argument('scratch'); p2.add_argument('--writer', choices=list(QUOTA) + ['fable'])
    p7 = sub.add_parser('prepare-wide'); p7.add_argument('sample'); p7.add_argument('scratch'); p7.add_argument('--writers', default='opus,sonnet')
    p6 = sub.add_parser('carve-fable'); p6.add_argument('scratch'); p6.add_argument('--per-writer', type=int, default=60)
    p3 = sub.add_parser('prepare-gap'); p3.add_argument('exclude'); p3.add_argument('scratch')
    p4 = sub.add_parser('segment-more'); p4.add_argument('parquet')
    p5 = sub.add_parser('prepare-scale'); p5.add_argument('exclude'); p5.add_argument('scratch')
    a = ap.parse_args()
    if a.cmd == 'segment':
        segment(a.parquet)
    elif a.cmd == 'prepare':
        prepare(a.exclude, a.scratch, a.overlap)
    elif a.cmd == 'prepare-gap':
        prepare_gap(a.exclude, a.scratch)
    elif a.cmd == 'prepare-wide':
        prepare_wide(a.sample, a.scratch, a.writers)
    elif a.cmd == 'carve-fable':
        carve_fable(a.scratch, a.per_writer)
    elif a.cmd == 'segment-more':
        segment_more(a.parquet)
    elif a.cmd == 'prepare-scale':
        prepare_scale(a.exclude, a.scratch)
    else:
        ingest(a.scratch, a.writer)
