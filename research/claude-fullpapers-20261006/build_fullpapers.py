"""Claude-written full research manuscripts from human title+abstract seeds (Claude counterpart of
open-text-detector/synthetic-research-papers-600).

Seeds: title+abstract pairs from papers dated 2022 or earlier, stratified evenly over (conference, year) cells:
  - paired-data TRAIN-split papers (ACL/ICML/NeurIPS 2013-2021; abstracts from research/data/paper-gap10000-v3-luna-20260930)
  - pre-2022 archive papers (open-text-detector/paper-text; ICLR 2020-22, CoRL 2021-22, NeurIPS 2022, TMLR 2022;
    NeurIPS 2021 archive papers are excluded because that venue/year overlaps the paired and evaluation papers)
Excluded: anything flagged by build_llm_edits.never_train_hit (held-out eval papers/paragraphs), sweep-eval /
selection / calibration papers, any paper id in the frozen space-suite-v1 package, and earlier full-paper seeds.

Usage:
  build_fullpapers.py prepare ARCHIVE_PARQUET EXCLUDE_IDS_JSON SCRATCH_DIR   # seeds + batches + INSTRUCTIONS.md
  build_fullpapers.py ingest SCRATCH_DIR [--writer sonnet|opus|haiku]       # works on partial outputs
Ingest writes claude-fullpapers-v1.jsonl.gz (same columns as synthetic-research-papers-600, readable by
current-data-v1/prepare.py) and claude-fullpapers-v1-train.jsonl.gz (training rows, dataset 'fullpapers', body only).
"""
import argparse, gzip, hashlib, json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAIRED = ROOT / 'research/data/paper-gap10000-v3-luna-20260930'
SUITE = ROOT / 'research/evaluation/space-suite-v1/package'
OLD_SEEDS = ROOT / 'research/data/synthetic-papers-600-hf-20261003/papers.jsonl'
QUOTA = {'sonnet': 120, 'opus': 120, 'haiku': 60}
CYCLE = ['sonnet', 'opus', 'haiku', 'sonnet', 'opus']
MODEL = {'sonnet': 'claude-sonnet-5-5', 'opus': 'claude-opus-5-5', 'haiku': 'claude-haiku-4-5'}
sys.path.insert(0, str(ROOT / 'research/llm-sentence-edits-20261005'))


def clean_abstract(a):
    a = re.sub(r'\s+', ' ', a).strip()
    a = re.sub(r'(Under review|Published) as a conference paper at ICLR \d{4}|Published in Transactions on Machine Learning Research \(\d+/\d+\)', '', a)
    a = re.sub(r'(\w)- (?=[a-z])', r'\1', a)  # line-break hyphens
    a = re.sub(r'\s*Keywords?:.*$', '', a)
    return re.sub(r'\s+', ' ', a).strip()


def archive_abstract(text):
    txt = text[:12000]
    m = re.search(r'\n\s*A\s?BSTRACT\s*\n|\n\s*Abstract\s*\n|\nAbstract[.:—-]?\s', txt)
    if not m:
        return None
    rest = txt[m.end():]
    e = re.search(r'\n\s*(1\.?\s*\n?\s*)?(I\s?NTRODUCTION|Introduction)\b', rest)
    return clean_abstract(rest[:e.start()]) if e else None


def eligible_text(a):
    w = len(a.split())
    return 60 <= w <= 400 and '�' not in a


RECENT = {('iclr', 2026), ('neurips', 2025), ('icml', 2025), ('colm', 2025)}
HALF = 150
HELDOUT_FRAC = .15
FOOTER = re.compile(r'\d+\s*(st|nd|rd|th)\s+Conference on Neural Information Processing Systems.*$|Proceedings of the \d+\s*(st|nd|rd|th) International Conference on Machine Learning.*$|Published as a conference paper at (ICLR|COLM) \d{4}|Under review as a conference paper at ICLR \d{4}')


def clean_recent(a):
    a = FOOTER.sub('', a).strip()
    return re.sub(r'\s+\d{1,2}$', '', a).strip()


def bad_recent(a):
    return bool(re.search(r'Conference on|Copyright|Proceedings|@|[Ee]qual contribution|Correspond|Preprint', a))


def norm_title(t):
    return re.sub(r'\W+', '', t.lower())


def build_pool(parquet, exclude_json):
    import pyarrow.parquet as pq
    from build_llm_edits import never_train_hit
    ex = set(json.loads(Path(exclude_json).read_text())['paper_ids'])  # sweep-eval + selection + calibration paper ids
    suite = set()
    for f in SUITE.glob('*.jsonl.gz'):
        for l in gzip.open(f, 'rt'):
            x = json.loads(l)
            for k in ('paper_id', 'group_id', 'forum_id'):
                if x.get(k) not in (None, 'None', ''):
                    suite.add(str(x[k]).split('/')[0].split(':')[-1])
    olds = [json.loads(l) for l in open(OLD_SEEDS)]
    old_ids = {o['source_paper_id'].split(':')[-1] for o in olds}; old_titles = {norm_title(o['title']) for o in olds}
    stats = Counter(); pool = defaultdict(list)

    def consider(s):
        ids = {s['source_paper_id'], str(s.get('forum_id'))}
        if ids & ex:
            stats['excluded_eval_selection_calibration'] += 1
        elif ids & suite:
            stats['excluded_space_suite'] += 1
        elif ids & old_ids or norm_title(s['title']) in old_titles:
            stats['excluded_previous_codex_seed'] += 1
        elif never_train_hit({'paper_id': s['source_paper_id'], 'text': s['abstract']}):
            stats['excluded_never_train'] += 1
        elif not eligible_text(s['abstract']) or (s['half'] == 'recent' and bad_recent(s['abstract'])):
            stats['excluded_abstract_quality'] += 1
        else:
            pool[(s['half'], s['conference'].lower(), s['year'])].append(s)

    papers = {json.loads(l)['paper_id']: json.loads(l) for l in open(PAIRED / 'papers.jsonl')}
    ab = {}
    for l in open(PAIRED / 'passages.jsonl'):
        x = json.loads(l); ab.setdefault(x['paper_id'], x['abstract'])
    for pid, p in papers.items():
        if p['split'] == 'train' and pid in ab:
            consider({'source_paper_id': pid, 'title': p['title'], 'abstract': clean_abstract(ab[pid]), 'conference': p['conference'],
                      'year': p['year'], 'source_url': p.get('abstract_url') or p.get('pdf_url'), 'seed_source': 'paired_train',
                      'half': 'pre2023', 'seed_label': 'seed_human_gold'})
    t = pq.read_table(parquet, columns=['paper_id', 'forum_id', 'title', 'conference', 'year', 'text'])
    for r in t.to_pylist():
        k = (r['conference'], r['year'])
        if k in RECENT:
            half = 'recent'
        elif r['year'] <= 2022:
            half = 'pre2023'
        else:
            continue
        a = archive_abstract(r['text'])
        if not a:
            stats['archive_no_abstract'] += 1; continue
        if half == 'recent':
            a = clean_recent(a)
        consider({'source_paper_id': r['paper_id'], 'forum_id': r['forum_id'], 'title': re.sub(r'\s+', ' ', r['title']).strip(),
                  'abstract': a, 'conference': r['conference'], 'year': r['year'],
                  'source_url': f"https://openreview.net/forum?id={r['forum_id']}" if r['forum_id'] else None, 'seed_source': 'archive',
                  'half': half, 'seed_label': 'seed_human_gold' if half == 'pre2023' else 'seed_human_unverified'})
    return pool, stats


def prepare(parquet, exclude_json, scratch, abstract_overlap=None):
    """Pass 1 (no abstract_overlap): write abstract snippets for the training-window check. Pass 2: select seeds."""
    rng = random.Random(20261006)
    pool, stats = build_pool(parquet, exclude_json)
    scratch = Path(scratch)
    if abstract_overlap is None:
        sn = {s['source_paper_id']: [x for x in (lambda n: (n[:60], n[len(n) // 2:len(n) // 2 + 60]))(re.sub(r'\W+', '', s['abstract'].lower())) if len(x) == 60]
              for v in pool.values() for s in v}
        (scratch / 'abstract-snippets.json').write_text(json.dumps(sn)); print(len(sn), 'abstract snippets written'); return
    in_training = set(json.loads(Path(abstract_overlap).read_text()))
    seeds = []
    for half in ('pre2023', 'recent'):
        cells = sorted(c for c in pool if c[0] == half); alloc = {c: 0 for c in cells}
        while sum(alloc.values()) < HALF:
            open_ = [c for c in cells if alloc[c] < len(pool[c])]
            if not open_:
                raise SystemExit(f'not enough eligible seeds in {half}')
            lo = min(alloc[c] for c in open_)
            for c in sorted([c for c in open_ if alloc[c] == lo], key=lambda c: rng.random()):
                if sum(alloc.values()) < HALF:
                    alloc[c] += 1
        for c in cells:
            rng.shuffle(pool[c])
        order, k = [], 0
        while len(order) < HALF:
            for c in cells:
                if k < alloc[c]:
                    order.append(pool[c][k])
            k += 1
        n_hold = round(HELDOUT_FRAC * HALF)
        # held-out: systematic sample over the cell-sorted list (at most ~1 per cell in the small pre-2023 cells; even in
        # the four recent cells). A seed whose abstract already appears in training data is never held out, so the
        # never-train list cannot contradict existing training data.
        ranked = sorted(range(HALF), key=lambda i: ((order[i]['conference'].lower(), order[i]['year']), i))
        step = HALF / n_hold; held = set(); off = rng.random() * step
        for m in range(n_hold):
            j = int(off + m * step)
            while j < HALF and (order[ranked[j]]['source_paper_id'] in in_training or ranked[j] in held):
                j += 1
            if j < HALF:
                held.add(ranked[j])
        # writers by the 2:2:1 cycle, run separately over held-out and training seeds so both are writer-balanced
        k_ = {'heldout': 0, 'train': 0}
        for i, s in enumerate(order):
            sp = 'heldout' if i in held else 'train'
            seeds.append({**s, 'writer': CYCLE[k_[sp] % len(CYCLE)], 'split': sp}); k_[sp] += 1
    for i, s in enumerate(seeds):
        s['seed_id'] = f'claude-fp-{i + 1:03d}'
    with open(HERE / 'seeds.jsonl', 'w') as f:
        for s in seeds:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    # per-paper work dirs + launch lists
    work = scratch / 'work'; launch = scratch / 'launch'; work.mkdir(parents=True, exist_ok=True); launch.mkdir(exist_ok=True)
    for s in seeds:
        d = work / s['seed_id']; d.mkdir(exist_ok=True)
        (d / 'seed.json').write_text(json.dumps({'seed_id': s['seed_id'], 'title': s['title'], 'abstract': s['abstract'], 'conference': s['conference'],
                                                 'year': s['year'], 'writer_model': MODEL[s['writer']]}, ensure_ascii=False, indent=1))
    for w in QUOTA:
        (launch / f'{w}.txt').write_text('\n'.join(str(work / s['seed_id']) for s in seeds if s['writer'] == w) + '\n')
    # never-train list for held-out seeds: seed ids + abstract hashes now; generated-body hashes are added by ingest
    hold = [s for s in seeds if s['split'] == 'heldout']
    write_never_train([s['seed_id'] for s in hold], [s['abstract'] for s in hold], {})
    cellstats = {}
    for s in seeds:
        c = cellstats.setdefault(f"{s['half']} {s['conference'].lower()} {s['year']}", {'eligible': len(pool[(s['half'], s['conference'].lower(), s['year'])]), 'chosen': 0, 'heldout': 0, **{w: 0 for w in QUOTA}})
        c['chosen'] += 1; c[s['writer']] += 1; c['heldout'] += s['split'] == 'heldout'
    out = {**stats, 'seeds': len(seeds), 'by_half_writer': {h: dict(Counter(s['writer'] for s in seeds if s['half'] == h)) for h in ('pre2023', 'recent')},
           'heldout': {h: dict(Counter(s['writer'] for s in hold if s['half'] == h)) for h in ('pre2023', 'recent')},
           'by_source': dict(Counter(s['seed_source'] for s in seeds)), 'cells': dict(sorted(cellstats.items()))}
    (HERE / 'prepare-stats.json').write_text(json.dumps(out, indent=1)); print(json.dumps({k: v for k, v in out.items() if k != 'cells'}, indent=1))


NEVER_TRAIN = HERE / 'heldout-never-train.json'


def write_never_train(seed_ids, texts, extra_paragraphs):
    """Same format and check as research/llm-sentence-edits-20261005/heldout-eval-never-train.json."""
    import time
    old = json.loads(NEVER_TRAIN.read_text()) if NEVER_TRAIN.exists() else {'paper_ids': [], 'paragraph_sha256': [], 'paragraph_shingle_sha256_12': []}
    paras = list(texts) + [p for v in extra_paragraphs.values() for p in v]
    def nh(t):
        return hashlib.sha256(re.sub(r'\W+', '', t.lower()).encode()).hexdigest()
    hs = {nh(p) for p in paras if p.strip()}
    sh = {hashlib.sha256(n[i:i + 60].encode()).hexdigest()[:12] for p in paras for n in [re.sub(r'\W+', '', p.lower())] for i in range(0, len(n) - 59, 10)}
    NEVER_TRAIN.write_text(json.dumps({
        'purpose': 'Held-out Claude full-paper evaluation items (claude-fullpapers-v1 split=heldout). Never train on these.',
        'updated_pdt': time.strftime('%Y-%m-%d %H:%M', time.localtime()),
        'check': 'build_llm_edits.never_train_hit(row) loads this file together with heldout-eval-never-train.json.',
        'normalization': "re.sub(r'\\W+', '', text.lower())",
        'paper_ids': sorted(set(old['paper_ids']) | set(seed_ids)), 'paragraph_sha256': sorted(set(old['paragraph_sha256']) | hs),
        'paragraph_shingle_sha256_12': sorted(set(old['paragraph_shingle_sha256_12']) | sh)}, separators=(',', ':')))


# ---------- ingest ----------

def fmt_problems(md):
    p = []
    if re.search(r'(?<!\\)\$', md):
        p.append('dollar math')
    if md.count('\\(') != md.count('\\)') or md.count('\\[') != md.count('\\]'):
        p.append('unbalanced math delimiters')
    if md.count('```') % 2:
        p.append('unbalanced code fence')
    if not re.search(r'^## References', md, re.M):
        p.append('no References section')
    return p


def ingest(scratch, writer=None):
    from build_llm_edits import never_train_hit
    scratch = Path(scratch)
    seeds = {json.loads(l)['seed_id']: json.loads(l) for l in open(HERE / 'seeds.jsonl')}
    rows, train, rejects, held_paras = [], [], Counter(), {}
    for sid, s in seeds.items():
        if writer and s['writer'] != writer:
            continue
        d = scratch / 'work' / sid
        if not all((d / f).exists() for f in ('draft.md', 'review.json', 'paper.md', 'metadata.json')):
            rejects['not_finished'] += 1; continue
        rv = json.loads((d / 'review.json').read_text())
        if not rv.get('feedback') or not all('response' in f for f in rv['feedback']):
            rejects['revision_not_done'] += 1; continue
        md = (d / 'paper.md').read_text(); meta = json.loads((d / 'metadata.json').read_text())
        title_line = f"# {s['title']}"
        abs_md = meta.get('abstract_markdown') or s['abstract']
        if not md.startswith('# ') or '## Abstract' not in md:
            rejects['bad_header'] += 1; continue
        if abs_md not in md:
            rejects['abstract_not_verbatim'] += 1; continue
        words = len(md.split())
        if words < 2500:
            rejects['too_short'] += 1; continue
        probs = fmt_problems(md)
        if probs:
            rejects['format:' + probs[0]] += 1; continue
        start = md.index(abs_md) + len(abs_md); body = md[start:]
        if s['split'] == 'train' and never_train_hit({'text': body, 'seed_id': sid}):
            rejects['never_train_hit'] += 1; continue
        t_end = md.index('\n') if '\n' in md else len(md)
        a0 = md.index(abs_md)
        human_spans = [{'section': 'title', 'start': 2, 'end': t_end}, {'section': 'abstract', 'start': a0, 'end': start}]
        model = MODEL[s['writer']]
        row = {'paper_id': sid, 'seed_id': sid, 'title': s['title'], 'abstract': s['abstract'], 'rendered_title': md[2:t_end].strip(),
               'rendered_abstract': abs_md, 'paper_markdown': md, 'model': model, 'final_revision_model': model, 'writer_model': model,
               'models_used': [model], 'model_label_source': 'assigned_writer', 'mixed_writer_models': False,
               'revision_count': int(meta.get('revision_count', 1)), 'synthetic_manuscript': True, 'experiments_actually_run': False,
               'results_and_methods_may_be_invented': True, 'human_supplied_sections': ['title', 'abstract'],
               'human_spans': human_spans, 'seed_rendering_changes': meta.get('seed_rendering_changes', []),
               'source_paper_id': s['source_paper_id'], 'conference': s['conference'], 'year': s['year'], 'source_url': s.get('source_url'),
               'paper_sha256': hashlib.sha256(md.encode()).hexdigest(), 'word_count': words,
               'generation_history_json': json.dumps(meta.get('generation_history', [])), 'source_batch': 'claude-fullpapers-v1',
               'markdown_path': f"work/{sid}/paper.md", 'split': s['split'], 'seed_label': s['seed_label'], 'seed_half': s['half'],
               'reviewer_model': rv.get('reviewer_model'), 'review_feedback_items': len(rv['feedback'])}
        rows.append(row)
        if s['split'] == 'heldout':
            held_paras[sid] = [p for p in re.split(r'\n\s*\n', body) if len(p.split()) >= 8]
            continue
        g = 'paper:' + s['source_paper_id']
        train.append({'id': 'fullpaper-' + sid, 'paper_id': s['source_paper_id'], 'group': g, 'dataset': 'fullpapers', 'text': body,
                      'label': 1, 'model': model, 'excluded_prefix_chars': start, 'seed_id': sid, 'seed_label': s['seed_label'],
                      'regions': [{'start': 0, 'end': len(body), 'label': 1}]})
    keep_old = lambda path: [json.loads(l) for l in gzip.open(path, 'rt')] if path.exists() and writer else []
    for name, new, key in (('claude-fullpapers-v1.jsonl.gz', rows, 'model'), ('claude-fullpapers-v1-train.jsonl.gz', train, 'model')):
        path = HERE / name
        old = [r for r in keep_old(path) if r[key] != MODEL[writer]] if writer else []
        with gzip.open(path, 'wt') as f:
            for r in old + new:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    if held_paras:
        # Held-out manuscripts are blocked by seed id; their generated bodies are not shingled into the list, because shared
        # references, headings and notation made every other generated paper match them (Oct 6 fix).
        write_never_train(list(held_paras), [], {})
    rep = {'writer': writer or 'all', 'accepted': len(rows), 'train_rows': len(train), 'heldout_rows': len(held_paras), 'rejects': dict(rejects),
           'by_model': dict(Counter(r['model'] for r in rows)), 'median_words': sorted(r['word_count'] for r in rows)[len(rows) // 2] if rows else None}
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    p1 = sub.add_parser('prepare'); p1.add_argument('parquet'); p1.add_argument('exclude'); p1.add_argument('scratch')
    p1.add_argument('--abstract-overlap', default=None)
    p2 = sub.add_parser('ingest'); p2.add_argument('scratch'); p2.add_argument('--writer', choices=list(QUOTA))
    a = ap.parse_args()
    prepare(a.parquet, a.exclude, a.scratch, a.abstract_overlap) if a.cmd == 'prepare' else ingest(a.scratch, a.writer)
