"""Build a page-level review kit (images + texts + labels) for visual audits of omissions.

Selects worst, edge-case and standard pages from a scored prototype run and writes, per
page, a 130 dpi JPEG and a JSON record with the cleaned baseline text, the text with
omission markers, error spans, omitted spans, detected regions and MinerU blocks.
Run on the training Space.
"""
import collections, glob, json, os, random, sys, tarfile
from pathlib import Path

B = Path('/data/workspace/baseline-omissions-20261005')
sys.path.insert(0, str(B / 'code/backend')); sys.path.insert(0, str(B / 'vendor'))
sys.path.insert(0, '/data/workspace/datasets/iclr2027-round3-7000-20261003'); sys.path.insert(0, str(B / 'code/scripts'))
import omission_prototype as op
import omission_limitations as ol

RUN = os.environ.get('OMIT_RUN', 'prototype-v4'); OUT = B / os.environ.get('KIT_NAME', 'review-kit')


def paper_traits(p):
    c = collections.Counter({tuple(k.split('|')): v for k, v in p['counts'].items()})
    tot = lambda t: sum(v for (a, b), v in c.items() if a == t)
    return {'math': p['omissions'].get('math_display', 0) + p['omissions'].get('math_inline', 0),
            'table': tot('table'), 'numeric': p['omissions'].get('numeric', 0), 'outside': tot('none'),
            'body_font': p.get('body_font', '')}


def choose():
    papers = json.load(open(B / RUN / 'papers.json')); by = {p['pdf_sha256']: p for p in papers}
    lim = json.load(open(B / 'limitations/dashboard.json'))
    picks = []  # (sha, group, reason, page_strategy)
    for p in lim['papers']:
        if p['group'].startswith('worst') or p['group'] == 'pinned':
            picks.append((p['pdf_sha256'], 'worst', p['group'] + ': ' + p['reason'], 'errors'))
    used = {s for s, *_ in picks}
    traits = {s: paper_traits(p) for s, p in by.items()}
    def top(key, n, label, cond=lambda s: True):
        for s in sorted(traits, key=lambda s: -traits[s][key] if isinstance(traits[s][key], (int, float)) else 0):
            if len([x for x in picks if x[2].startswith(label)]) >= n: break
            if s in used or not cond(s): continue
            used.add(s); picks.append((s, 'edge', f"{label}: {traits[s][key]}", key))
    top('math', 2, 'math-heavy'); top('table', 2, 'table-heavy'); top('numeric', 2, 'number-heavy'); top('outside', 2, 'text outside MinerU blocks')
    odd = [s for s in traits if not traits[s]['body_font'].startswith(('Nimbus', 'Times', 'TeXGyreTermes', 'ptm', 'STIX')) and s not in used]
    random.seed(7)
    for s in random.sample(odd, min(3, len(odd))):
        used.add(s); picks.append((s, 'edge', f"unusual body font: {traits[s]['body_font']}", 'mixed'))
    rest = [s for s in by if s not in used]
    for s in random.sample(rest, 14):
        used.add(s); picks.append((s, 'standard', 'random paper', 'random'))
    return picks


def page_choice(paper, strategy, n):
    pages = paper['pages']; idx = list(range(len(pages)))
    err = lambda i: sum(v for k, v in pages[i]['counts'].items() if k != 'words')
    om = lambda i, k: len(pages[i]['omit_boxes'].get(k, []))
    if strategy == 'errors': return sorted(idx, key=lambda i: -err(i))[:n]
    if strategy == 'math': return sorted(idx, key=lambda i: -(om(i, 'math_display') + om(i, 'math_inline')))[:n]
    if strategy == 'numeric': return sorted(idx, key=lambda i: -om(i, 'numeric'))[:n]
    if strategy == 'table': return sorted(idx, key=lambda i: -sum(1 for r in pages[i]['regions'] if r['kind'] == 'table'))[:n]
    random.seed(paper['pdf_sha256']); body = [i for i in idx if pages[i]['counts'].get('words', 0) > 50]
    return sorted(random.sample(body or idx, min(n, len(body or idx))))


if __name__ == '__main__':
    import pyarrow.parquet as pq, pymupdf
    OUT.mkdir(parents=True, exist_ok=True)
    fixed = json.load(open(os.environ['KIT_PAGES'])) if os.environ.get('KIT_PAGES') else None
    if fixed:   # reuse an earlier kit's exact pages (same paper, same page) for before/after comparison
        picks = []; seen_ = set()
        for m in fixed:
            if m['pdf_sha256'] not in seen_: seen_.add(m['pdf_sha256']); picks.append((m['pdf_sha256'], m['group'], m['why_selected'], 'fixed'))
        fixed_pages = collections.defaultdict(list)
        for m in fixed: fixed_pages[m['pdf_sha256']].append(m['page'] - 1)
    else:
        picks = choose()
    want = {s for s, *_ in picks}; rows = {}
    for f in sorted(glob.glob(str(op.SRC / 'data/train-*.parquet'))):
        for r in pq.read_table(f, columns=['id', 'title', 'pdf_sha256', 'positions_path']).to_pylist():
            if r['pdf_sha256'] in want: rows[r['pdf_sha256']] = dict(r)
    for f in sorted(glob.glob(str(op.CLEAN / 'data/train-*.parquet'))):
        for r in pq.read_table(f, columns=['pdf_sha256', 'clean_positions_path']).to_pylist():
            if r['pdf_sha256'] in rows: rows[r['pdf_sha256']]['clean_positions_path'] = r['clean_positions_path']
    from concurrent.futures import ProcessPoolExecutor
    specs = [{'pdf_sha256': s, 'id': rows[s]['id'], 'group': g, 'reason': why} for s, g, why, _ in picks]
    with ProcessPoolExecutor(8) as pool:
        details = list(pool.map(ol.detail, specs, [rows[s] for s, *_ in picks]))
    manifest = []
    for (s, group, why, strat), paper in zip(picks, details):
        n = 3 if group != 'standard' else 2
        doc = pymupdf.open(op.SRC / f'objects/{s}.pdf')
        for i in (fixed_pages[s] if fixed else page_choice(paper, strat, n)):
            pg = paper['pages'][i]; name = f"{group}-{paper['id']}-p{i + 1}"
            doc[i].get_pixmap(dpi=130).save(str(OUT / f'{name}.jpg'), jpg_quality=82)
            def snippet(kind):
                return [pg['clean'][a:b] for a, b, k in pg['spans'] if k == kind]
            rec = {'name': name, 'group': group, 'why_selected': why, 'paper_id': paper['id'], 'title': paper.get('title'),
                   'page': i + 1, 'page_count': paper['page_count'], 'body_font': paper.get('body_font'),
                   'baseline_text': pg['clean'], 'omission_text': pg['omit'],
                   'flagged_errors_vs_mineru': {k: snippet(k) for k in ol.ERR if snippet(k)} if hasattr(ol, 'ERR') else {
                       k: snippet(k) for k in ('dropped_text', 'missed_figure', 'missed_equation', 'table_misfire') if snippet(k)},
                   'omitted_words': {k: snippet('omitted_' + k) for k in ('figure', 'math_display', 'math_inline', 'numeric') if snippet('omitted_' + k)},
                   'detected_regions': pg['regions'], 'mineru_blocks': [{'type': b['type'], 'text': b['text'][:300]} for b in pg['blocks'] if b['text'].strip()]}
            (OUT / f'{name}.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1))
            manifest.append({'name': name, 'group': group, 'paper_id': paper['id'], 'pdf_sha256': s, 'page': i + 1, 'why_selected': why})
        doc.close()
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    with tarfile.open(B / f'{OUT.name}.tar', 'w') as t: t.add(OUT, arcname=OUT.name)
    print(collections.Counter(m['group'] for m in manifest), 'pages', len(manifest))
