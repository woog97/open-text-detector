"""Pick worst/average/best papers for the omission prototype and build dashboard detail.

Reads prototype-v2 per-paper counts, selects examples by failure type, then for each
selected paper writes per-page cleaned text, omission text, error spans/boxes (missed
figure text, missed equations, wrongly dropped real words, table misfires), MinerU
blocks for comparison, and page JPEGs. Run on the training Space.
"""
import collections, glob, json, os, re, sys, tarfile
from pathlib import Path

B = Path('/data/workspace/baseline-omissions-20261005')
sys.path.insert(0, str(B / 'code/backend')); sys.path.insert(0, str(B / 'vendor'))
sys.path.insert(0, '/data/workspace/datasets/iclr2027-round3-7000-20261003'); sys.path.insert(0, str(B / 'code/scripts'))
import omission_prototype as op
from open_text_detector_backend.omissions import page_features, label_words, classify, render, DEFAULT_POLICY
from result_codec import decode

OUT = B / 'limitations'
REAL = re.compile(r"[A-Za-z][a-z'\-]{2,}[,.;:)]?")
DROP = {'figure', 'math_display', 'math_inline', 'numeric', 'table', 'hidden', 'duplicate'}


def paper_scores(p):
    c = collections.Counter()
    for k, v in p['counts'].items(): c[tuple(k.split('|'))] = v
    tot = lambda t: sum(v for (a, b), v in c.items() if a == t)
    fig_total, eq_total = tot('figure'), tot('equation')
    fig_missed = c[('figure', 'kept')] + c[('figure', 'table')]
    eq_missed = c[('equation', 'kept')] + c[('equation', 'table')]
    dropped = sum(v for (a, b), v in c.items() if a in ('prose_word', 'caption_word') and b in DROP)
    real_total = tot('prose_word') + tot('caption_word')
    misfire = sum(v for (a, b), v in c.items() if a == 'table' and b in DROP - {'table'})
    errors = fig_missed + eq_missed + 5 * dropped + misfire
    return {'pdf_sha256': p['pdf_sha256'], 'id': p['id'], 'fig_total': fig_total, 'fig_missed': fig_missed,
            'fig_missed_rate': fig_missed / fig_total if fig_total else 0, 'eq_total': eq_total, 'eq_missed': eq_missed,
            'dropped_real': dropped, 'dropped_rate': dropped / real_total if real_total else 0, 'table_misfire': misfire,
            'error_rate': errors / max(1, fig_total + eq_total + 200)}


def select(scores):
    used = set(); picks = []
    def take(group, label, items, n):
        for s in items:
            if len([p for p in picks if p['group'] == group]) >= n: break
            if s['pdf_sha256'] in used: continue
            used.add(s['pdf_sha256']); picks.append({**s, 'group': group, 'reason': label(s)})
    rich = [s for s in scores if s['fig_total'] >= 200 and s['eq_total'] >= 100]
    take('worst_figure', lambda s: f"{s['fig_missed_rate']:.0%} of figure text kept",
         sorted([s for s in scores if s['fig_total'] >= 200], key=lambda s: -s['fig_missed_rate']), 3)
    take('worst_dropped', lambda s: f"{s['dropped_real']} real words dropped",
         sorted(scores, key=lambda s: -s['dropped_real']), 3)
    take('worst_misfire', lambda s: f"{s['table_misfire']} table words omitted, {s['eq_missed']} equation words kept",
         sorted(scores, key=lambda s: -(s['table_misfire'] + s['eq_missed'])), 2)
    ordered = sorted(rich, key=lambda s: s['error_rate'])
    mid = len(ordered) // 2
    take('average', lambda s: f"error rate {s['error_rate']:.1%} (median)", ordered[mid - 2:mid + 3], 3)
    take('best', lambda s: f"error rate {s['error_rate']:.1%}", ordered, 3)
    return picks


def strings(x, out):
    if isinstance(x, str): out.append(x)
    elif isinstance(x, dict):
        for k, v in x.items():
            if k not in ('type', 'bbox', 'index', 'score', 'id', 'page_idx', 'image_path', 'image_base64'): strings(v, out)
    elif isinstance(x, list):
        for v in x: strings(v, out)
    return out


def detail(pick, row):
    import pymupdf, zstandard
    sha = pick['pdf_sha256']
    src = decode((op.SRC / row['positions_path']).read_bytes()); clean = decode((op.CLEAN / row['clean_positions_path']).read_bytes())
    doc = pymupdf.open(op.SRC / f'objects/{sha}.pdf'); feats = page_features(doc)
    labels, body = label_words(src, feats); kinds = classify(clean, labels)
    text, omissions = render(clean, kinds, DEFAULT_POLICY)
    layout = op.MD / f'{sha}.middle.json.zst'   # MinerU reference is optional (held-out papers mostly lack it)
    middle = json.loads(zstandard.ZstdDecompressor().decompress(layout.read_bytes(), max_output_size=2**30)) if layout.exists() else {'pages': []}
    boxes = op.mineru_boxes(middle); pages = {p['page']: p for p in src['pages']}
    # Group output tokens by their source page; never split on form feeds, because the
    # cleaner emits no page break for pages that kept no text.
    ctext = clean['text']; n_pages = len(src['pages'])
    by_page = collections.defaultdict(list)
    for j, (m, k) in enumerate(zip(clean['mapping'], kinds)): by_page[m['page']].append(j)
    clean_pages, omit_pages = [''] * n_pages, [''] * n_pages
    spans = [[] for _ in range(n_pages)]; err_boxes = [collections.defaultdict(list) for _ in range(n_pages)]
    om_boxes = [collections.defaultdict(list) for _ in range(n_pages)]; page_err = [collections.Counter() for _ in range(n_pages)]
    for page_no, idx in by_page.items():
        i0 = page_no - 1
        a0 = clean['mapping'][idx[0]]['start']; b0 = clean['mapping'][idx[-1]]['end']
        clean_pages[i0] = ctext[a0:b0].replace('\f', '')
        sub = {'text': ctext[a0:b0], 'mapping': [{**clean['mapping'][j], 'start': clean['mapping'][j]['start'] - a0, 'end': clean['mapping'][j]['end'] - a0} for j in idx]}
        omit_pages[i0] = render(sub, [kinds[j] for j in idx], DEFAULT_POLICY)[0].replace('\f', '')
        for j in idx:
            m = clean['mapping'][j]; k = kinds[j]
            r = src['rectangles'][m['word_index']]; pg = pages[r['page']]
            box = [round(r['x0'] / pg['width'], 4), round(r['y0'] / pg['height'], 4), round(r['x1'] / pg['width'], 4), round(r['y1'] / pg['height'], 4)]
            cx = (box[0] + box[2]) / 2; cy = (box[1] + box[3]) / 2
            truth = next((t for t, (x0, y0, x1, y1) in boxes.get(r['page'], []) if x0 <= cx <= x1 and y0 <= cy <= y1), 'none')
            lab = labels.get(m['word_index'], {})
            real = lab.get('font') == 'body' and REAL.fullmatch(lab.get('text', '')) is not None
            err = None
            if truth == 'figure' and k is None: err = 'missed_figure'
            elif truth == 'equation' and k is None: err = 'missed_equation'
            elif truth in ('prose', 'caption') and real and k in DROP: err = 'dropped_text'
            elif truth == 'table' and k in DROP - {'table'}: err = 'table_misfire'
            a = m['start'] - a0; b = m['end'] - a0
            if err:
                spans[i0].append([a, b, err]); err_boxes[i0][err].append(box); page_err[i0][err] += 1
            elif k in DROP:
                spans[i0].append([a, b, 'omitted_' + k])
            if k in DROP: om_boxes[i0][k].append(box)
            page_err[i0]['words'] += 1
    n_seg = n_pages
    blocks = []
    for pg in middle['pages']:
        blocks.append([{'type': b.get('type'), 'text': ' '.join(s for s in strings(b.get('content', []), []) if s.strip())} for b in pg.get('blocks', [])])
    # page images (90 dpi JPEG)
    imgdir = OUT / 'pages' / sha; imgdir.mkdir(parents=True, exist_ok=True); sizes = []
    for i, page in enumerate(doc):
        if not (imgdir / f'{i + 1}.jpg').exists():
            pix = page.get_pixmap(dpi=90); pix.save(str(imgdir / f'{i + 1}.jpg'), jpg_quality=72)
        sizes.append([i])
    doc.close()
    regions = {i + 1: [r for r in f['regions'] if r['kind'] in ('figure', 'table')] for i, f in enumerate(feats)}
    return {**pick, 'title': row.get('title') or pick['id'], 'body_font': body, 'page_count': len(sizes), 'image_sizes': sizes,
            'pages': [{'clean': clean_pages[i] if i < len(clean_pages) else '', 'omit': omit_pages[i] if i < len(omit_pages) else '',
                       'spans': spans[i] if i < n_seg else [], 'error_boxes': dict(err_boxes[i]) if i < n_seg else {},
                       'omit_boxes': dict(om_boxes[i]) if i < n_seg else {}, 'counts': dict(page_err[i]) if i < n_seg else {},
                       'blocks': blocks[i] if i < len(blocks) else [], 'regions': regions.get(i + 1, [])}
                      for i in range(len(sizes))]}


if __name__ == '__main__':
    import pyarrow.parquet as pq
    OUT.mkdir(parents=True, exist_ok=True)
    papers = json.load(open(B / os.environ.get('OMIT_RUN', 'prototype-v2') / 'papers.json'))
    scores = [paper_scores(p) for p in papers]
    picks = select(scores)
    pinned = [x for x in os.environ.get('OMIT_PIN', '').split(',') if x]
    for s_ in scores:
        if s_['id'] in pinned and s_['pdf_sha256'] not in {p['pdf_sha256'] for p in picks}:
            picks.append({**s_, 'group': 'pinned', 'reason': f"pinned for inspection; error rate {s_['error_rate']:.1%}"})
    json.dump({'scores': scores, 'picks': picks}, open(OUT / 'selection.json', 'w'))
    want = {p['pdf_sha256'] for p in picks}; rows = {}
    for f in sorted(glob.glob(str(op.SRC / 'data/train-*.parquet'))):
        for r in pq.read_table(f, columns=['id', 'title', 'pdf_sha256', 'positions_path']).to_pylist():
            if r['pdf_sha256'] in want: rows[r['pdf_sha256']] = dict(r)
    for f in sorted(glob.glob(str(op.CLEAN / 'data/train-*.parquet'))):
        for r in pq.read_table(f, columns=['pdf_sha256', 'clean_positions_path']).to_pylist():
            if r['pdf_sha256'] in rows: rows[r['pdf_sha256']]['clean_positions_path'] = r['clean_positions_path']
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(8) as pool:
        out = list(pool.map(detail, picks, [rows[p['pdf_sha256']] for p in picks]))
    metrics = json.load(open(B / os.environ.get('OMIT_RUN', 'prototype-v2') / 'metrics.json')); metrics.pop('counts', None)
    json.dump({'papers': out, 'metrics': metrics, 'pool': len(scores)}, open(OUT / 'dashboard.json', 'w'))
    for p in picks: print(p['group'], p['id'], p['reason'])
    print('papers', len(out), 'pages', sum(p['page_count'] for p in out))
