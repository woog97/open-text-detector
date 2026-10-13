"""Prototype: baseline text with figure/math omissions, scored against MinerU layout labels.

Run on the training Space with PyMuPDF available. Reads the cleaned baseline (v2) and
original positions for each paper, the PDF, and MinerU's layout JSON as an answer key.
Writes per-paper rendered text, aggregate metrics, and dashboard detail for --detail ids.
"""
import argparse, collections, glob, json, os, random, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from open_text_detector_backend.omissions import page_features, label_words, classify, render, DEFAULT_POLICY

SRC = Path('/data/workspace/datasets/iclr2027-complete-20261004')
CLEAN = Path('/data/workspace/datasets/iclr2027-clean-text-v2-20261005')
MD = Path('/data/workspace/datasets/iclr2027-mineru-markdown-20261005/objects')
LEAF = {'chart_body': 'figure', 'image_body': 'figure', 'equation': 'equation', 'table_body': 'table',
        'chart_caption': 'caption', 'image_caption': 'caption', 'table_caption': 'caption',
        'text': 'prose', 'ref_text': 'prose', 'paragraph_title': 'prose', 'list': 'prose', 'doc_title': 'prose', 'page_footnote': 'prose'}


def mineru_boxes(middle):
    out = collections.defaultdict(list)
    def walk(node, page):
        if isinstance(node, dict):
            t = node.get('type')
            if t in LEAF and len(node.get('bbox') or []) == 4: out[page].append((LEAF[t], node['bbox']))
            for v in node.values(): walk(v, page)
        elif isinstance(node, list):
            for v in node: walk(v, page)
    for p in middle['pages']: walk(p.get('blocks', []), p['page_idx'] + 1)
    return out


def one(sha, rows, decode, detail):
    import pymupdf, zstandard
    src = decode((SRC / rows['positions_path']).read_bytes())
    clean = decode((CLEAN / rows['clean_positions_path']).read_bytes())
    doc = pymupdf.open(SRC / f'objects/{sha}.pdf'); feats = page_features(doc); doc.close()
    labels, body = label_words(src, feats)
    kinds = classify(clean, labels)
    text, omissions = render(clean, kinds, DEFAULT_POLICY)
    middle = json.loads(zstandard.ZstdDecompressor().decompress((MD / f'{sha}.middle.json.zst').read_bytes(), max_output_size=2**30))
    boxes = mineru_boxes(middle); pages = {p['page']: p for p in src['pages']}
    import re
    c = collections.Counter()
    for m, k in zip(clean['mapping'], kinds):
        lab = labels.get(m['word_index'], {})
        real_word = lab.get('font') == 'body' and re.fullmatch(r"[A-Za-z][a-z'\-]{2,}[,.;:)]?", lab.get('text', '')) is not None
        r = src['rectangles'][m['word_index']]; pg = pages[r['page']]
        cx = (r['x0'] + r['x1']) / 2 / pg['width']; cy = (r['y0'] + r['y1']) / 2 / pg['height']
        truth = next((t for t, (x0, y0, x1, y1) in boxes.get(r['page'], []) if x0 <= cx <= x1 and y0 <= cy <= y1), 'none')
        c[(truth, k or 'kept')] += 1
        if real_word: c[(truth + '_word', k or 'kept')] += 1
    out = {'pdf_sha256': sha, 'id': rows['id'], 'body_font': body, 'counts': {f'{a}|{b}': n for (a, b), n in c.items()},
           'omissions': collections.Counter(o['type'] for o in omissions), 'chars_before': len(clean['text']), 'chars_after': len(text)}
    if detail:
        per_page = collections.defaultdict(lambda: collections.defaultdict(list))
        for m, k in zip(clean['mapping'], kinds):
            if not k: continue
            r = src['rectangles'][m['word_index']]; pg = pages[r['page']]
            per_page[r['page']][k].append([round(r['x0'] / pg['width'], 4), round(r['y0'] / pg['height'], 4), round(r['x1'] / pg['width'], 4), round(r['y1'] / pg['height'], 4)])
        out['text'] = text
        out['boxes'] = {p: dict(v) for p, v in per_page.items()}
        out['regions'] = {i + 1: f['regions'] for i, f in enumerate(feats)}
        out['omission_samples'] = omissions[:400]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--detail', type=Path, required=True, help='JSON list of pdf_sha256 to keep full detail for')
    ap.add_argument('--extra', type=int, default=180)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--vendor', default='/data/workspace/baseline-omissions-20261005/vendor')
    args = ap.parse_args()
    sys.path.insert(0, args.vendor); sys.path.insert(0, '/data/workspace/datasets/iclr2027-round3-7000-20261003')
    from result_codec import decode
    import pyarrow.parquet as pq
    detail = json.loads(args.detail.read_text())
    parsed = {Path(p).name[:-len('.middle.json.zst')] for p in glob.glob(str(MD / '*.middle.json.zst'))}
    random.seed(20261005)
    extra = random.sample(sorted(parsed - set(detail)), args.extra)
    wanted = set(detail) | set(extra)
    rows = {}
    for f in sorted(glob.glob(str(SRC / 'data/train-*.parquet'))):
        for r in pq.read_table(f, columns=['id', 'pdf_sha256', 'positions_path']).to_pylist():
            if r['pdf_sha256'] in wanted and r['positions_path']: rows[r['pdf_sha256']] = dict(r)
    for f in sorted(glob.glob(str(CLEAN / 'data/train-*.parquet'))):
        for r in pq.read_table(f, columns=['pdf_sha256', 'clean_positions_path']).to_pylist():
            if r['pdf_sha256'] in rows: rows[r['pdf_sha256']]['clean_positions_path'] = r['clean_positions_path']
    results = []; errors = []; t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor, as_completed
    todo = [s for s in sorted(wanted) if s in rows and 'clean_positions_path' in rows[s]]
    errors += [{'pdf_sha256': s, 'error': 'no cleaned v2 text'} for s in sorted(wanted) if s not in todo]
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(one, s, rows[s], decode, s in detail): s for s in todo}
        for n, f in enumerate(as_completed(futures), 1):
            try: results.append(f.result())
            except Exception as e: errors.append({'pdf_sha256': futures[f], 'error': repr(e)[:300]})
            if n % 20 == 0 or n == len(todo): print(json.dumps({'done': n, 'of': len(todo), 'seconds': round(time.time() - t0)}), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    total = collections.Counter()
    for r in results:
        for k, v in r['counts'].items(): total[k] += v
    def frac(truth, kinds):
        n = sum(v for k, v in total.items() if k.split('|')[0] == truth)
        return round(sum(v for k, v in total.items() if k.split('|')[0] == truth and k.split('|')[1] in kinds) / max(1, n), 4), n
    def prec(kind, truths):
        n = sum(v for k, v in total.items() if k.split('|')[1] == kind)
        return round(sum(v for k, v in total.items() if k.split('|')[1] == kind and k.split('|')[0] in truths) / max(1, n), 4), n
    metrics = {
        'papers': len(results), 'errors': errors, 'seconds': round(time.time() - t0),
        'figure_recall': frac('figure', {'figure', 'numeric'}), 'table_recall': frac('table', {'table'}), 'equation_recall': frac('equation', {'math_display', 'math_inline'}),
        'prose_lost_to_figure_or_display': frac('prose', {'figure', 'math_display', 'numeric'}), 'prose_inline_math': frac('prose', {'math_inline'}),
        'prose_real_words_lost': frac('prose_word', {'figure', 'math_display', 'math_inline', 'numeric', 'table', 'hidden', 'duplicate'}), 'caption_real_words_lost': frac('caption_word', {'figure', 'math_display', 'math_inline', 'numeric', 'table', 'hidden', 'duplicate'}),
        'numeric_omitted_outside_figures': frac('prose', {'numeric'}), 'table_words_as_numbers': frac('table', {'numeric'}),
        'table_words_as_equation': frac('table', {'math_display'}),
        'caption_lost': frac('caption', {'figure', 'math_display', 'math_inline', 'numeric', 'table', 'hidden', 'duplicate'}), 'table_omitted_as_figure': frac('table', {'figure'}),
        'figure_precision_vs_mineru_figure': prec('figure', {'figure'}), 'display_precision_vs_mineru_equation': prec('math_display', {'equation'}),
        'counts': dict(total)}
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=1))
    (args.output / 'detail.json').write_text(json.dumps([r for r in results if 'text' in r]))
    (args.output / 'papers.json').write_text(json.dumps([{k: v for k, v in r.items() if k not in ('text', 'boxes', 'regions', 'omission_samples')} for r in results]))
    print(json.dumps({k: v for k, v in metrics.items() if k != 'counts'}, indent=1))


if __name__ == '__main__':
    main()
