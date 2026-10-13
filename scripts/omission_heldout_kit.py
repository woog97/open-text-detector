"""Review kit for the frozen held-out set: one random text page per paper, baseline vs omissions.

Evaluation only; never use these pages to design rules. Writes, per page, a 130 dpi JPEG
and JSON with the baseline text (control) and the text with omission markers.
"""
import glob, json, os, random, sys, tarfile
from pathlib import Path

B = Path('/data/workspace/baseline-omissions-20261005')
sys.path.insert(0, str(B / 'code/backend')); sys.path.insert(0, str(B / 'vendor'))
sys.path.insert(0, '/data/workspace/datasets/iclr2027-round3-7000-20261003'); sys.path.insert(0, str(B / 'code/scripts'))
import omission_prototype as op
import omission_limitations as ol

OUT = B / os.environ.get('KIT_NAME', 'heldout-kit')

if __name__ == '__main__':
    import pyarrow.parquet as pq, pymupdf
    OUT.mkdir(parents=True, exist_ok=True)
    held = json.load(open(B / os.environ.get('HELDOUT', 'heldout-100.json')))['papers']; want = {p['pdf_sha256'] for p in held}; rows = {}
    for f in sorted(glob.glob(str(op.SRC / 'data/train-*.parquet'))):
        for r in pq.read_table(f, columns=['id', 'title', 'pdf_sha256', 'positions_path']).to_pylist():
            if r['pdf_sha256'] in want: rows[r['pdf_sha256']] = dict(r)
    for f in sorted(glob.glob(str(op.CLEAN / 'data/train-*.parquet'))):
        for r in pq.read_table(f, columns=['pdf_sha256', 'clean_positions_path']).to_pylist():
            if r['pdf_sha256'] in rows: rows[r['pdf_sha256']]['clean_positions_path'] = r['clean_positions_path']
    specs = [{'pdf_sha256': p['pdf_sha256'], 'id': p['id'], 'group': 'heldout', 'reason': 'random held-out paper'} for p in held if p['pdf_sha256'] in rows]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(8) as pool:
        details = list(pool.map(ol.detail, specs, [rows[s['pdf_sha256']] for s in specs]))
    manifest = []; summary = []
    for spec, paper in zip(specs, details):
        words = [len(pg['clean'].split()) for pg in paper['pages']]
        omitted = sum(len(b) for pg in paper['pages'] for b in pg['omit_boxes'].values())
        summary.append({'paper_id': paper['id'], 'baseline_words': sum(words), 'omitted_words': omitted,
                        'omitted_share': omitted / max(1, sum(words))})
        textpages = [i for i, n in enumerate(words) if n >= 50] or list(range(len(words)))
        i = random.Random(spec['pdf_sha256']).choice(textpages)
        pg = paper['pages'][i]; name = f"heldout-{paper['id']}-p{i + 1}"
        doc = pymupdf.open(op.SRC / f"objects/{spec['pdf_sha256']}.pdf")
        doc[i].get_pixmap(dpi=130).save(str(OUT / f'{name}.jpg'), jpg_quality=82); doc.close()
        rec = {'name': name, 'paper_id': paper['id'], 'title': paper.get('title'), 'page': i + 1, 'page_count': paper['page_count'],
               'body_font': paper.get('body_font'), 'baseline_text': pg['clean'], 'omission_text': pg['omit'],
               'omitted_words': {k: [pg['clean'][a:b] for a, b, kk in pg['spans'] if kk == 'omitted_' + k]
                                 for k in ('figure', 'math_display', 'math_inline', 'numeric', 'table', 'hidden', 'duplicate')},
               'detected_regions': pg['regions']}
        (OUT / f'{name}.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        manifest.append({'name': name, 'paper_id': paper['id'], 'page': i + 1})
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    (OUT / 'paper-summary.json').write_text(json.dumps(summary, indent=1))
    with tarfile.open(B / f'{OUT.name}.tar', 'w') as t: t.add(OUT, arcname=OUT.name)
    share = sorted(s['omitted_share'] for s in summary)
    print('pages', len(manifest), 'omitted share per paper: median', round(share[len(share) // 2], 3), 'p90', round(share[int(len(share) * .9)], 3), 'max', round(share[-1], 3))
