"""Pick one ICLR 2027 Atlas paper per 5% bin of the final MoE's body-mode flagged share (bins 50-55% ... 95-100%),
cut each before its bibliography and at the end of page 9 at the latest, and recount our flags on the cut text.
Usage: prep.py OUT_DIR  (OUT_DIR gets flags.json/catalogue.json downloads and papers.json; paper text stays out of git)
"""
import gzip, json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'research/atlas-bib-exclusion-20261008')]
from upload_paper_pdfs import request, BASE  # noqa: E402
from bib import bibliography_spans  # noqa: E402

SEED, MAX_PAGE, MOE = 20261009, 9, 'qwen36-35b-a3b'
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)

def cached(name, url):
    p = out / name
    if not p.exists(): p.write_text(json.dumps(request(url)))
    return json.loads(p.read_text())

flags = cached('flags.json', BASE + '/indexes/flags/current.json')
cat = {i['id']: i for i in cached('catalogue.json', BASE + '/atlas-public/catalogue.json')['items'] if i['collection'] == 'iclr/2027'}
T = flags['thresholds']['body']
bins = {}
for pid in sorted(cat):
    e = flags['papers'].get(pid)
    if not e: continue
    fl, n = e['modes']['body']
    if n: bins.setdefault(min(int(100 * fl / n // 5), 19), []).append(pid)
rng = random.Random(SEED); papers = []
for b in range(10, 20):
    pid = rng.choice(bins[b]); fl, n = flags['papers'][pid]['modes']['body']
    raw = request(BASE + '/' + cat[pid]['detail_key'], raw=True)
    d = json.loads(gzip.decompress(raw) if raw[:2] == b'\x1f\x8b' else raw)
    rep = next(r for r in d['reports'] if r['id'].startswith(MOE)); text = rep['text']
    rects = d['position_maps'][rep['text_sha256']]['rectangles']
    page_cut = min((r['start'] for r in rects if r['page'] > MAX_PAGE), default=len(text))
    spans = bibliography_spans(text); bib_cut = spans[0][0] if spans else len(text)
    cut = min(page_cut, bib_cut)
    segs = [s for s in rep['result']['segments'] if s['end'] <= cut]
    cut = segs[-1]['end'] if segs else 0  # end on a sentence boundary
    counted = [s for s in segs if any(ch.isalnum() for ch in text[s['start']:s['end']])]
    papers.append(dict(id=pid, bin=f'{5*b}-{5*b+5}%', title=cat[pid]['title'], filename=cat[pid]['filename'],
                       atlas_body_share=fl / n, atlas_body=[fl, n], pages=max(r['page'] for r in rects),
                       cut_reason='bibliography' if bib_cut <= page_cut else f'page {MAX_PAGE}',
                       cut_page=max((r['page'] for r in rects if r['start'] < cut), default=0),
                       text=text[:cut], words=len(text[:cut].split()),
                       segments=[dict(start=s['start'], end=s['end'], score=s['score'], flagged=s['score'] > T) for s in segs],
                       cut_share=sum(s['score'] > T for s in counted) / max(len(counted), 1), cut_counted=len(counted)))
    p = papers[-1]
    print(f"{p['bin']:>8} {pid} pages {p['pages']:>2} cut {p['cut_reason']:>12} at p{p['cut_page']} words {p['words']:>5} "
          f"atlas {p['atlas_body_share']:.0%} cut {p['cut_share']:.0%}  {p['title'][:60]}", flush=True)
(out / 'papers.json').write_text(json.dumps(dict(seed=SEED, threshold_body=T, papers=papers)))
print('total words', sum(p['words'] for p in papers))
