"""Join papers.json with the Pangram results into the page data and render page.html.
Usage: build.py OUT_DIR   (writes OUT_DIR/page.html; the page embeds paper text, so it stays out of git)
"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
out = Path(sys.argv[1]); prep = json.loads((out / 'papers.json').read_text())
flags = json.loads((out / 'flags.json').read_text()); cat = {i['id'] for i in json.loads((out / 'catalogue.json').read_text())['items'] if i['collection'] == 'iclr/2027'}
counts = [0] * 20
for pid, e in flags['papers'].items():
    fl, n = e['modes']['body']
    if pid in cat and n: counts[min(int(100 * fl / n // 5), 19)] += 1


def offset_map(sent, got, pid):
    """Pangram drops the page-break form feeds ('\\x0c\\n') from the text it returns; map its offsets back onto ours.
    Only whitespace and form feeds may be missing, or the page refuses to build."""
    m, i = [], 0
    for ch in got:
        while i < len(sent) and sent[i] != ch:
            if sent[i] not in '\x0c\n \t': raise SystemExit(f'{pid}: Pangram text differs beyond page breaks at {i}')
            i += 1
        m.append(i); i += 1
    if i < len(sent) and sent[i:].strip('\x0c\n \t'): raise SystemExit(f'{pid}: Pangram text is missing a tail')
    m.append(len(sent)); return m


rows = []
for p in prep['papers']:
    f = out / 'pangram' / f"{p['id']}.json"
    r = json.loads(f.read_text()).get('result', {}) if f.exists() else {}
    if r.get('stage') != 'STAGE_SUCCESS': r = None
    else: m = offset_map(p['text'], r['text'], p['id'])
    rows.append(dict(id=p['id'], bin=p['bin'], title=p['title'], filename=p['filename'], pages=p['pages'],
                     cut_reason=p['cut_reason'], cut_page=p['cut_page'], words=p['words'], text=p['text'],
                     ours=dict(full=p['atlas_body_share'], full_counts=p['atlas_body'], cut=p['cut_share'], counted=p['cut_counted'],
                               segs=[[s['start'], s['end'], round(s['score'], 5), int(s['flagged'])] for s in p['segments']]),
                     pangram=None if r is None else dict(
                         fraction_ai=r['fraction_ai'], fraction_ai_assisted=r['fraction_ai_assisted'], fraction_human=r['fraction_human'],
                         headline=r['headline'], prediction=r['prediction'], short=r['prediction_short'], version=r.get('version'),
                         windows=[[m[w['start_index']], m[w['end_index']], w['label'], round(w['ai_assistance_score'], 4), w['confidence'],
                                   w['word_count'], int(bool(w.get('is_humanized')))] for w in r['windows']])))
data = dict(threshold=prep['threshold_body'], seed=prep['seed'], bin_counts=counts, total=sum(counts), papers=rows)
html = (HERE / 'page.template.html').read_text().replace('/*DATA*/null', json.dumps(data, separators=(',', ':')).replace('</', '<\\/'))
(out / 'page.html').write_text(html)
done = [r for r in rows if r['pangram']]
print(f'{len(done)}/{len(rows)} with Pangram results; page {len(html)/1e6:.2f} MB')
for r in rows:
    pg = r['pangram']
    print(f"{r['bin']:>8}  ours full {r['ours']['full']:.0%} cut {r['ours']['cut']:.0%}  pangram " +
          (f"AI {pg['fraction_ai']:.0%} assisted {pg['fraction_ai_assisted']:.0%} {pg['headline']}" if pg else '-'))
