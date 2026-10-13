import gzip, json, re
from pathlib import Path
R = Path('/tmp/pangram-splice-20261006'); base = R / 'runs/qwen35-4b/prepared-v2'
SENT = re.compile(r'\S.*?(?:[.!?](?=\s|$)|$)', re.S)
def sents(t):
    return {' '.join(w) for m in SENT.finditer(t) for w in [re.findall(r'[a-z0-9]+', m.group().lower())] if len(w) >= 10}
tr = set(); trk = set()
for l in gzip.open(base / 'stage2-epoch0.jsonl.gz', 'rt'):
    r = json.loads(l)
    if r['dataset'] == 'papers': tr |= sents(r['text']); trk.add(r['paper_id'])
ev = set(); evk = set()
for l in gzip.open(R / 'sweeps/sweep-eval-rows.jsonl.gz', 'rt'):
    r = json.loads(l); ev |= sents(r['text']); evk.add(r['id'].split('/')[0])
cal = set(); calk = set(); sel = set()
for name, S in [('calibration-windows', cal), ('selection-windows', sel)]:
    for l in gzip.open(base / f'{name}.jsonl.gz', 'rt'):
        r = json.loads(l); S |= sents(r['text']); calk.add(r.get('paper_id'))
sp = [json.loads(l) for l in gzip.open(R / 'runs/qwen35-4b-sph/prepared-v2/stage2-epoch0.jsonl.gz', 'rt')]
sp = [r for r in sp if r['dataset'] == 'papers_splice']
print('splices sharing a sentence with train papers rows (positive control):', sum(bool(sents(r['text']) & tr) for r in sp), 'of', len(sp))
print('train papers vs eval shared sentences:', len(tr & ev), ' vs cal:', len(tr & cal), ' vs sel:', len(tr & sel))
print('sizes', len(tr), len(ev), len(cal), len(sel), 'eval key sample', list(evk)[:3], 'cal key sample', list(calk)[:3], 'splice key sample', [r['paper_id'] for r in sp[:3]])
