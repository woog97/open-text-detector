"""Space-side: find calibration papers that share text with any scored model's training data.

Reference text: the original prepared-v2 mix (Atlas fast10 trained on 10% of it), the T2 mix, the T2.1 mix, the
splice pool (SPG), selection/calibration windows, and sweep-eval rows. Same matching rule as build_t21.py:
paper-key match, plus normalized sentences of 10+ words. Writes /tmp/pangram-humancal/overlap.json
(per-paper shared-sentence counts and key hits; no text).
"""
import gzip, json, re, sys
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq

T = Path('/tmp/pangram-t2-20261006'); H = Path('/tmp/pangram-humancal')
SENT = re.compile(r'\S.*?(?:[.!?](?=\s|$)|$)', re.S)
def sents(t): return {' '.join(w) for m in SENT.finditer(t) for w in [re.findall(r'[a-z0-9]+', m.group().lower())] if len(w) >= 10}

files = [p for run in ['qwen35-4b', 'qwen35-4b-t2', 'qwen35-4b-t21'] for p in sorted((T / 'runs' / run / 'prepared-v2').glob('*.jsonl.gz'))]
files += [T / 'sweeps/sweep-eval-rows.jsonl.gz', H / 'splices-v1.jsonl.gz']
held, keys = set(), set()
for f in files:
    for l in gzip.open(f, 'rt'):
        r = json.loads(l)
        for k in ('id', 'paper_id', 'group_id', 'group', 'source_id', 'doc_id'):
            v = r.get(k)
            if isinstance(v, str): keys.update(x for x in re.split(r'[/:|]', v) if len(x) >= 8)
        for k in ('text', 'host_text', 'original', 'context', 'replaced_human'):
            if isinstance(r.get(k), str): held |= sents(r[k])
print('reference sentences', len(held), 'keys', len(keys), flush=True)
out = {}
for f in sorted((H / 'clean/data').glob('train-*.parquet')):
    for r in pq.read_table(f, columns=['id', 'text']).to_pylist():
        s = sents(r['text']); shared = s & held
        out[r['id']] = {'sentences_10w': len(s), 'shared': len(shared), 'key_hit': r['id'] in keys}
(H / 'overlap.json').write_text(json.dumps({'reference_files': [str(f) for f in files], 'papers': out}))
c = Counter(min(v['shared'], 5) for v in out.values())
print(json.dumps({'papers': len(out), 'shared_hist(5=5+)': dict(sorted(c.items())), 'key_hits': sum(v['key_hit'] for v in out.values())}))
