"""Build the fixed sweep evaluation set (dev/test halves by paper/source group) from the frozen suite package."""
import gzip, hashlib, json, random
from collections import defaultdict
from pathlib import Path

PKG = Path(__file__).resolve().parents[3] / '..' / 'research' / 'evaluation' / 'space-suite-v1' / 'package'
PKG = PKG.resolve()
OUT = Path(__file__).resolve().parent / 'sweep-eval-rows.jsonl.gz'
DOC_GRAN = {'document_native_label', 'observed_generated_response'}
PAPER_SETS = {'paper_v3_target', 'paper_pilots_exploratory', 'human_paper_remaining', 'human_paper_context', 'opai', 'local_mixed'}


def load(p):
    return [json.loads(l) for l in gzip.open(PKG / f'{p}.jsonl.gz', 'rt')]


def half(group):
    return 'dev' if int(hashlib.sha256(group.encode()).hexdigest(), 16) % 2 == 0 else 'test'


def pick(rows, n, rng):
    """Sample whole groups until n rows, so groups stay intact."""
    by = defaultdict(list)
    for r in rows:
        by[str(r.get('group_id', r['id']))].append(r)
    groups = sorted(by); rng.shuffle(groups); out = []
    for g in groups:
        if len(out) >= n:
            break
        out.extend(by[g])
    return out


rng = random.Random(20261004)
wf = load('workflow'); co = load('comparison')
sel = []
for r in wf:
    if r['dataset'] == 'paper_workflow_reconstruction' and r['label'] == 'mixed' and r.get('view') == 'context':
        sel.append(('edit_context', r))
    elif r['dataset'] == 'paper_workflow_reconstruction' and r['label'] == 'ai':
        sel.append(('ai_standalone', r))
    elif r['dataset'] == 'human_paper_workflow_matched':
        sel.append(('human_untouched', r))
sel += [('human_paper', r) for r in pick([r for r in wf if r['dataset'] == 'human_paper_workflow_remaining'], 1200, rng)]
v3 = [r for r in co if r['dataset'] == 'paper_v3_target']
sel += [('v3', r) for r in pick([r for r in v3 if r['label'] == 'mixed'], 300, rng)]
sel += [('v3', r) for r in pick([r for r in v3 if r['label'] == 'human'], 300, rng)]
pub = [r for r in co if r['dataset'] not in PAPER_SETS and r['granularity'] in DOC_GRAN and r['label'] in ('human', 'ai')]
for lab in ('human', 'ai'):
    by_ds = defaultdict(list)
    for r in pub:
        if r['label'] == lab:
            by_ds[r['dataset']].append(r)
    per = max(1, 400 // len(by_ds)); chosen = []
    for ds in sorted(by_ds):
        chosen += pick(by_ds[ds], per, rng)
    rest = [r for r in pub if r['label'] == lab and r not in chosen]
    chosen += pick(rest, 400 - len(chosen), rng) if len(chosen) < 400 else []
    sel += [('public', r) for r in chosen[:400]]

seen = set(); n = {}
with gzip.open(OUT, 'wt') as f:
    for slot, r in sel:
        if r['id'] in seen:
            continue
        seen.add(r['id'])
        g = str(r.get('group_id', r['id']))
        keep = {k: r.get(k) for k in ['id', 'dataset', 'label', 'granularity', 'condition', 'view', 'text', 'regions', 'group_id', 'generator']}
        keep.update(slot=slot, split=half(g))
        f.write(json.dumps(keep) + '\n')
        n[(slot, keep['split'])] = n.get((slot, keep['split']), 0) + 1
print(len(seen), 'rows'); [print(k, v) for k, v in sorted(n.items())]
