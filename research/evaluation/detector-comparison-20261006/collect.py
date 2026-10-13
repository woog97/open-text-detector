"""Collect every scored model into compare-data.json (aggregated metrics only)."""
import json, glob, os, re, statistics as st, collections
D = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(D, '../../../benchmarks/otd/training/overnight-sweep-20261004/results/runs.json')
M = [('one', 'edits_one', 'recall_at_1pct'), ('two', 'edits_two', 'recall_at_1pct'), ('small', 'edits_small', 'recall_at_1pct'),
     ('para', 'edits_para', 'recall_at_1pct'), ('all', 'edits_all', 'recall_at_1pct'), ('small_auc', 'edits_small', 'auroc'),
     ('all_auc', 'edits_all', 'auroc'), ('v3', 'v3_sentences', 'recall_at_1pct'), ('standalone', 'standalone_rewrites', 'recall_at_1pct'),
     ('public_auc', 'public_docs', 'auroc')]
def pick(ev, half):
    h = ev.get(half, {}); return {k: (h.get(a) or {}).get(b) for k, a, b in M}
runs = {}  # tag -> {epoch: {dev:{}, test:{}}}
R = json.load(open(RUNS))
for t, r in R.items():
    for ck, ev in (r.get('eval') or {}).items():
        m = re.search(r'stage2-epoch(\d)', ck)
        if m and 'dev' in ev: runs.setdefault(t, {})[int(m.group(1))] = {'dev': pick(ev, 'dev'), 'test': pick(ev, 'test')}
for f in glob.glob(D + '/h200/*/eval/stage2-epoch*.json') + glob.glob(D + '/mixgrid/*/eval/stage2-epoch*.json'):  # mixgrid: Oct 7 MoE mix runs on the Space
    t = f.split('/')[-3]; ev = json.load(open(f)); e = int(re.search(r'epoch(\d)', f).group(1))
    runs.setdefault(t, {})[e] = {'dev': pick(ev, 'dev'), 'test': pick(ev, 'test')}
sp = json.load(open(D + '/space/space.json'))
if os.path.exists(D + '/space/t2.json'):  # T2-mix runs and re-scored SPG, persisted in /data/workspace/pangram-t2-20261006
    sp.update(json.load(open(D + '/space/t2.json')))
for f in glob.glob(D + '/extra/q4b-*/eval/stage2-epoch*.json'):  # T2.1 and its soft-label arm, pulled from /tmp/pangram-t2-20261006
    k = f[len(D) + 7:]
    if k.split('/')[0].startswith(('q4b-T21', 'q4b-T21S')): sp[k] = json.load(open(f))
for k, ev in sp.items():
    m = re.match(r'(q4b-[A-Za-z0-9]+-s\d)/eval/stage2-epoch(\d)\.json', k)
    if m: runs.setdefault(m.group(1), {})[int(m.group(2))] = {'dev': pick(ev, 'dev'), 'test': pick(ev, 'test')}
status = {k.split('/')[0]: v.get('state') for k, v in sp.items() if k.endswith('status.json')}
cross = {}
for f in glob.glob(D + '/h200/*/eval/cross-model-*.json') + glob.glob(D + '/h200/*/eval/heldout-edits-*.json') + glob.glob(D + '/h200/*/eval/heldout-writers-*.json') + glob.glob(D + '/h200/*/eval/gpt-6-luna-stage2-*.json') + glob.glob(D + '/space/cross/*.json') + glob.glob(D + '/mixgrid/*/eval/heldout-writers-*.json') + glob.glob(D + '/mixgrid/*/eval/cross-model-s500-*.json') + [f for f in glob.glob(D + '/extra/q4b-T21*/eval/*-stage2-epoch2.json') if re.search(r'/(heldout-writers|cross-model-s500)-', f)]:
    if f.endswith('-rows.json'): continue
    c = json.load(open(f)); e = re.search(r'stage2-epoch(\d)', f); ds = ('heldout2' if 'heldout-writers' in f else 'heldout' if ('heldout-edits' in f or 'gpt-6-luna-stage2' in f)
          else 'hetero500' if 'cross-model-s500' in f else 'hetero')
    key = f"{ds}|{c['tag']}@e{e.group(1) if e else '2'}"  # heldout2: strict 1,171-edit set; hetero500: 500-document sample
    cross[key] = {'groups': c['groups'], 'documents': c.get('documents'), 'thr_in': c.get('threshold_in_domain'),
                  'fpr_ctrl': c.get('control_fpr_at_in_domain_threshold'), 'fpr_retained': c.get('retained_human_fpr_at_in_domain_threshold'),
                  'fpr_all': c.get('human_fpr_at_in_domain_threshold')}
river = {}
if os.path.exists(D + '/river.json'):  # River's (sky.moo) Oct 6 hyperparameter sweep, same evaluation set and scorer
    rj = json.load(open(D + '/river.json'))
    for k, ev in rj.items():
        m = re.match(r'\./sweeps/([^/]+)/eval/(stage2-epoch\d|step\d+)\.json$', k)
        if m and 'dev' in ev: river.setdefault(m.group(1), {})[m.group(2)] = {'dev': pick(ev, 'dev'), 'test': pick(ev, 'test')}
ab = json.load(open(D + '/ab-t21.json')) if os.path.exists(D + '/ab-t21.json') else None  # ab_t21.py output (mix decision tab)
json.dump({'runs': runs, 'status': status, 'cross': cross, 'river': river, 'ab': ab}, open(D + '/compare-data.json', 'w'))
print(len(runs), 'runs;', len(cross), 'cross;', sorted(runs)[:80])
