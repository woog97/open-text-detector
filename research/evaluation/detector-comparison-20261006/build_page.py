"""Build the detector comparison page from compare-data.json (aggregated metrics only; no paper text)."""
import json, os, re, statistics as st, time
from collections import defaultdict

D = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(D + '/compare-data.json'))
runs, status, cross = data['runs'], data['status'], data['cross']

ARM = {'A': '2e-4 cosine (baseline)', 'B': '5e-4 constant (LoRA Without Regret)', 'C': '5e-4 constant, head LR 5e-4',
       'D': '5e-4 constant, sentence weight 1.0', 'E': '1e-3 constant', 'F': '5e-4 constant, head LR 5e-4, sentence weight 1.0',
       'G': '5e-4 constant, standard AdamW', 'H': '3e-4 cosine', 'I': '1e-4 cosine', 'J': '5e-4 cosine',
       'K': '2e-4 cosine, sentence weight 1.0', 'L': '2e-4 cosine, short-span oversampling'}
DATA = {'SPH': 'Splices replace half the paper pairs', 'SPG': 'Splices replace GRADTEX', 'LLE': 'LLM sentence edits replace GRADTEX',
        'MIX': 'Splices + LLM edits replace half the mirrors', 'Arep': 'Baseline rerun (2e-4 cosine)',
        'T2': 'T2 mix: multi-writer small edits, paragraph edits, AI sections, more human paper text',
        'T21': 'T2.1: T2 minus the T2.1 held-out split', 'T21S': 'T2.1 with soft (clause-level) labels'}
DIVERGED = {'q9b-B-s1', 'curve-4b-B', 'curve-9b-B', 'curve-9b-B2', 'moe-A-full'}
WRITERS = [('gpt-6-luna', 'GPT-6 Luna', 'in training'), ('gpt-6.1-sol', 'GPT-6.1 Sol', ''), ('claude-haiku-4-5-20251001', 'Claude Haiku 4.5', ''),
           ('claude-sonnet-5-5', 'Claude Sonnet 5.5', ''), ('claude-opus-5-5', 'Claude Opus 5.5', ''), ('Opus 3', 'Claude Opus 3', '')]
COHORTS = [('original', 'Fiction, reference, reports'), ('expansion', 'Expansion batch'), ('gpt_general', 'General (GPT writers)'),
           ('ml_papers', 'ML paper excerpts'), ('gpt_ml_papers', 'ML paper excerpts (GPT writers)')]
METRICS = [('one', 'One sentence', 'r'), ('two', 'Two sentences', 'r'), ('small', 'Small edits', 'r'), ('para', 'Paragraph', 'r'),
           ('all', 'All edits', 'r'), ('small_auc', 'Small-edit AUROC', 'a'), ('all_auc', 'All-edit AUROC', 'a'), ('v3', 'v3 sentences', 'r'),
           ('standalone', 'Standalone rewrites', 'r'), ('public_auc', 'Public docs AUROC', 'a')]


def agg(tags, epoch):
    out = {}
    for half in ('dev', 'test'):
        out[half] = {}
        for k, _, _ in METRICS:
            v = [runs[t][epoch][half].get(k) for t in tags if epoch in runs.get(t, {}) and runs[t][epoch][half].get(k) is not None]
            out[half][k] = {'mean': st.mean(v), 'sd': st.stdev(v) if len(v) > 1 else None, 'vals': v} if v else None
    return out


runs = {t: {int(e): v for e, v in r.items()} for t, r in runs.items()}
groups = defaultdict(list)
for t in runs:
    m = re.match(r'(q4b|q9b)-([A-Za-z0-9]+)-s(\d)', t)
    if m:
        groups[f'{m.group(1)}-{m.group(2)}'].append(t)
pending = {t.rsplit('-s', 1)[0] for t, s in status.items() if s != 'trained' and t not in runs}
rows = []


def add(section, key, label, tags, epoch, note='', model='', flag=''):
    rows.append({'section': section, 'key': key, 'label': label, 'model': model, 'n': len(tags), 'epoch': epoch, 'note': note, 'flag': flag,
                 'm': agg(tags, epoch) if tags else None})


for g in ['q4b-A', 'q4b-SPH', 'q4b-SPG', 'q4b-T2', 'q4b-T21', 'q4b-T21S', 'q4b-LLE', 'q4b-MIX', 'q4b-Arep']:
    arm = g.split('-')[1]
    if g in groups:
        add('data', g, DATA.get(arm, ARM.get(arm)), groups[g], 2, model='Qwen3.5-4B, 20%')
    elif g in pending:
        add('data', g, DATA[arm], [], 2, model='Qwen3.5-4B, 20%', flag='training')
for t, lab in [('moe-mixA', 'Mix A (AI-heavy: 32% human, 20% small edits), full length, 1e-4'), ('moe-mixB', 'Mix B (human-heavy: 75% human, 4% small edits), full length, 1e-4'), ('moe-T2-p20', 'T2 mix, 20% length, 1e-4'), ('moe-T2-pass1', 'T2 mix, one full-length pass, 1e-4'), ('moe-A-s1', '20% length, 2e-4'), ('moe-A-full-lr1e4', 'full length, 1e-4'), ('moe-A-full', 'full length, 2e-4 (diverged)')]:
    for e in sorted(runs.get(t, {})):
        add('moe', f'{t}-e{e}', f'{lab}, epoch {e}', [t], e, model='Qwen3.6-35B-A3B', flag='diverged' if t in DIVERGED else '')
for size, model in [('q4b', 'Qwen3.5-4B, 20%'), ('q9b', 'Qwen3.5-9B, 20%')]:
    for arm in sorted(ARM):
        g = f'{size}-{arm}'
        if g in groups:
            bad = [t for t in groups[g] if t in DIVERGED]
            add('lr', g, ARM[arm], groups[g], 2, model=model, flag='1 seed diverged' if bad else '')
for t, lab, model in [('curve-4b-A', '2e-4 cosine, seed 1', 'Qwen3.5-4B, full'), ('curve-4b-A2', '2e-4 cosine, seed 2', 'Qwen3.5-4B, full'),
                      ('curve-4b-J', '5e-4 cosine', 'Qwen3.5-4B, full'), ('curve-4b-B', '5e-4 constant', 'Qwen3.5-4B, full'),
                      ('curve-9b-A', '2e-4 cosine', 'Qwen3.5-9B, full'), ('curve-9b-B2', '5e-4 constant', 'Qwen3.5-9B, full')]:
    if t in runs:
        add('curve', t, lab, [t], max(runs[t]), model=model, flag='diverged' if t in DIVERGED else '')

RIVER = {'q4b-A-ref-s1': 'Reference (same recipe as baseline)', 'q4b-a128-s1': 'LoRA alpha 128', 'q4b-a256-s1': 'LoRA alpha 256',
         'q4b-a128-lr5e5-s1': 'LoRA alpha 128, LR 5e-5', 'q4b-r16a32-s1': 'LoRA rank 16, alpha 32', 'q4b-r32a64-s1': 'LoRA rank 32, alpha 64',
         'q4b-r256a512-s1': 'LoRA rank 256, alpha 512', 'q4b-drop05-s1': 'LoRA dropout 0.05', 'q4b-wd0-s1': 'Weight decay 0',
         'q4b-wd01-s1': 'Weight decay 0.1', 'q4b-eb16-s1': 'Effective batch 16', 'q4b-eb64-s1': 'Effective batch 64',
         'q4b-hlr1e3-s1': 'Head LR 1e-3', 'q4b-hlr2e4-s1': 'Head LR 2e-4', 'q4b-warm15-s1': 'Warmup 15%'}
_river = data.get('river', {})
_rr = {}
for tag, cks in _river.items():
    for ck, v in cks.items():
        _rr[f'river:{tag}:{ck}'] = v
for tag, lab in RIVER.items():
    if f'river:{tag}:stage2-epoch2' in _rr:
        runs[f'river:{tag}'] = {2: _rr[f'river:{tag}:stage2-epoch2']}
        add('river', f'river-{tag}', lab, [f'river:{tag}'], 2, model="Qwen3.5-4B, 20%, one seed (River's sweep)")
for tag, lab in [('moe-base-lr1e4-w10-s1', 'base data'), ('moe-mix-lr1e4-w10-s1', "base data + River's curated mix")]:
    for ck in ['step00500', 'step01000', 'stage2-epoch0', 'stage2-epoch1', 'stage2-epoch2']:
        if f'river:{tag}:{ck}' in _rr:
            k = f'river:{tag}:{ck}'; runs[k] = {0: _rr[k]}
            add('river', f'river-{tag}-{ck}', f'{lab}, {ck.replace("stage2-", "").replace("step0", "step ")}', [k], 0, model="Qwen3.6-35B-A3B, 50%, 1e-4 (River's sweep)")

from collections import defaultdict as _dd
_agg = _dd(list)
for key, c in sorted(cross.items()):
    ds, rest = key.split('|'); tag, ep = rest.split('@e')
    base = tag if tag.startswith('moe') else re.sub(r'-s\d$', '', tag)
    _agg[(ds, base, ep)].append(c)
NAMES = {'moe-mixA': 'MoE mix A', 'moe-mixB': 'MoE mix B', 'moe-T2-p20': 'MoE T2 20%', 'moe-T2-pass1': 'MoE T2 one pass', 'q4b-T2': '4B T2 mix', 'q4b-T21': '4B T2.1', 'q4b-T21S': '4B T2.1 soft labels', 'moe-A-s1': 'MoE 20%', 'moe-A-full-lr1e4': 'MoE full 1e-4', 'q4b-SPG': '4B splices (GRADTEX)', 'q4b-SPH': '4B splices (paper pairs)',
         'q4b-Arep': '4B baseline rerun', 'q4b-LLE': '4B LLM edits', 'q4b-MIX': '4B splices + LLM edits'}
def _mean(vals):
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None
xmodels = []
for (ds, base, ep), cs in sorted(_agg.items(), key=lambda x: (x[0][0], not x[0][1].startswith('moe'), x[0][1], x[0][2])):
    keys = set().union(*[c['groups'] for c in cs])
    groups = {k: {f: _mean([(c['groups'].get(k) or {}).get(f) for c in cs]) for f in ('auroc', 'auroc_vs_controls', 'recall_at_1pct_controls', 'recall_at_in_domain_threshold')} for k in keys}
    lab = NAMES.get(base, base) + (f', epoch {ep}' if base.startswith('moe') else '') + (f' ({len(cs)} seeds)' if len(cs) > 1 else '')
    xmodels.append({'key': f'{ds}|{base}@{ep}', 'ds': ds, 'label': lab, 'groups': groups,
                    'doc_auc': _mean([(c.get('documents') or {}).get('auroc_mixed_vs_controls') for c in cs]),
                    'fpr_ctrl': _mean([c.get('fpr_ctrl') for c in cs]), 'fpr_retained': _mean([c.get('fpr_retained') for c in cs]), 'fpr_all': _mean([c.get('fpr_all') for c in cs])})

HW = [('gpt-6-luna', 'GPT-6 Luna', 'in training'), ('claude-haiku-4-5', 'Claude Haiku 4.5', ''), ('claude-sonnet-5-5', 'Claude Sonnet 5.5', ''), ('claude-opus-5-5', 'Claude Opus 5.5', '')]
HT = [('rewrite_one', 'Rewrite one sentence'), ('rewrite_two', 'Rewrite two sentences'), ('insert_one', 'Insert one sentence'), ('split_one', 'Split one sentence')]
payload = {'rows': rows, 'metrics': METRICS, 'xmodels': xmodels, 'writers': WRITERS, 'cohorts': COHORTS, 'hwriters': HW, 'htypes': HT,
           'ab': data.get('ab'), 'mixNotes': json.load(open(D + '/mix-notes.json')) if os.path.exists(D + '/mix-notes.json') else [], 'updated': time.strftime('%-I:%M %p', time.localtime()) + ' PDT, Oct 6'}
html = open(D + '/template.html').read().replace('/*DATA*/null', json.dumps(payload, separators=(',', ':')))
open(D + '/detector-comparison.html', 'w').write(html)
print('rows', len(rows), 'xmodels', len(xmodels), 'bytes', len(html))
