"""Mainline candidates page data: the three full-size 35B MoE runs (T2.1 72k = the final MoE, mixA, mixB) plus the Oct 8
Qwen3.5-4B T2.1 single-epoch run as a reference, every one scored on the FAST path (FAST_EVAL=1 eval_final.py, H200).
Inputs: ../moe-slow-fast-20261008/fast/<tag>/ (MoE runs) and fast/<tag>/ here (4B; rsynced from H200 sweeps/fast-<tag>/eval/).
Per test: recall at the model's in-domain 1%-FPR threshold, AUROC, human false-positive rate on the set's human controls,
plus per-writer and per-cohort breakdowns. -> data.json"""
import json
from pathlib import Path
H = Path(__file__).resolve().parent; MF = H.parent / 'moe-slow-fast-20261008/fast'
MODELS = [
    {'tag': 'final-moe', 'ck': 'stage2-epoch2', 'name': 'T2.1 72k', 'sub': 'Final MoE · Atlas default', 'rows': 72000, 'dir': MF / 'final-moe', 'main': True},
    {'tag': 'moe-mixA-full', 'ck': 'stage2-epoch0', 'name': 'mixA', 'sub': '145.6k rows, one pass', 'rows': 145569, 'dir': MF / 'moe-mixA-full', 'main': True},
    {'tag': 'moe-mixB-full', 'ck': 'stage2-epoch0', 'name': 'mixB', 'sub': '577.9k rows, one pass', 'rows': 577869, 'dir': MF / 'moe-mixB-full', 'main': True},
    {'tag': 'q4b-t21-1ep', 'ck': 'stage2-epoch0', 'name': '4B · T2.1 72k', 'sub': 'Qwen3.5-4B reference, one pass', 'rows': 72000, 'dir': H / 'fast/q4b-t21-1ep', 'main': False},
]
SWEEP = [('edits_small', 'Small edits (1–2 sentences)'), ('edits_one', 'One-sentence edits'), ('edits_two', 'Two-sentence edits'),
         ('edits_para', 'Paragraph edits'), ('standalone_rewrites', 'Standalone rewrites'), ('public_docs', 'Public documents')]
SETS = [('heldout-writers', 'Held-out writers', 'Small edits by Claude and GPT writers held out of training'),
        ('wide1-heldout', 'wide1 held-out', 'DeepSeek (held-out papers), GLM and MiMo (never trained on)'),
        ('t21-heldout-v2', 'T2.1 held-out', 'Edits, paragraphs and sections from the T2.1 held-out papers'),
        ('cross-model-s500', 'Cross-model', 'Full-text rewrites by six models'),
        ('fable-humanizer', 'Fable humanizer', 'Claude text before and after Fable 5.1 humanizing'),
        ('fable-cowrite', 'Fable co-writing', 'Three rounds of human–Fable co-writing')]


def sweep(m, split):
    p = m['dir'] / f"{m['ck']}.json"
    if not p.exists(): return None
    d = json.loads(p.read_text())[split]
    return {k: {'r': d[k].get('recall_at_1pct'), 'auc': d[k].get('auroc'), 'n': d[k].get('n')} for k, _ in SWEEP}


def heldout(m, s):
    p = m['dir'] / f"{s}-{m['ck']}.json"
    if not p.exists(): return None
    d = json.loads(p.read_text()); g = d['groups']
    one = lambda v: {'r': v['recall_at_in_domain_threshold'], 'auc': v.get('auroc_vs_controls'), 'n': v['n_ai_sentences']}
    return {'all': one(g['all:all']), 'cfpr': d.get('control_fpr_at_in_domain_threshold'), 'n_ctrl': d.get('n_control_sentences'),
            'writers': {k.split(':', 1)[1]: one(v) for k, v in g.items() if k.startswith('writer:')},
            'cohorts': {k.split(':', 1)[1]: one(v) for k, v in g.items() if k.startswith('cohort:')}}


out = {'models': [], 'sweep_tests': SWEEP, 'sets': SETS}
for m in MODELS:
    r = {k: v for k, v in m.items() if k != 'dir'}
    r['scored'] = (m['dir'] / 'summary.json').exists()
    r['sweep'] = {s: sweep(m, s) for s in ('test', 'dev')}
    r['sets'] = {s: heldout(m, s) for s, _, _ in SETS}
    out['models'].append(r)
(H / 'data.json').write_text(json.dumps(out, indent=1))
print({m['tag']: m['scored'] for m in out['models']})
