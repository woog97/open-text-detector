"""Slow vs fast scoring path for every Qwen3.6 35B-A3B MoE run, and a like-for-like comparison of the runs -> data.json.

Slow path: sweep_eval / cross_model_eval as used for every result so far (BF16, padded 8-window batches, all heads).
Fast path: FAST_EVAL=1 eval_final.py (fast_eval.py: packed windows, token head only, routed-token expert LoRA, unmerged),
scored on the H200 on Oct 8 (GPUs 2+3) from sweeps/fast-<tag>/, all on the same base weights.
Inputs:
  ../final-runs-20261007/eval/<tag>/          slow, scored Oct 7-8 on the H200 with eval_final.py (final-moe, mixA, mixB)
  ../detector-comparison-20261006/h200/<tag>/eval/   slow, scored Oct 5-6 (older runs; sweep + heldout-writers + cross-model only)
  fast/<tag>/                                 fast, rsynced from the H200 sweeps/fast-<tag>/eval/
  agreement.json                              per-sentence slow-vs-fast score agreement (agreement.py on the H200)
"""
import json
from pathlib import Path

H = Path(__file__).resolve().parent
FR = H.parent / 'final-runs-20261007/eval'; DC = H.parent / 'detector-comparison-20261006/h200'
RUNS = [  # tag, checkpoint, name, note, slow dir, slow scored when
    ('moe-mixA-full', 'stage2-epoch0', 'mixA full', 'Oct 8 · 145.6k rows, one pass', FR, 'Oct 8, H200'),
    ('moe-mixB-full', 'stage2-epoch0', 'mixB full', 'Oct 8 · 577.9k rows, one pass', FR, 'Oct 8, H200'),
    ('final-moe', 'stage2-epoch2', 'Final MoE', 'Oct 7 · 72k final mix; Atlas default', FR, 'Oct 7, H200'),
    ('moe-T2-p20', 'stage2-epoch2', 'T2 · 20% × 3 epochs', 'Oct 6 · T2 mix', DC, 'Oct 6'),
    ('moe-T2-pass1', 'stage2-epoch0', 'T2 · one pass', 'Oct 6 · T2 mix, 24k rows', DC, 'Oct 6'),
    ('moe-A-s1', 'stage2-epoch2', 'Old mix · 20%', 'Oct 4–5 · pre-splice mix', DC, 'Oct 5'),
    ('moe-A-full-lr1e4', 'stage2-epoch0', 'Old mix · full length', 'Oct 5–6 · best epoch', DC, 'Oct 6'),
]
SETS = [('heldout-writers', 'Held-out writers'), ('wide1-heldout', 'wide1 held-out (DeepSeek, GLM, MiMo)'), ('t21-heldout-v2', 'T2.1 held-out'),
        ('cross-model-s500', 'Cross-model'), ('fable-humanizer', 'Fable humanizer'), ('fable-cowrite', 'Fable co-writing')]
SW = [('edits_small', 'Small edits'), ('edits_para', 'Paragraph edits'), ('standalone_rewrites', 'Standalone rewrites'), ('public_docs', 'Public documents')]


def sweep(p):
    if not p.exists(): return None
    d = json.loads(p.read_text())
    return {s: {k: {'r': d[s][k]['recall_at_1pct'], 'auc': d[s][k]['auroc'], 'n': d[s][k].get('n')} for k, _ in SW} for s in ('dev', 'test')}


def heldout(p):
    if not p.exists(): return None
    d = json.loads(p.read_text()); g = d['groups']
    return {'r': g['all:all']['recall_at_in_domain_threshold'], 'auc': g['all:all'].get('auroc_vs_controls'), 'n': g['all:all']['n_ai_sentences'],
            'cfpr': d.get('control_fpr_at_in_domain_threshold'), 'n_ctrl': d.get('n_control_sentences'),
            'writers': {k.split(':', 1)[1]: {'r': v['recall_at_in_domain_threshold'], 'auc': v.get('auroc_vs_controls'), 'n': v['n_ai_sentences']}
                        for k, v in g.items() if k.startswith('writer:')}}


def scores(base, ck):
    if not base.exists(): return None
    out = {'sweep': sweep(base / f'{ck}.json')}
    for s, _ in SETS:
        r = heldout(base / f'{s}-{ck}.json')
        if r: out[s] = r
    return out


models = []
for tag, ck, name, note, sdir, when in RUNS:
    m = {'tag': tag, 'ck': ck, 'name': name, 'note': note, 'slow_when': when,
         'slow': scores(sdir / tag / ('eval' if sdir == DC else ''), ck), 'fast': scores(H / 'fast' / tag, ck)}
    if tag in ('moe-mixA-full', 'moe-mixB-full'):  # learning curve (slow path): checkpoints at 1/3, 2/3 and the end of the pass
        m['curve'] = [(p.stem, sweep(p)) for p in sorted((FR / tag).glob('step*.json')) if '-' not in p.stem] + [(ck, sweep(FR / tag / f'{ck}.json'))]
    models.append(m)
ag = H / 'agreement.json'
out = {'models': models, 'sets': SETS, 'sweep_metrics': SW, 'agreement': json.loads(ag.read_text()) if ag.exists() else None}
(H / 'data.json').write_text(json.dumps(out, indent=1))
for m in models:
    f, s = (m['fast'] or {}).get('sweep'), (m['slow'] or {}).get('sweep')
    print(f"{m['tag']:<18} slow test small {s and s['test']['edits_small']['r']}  fast {f and f['test']['edits_small']['r']}  fast sets {len([k for k in (m['fast'] or {}) if k != 'sweep'])}")
