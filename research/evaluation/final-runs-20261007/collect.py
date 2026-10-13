"""Collect the Oct 7 final-run scores, the earlier MoE runs and the T2.1/T2/SPG references into data.json.

Inputs (all local):
  eval/<tag>/*.json      final runs, downloaded from bucket workspace/final-20261007/eval/
  eval/report.json       3-seed reference means (T21, T2, SPG) built by final-20261007/eval_report.py
  ../detector-comparison-20261006/h200/<moe run>/eval/*.json   earlier MoE runs (same scorers)
Recall is at the in-domain threshold (1% FPR on sweep-eval dev human sentences); AUROC on the held-out style sets is
against their human controls, matching eval_report.py.
"""
import json
from pathlib import Path

H = Path(__file__).resolve().parent
OLD = H.parent / 'detector-comparison-20261006/h200'
SETS = ['heldout-writers', 't21-heldout-v2', 'cross-model-s500', 'fable-humanizer', 'fable-cowrite', 'wide1-heldout']


def sweep(path):
    if not path.exists(): return None
    d = json.loads(path.read_text()); t, v = d['test'], d['dev']
    return {'test_small_r': t['edits_small']['recall_at_1pct'], 'test_small_auc': t['edits_small']['auroc'],
            'dev_small_r': v['edits_small']['recall_at_1pct'], 'dev_small_auc': v['edits_small']['auroc'],
            'test_para_r': t['edits_para']['recall_at_1pct'], 'test_standalone_auc': t['standalone_rewrites']['auroc'],
            'test_public_auc': t['public_docs']['auroc'], 'n_test_small': t['edits_small']['n'], 'n_dev_small': v['edits_small']['n']}


def heldout(path):
    if not path.exists(): return None
    d = json.loads(path.read_text()); g = d['groups']
    out = {'recall': g['all:all']['recall_at_in_domain_threshold'], 'auroc': g['all:all'].get('auroc_vs_controls'),
           'n': g['all:all']['n_ai_sentences'], 'control_fpr': d.get('control_fpr_at_in_domain_threshold')}
    out['writers'] = {k.split(':', 1)[1]: {'recall': v['recall_at_in_domain_threshold'], 'auroc': v.get('auroc_vs_controls'), 'n': v['n_ai_sentences']}
                      for k, v in g.items() if k.startswith(('writer:', 'w:'))}
    for k in ('small', 'para'):
        for key in (f'kind:{k}', k):
            if key in g: out[k] = {'recall': g[key]['recall_at_in_domain_threshold'], 'auroc': g[key].get('auroc_vs_controls')}
    return out


def model(base, ck, sweep_ck=None):
    m = {'sweep': sweep(base / f'{sweep_ck or ck}.json')}
    for s in SETS:
        r = heldout(base / f'{s}-{ck}.json')
        if r: m[s] = r
    return m


def ref(r):
    sw = r['sweep']
    m = {'sweep': {'test_small_r': sw['test']['small_recall_1pct'], 'test_small_auc': sw['test']['small_auroc'],
                   'dev_small_r': sw['dev']['small_recall_1pct'], 'dev_small_auc': sw['dev']['small_auroc'],
                   'test_para_r': sw['test']['para_recall_1pct']}}
    for s in SETS:
        x = r.get(s)
        if x and 'all' in x:
            m[s] = {'recall': x['all']['recall'], 'auroc': x['all']['auroc'], 'n': x['all']['n'], 'control_fpr': x.get('control_fpr'),
                    'writers': {k[2:]: v for k, v in x.items() if k.startswith('w:')}}
    return m


models = []
NEW = [('final-moe', 'stage2-epoch2', 'MoE', 'Last checkpoint (epoch 2)'),
       ('final-moe', 'stage2-epoch1', 'MoE', 'Loss-selected (epoch 1)'),
       ('final-qwen35-4b', 'stage2-epoch2', '4B', 'Last checkpoint (epoch 2)'),
       ('final-qwen35-4b', 'stage2-epoch1', '4B', 'Loss-selected (epoch 1)'),
       ('final-eg2', 'stage2-epoch2', 'EmbeddingGemma 2', 'Last checkpoint (epoch 2)'),
       ('final-eg2', 'stage2-epoch1', 'EmbeddingGemma 2', 'Loss-selected (epoch 1)')]
for tag, ck, name, note in NEW:
    models.append({'id': f'{tag}:{ck}', 'group': 'final', 'name': name, 'note': note, **model(H / 'eval' / tag, ck)})
PAST = [('moe-T2-p20', 'stage2-epoch2', 'MoE T2 · 20% × 3 epochs', 'Oct 6, best earlier MoE; T2 mix'),
        ('moe-T2-pass1', 'stage2-epoch0', 'MoE T2 · one full pass', 'Oct 6; T2 mix, one 24k-row pass'),
        ('moe-A-s1', 'stage2-epoch2', 'MoE old mix · 20%', 'Oct 4–5; pre-splice mix'),
        ('moe-A-full-lr1e4', 'stage2-epoch0', 'MoE old mix · full length', 'Oct 5–6; best epoch (overfit after)')]
for tag, ck, name, note in PAST:
    m = model(OLD / tag / 'eval', ck)
    if tag == 'moe-A-s1' and 'cross-model-s500' not in m and (OLD / tag / 'eval/cross-model-s500-stage2-epoch2.json').exists():
        m['cross-model-s500'] = heldout(OLD / tag / 'eval/cross-model-s500-stage2-epoch2.json')
    models.append({'id': f'{tag}:{ck}', 'group': 'past', 'name': name, 'note': note, **m})
R = json.loads((H / 'eval/report.json').read_text())['reference']
for k, name, note in [('T21', 'T2.1 4B', '3-seed mean, Oct 6'), ('T2', 'T2 4B', '3-seed mean, Oct 6'), ('SPG', 'SPG 4B', '3-seed mean, Oct 5–6')]:
    models.append({'id': k, 'group': 'reference', 'name': name, 'note': note, **ref(R[k])})
(H / 'data.json').write_text(json.dumps({'models': models}, indent=1))
for m in models:
    s = m.get('sweep') or {}
    print(f"{m['id']:<34} test {s.get('test_small_r', float('nan')):.3f}  hw {m.get('heldout-writers', {}).get('recall', float('nan')):.3f}  xm {m.get('cross-model-s500', {}).get('recall', float('nan')):.3f}")
