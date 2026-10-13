"""Export the aggregate calibration for the Paper Atlas calibration page (no per-paper data).

Usage: export_calibration.py GROUP_SUMMARY.json
Reads results.json (this directory) and the year-diagnostic group summary; writes
app/lib/model-calibration/qwen35-4b-fast10.json.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / 'app/lib/model-calibration/qwen35-4b-fast10.json'
R = json.loads((HERE / 'results.json').read_text())
groups = json.loads(Path(sys.argv[1]).read_text())
s, d = R['sentence'], R['document']


def curve(name, c, tpr1):
    return {'name': name, 'auc': round(c['auc'], 5), 'n_ai': c['n_ai'], 'n_human': c['n_human'], 'tpr_at_1pct': round(tpr1, 5),
            'fpr': c['fpr'], 'tpr': c['tpr']}


def group_label(g):
    if g.startswith('HUMAN'): return {'label': 'Human papers, 2022 or earlier', 'kind': 'human', 'year': 2022}
    if g.startswith('ICLR 2027'): return {'label': 'ICLR 2027 submissions', 'kind': 'iclr', 'year': 2027}
    parts = g.split('|')
    source = {'archive': 'archive', 'new-download': 'OpenReview download'}[parts[0]]
    venue = {'iclr': 'ICLR', 'tmlr': 'TMLR', 'emnlp': 'EMNLP', 'corl': 'CoRL', 'neurips': 'NeurIPS'}.get(parts[1], parts[1])
    label = f'{venue} {parts[2]}'
    if len(parts) > 3: label += f' ({parts[3]})'
    return {'label': f'{label}, {source}', 'kind': 'iclr' if parts[1] == 'iclr' else 'other', 'year': int(parts[2])}


th = s['thresholds_by_target_fpr']
data = {
    'model': {'id': 'qwen35-4b', 'name': 'Qwen3.5-4B · Experimental',
              'detail': 'LoRA classifier trained on 10% of the training mix (checkpoint stage2-epoch0); 510-token windows without overlap, LoRA merged, BF16.'},
    'calibration_set': {'papers': s['n_human_papers'], 'sentences': s['n_human_sentences'],
                        'description': 'Papers from 2020–2022 in the clean-text archive (human by project rule), after removing 148 papers that share text with the model’s training data.',
                        'method': 'Thresholds are set on a random half of the papers and checked on the other half.'},
    'sentence_score': 'Mean AI probability of the tokens in the sentence (raw model output, not rescaled).',
    'thresholds': [
        {'target_fpr': float(k), 'threshold': round(v['threshold_all_human'], 5), 'heldout_fpr': round(v['heldout_fpr'], 6),
         'heldout_ci95': [round(x, 6) for x in v['heldout_ci95']], 'heldout_sentences': s['heldout_fpr']['n'],
         'stable': float(k) == 0.01 or abs(v['heldout_fpr'] - float(k)) <= 0.5 * float(k)}
        for k, v in sorted(th.items(), key=lambda kv: float(kv[0]))],
    'sentence_curves': [curve('One-sentence edits', s['roc_vs_human_papers']['papers_one'], s['tpr_at_threshold']['papers_one']),
                        curve('Two-sentence edits', s['roc_vs_human_papers']['papers_two'], s['tpr_at_threshold']['papers_two']),
                        curve('Paragraph edits', s['roc_vs_human_papers']['papers_para'], s['tpr_at_threshold']['papers_para'])],
    'document_curves': [curve('Document head', d['document_prob'], d['document_prob']['tpr_at_threshold']),
                        curve('Mean token score', d['token_prob_mean'], d['token_prob_mean']['tpr_at_threshold'])],
    'document_threshold': {'threshold': round(d['document_prob']['threshold_1pct'], 5), 'human_fpr_ci95': d['document_prob']['human_fpr_ci95'],
                           'note': d['document_prob']['note']},
    'years': sorted([{**group_label(g['group']), 'papers': g['n'], 'flagged_share_median': round(g['flag_p50'], 5),
                      'flagged_share_p25': round(g['flag_p25'], 5), 'flagged_share_p75': round(g['flag_p75'], 5),
                      'over_human_p99': round(g['over_h99'], 5), 'document_score_median': round(g['doc_p50'], 5)} for g in groups],
                    key=lambda g: (g['year'], g['kind'] != 'human', g['label'])),
    'caveats': ['The 1% threshold rests on 215 human papers; the 0.1% threshold is unstable at this size.',
                'Flags measure resemblance to AI-generated text. They cannot separate AI use from human writing that has adopted LLM-like phrasing.',
                'This is an early model trained on 10% of the data. Classifications are model outputs, not conclusions about authorship.'],
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(data, separators=(',', ':')))
print('wrote', OUT, OUT.stat().st_size, 'bytes')
