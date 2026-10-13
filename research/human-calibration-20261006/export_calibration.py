"""Export the expanded calibration to the Paper Atlas calibration page (aggregates only, no per-paper data).

Usage: export_calibration.py RESULTS.json GROUP_SUMMARY.json
  RESULTS.json        output of analyze_calibration.py
  GROUP_SUMMARY.json  output of year_groups_space.py at the new thresholds
Writes app/lib/model-calibration/qwen35-4b-fast10.json (same schema as the Oct 5 export, plus threshold
intervals and a table of candidate models).
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / 'app/lib/model-calibration/qwen35-4b-fast10.json'
R = json.loads(Path(sys.argv[1]).read_text())
groups = json.loads(Path(sys.argv[2]).read_text())
A = R['models']['atlas-fast10']; s, d = A['sentence'], A['document']
VENUE = {'iclr': 'ICLR', 'tmlr': 'TMLR', 'emnlp': 'EMNLP', 'corl': 'CoRL', 'neurips': 'NeurIPS'}
CANDIDATES = {'q4b-T2-s1': ('T2 mix (seed 1)', 'Small edits from Luna, Claude and GPT, paragraph edits, AI sections and Claude full papers'),
              'q4b-SPG-s1': ('SPG mix (seed 1)', 'Original mix with every GRADTEX row replaced by splices (single AI edits inside human text)')}
r5 = lambda v: round(v, 5)


def curve(name, c):
    return {'name': name, 'auc': r5(c['auc']), 'n_ai': c['n_ai'], 'n_human': c['n_human'], 'tpr_at_1pct': r5(c['tpr_at_threshold']),
            'fpr': c['fpr'], 'tpr': c['tpr']}


def year_label(g):
    if g.startswith('ICLR 2027'): return {'label': 'ICLR 2027 submissions', 'kind': 'iclr', 'year': 2027}
    source, venue, year = g.split('|')[:3]
    src = {'archive': 'archive', 'openreview-2026-10': 'OpenReview download'}.get(source, source)
    return {'label': f'{VENUE.get(venue, venue)} {year}, {src}', 'kind': 'iclr' if venue == 'iclr' else 'other', 'year': int(year)}


t1 = next(t for t in s['thresholds'] if t['target_fpr'] == 0.01)
human_rows = [{'label': f"{VENUE.get(g['venue'], g['venue'])} {g['year']}" if g['venue'] != 'archive' else 'Oct 5 set (archive, ≤2022)',
               'kind': 'human', 'year': g['year'], 'papers': g['papers'], 'flagged_share_median': r5(g['flagged_share_p50']),
               'flagged_share_p25': r5(g['flagged_share_p25']), 'flagged_share_p75': r5(g['flagged_share_p75']),
               'over_human_p99': None, 'document_score_median': r5(g['document_prob_p50']) if g['document_prob_p50'] is not None else None}
              for g in A['groups']]
other_rows = [{**year_label(g['group']), 'papers': g['n'], 'flagged_share_median': r5(g['flag_p50']), 'flagged_share_p25': r5(g['flag_p25']),
               'flagged_share_p75': r5(g['flag_p75']), 'over_human_p99': r5(g['over_h99']), 'document_score_median': r5(g['doc_p50'])}
              for g in groups if not g['group'].startswith('HUMAN')]
ov = R['overlap']
data = {
    'model': {'id': 'qwen35-4b', 'name': 'Qwen3.5-4B · Experimental',
              'detail': 'LoRA classifier trained on 10% of the training mix (checkpoint stage2-epoch0); 510-token windows without overlap, LoRA merged, BF16.'},
    'calibration_set': {
        'papers': s['n_papers'], 'sentences': s['n_sentences'],
        'description': (f"{s['n_papers']:,} human papers (human by project rule: 2022 or earlier). {ov['kept']:,} are ICLR 2019–2022, NeurIPS 2021–2022 "
                        f"and CoRL 2021–2022 papers sampled by venue and year from the site's PDF collection, after removing {ov['excluded']} that share "
                        f"text with a scored model’s training data; {s['n_papers'] - ov['kept']} are the 2020–2022 archive papers used before October 7."),
        'method': ('Thresholds are set on all human papers. Each is checked by setting it on a random half of the papers and measuring the flag rate on '
                   'the other half. Flags cluster within papers, so every interval resamples whole papers rather than treating sentences as independent.')},
    'sentence_score': 'Mean AI probability of the tokens in the sentence (raw model output, not rescaled).',
    'thresholds': [{'target_fpr': t['target_fpr'], 'threshold': r5(t['threshold']), 'threshold_ci95': [r5(x) for x in t['threshold_ci95']],
                    'heldout_fpr': round(t['heldout_fpr'], 6), 'heldout_ci95': [round(x, 6) for x in t['heldout_ci95_paper_bootstrap']],
                    'heldout_sentences': t['heldout_sentences'], 'heldout_papers': t['heldout_papers'],
                    'stable': t['heldout_ci95_paper_bootstrap'][0] <= t['target_fpr'] <= t['heldout_ci95_paper_bootstrap'][1]
                              and t['heldout_ci95_paper_bootstrap'][1] <= 2 * t['target_fpr']}
                   for t in s['thresholds']],
    'sentence_curves': [curve('One-sentence edits', A['sentence_roc']['papers_one']), curve('Two-sentence edits', A['sentence_roc']['papers_two']),
                        curve('Paragraph edits', A['sentence_roc']['papers_para'])],
    'document_curves': [curve('Document head', A['document_roc']['document_prob']), curve('Mean token score', A['document_roc']['token_prob_mean'])],
    'document_threshold': {'threshold': r5(d['document_prob']['threshold_1pct']), 'threshold_ci95': [r5(x) for x in d['document_prob']['threshold_ci95']],
                           'human_fpr_ci95': [round(x, 5) for x in d['document_prob']['heldout_ci95']], 'heldout_fpr': round(d['document_prob']['heldout_fpr'], 5),
                           'note': d['document_prob']['note']},
    'years': sorted(human_rows, key=lambda g: (g['year'], g['label'])) + sorted(other_rows, key=lambda g: (g['year'], g['kind'] != 'human', g['label'])),
    'candidates': [{'id': k, 'name': CANDIDATES[k][0], 'detail': CANDIDATES[k][1], 'papers': m['sentence']['n_papers'],
                    'threshold_1pct': r5(next(t['threshold'] for t in m['sentence']['thresholds'] if t['target_fpr'] == 0.01)),
                    'threshold_ci95': [r5(x) for x in next(t['threshold_ci95'] for t in m['sentence']['thresholds'] if t['target_fpr'] == 0.01)],
                    'heldout_fpr': round(next(t['heldout_fpr'] for t in m['sentence']['thresholds'] if t['target_fpr'] == 0.01), 6),
                    'heldout_ci95': [round(x, 6) for x in next(t['heldout_ci95_paper_bootstrap'] for t in m['sentence']['thresholds'] if t['target_fpr'] == 0.01)],
                    'document_threshold': r5(m['document']['document_prob']['threshold_1pct']),
                    'document_threshold_ci95': [r5(x) for x in m['document']['document_prob']['threshold_ci95']]}
                   for k, m in R['models'].items() if k in CANDIDATES],
    'caveats': [f"Thresholds rest on {s['n_papers']:,} human papers from 2019–2022, mostly machine learning venues. Topics and style differ from 2025–2027 writing, so false-positive rates on recent human papers may be higher than shown.",
                'Flags measure resemblance to AI-generated text. They cannot separate AI use from human writing that has adopted LLM-like phrasing.',
                'This is an early model trained on 10% of the data. Classifications are model outputs, not conclusions about authorship.'],
}
OUT.write_text(json.dumps(data, separators=(',', ':')))
print('wrote', OUT, OUT.stat().st_size, 'bytes')
