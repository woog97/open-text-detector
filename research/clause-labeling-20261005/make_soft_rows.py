"""Turn relabel units (space_relabel.py) into training rows with soft n-gram labels.

Inside each pair's AI region, every spaCy clause unit gets Human / AI-Assisted / AI-Generated from
L (lexical containment in the source paragraph) and E (embedding similarity), using the thresholds
tuned on the synthetic test (spaCy splitter, report convention: light polish counts as Human):
  Human if L >= 0.72; AI-Assisted if L >= 0.35 or E >= 0.70; else AI-Generated.
The trainer's labels are binary, so `regions` maps Human -> 0 and Assisted/Generated -> 1; text outside
the AI region keeps label 0. The three-way labels are kept in `soft_regions`
(0 human, 1 assisted, 2 generated) for a future three-class head.

Usage: make_soft_rows.py PAIRS.jsonl UNITS.jsonl OUT.jsonl.gz [STATS.json]
PAIRS rows come from build_relabel_inputs.py (T, S, region, s_par, ...).
"""
import gzip, json, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'llm-sentence-edits-20261005'))
import label_eval as le
from build_llm_edits import never_train_hit  # re-checked at build time: the list can grow after generation

TH = (0.72, 0.35, 0.70)
NAMES = ('human', 'assisted', 'generated')


def label_units(t_text, units, E, s_text):
    sg = le.grams(s_text); le._NORM[id(sg)] = le.norm(s_text)
    L = [le.lexical(t_text[a:b], sg) for a, b in units]
    return le.predict(units, L, E, TH), L


def main():
    pairs_path, units_path, out_path = sys.argv[1:4]
    stats_path = sys.argv[4] if len(sys.argv) > 4 else None
    units = {r['id']: r for r in map(json.loads, open(units_path))}
    stats = defaultdict(Counter); n = 0
    with gzip.open(out_path, 'wt') as out:
        for p in map(json.loads, open(pairs_path)):
            u = units.get(p['id'])
            if u is None:
                stats['_missing']['rows'] += 1; continue
            T, (ra, rb) = p['T'], p['region']
            sa, sb = p['s_par']
            labs, L = label_units(T, u['t_units'], u['E'], p['S'][sa:sb])
            # Start from the original binary labels, then overwrite each unit inside the region.
            soft = [(0, ra, 0), (ra, rb, 2), (rb, len(T), 0)]
            for (a, b), v in zip(u['t_units'], labs):
                soft.append((a, b, v))
            chars = [0] * len(T)
            for a, b, v in soft:
                chars[a:b] = [v] * (b - a)
            spans = []
            for i, v in enumerate(chars):
                if spans and spans[-1][2] == v and spans[-1][1] == i:
                    spans[-1][1] = i + 1
                else:
                    spans.append([i, i + 1, v])
            soft_regions = [{'start': a, 'end': b, 'label': v} for a, b, v in spans]
            binary = []
            for s in soft_regions:
                v = int(s['label'] > 0)
                if binary and binary[-1]['label'] == v:
                    binary[-1]['end'] = s['end']
                else:
                    binary.append({'start': s['start'], 'end': s['end'], 'label': v})
            key = f"{p['dataset']}:{p.get('edit_type', 'paragraph')}"
            for a, b, v in spans:
                if ra <= a and b <= rb:
                    stats[key][NAMES[v]] += sum(1 for ch in T[a:b] if not ch.isspace())
            stats[key]['rows'] += 1
            stats[key]['rows_any_human_inside'] += any(v == 0 for v in labs)
            out.write(json.dumps({
                'id': p['id'] + '/soft', 'source_id': p['id'], 'paper_id': p['paper_id'], 'group': f"paper:{p['paper_id']}",
                'dataset': p['dataset'] + '_soft', 'split': p['split'], 'never_train': bool(p['never_train'] or never_train_hit({'paper_id': p['paper_id'], 'text': T})),
                'edit_type': p.get('edit_type'), 'text': T, 'target_start': ra, 'target_end': rb,
                'regions': binary, 'soft_regions': soft_regions,
                'soft_units': [{'start': a, 'end': b, 'L': round(l, 4), 'E': e, 'label': NAMES[v]}
                               for (a, b), l, e, v in zip(u['t_units'], L, u['E'], labs)],
                'truth_regions': p.get('truth_regions'), 'edits': p.get('edits'), 'generator': p.get('generator'),
                'labeler': {'splitter': 'spacy en_core_web_trf 3.8 (split_spacy.py)', 'embedding': 'Qwen/Qwen3-Embedding-0.6B',
                            'thresholds': dict(zip(('human_L', 'assisted_L', 'assisted_E'), TH))},
            }, ensure_ascii=False) + '\n')
            n += 1
    summary = {}
    for k, c in sorted(stats.items()):
        tot = sum(c[x] for x in NAMES)
        summary[k] = dict(rows=c['rows'], rows_with_human_clause_inside=c['rows_any_human_inside'],
                          **{f'{x}_share': round(c[x] / tot, 4) for x in NAMES} if tot else {})
    print(json.dumps(summary, indent=1)); print(n, 'rows ->', out_path)
    if stats_path:
        Path(stats_path).write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
