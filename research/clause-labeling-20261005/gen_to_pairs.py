"""Convert generated edit paragraphs (gen_synthetic.py --production) into relabel pairs.

Each accepted paragraph is put back into its document context: for gap10000 sources the human
window from the dataset (paragraph replaced in place), for candidate-pool sources the stored
before/after paragraphs. The AI region is the whole edited paragraph (every clause gets a soft
label, as in the report); the exact construction labels go in `truth_regions`
(0 human, 1 assisted, 2 generated) in document offsets.

Usage: gen_to_pairs.py GEN_PAIRS.jsonl NAME OUT.jsonl [--gap-dataset DATASET.jsonl]
"""
import argparse, ast, json

CODE = {'human': 0, 'assisted': 1, 'generated': 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('gen'); ap.add_argument('name'); ap.add_argument('out'); ap.add_argument('--gap-dataset')
    a = ap.parse_args()
    windows = {}
    if a.gap_dataset:
        for r in map(json.loads, open(a.gap_dataset)):
            if r['operation'] == 'paragraph_generate':
                reg = next(x for x in (r['regions'] if isinstance(r['regions'], list) else ast.literal_eval(r['regions']))
                           if x['label'] == 'ai_rewritten')
                windows[r['passage_id']] = (reg['source_start'], reg['source_end'])
        human = {}
        for r in map(json.loads, open(a.gap_dataset)):
            if r['operation'] == 'human_original':
                human[r['passage_id']] = r['text']
    n = 0
    with open(a.out, 'w') as f:
        for p in map(json.loads, open(a.gen)):
            if p['state'] != 'ok' or not p['plan']:
                continue
            if 'context_before' in p:
                before, after = p['context_before'] + '\n\n', '\n\n' + p['context_after']
            else:
                lo, hi = windows[p['passage_id']]; doc = human[p['passage_id']]
                assert doc[lo:hi] == p['source'], p['passage_id']
                before, after = doc[:lo], doc[hi:]
            S = before + p['source'] + after; T = before + p['target'] + after
            o = len(before)
            truth = [{'start': 0, 'end': o, 'label': 0}] + \
                    [{'start': t['start'] + o, 'end': t['end'] + o, 'label': CODE[t['label']], 'op': t['op']} for t in p['truth']] + \
                    [{'start': o + len(p['target']), 'end': len(T), 'label': 0}]
            f.write(json.dumps({'id': f"{p['passage_id']}/{a.name}", 'dataset': a.name, 'split': p['split'], 'paper_id': p['paper_id'],
                                'never_train': False, 'edit_type': p['budget'], 'T': T, 'S': S,
                                'region': [o, o + len(p['target'])], 't_par': [o, o + len(p['target'])],
                                's_par': [o, o + len(p['source'])], 'truth_regions': [t for t in truth if t['end'] > t['start']],
                                'edits': [{k: e[k] for k in ('i', 'op', 'similarity')} for e in p['edits']],
                                'generator': p.get('model'), 'service_tier': p.get('reported_service_tier')}, ensure_ascii=False) + '\n')
            n += 1
    print(n, 'pairs ->', a.out)


if __name__ == '__main__':
    main()
