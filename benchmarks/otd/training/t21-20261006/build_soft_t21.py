"""Soft-label arm for T2.1: relabel inputs for the paired rows T2.1 trains on that have no soft labels yet.

Existing soft rows (soft-rows-existing.jsonl.gz) cover the Luna rows (gap10000 rewrites, llm-edits-v1, splices).
This adds the Claude/GPT small edits, the Claude/GPT paragraph edits and the Claude sections, restricted to the
ids in the first 25% of each key per stage-2 epoch of the T2.1 prepared dir (the trainer uses the first 20%).
Pairs follow build_relabel_inputs.py; paragraph edits and sections pair with their matched human row.

Usage: build_soft_t21.py USED_IDS.json OUT_DIR [N_SHARDS]
Writes OUT_DIR/pairs.jsonl (for make_soft_rows.py), OUT_DIR/compact-{i}.jsonl (for space_relabel.py), coverage.json.
"""
import ast, gzip, json, sys
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parents[4] / 'research'
sys.path.insert(0, str(R / 'clause-labeling-20261005'))
from build_relabel_inputs import from_edits, par_bounds
from build_llm_edits import never_train_hit

lit = lambda v: v if isinstance(v, list) else ast.literal_eval(v)


def from_matched(path, ai_ds, hu_ds, key, stats):
    rows = [json.loads(l) for l in gzip.open(path, 'rt')]
    hum = {r[key]: r for r in rows if r['dataset'] == hu_ds}
    for r in rows:
        if r['dataset'] != ai_ds: continue
        ai = [g for g in lit(r['regions']) if g['label'] == 1]
        a, b = ai[0]['start'], ai[-1]['end']; T = r['text']; h = hum.get(r['id'] if key == 'matched_id' else r[key])
        if len(ai) != 1 or h is None or h['text'][:a] != T[:a] or not h['text'].endswith(T[b:]):
            stats[f'{ai_ds}:unpaired'] += 1; continue
        S = h['text']; sb = len(S) - (len(T) - b)
        yield {'id': r['id'], 'dataset': ai_ds, 'split': r.get('split', 'train'), 'paper_id': r['paper_id'], 'T': T, 'S': S,
               'region': [a, b], 't_par': par_bounds(T, a, b), 's_par': par_bounds(S, a, sb),
               'edit_type': r.get('item_kind'), 'generator': r.get('generator')}


def main():
    used, out = json.load(open(sys.argv[1])), Path(sys.argv[2]); n_sh = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    out.mkdir(parents=True, exist_ok=True); stats = Counter()
    want = {i.split('@')[0] for ep in used.values() for k, ids in ep.items() if k in ('edit_claude', 'edit_gpt', 'para_e', 'sections_ai') for i in ids}
    luna = {i.split('@')[0] for ep in used.values() for k, ids in ep.items() if k in ('edit_luna', 'para_luna') for i in ids}
    have = {json.loads(l)['source_id'] for l in gzip.open(R / 'data/soft-relabel-20261006/soft-rows-existing.jsonl.gz', 'rt')}
    stats['luna_used'] = len(luna); stats['luna_covered_by_existing'] = len(luna & have)
    gens = [from_edits(R / 'llm-sentence-edits-20261005/llm-edits-claude-v1.jsonl.gz', 'claude_edit', stats),
            from_edits(R / 'claude-hosted-edits-20261006/llm-edits-claude-hsm-v1.jsonl.gz', 'hsm_edit', stats),
            from_matched(R / 'claude-hosted-edits-20261006/paragraph-edits-claude-hsm-v1-train.jsonl.gz', 'hsm_paragraph_edit', 'hsm_paragraph_edit_human', 'matched_id', stats),
            from_matched(R / 'claude-sections-20261006/claude-sections-v1-train.jsonl.gz', 'papers_section', 'papers_section_human', 'item_id', stats)]
    pairs, seen = [], set()
    for g in gens:
        for p in g:
            if p['id'] in want and p['id'] not in seen:
                p['never_train'] = bool(never_train_hit({'paper_id': p['paper_id'], 'text': p['T']}))
                seen.add(p['id']); pairs.append(p); stats[f"{p['dataset']}:ok"] += 1
    stats['wanted'] = len(want); stats['missing'] = len(want - seen)
    with open(out / 'pairs.jsonl', 'w') as f:
        for p in pairs: f.write(json.dumps(p, ensure_ascii=False) + '\n')
    pairs.sort(key=lambda p: -(p['t_par'][1] - p['t_par'][0]))  # deal long rows round-robin so shards finish together
    sh = [open(out / f'compact-{i}.jsonl', 'w') for i in range(n_sh)]
    for k, p in enumerate(pairs):
        (a, b), (sa, sb) = p['t_par'], p['s_par']
        sh[k % n_sh].write(json.dumps({'id': p['id'], 't_par_text': p['T'][a:b], 't_off': a, 'region_rel': [p['region'][0] - a, p['region'][1] - a],
                                      's_par_text': p['S'][sa:sb]}, ensure_ascii=False) + '\n')
    (out / 'coverage.json').write_text(json.dumps(dict(sorted(stats.items())), indent=1)); print(json.dumps(dict(sorted(stats.items())), indent=1))


if __name__ == '__main__':
    main()
