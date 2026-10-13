"""T2.1 held-out split (user decision, Oct 6 evening). v2 (7:45 PM): re-run after more Fable data landed; the v1 held-out papers
stay held out (asserted), new Fable papers are added, held-out sections are now evaluation rows, and rows carry their source.
Outputs get a -v2 suffix; v1 files are kept because T2.1 was built against them.

- Hold out 15% of the papers that host Claude Opus 5.5 or Sonnet 5.5 small edits (random by paper).
- Keep GPT-6 Sol and Claude Fable 5.1 entirely out of training: every paper hosting one of their rows is held out.
- Every row on a held-out paper (any writer, any pool) is dropped from training via t21-heldout-never-train.json
  (paper ids + hashed shingles of the human text of every held-out row, so re-hosted copies under other id schemes match).

Writes, next to this file:
  t21-heldout-never-train.json   same format as the other never-train lists (never_train_hit loads it)
  t21-heldout-score-rows.jsonl.gz rows in the heldout-score-rows format (human_controls + mixed) for cross_model_eval.py
  t21-split-stats.json
"""
import ast, collections, gzip, hashlib, json, random, re, time
from pathlib import Path

HERE = Path(__file__).resolve().parent; P = HERE.parent
FRACTION, SEED = 0.15, 20261006
UNSEEN = {'openai/gpt-6-sol', 'claude-fable-5-1'}
SLICED = {'claude-opus-5-5', 'claude-sonnet-5-5'}


def rd(f):
    return [json.loads(l) for l in gzip.open(P / f, 'rt')]


def regs(r):
    x = r['regions']; x = ast.literal_eval(x) if isinstance(x, str) else x
    return [{'start': int(g['start']), 'end': int(g['end']), 'label': int(g['label'])} for g in x]


small = rd('llm-sentence-edits-20261005/llm-edits-claude-v1.jsonl.gz') + rd('claude-hosted-edits-20261006/llm-edits-claude-hsm-v1.jsonl.gz')
para_all = rd('claude-hosted-edits-20261006/paragraph-edits-claude-hsm-v1-train.jsonl.gz')
para = [r for r in para_all if r['dataset'] == 'hsm_paragraph_edit']
para_human = {r['host_id']: r for r in para_all if r['dataset'] == 'hsm_paragraph_edit_human'}
sections = rd('claude-sections-20261006/claude-sections-v1-train.jsonl.gz')
V1 = {p.split(':', 1)[-1] for p in json.loads((HERE / 't21-heldout-never-train.json').read_text())['paper_ids']}

# 1. Held-out papers
sliced_papers = sorted({r['paper_id'] for r in small if r['generator'] in SLICED})
rng = random.Random(SEED); rng.shuffle(sliced_papers)
held = set(sliced_papers[:round(FRACTION * len(sliced_papers))])
n_sliced = len(held)
for r in small + para + sections:
    if (r.get('generator') or r.get('model')) in UNSEEN: held.add(r['paper_id'])
assert V1 <= held, len(V1 - held)
new_papers = held - V1

# 2. Evaluation rows: every AI edit on a held-out paper, plus the human original of each host paragraph
def original(r):
    """Human version of a small-edit row: the AI span(s) replaced by the human text they replaced."""
    t = r['text']; ai = [g for g in regs(r) if g['label'] == 1]
    a, b = min(g['start'] for g in ai), max(g['end'] for g in ai)
    return t[:a] + (r.get('replaced_human') or '') + t[b:]

rows, humans, mismatch = [], {}, 0
for r in small:
    if r['paper_id'] not in held: continue
    host = r.get('host_id') or r['id'].rsplit('/', 1)[0]
    o = original(r)
    if host in humans and re.sub(r'\s+', '', humans[host]) != re.sub(r'\s+', '', o): mismatch += 1
    humans.setdefault(host, o)
    rows.append({'id': r['id'], 'text': r['text'], 'regions': regs(r), 'slot': 'mixed', 'split': 'heldout', 'label': 'mixed',
                 'condition': r['edit_type'], 'writer': r['generator'].replace('openai/', ''), 'paper_id': r['paper_id'],
                 'source': r.get('source') if r.get('source') in ('pmc', 'pes2o', 'arxiv') else 'paired-ml', 'new_in_v2': r['paper_id'] in new_papers})
for r in para:
    if r['paper_id'] not in held: continue
    rows.append({'id': r['id'], 'text': r['text'], 'regions': regs(r), 'slot': 'mixed', 'split': 'heldout', 'label': 'mixed',
                 'condition': 'paragraph_' + r['item_kind'], 'writer': r['generator'].replace('openai/', ''), 'paper_id': r['paper_id'],
                 'source': r.get('source'), 'new_in_v2': r['paper_id'] in new_papers})
    h = para_human.get(r['host_id'])
    if h: humans.setdefault(r['host_id'] + '#para', h['text'])
sec_h = {r['matched_item_id']: r for r in sections if r['dataset'] == 'papers_section_human'}
for r in sections:
    if r['dataset'] != 'papers_section' or r['paper_id'] not in held: continue
    rows.append({'id': r['id'], 'text': r['text'], 'regions': regs(r), 'slot': 'mixed', 'split': 'heldout', 'label': 'mixed',
                 'condition': 'section_' + r['item_kind'], 'writer': r['generator'].replace('openai/', ''), 'paper_id': r['paper_id'],
                 'source': 'paired-ml' if r.get('source') == 'paired' else r.get('source'), 'new_in_v2': r['paper_id'] in new_papers})
    h = sec_h.get(r['item_id'])
    if h: humans.setdefault(h['id'].rsplit('/', 1)[0] + '#section', h['text'])
pid = {r['id'].rsplit('/', 1)[0]: r['paper_id'] for r in rows}
src = {r['id'].rsplit('/', 1)[0]: r['source'] for r in rows}
hum = [{'id': f'{h}/original', 'text': t, 'regions': [{'start': 0, 'end': len(t), 'label': 0}], 'slot': 'human_controls', 'split': 'heldout',
        'label': 'human', 'condition': 'original', 'writer': 'human', 'paper_id': pid.get(h.split('#')[0], ''), 'source': src.get(h.split('#')[0])}
       for h, t in humans.items()]

# 3. Never-train list: paper ids (raw and prefixed forms) + human text of every held-out row
texts = list(humans.values()) + [r['text'][g['start']:g['end']] for r in rows for g in r['regions'] if g['label'] == 0]
norm = lambda x: re.sub(r'\W+', '', x.lower())
sh = {hashlib.sha256(n[i:i + 60].encode()).hexdigest()[:12] for x in texts for n in [norm(x)] for i in range(0, len(n) - 59, 10)}
ids = held | {'paper:' + p for p in held} | {'human:' + p for p in held}
(HERE / 't21-heldout-never-train-v2.json').write_text(json.dumps({
    'purpose': 'T2.1 held-out split: 15% of Opus/Sonnet small-edit papers + every paper hosting GPT-6 Sol or Fable rows. Never train on these.',
    'created_pdt': time.strftime('%Y-%m-%d %H:%M'), 'check': 'never_train_hit(row) loads this with the other held-out lists.',
    'normalization': "re.sub(r'\\W+', '', text.lower())", 'paper_ids': sorted(ids),
    'paragraph_sha256': sorted({hashlib.sha256(norm(x).encode()).hexdigest() for x in humans.values()}),
    'paragraph_shingle_sha256_12': sorted(sh)}, separators=(',', ':')))
with gzip.open(HERE / 't21-heldout-score-rows-v2.jsonl.gz', 'wt') as f:
    for r in hum + rows: f.write(json.dumps(r) + '\n')

C = collections.Counter
stats = {'held_papers': len(held), 'held_from_opus_sonnet_slice': n_sliced, 'opus_sonnet_papers': len(sliced_papers),
         'eval_ai_rows_by_writer_condition': {f'{w}|{c}': n for (w, c), n in sorted(C((r['writer'], r['condition']) for r in rows).items())},
         'eval_ai_rows_by_writer': dict(C(r['writer'] for r in rows)), 'human_controls': len(hum), 'host_reconstruction_mismatches': mismatch,
         'training_rows_removed': {'small_edits': dict(C(r['generator'] for r in small if r['paper_id'] in held)),
                                   'paragraph_edits': dict(C(r['generator'] for r in para if r['paper_id'] in held)),
                                   'sections': dict(C(r.get('model') or r.get('generator') for r in sections if r['paper_id'] in held))},
         'shingles': len(sh), 'v1_papers': len(V1), 'new_papers_in_v2': len(new_papers),
         'eval_ai_rows_by_writer_source': {f'{w}|{s}': n for (w, s), n in sorted(C((r['writer'], r['source']) for r in rows).items())},
         'eval_ai_rows_new_in_v2': sum(r['new_in_v2'] for r in rows)}
(HERE / 't21-v2-new-papers.json').write_text(json.dumps(sorted(new_papers)))
(HERE / 't21-split-stats-v2.json').write_text(json.dumps(stats, indent=1)); print(json.dumps(stats, indent=1))
