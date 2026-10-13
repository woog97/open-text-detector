"""Space side (CPU): bundle every evaluation input for the final runs into /data/workspace/final-20261007/eval/inputs.tgz.

Sets (same files the T2/T2.1 runs were scored on, from /tmp/pangram-t2-20261006):
  sweep-eval-rows.jsonl.gz                 fixed sweep eval set (dev/test halves)
  heldout-score-rows.jsonl.gz              older held-out writers set (heldout-writers)
  t21-heldout-score-rows-v2.jsonl.gz       T2.1 held-out split v2 (GPT-6 Sol, Fable, 15% Opus/Sonnet papers)
  {polish,humanizer,cowrite}-eval-v1       Fable sets
  hetero/                                  heterogeneous-ai-spans (cross-model-s500)
New: wide1-heldout-rows.jsonl.gz, built here from wide1 accepted.jsonl split=heldout (DeepSeek 15% papers, GLM, MiMo):
  mixed rows = small edits (condition = edit_type) and paragraph edits (condition = para_<draft|rewrite>);
  human_controls = matched human paragraphs (job e) and the original host windows of the small edits (AI span replaced
  by replaced_human, i.e. the untouched human text).
"""
import gzip, io, json, tarfile
from pathlib import Path
T = Path('/tmp/pangram-t2-20261006'); W = Path('/data/workspace/corpus-sampling-20261007/wide1'); O = Path('/data/workspace/final-20261007/eval')
O.mkdir(parents=True, exist_ok=True)
rows = []
for w in ('deepseek', 'glm', 'mimo'):
    for job in ('d', 'e'):
        for r in map(json.loads, open(W / f'out-{w}-{job}/accepted.jsonl')):
            if r.get('split') != 'heldout': continue
            if r['dataset'] == 'hsm_llm_edit':
                ai = [g for g in r['regions'] if g['label'] == 1]
                a, b = min(g['start'] for g in ai), max(g['end'] for g in ai)
                rows.append({'id': r['id'], 'text': r['text'], 'regions': r['regions'], 'slot': 'mixed', 'split': 'heldout', 'label': 'mixed',
                             'condition': r['edit_type'], 'writer': r['generator'], 'paper_id': r['paper_id'], 'job': 'small'})
                orig = r['text'][:a] + (r.get('replaced_human') or '') + r['text'][b:]
                if orig.strip() and orig != r['text']:
                    rows.append({'id': r['id'] + '/orig', 'text': orig, 'regions': [{'start': 0, 'end': len(orig), 'label': 0}], 'slot': 'human_controls',
                                 'split': 'heldout', 'label': 'human_controls', 'condition': 'original', 'writer': 'human', 'paper_id': r['paper_id'], 'job': 'small'})
            elif r['dataset'] == 'hsm_paragraph_edit':
                rows.append({'id': r['id'], 'text': r['text'], 'regions': r['regions'], 'slot': 'mixed', 'split': 'heldout', 'label': 'mixed',
                             'condition': 'para_' + str(r.get('item_kind')), 'writer': r['generator'], 'paper_id': r['paper_id'], 'job': 'para'})
            elif r['dataset'] == 'hsm_paragraph_edit_human':
                rows.append({'id': r['id'], 'text': r['text'], 'regions': [{'start': 0, 'end': len(r['text']), 'label': 0}], 'slot': 'human_controls',
                             'split': 'heldout', 'label': 'human_controls', 'condition': 'original', 'writer': 'human', 'paper_id': r['paper_id'], 'job': 'para'})
ids = [r['id'] for r in rows]; assert len(ids) == len(set(ids))
b = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode()
from collections import Counter
print('wide1 rows', len(rows), dict(Counter((r['slot'], r['writer'], r['job']) for r in rows)))
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode='w:gz') as t:
    t.add(T / 'sweeps/sweep-eval-rows.jsonl.gz', 'sweep-eval-rows.jsonl.gz')
    for n in ['heldout-score-rows.jsonl.gz', 't21-heldout-score-rows-v2.jsonl.gz', 'polish-eval-v1.jsonl.gz', 'humanizer-eval-v1.jsonl.gz', 'cowrite-eval-v1.jsonl.gz']:
        t.add(T / 'inputs' / n, n)
    t.add(T / 'hetero/data', 'hetero/data')
    ti = tarfile.TarInfo('wide1-heldout-rows.jsonl.gz'); g = gzip.compress(b); ti.size = len(g); t.addfile(ti, io.BytesIO(g))
(O / 'inputs.tgz').write_bytes(buf.getvalue()); print('inputs.tgz', len(buf.getvalue()))
