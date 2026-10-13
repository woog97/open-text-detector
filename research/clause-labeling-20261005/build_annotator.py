"""Inject the gold-set sentences and draft cuts into the annotator page.

Usage: build_annotator.py SENTENCES.jsonl DRAFTS.jsonl OUT.html
Only gold_id, text and draft cuts go into the page; side (human/AI) and tags are left
out so the annotator is not primed by them. OUT should be outside git (it holds paper text).
"""
import json, sys
from pathlib import Path

sent_path, draft_path, out = map(Path, sys.argv[1:4])
sents = [json.loads(l) for l in open(sent_path)]
drafts = {d['gold_id']: d['cuts'] for d in map(json.loads, open(draft_path))}
items = []
for s in sents:
    cuts = sorted(set(drafts.get(s['gold_id'], [])))
    t = s['text']
    assert all(0 < c < len(t) and t[c - 1].isspace() and not t[c].isspace() for c in cuts), s['gold_id']
    items.append({'id': s['gold_id'], 'text': t, 'draft': cuts})
missing = [s['gold_id'] for s in sents if s['gold_id'] not in drafts]
data = json.dumps(items, ensure_ascii=False).replace('</', '<\\/')
tpl = (Path(__file__).resolve().parent / 'annotator.template.html').read_text()
out.write_text(tpl.replace('__DATA__', data))
print(f'{len(items)} items, {len(missing)} without a draft, {sum(len(i["draft"]) for i in items)} draft cuts -> {out}')
