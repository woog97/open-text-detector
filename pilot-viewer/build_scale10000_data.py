"""Publish the completed 10,000-paragraph collection as conference/year slices."""
import json
from collections import Counter
import build_data as viewer

SOURCE = viewer.ROOT / 'research/data/paper-gap10000-v3-luna-20260930'

def records(name):
    with (SOURCE / name).open() as handle:
        for line in handle:
            yield json.loads(line)

def read(name):
    return json.loads((SOURCE / name).read_text())

def costs(attempts):
    usage = [a['response'].get('usage', {}) for a in attempts]
    total = {'calls': len(usage), 'attempts': len(usage),
             'cost_usd': sum(u.get('cost', 0) for u in usage),
             'cost_coverage': sum('cost' in u for u in usage),
             'input_tokens': sum(u.get('prompt_tokens', 0) for u in usage),
             'output_tokens': sum(u.get('completion_tokens', 0) for u in usage),
             'reasoning_tokens': sum(u.get('completion_tokens_details', {}).get('reasoning_tokens', 0) for u in usage)}
    total['provider_reported_cost_usd'] = total['cost_usd']
    return total

def build():
    report = read('scale-report.json')
    assert report['paragraphs'] == 10000 and report['costs']['new']['account_matches_ledger']
    papers = list(records('papers.jsonl'))
    entries = []
    viewer.SOURCE = SOURCE
    for conference,year in sorted({(p['conference'],p['year']) for p in papers}):
        selected = [p for p in papers if p['year'] == year and p['conference']==conference]
        paper_ids = {p['paper_id'] for p in selected}
        passages = [p for p in records('passages.jsonl') if p['paper_id'] in paper_ids]
        passage_ids = {p['passage_id'] for p in passages}
        cache = {'papers.jsonl': selected, 'passages.jsonl': passages}
        def rows(name):
            if name not in cache:
                cache[name] = [r for r in records(name) if r.get('passage_id') in passage_ids]
            return cache[name]
        corrections=read('source-corrections.json') if (SOURCE/'source-corrections.json').exists() else []
        retired_papers={r['original_paper']['paper_id'] for r in corrections if r.get('original_paper',{}).get('year')==year}
        retired_attempts=[a for a in records('retired-source-attempts.jsonl') if a['passage_id'].split('/')[0] in retired_papers] if (SOURCE/'retired-source-attempts.jsonl').exists() else []
        root_attempts = rows('attempts.jsonl')
        review_attempts = rows('fidelity-review/attempts.jsonl')
        base_cost = costs(root_attempts)
        base_cost['by_operation'] = {s: costs([a for a in root_attempts if a['stage'] == s]) for s in ['outline', 'writer', 'audit']}
        ds = rows('dataset.jsonl')
        manifest = read('manifest.json')
        manifest['split'] = {s + '_papers': sum(p['split'] == s for p in selected) for s in ['train', 'validation', 'test']}
        manifest['viewer_scope'] = {'year': year, 'papers': len(selected), 'paragraphs': len(passages)}
        fidelity = rows('fidelity-review/responses.jsonl')
        fs = {**read('fidelity-review/summary.json'), 'papers': len(selected), 'paragraphs': len(passages),
              'verdicts': dict(Counter(r['output']['verdict'] for r in fidelity)), 'costs': costs(review_attempts),
              'dimensions': {d: dict(Counter(r['output']['dimensions'][d] for r in fidelity)) for d in read('fidelity-review/summary.json')['dimensions']}}
        overrides = {'manifest.json': manifest, 'costs.json': base_cost,
                     'summary.json': {'unique_texts': len({r['text_sha256'] for r in ds}), 'eligible_rows': sum(r['eligible_for_pilot_training'] for r in ds)},
                     'content-quality-audit.json': {'automated_reviewed': len(rows('audit-responses.jsonl'))},
                     'fidelity-review/summary.json': fs}
        viewer.rows = rows
        viewer.read = lambda name: overrides[name] if name in overrides else read(name)
        filename = f'data-gap10000-{conference.lower()}-{year}.json'
        viewer.build(filename)
        path = viewer.DEST / filename
        data = json.loads(path.read_text())
        data['scale_scope'] = {'year': year, 'collection_paragraphs': 10000, 'new_paragraphs': 7500,
                               'collection_new_costs': report['costs']['new'],
                               'view_new_costs': costs([a for a in root_attempts + review_attempts + retired_attempts if not a.get('reused_from')]),
                               'reused_paragraphs': sum(p['passage_id'] in set(manifest['reused_passage_ids']) for p in passages)}
        for item in data['examples']:
            item['development_exposed'] = next(p['development_exposed'] for p in passages if p['passage_id'] == item['passage_id'])
        path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')))
        entries.append({'id': f'gap10000-{conference.lower()}-{year}', 'label': f'Luna · V3 10k · {conference} {year} · {len(passages)} paragraphs', 'file': filename})
        del data, ds, cache
    registry = viewer.DEST / 'runs.json'
    prior = json.loads(registry.read_text())
    registry.write_text(json.dumps([r for r in prior if not r['id'].startswith('gap10000-')] + entries))

if __name__ == '__main__':
    build()
