"""Create a read-only browser payload from the completed pilot; no API access."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'research/data/paper-pilot10-gpt61-sol-20260929'
DEST = Path(__file__).parent / 'dist'

def read(name):
    return json.loads((SOURCE / name).read_text())

def rows(name):
    return [json.loads(line) for line in (SOURCE / name).read_text().splitlines()]

def build(filename):
    dataset = rows('dataset.jsonl')
    gap = read('manifest.json').get('schema') == 'paragraph-reconstruction-v1'
    final_ids = {row['generation_id'] for row in dataset if row['generation_id']}
    pipeline_final_ids = {r['generation_id'] for stage in ['outline','audit'] for r in rows(stage+'-responses.jsonl')} if gap else set()
    attempts = []
    for i, item in enumerate(rows('attempts.jsonl')):
        response = item['response']
        choice = (response.get('choices') or [{}])[0]
        request = item['request']
        assert set(request) <= {'model', 'messages', 'max_tokens', 'reasoning', 'response_format', 'provider', 'service_tier'}
        attempts.append({
            'ordinal': i + 1, 'request_id': item['request_id'].removesuffix('/writer')+'/paragraph_generate' if gap and item.get('stage')=='writer' else item['request_id'], 'started': item['started_utc'],
            'stage': item.get('stage'), 'passage_id': item.get('passage_id'),
            'status': 'final' if response.get('id') in final_ids else 'pipeline_final' if response.get('id') in pipeline_final_ids else 'superseded' if item['mechanical_valid'] else 'invalid',
            'validation_error': item['validation_error'], 'http_status': item['http_status'],
            'request': request, 'request_sha256': item['request_sha256'],
            'generation_id': response.get('id'), 'model': response.get('model'),
            'provider': response.get('provider'), 'finish_reason': choice.get('finish_reason'),
            'response_content': (choice.get('message') or {}).get('content'),
            'usage': response.get('usage') or {},
        })
    examples = []
    for row in dataset:
        keep = {k: row[k] for k in ['id', 'passage_id', 'paper_id', 'split', 'operation', 'text', 'regions', 'edits',
                 'sentences', 'quality_status', 'quality_flags', 'eligible_for_pilot_training', 'generation_id', 'sample_weight', 'text_sha256']}
        keep['token_counts'] = dict(Counter(t['label'] for t in row['tokens']))
        keep['token_count'] = len(row['tokens'])
        examples.append(keep)
    papers = [{k: r[k] for k in ['paper_id', 'title', 'authors', 'conference', 'year', 'abstract_url', 'pdf_url', 'split', 'pdf_sha256', 'human_label_basis']} for r in rows('papers.jsonl')]
    data = {'papers': papers, 'passages': rows('passages.jsonl'), 'examples': examples,
            'attempts': attempts, 'summary': read('summary.json'), 'costs': read('costs.json'),
            'manifest': read('manifest.json'), 'quality': read('content-quality-audit.json') if (SOURCE / 'content-quality-audit.json').exists() else {},
            'experiment_report': read('quality-comparison-report.json') if (SOURCE / 'quality-comparison-report.json').exists() else None,
            'blinded_comparisons': rows('blinded-quality-comparisons.jsonl') if (SOURCE / 'blinded-quality-comparisons.jsonl').exists() else []}
    if (SOURCE/'section-contexts.jsonl').exists():
        data['section_contexts']=rows('section-contexts.jsonl')
    if gap:
        data['operation_labels']={'human_original':'Original control','paragraph_generate':'Generated paragraph'}
        data['pipeline']={'outlines':rows('outline-responses.jsonl'),'writers':rows('writer-responses.jsonl'),'audits':rows('audit-responses.jsonl'),'reviews':rows('quality-reviews.jsonl')}
        if (SOURCE/'fidelity-review/summary.json').exists():
            data['fidelity']={'summary':read('fidelity-review/summary.json'),'results':rows('fidelity-review/responses.jsonl'),'attempts':rows('fidelity-review/attempts.jsonl')}
            if (SOURCE/'fidelity-comparison.json').exists():
                data['fidelity']['comparison']=read('fidelity-comparison.json')
        if (SOURCE/'expansion-report.json').exists():
            data['collection']=read('expansion-report.json')
        if (SOURCE/'comparison-report.json').exists():
            data['matched_evaluation']=read('comparison-report.json')
            data['matched_evaluation']['prior_writers']=[json.loads(line) for line in (Path(data['manifest']['prior_run'])/'writer-responses.jsonl').read_text().splitlines()]
            data['matched_evaluation']['quality_attempts']=rows('paired-quality/attempts.jsonl')
            if (SOURCE/'paired-quality-reversed/attempts.jsonl').exists():
                data['matched_evaluation']['quality_attempts']=[{**a,'order':'first'} for a in data['matched_evaluation']['quality_attempts']]+[{**a,'order':'reversed'} for a in rows('paired-quality-reversed/attempts.jsonl')]
            if (SOURCE/'paired-section-quality/attempts.jsonl').exists():
                data['matched_evaluation']['section_quality_attempts']=rows('paired-section-quality/attempts.jsonl')
            data['matched_evaluation']['similarity_pairs']={'old':rows('baseline-similarity.jsonl'),'new':rows('similarity.jsonl')}
    assert len(examples) == (len(data['passages'])*2 if gap else len(papers)*20) and len(final_ids) == (len(data['passages']) if gap else len(papers)*15)
    assert sum(a['status'] == 'final' for a in attempts) == len(final_ids)
    assert all(e['operation'] == 'human_original' or any(a['generation_id'] == e['generation_id'] for a in attempts) for e in examples)
    raw = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    assert 'sk-or-' not in raw and 'Authorization: Bearer' not in raw
    DEST.mkdir(exist_ok=True)
    (DEST / filename).write_text(raw)
    print(f'Prepared {filename}: {len(papers)} papers, {len(examples)} examples, {len(attempts)} attempts; {len(raw.encode()):,} bytes.')

if __name__ == '__main__':
    runs = []
    for run_id, folder, label, filename in [
        ('sol10', 'paper-pilot10-gpt61-sol-20260929', 'GPT-6.1 Sol · 10 papers', 'data.json'),
        ('luna50', 'paper-luna50-20260929', 'GPT-6 Luna · 50 papers', 'data-luna50.json'),
        ('luna50v2', 'paper-luna50-quality-v2-20260929', 'Luna · quality-first · same 50 papers', 'data-luna50v2.json'),
        ('gap50', 'paper-gap50-luna-20260929', 'Luna · Reconstruction v1 · 50 papers', 'data-gap50.json'),
        ('gap50v2', 'paper-gap50-luna-v2-20260929', 'Luna · Reconstruction v2 · 50 papers', 'data-gap50v2.json'),
        ('gap250', 'paper-gap250-luna-v2-20260929', 'Luna · 5 paragraphs × 50 papers', 'data-gap250.json'),
        ('gap250v3', 'paper-gap250-luna-v3-20260930', 'Luna · Reconstruction v3 · 250 paragraphs', 'data-gap250v3.json'),
        ('gap250v4', 'paper-gap250-luna-v4-20260930', 'Luna · Section context v4 · 250 paragraphs', 'data-gap250v4.json'),
    ]:
        SOURCE = ROOT / 'research/data' / folder
        if not (SOURCE / 'dataset.jsonl').exists():
            continue
        build(filename)
        runs.append({'id': run_id, 'label': label, 'file': filename})
    if (DEST / 'runs.json').exists():
        runs.extend(r for r in json.loads((DEST / 'runs.json').read_text()) if r['id'].startswith(('gap2500-','gap10000-')) and (DEST / r['file']).exists())
    (DEST / 'runs.json').write_text(json.dumps(runs))
