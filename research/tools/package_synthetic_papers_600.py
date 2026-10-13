"""Combine final manuscripts without rewriting them or modifying source batches."""
import collections
import hashlib
import json
import re
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research/data/synthetic-papers-600-hf-20261003'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def main():
    OUT.mkdir(exist_ok=True)
    (OUT / 'data').mkdir(exist_ok=True)
    seeds, sources = {}, {}
    for name in ['abstracts-100-four-venues-20261002', 'abstracts-500-additional-four-venues-20261002']:
        folder = ROOT / 'research/data' / name
        for row in read_rows(folder / 'title-abstract-seeds.jsonl'):
            seeds[row['seed_id']] = row
        for row in read_rows(folder / 'metadata.jsonl'):
            sid = row.get('seed_id', f"paper-{row['row']:03d}")
            sources[sid] = row
    paths = list((ROOT / 'research/data').glob('synthetic-full-papers*/papers/*/paper.md'))
    paths += list((ROOT / 'research/data/synthetic-full-papers-100-luna-20261002/runs/125-135/papers').glob('*/paper.md'))
    papers = {}
    for path in paths:
        sid = path.parent.name
        assert sid not in papers, ('duplicate', sid)
        papers[sid] = path
    expected = [f'paper-{n:03d}' for n in range(1, 601)]
    assert set(papers) == set(expected) == set(seeds) == set(sources)
    rows, provenance, notes = [], [], []
    for number, sid in enumerate(expected, 1):
        path = papers[sid]
        data = path.read_bytes()
        text = data.decode('utf-8')
        meta = json.loads((path.parent / 'metadata.json').read_text())
        assert meta.get('synthetic_manuscript') is True, sid
        assert meta.get('experiments_actually_run') is False, sid
        assert text.startswith('# ') and len(text.split()) > 1000, sid
        sha = digest(data)
        matches = []
        for field in ['files', 'file_sha256', 'sha256', 'file_hashes_sha256', 'file_hashes', 'hashes', 'files_sha256']:
            value = meta.get(field, {})
            if isinstance(value, dict):
                for key in ['paper.md', 'paper_sha256', 'final_sha256']:
                    if key in value:
                        item = value[key]
                        matches.append(item.get('sha256') if isinstance(item, dict) else item)
        if not matches:
            matches = [meta[k] for k in ['paper_sha256', 'final_sha256'] if meta.get(k)]
        assert matches and all(value == sha for value in matches), ('paper hash', sid)
        revision = meta['revision_count']
        snapshots = [path.parent / f'revision-{revision:02d}.md', path.parent / f'revision-{revision}.md']
        snapshots = [p for p in snapshots if p.exists()]
        assert snapshots and any(p.read_bytes() == data for p in snapshots), ('snapshot', sid)
        if meta.get('final_sha256') and meta['final_sha256'] != sha:
            notes.append({'seed_id': sid, 'note': 'Legacy top-level final_sha256 is stale; per-file hash, final snapshot and completed batch export agree with uploaded final.', 'legacy_sha256': meta['final_sha256'], 'uploaded_sha256': sha})
        history = meta.get('generation_history', [])
        actual_history = [h for h in history if h.get('manuscript_modified') is not False and 'review' not in h.get('stage', '').lower()]
        models = list(dict.fromkeys(h.get('model', h.get('model_requested')).lower() for h in actual_history if h.get('model', h.get('model_requested'))))
        revision_history = [h for h in actual_history if 'revis' in h.get('stage', '').lower() and h.get('model', h.get('model_requested'))]
        model = revision_history[-1].get('model', revision_history[-1].get('model_requested')) if revision_history else meta['model_requested']
        model = model.lower()
        assert model in {'gpt-6.1-sol', 'gpt-6-sol', 'gpt-6-luna'}
        if models:
            assert model in models, ('history model', sid, models, model)
        else:
            models = [model]
        abstract_match = re.search(r'^##\s+(?:\d+[. ]+)?Abstract\s*\n(.*?)(?=^##\s|\Z)', text, re.M | re.S | re.I)
        assert abstract_match, ('abstract', sid)
        rendered_abstract = abstract_match.group(1).strip()
        source = sources[sid]
        row = {'paper_id': number, 'seed_id': sid, 'title': seeds[sid]['title'], 'abstract': seeds[sid]['abstract'],
               'rendered_title': text.splitlines()[0][2:], 'rendered_abstract': rendered_abstract,
               'paper_markdown': text, 'model': model, 'final_revision_model': model, 'writer_model': model, 'models_used': models,
               'model_label_source': 'last_revision_generation_history' if revision_history else 'generation_metadata_model_requested',
               'mixed_writer_models': len(models) > 1, 'revision_count': revision,
               'synthetic_manuscript': True, 'experiments_actually_run': False,
               'results_and_methods_may_be_invented': True, 'human_supplied_sections': ['title', 'abstract'],
               'source_paper_id': source['paper_id'], 'conference': source['conference'], 'year': source['year'],
               'source_url': source['forum_url'], 'paper_sha256': sha, 'word_count': len(text.split()),
               'generation_history_json': json.dumps(history, ensure_ascii=False),
               'source_batch': str(path.parents[2].relative_to(ROOT)), 'markdown_path': f'papers/{sid}.md'}
        rows.append(row)
        provenance.append({'seed_id': sid, 'paper_sha256': sha, 'source_metadata': source, 'generation_metadata': meta})
    for filename, records in [('papers.jsonl', rows), ('provenance.jsonl', provenance)]:
        (OUT / filename).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, OUT / 'data/train-00000-of-00001.parquet', compression='zstd')
    assert pq.read_table(OUT / 'data/train-00000-of-00001.parquet').to_pylist() == rows
    with zipfile.ZipFile(OUT / 'markdown-papers.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for sid in expected:
            archive.writestr(f'papers/{sid}.md', papers[sid].read_bytes())
    with zipfile.ZipFile(OUT / 'markdown-papers.zip') as archive:
        assert archive.testzip() is None and len(archive.namelist()) == 600
        for row in rows:
            assert digest(archive.read(row['markdown_path'])) == row['paper_sha256']
    counts = dict(collections.Counter(row['writer_model'] for row in rows))
    validation = {'papers': 600, 'unique_ids': 600, 'missing_ids': [], 'writer_model_counts': counts,
                  'conference_counts': dict(collections.Counter(r['conference'] for r in rows)),
                  'mixed_writer_papers': [r['seed_id'] for r in rows if r['mixed_writer_models']],
                  'legacy_metadata_notes': notes, 'checks_passed': True,
                  'checks': 'Exact ID coverage; final per-file hashes; revision snapshots; synthetic provenance; Parquet and ZIP readback. Manuscripts preserved byte-for-byte. No additional scientific review or format rewrite.'}
    (OUT / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps(validation))

if __name__ == '__main__':
    main()
