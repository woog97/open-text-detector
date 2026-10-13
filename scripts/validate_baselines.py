"""Audit the collected files without network or model inference."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
from collect_baselines import title_key, CAP
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
expected = {'iclr_2023': 100, 'iclr_2026': 100, 'human_pg19': 100, 'ai_mdta_2025': 6397}
report = {'validated_at': datetime.now(timezone.utc).isoformat(), 'collections': {}}
all_ids, all_hashes = set(), set()
for name, count in expected.items():
    folder = ROOT / 'research/data' / name
    m = json.loads((folder / 'manifest.json').read_text())
    total = (folder / 'manifest.json').stat().st_size
    for relative, info in m['files'].items():
        path = ROOT / relative
        with path.open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == info['sha256'], relative
        assert path.stat().st_size == info['bytes']
        total += path.stat().st_size
    if name in ('human_pg19', 'ai_mdta_2025'):
        assert total < CAP
    rows = [json.loads(line) for path in folder.glob('*.jsonl') for line in path.open()]
    assert len(rows) == count
    ids, hashes = {r['id'] for r in rows}, {r['text_sha256'] for r in rows}
    assert len(ids) == count and len(hashes) == count
    assert not (all_ids & ids or all_hashes & hashes)
    all_ids.update(ids)
    all_hashes.update(hashes)
    for r in rows:
        assert hashlib.sha256(r['text'].encode()).hexdigest() == r['text_sha256']
    if name.startswith('iclr_'):
        year = int(name[-4:])
        soup = BeautifulSoup((ROOT / f'research/sources/iclr{year}.html').read_text(), 'html.parser')
        titles = {title_key(a.get_text(' ', strip=True)) for a in soup.select('a[href]')
                  if a['href'].startswith(f'/virtual/{year}/poster/')}
        assert all(r['year'] == year and r['label'] is None and not r['authorship_verified']
                   and title_key(r['title']) in titles and len(r['text'].split()) >= 1000 for r in rows)
    if name == 'human_pg19':
        assert all(r['publication_date'] < 1919 and r['label'] == 'human' for r in rows)
    if name == 'ai_mdta_2025':
        assert all(r['generator'] in {'gemma-3-12b', 'qwen2.5-vl-7b'} and r['label'] == 'ai'
                   and r['source_field'] == 'model_responses' and len(r['text'].split()) >= 50 for r in rows)
    report['collections'][name] = {'rows': count, 'unique_ids': len(ids), 'unique_texts': len(hashes),
                                   'verified_bytes_including_sources_and_manifest': total, 'passed': True}
(ROOT / 'research/validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
