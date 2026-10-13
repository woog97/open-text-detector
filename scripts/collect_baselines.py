"""Reproducible, size-capped baseline collection. Run from the project root.

ICLR full text uses the public ReviewBench OCR archive because OpenReview
requires a challenge. Papers are independently matched to official programs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata
import urllib.request

import pyarrow.parquet as pq
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'research/data'
CACHE = ROOT / 'research/cache'
SOURCES = ROOT / 'research/sources'
SEED = 'pangram-baseline-2026-09-22-v1'
REVIEW_REV = '7d1b399bd7297318a2d6284b5349326166702530'
PG_REV = 'c021754c8e01c5b1cc83a1f549c1f97fbbb756b8'
MDTA_REV = 'fffb86b767e23367d09d27b32d34c3c586cf5219'
CAP = 100_000_000


def download(url, target, cap=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return target
    part = target.with_suffix(target.suffix + '.part')
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'OpenTextDetectorResearch/1.0 (public baseline collection)'})
            with urllib.request.urlopen(request, timeout=90) as response, part.open('wb') as out:
                count = 0
                while block := response.read(1024 * 1024):
                    count += len(block)
                    if cap and count >= cap:
                        raise ValueError(f'Download exceeds byte cap: {target.name}')
                    out.write(block)
            part.replace(target)
            print(f'Downloaded {target.name}: {count:,} bytes', flush=True)
            return target
        except Exception:
            part.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(min(2 ** attempt, 8))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_rows(path, rows, cap=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as out:
        for row in rows:
            line = (json.dumps(row, ensure_ascii=False) + '\n').encode()
            if cap and out.tell() + len(line) >= cap:
                raise ValueError('Output exceeds dataset byte cap')
            out.write(line)


def manifest(folder, details, raw_files=(), cap=None):
    files = [p for p in folder.rglob('*') if p.is_file() and p.name != 'manifest.json'] + list(raw_files)
    artifacts = {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files}
    total = sum(v['bytes'] for v in artifacts.values())
    payload = dict(details, collected_at=datetime.now(timezone.utc).isoformat(), files=artifacts,
                   total_artifact_bytes=total, seed=SEED)
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + '\n').encode()
    if cap and total + len(encoded) >= cap:
        raise ValueError(f'{folder.name} exceeds {cap} bytes including cached source')
    (folder / 'manifest.json').write_bytes(encoded)
    print(folder.name, 'complete:', details.get('rows'), 'rows;', total + len(encoded), 'bytes', flush=True)


def title_key(s):
    return re.sub(r'[^a-z0-9]', '', unicodedata.normalize('NFKD', s).lower())


def iclr():
    official = {}
    for year in (2023, 2026):
        url = f'https://iclr.cc/virtual/{year}/papers.html'
        path = download(url, SOURCES / f'iclr{year}.html')
        soup = BeautifulSoup(path.read_text(), 'html.parser')
        official[year] = {title_key(a.get_text(' ', strip=True)): 'https://iclr.cc' + a['href']
                          for a in soup.select('a[href]') if a['href'].startswith(f'/virtual/{year}/poster/')}
    urls = [f'https://huggingface.co/datasets/Samarth0710/reviewbench/resolve/{REVIEW_REV}/data/iclr-{i:05d}-of-00005.parquet' for i in range(5)]
    def fetch(url):
        return download(url, CACHE / 'iclr' / url.rsplit('/', 1)[-1])
    with ThreadPoolExecutor(max_workers=3) as pool:
        paths = list(pool.map(fetch, urls))
    candidates = {2023: {}, 2026: {}}
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=100, columns=[
                'forum_id', 'year', 'title', 'abstract', 'authors', 'venue', 'decision', 'markdown']):
            for r in batch.to_pylist():
                year = r['year']
                if year not in candidates or title_key(r['title']) not in official[year]:
                    continue
                text = r.get('markdown') or ''
                if len(text.split()) < 1000:
                    continue
                candidates[year][r['forum_id']] = r
    for year, pool in candidates.items():
        chosen = sorted(pool.values(), key=lambda r: hashlib.sha256((SEED + r['forum_id']).encode()).hexdigest())[:100]
        assert len(chosen) == 100, (year, len(pool))
        rows = []
        for r in chosen:
            rows.append({'id': r['forum_id'], 'title': r['title'], 'year': year,
                         'authors': r['authors'], 'abstract': r['abstract'], 'text': r['markdown'],
                         'label': None, 'cohort': f'iclr_{year}',
                         'baseline_role': 'historical_human_proxy' if year == 2023 else 'contemporary_comparison',
                         'authorship_verified': False, 'decision': r['decision'], 'venue': r['venue'],
                         'source_url': f"https://openreview.net/forum?id={r['forum_id']}",
                         'pdf_url': f"https://openreview.net/pdf?id={r['forum_id']}",
                         'official_program_url': official[year][title_key(r['title'])],
                         'text_source': 'Samarth0710/reviewbench', 'source_revision': REVIEW_REV,
                         'text_format': 'OCR markdown',
                         'text_sha256': hashlib.sha256(r['markdown'].encode()).hexdigest()})
        folder = DATA / f'iclr_{year}'
        write_rows(folder / 'papers.jsonl', rows)
        manifest(folder, {'rows': 100, 'source': 'Samarth0710/reviewbench', 'revision': REVIEW_REV,
                          'official_program': f'https://iclr.cc/virtual/{year}/papers.html',
                          'eligible_candidates': len(pool),
                          'selection': '100 lowest seeded SHA256 forum IDs among official-program matches with >=1000 words; official program membership is the acceptance check',
                          'format': 'Full OCR text, not original PDFs; may contain OCR errors, references and appendices',
                          'label_policy': 'Year cohorts are not ground-truth AI/human labels',
                          'version_caveat': '2023 archived versions may include revisions after ChatGPT launch',
                          'license': 'ReviewBench CC-BY-4.0; underlying authors retain their paper rights'})
    # Shards are reproducible download cache, excluded from the final two capped HF datasets.


def human():
    filename = 'test-00000-of-00001-29a571947c0b5ccc.parquet'
    raw = download(f'https://huggingface.co/datasets/emozilla/pg19/resolve/{PG_REV}/data/{filename}', CACHE / 'human_pg19' / filename, CAP)
    folder = DATA / 'human_pg19'
    rows = []
    for i, r in enumerate(pq.read_table(raw).to_pylist()):
        assert r['publication_date'] < 1919
        text = r['text']
        rows.append({'id': f'pg19-test-{i:03d}', 'title': r['short_book_title'], 'text': text,
                     'publication_date': r['publication_date'], 'source_url': r['url'], 'label': 'human',
                     'label_evidence': 'Published before 1919; PG-19 test split',
                     'source_dataset': 'emozilla/pg19', 'original_dataset': 'deepmind/pg19',
                     'source_revision': PG_REV, 'source_split': 'test',
                     'text_sha256': hashlib.sha256(text.encode()).hexdigest()})
    assert len(rows) == 100
    write_rows(folder / 'books.jsonl', rows, CAP - raw.stat().st_size - 1_000_000)
    manifest(folder, {'rows': len(rows), 'source': 'emozilla/pg19', 'original_source': 'deepmind/pg19',
                      'revision': PG_REV, 'split': 'test', 'license': 'Apache-2.0 compilation; historical book rights vary by jurisdiction',
                      'selection': 'Entire 100-book test split; no train split downloaded',
                      'caveats': 'Historical literary style differs from contemporary scientific prose; OCR and editorial artifacts may remain',
                      'byte_cap': CAP}, [raw], CAP)


def ai():
    folder = DATA / 'ai_mdta_2025'
    generators = {
        'gemma-3-12b': {'release_date': '2025-03-12', 'source': 'https://blog.google/technology/developers/gemma-3/'},
        'qwen2.5-vl-7b': {'release_date': '2025-01-28', 'source': 'https://qwenlm.github.io/blog/qwen2.5-vl/'},
    }
    rows, raws, seen = [], [], set()
    for domain in ('open_qa', 'wiki_csai'):
        raw = download(f'https://huggingface.co/datasets/nsp909/MDTA/resolve/{MDTA_REV}/{domain}.jsonl', CACHE / 'ai_mdta_2025' / f'{domain}.jsonl', 40_000_000)
        raws.append(raw)
        with raw.open() as handle:
            for line in handle:
                r = json.loads(line)
                for generator, release in generators.items():
                    for temperature, text in r.get('model_responses', {}).get(generator, {}).items():
                        if not isinstance(text, str) or len(text.split()) < 50:
                            continue
                        digest = hashlib.sha256(text.encode()).hexdigest()
                        if digest in seen:
                            continue
                        seen.add(digest)
                        rows.append({'id': f"{domain}-{r['question_index']}-{generator}-{temperature}",
                                     'text': text, 'prompt': r['question'], 'label': 'ai', 'generator': generator,
                                     'generator_release_date': release['release_date'], 'generator_release_source': release['source'],
                                     'temperature': temperature, 'domain': domain, 'question_index': r['question_index'],
                                     'source_dataset': 'nsp909/MDTA', 'source_revision': MDTA_REV,
                                     'source_field': 'model_responses', 'text_sha256': digest})
    assert rows and set(r['generator'] for r in rows) == set(generators)
    write_rows(folder / 'responses.jsonl', rows, CAP - sum(p.stat().st_size for p in raws) - 1_000_000)
    manifest(folder, {'rows': len(rows), 'source': 'nsp909/MDTA', 'revision': MDTA_REV, 'license': 'CC-BY-SA-4.0',
                      'generators': generators, 'selection': 'All unique >=50-word standard responses from the two 2025 generators in open_qa and wiki_csai; three temperatures',
                      'excluded': 'Older Llama/Ministral generators, human answers, adversarial rewrites, other domains',
                      'caveats': 'Labels and generator identity supplied by dataset author; near-duplicates may share prompts. Split by question, not row.',
                      'counts_by_generator': {g: sum(r['generator'] == g for r in rows) for g in generators},
                      'byte_cap': CAP}, raws, CAP)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('collection', choices=['all', 'iclr', 'human', 'ai'], default='all', nargs='?')
    args = parser.parse_args()
    for name, fn in [('human', human), ('ai', ai), ('iclr', iclr)]:
        if args.collection in ('all', name):
            fn()
