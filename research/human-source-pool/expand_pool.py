"""Checkpointed source-quota top-up. CPU/network only; originals are preserved.

SQLite stores accepted passages, exact deduplication and scan cursors atomically.
An immutable earlier release is imported once; it is never overwritten.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gzip
from itertools import islice
import json
from pathlib import Path
import random
import sqlite3
import time
from urllib.parse import unquote, urlparse

import requests
from collect_pool import (OPEN_LICENSE, LENGTH_WEIGHTS, apportion, atomic_json,
                          date_year, digest, make_passages, now, package)

VERSION = 'candidate-topup-v2'

def connect(path):
    db = sqlite3.connect(path, timeout=120)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.executescript('''
      CREATE TABLE IF NOT EXISTS documents (hash TEXT PRIMARY KEY, raw BLOB);
      CREATE TABLE IF NOT EXISTS passages (id TEXT PRIMARY KEY, norm TEXT UNIQUE, source TEXT, doc TEXT, bin INTEGER, row TEXT);
      CREATE INDEX IF NOT EXISTS source_index ON passages(source);
      CREATE INDEX IF NOT EXISTS source_bin_index ON passages(source,bin);
      CREATE INDEX IF NOT EXISTS doc_index ON passages(source,doc);
      CREATE TABLE IF NOT EXISTS cursors (source TEXT, file TEXT, position INTEGER, done INTEGER, PRIMARY KEY(source,file));
      CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
    ''')
    return db

def namespace_ok(sid, row):
    meta = row.get('metadata') or {}
    if sid == 'wiki_talk':
        return str(meta.get('namespace')) == '1'
    if sid in ('wikipedia', 'wikibooks', 'wikisource', 'wikinews', 'wikivoyage'):
        return str(meta.get('namespace')) == '0'
    return True

def claimed_date(sid, row):
    meta = row.get('metadata') or {}
    value = row.get('created') or meta.get('year')
    if not value and sid in ('globalvoices', 'foodista'):
        import re
        match = re.search(r'/(20\d{2}|19\d{2})/(\d{2})/(\d{2})/', str(meta.get('url', '')))
        if match:
            return '-'.join(match.groups()), 'dated_original_url_unverified_version'
    return str(value or ''), 'source_claimed_date_unverified_version'

def add(db, row, raw, quota):
    sid = row['source_id']
    if db.execute('SELECT count(*) FROM passages WHERE source=?', (sid,)).fetchone()[0] >= quota:
        return False
    normalized = digest(' '.join(row['text'].casefold().split()))
    inserted = db.execute('INSERT OR IGNORE INTO passages VALUES (?,?,?,?,?,?)',
                          (row['record_id'], normalized, sid, row['parent_document_id'], row['length_bin'], json.dumps(row, ensure_ascii=False))).rowcount
    if inserted:
        db.execute('INSERT OR IGNORE INTO documents VALUES (?,?)',
                   (row['raw_text_sha256'], gzip.compress(json.dumps(raw, ensure_ascii=False).encode())))
    return bool(inserted)

def import_previous(db, previous, quotas):
    import pyarrow.parquet as pq
    if db.execute("SELECT value FROM settings WHERE key='seed_complete'").fetchone():
        return
    for shard in sorted((previous / 'release/data').glob('*.parquet')):
        sid = shard.stem
        raw = {}
        with gzip.open(previous / 'sources' / sid / 'documents.jsonl.gz', 'rt') as stream:
            for line in stream:
                doc = json.loads(line)
                raw[doc['raw_text_sha256']] = doc
        with db:
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=512):
                for row in batch.to_pylist():
                    doc = raw[row['raw_text_sha256']]
                    if namespace_ok(sid, doc['record']):
                        add(db, row, doc, quotas[sid])
        print(json.dumps({'imported': sid, 'count': db.execute('SELECT count(*) FROM passages WHERE source=?', (sid,)).fetchone()[0]}), flush=True)
    with db:
        db.execute("INSERT INTO settings VALUES ('seed_complete','true')")

def counts(db):
    return dict(db.execute('SELECT source,count(*) FROM passages GROUP BY source'))

def process_row(db, sid, spec, category, quota, row, filename, index, exclusions):
    meta = row.get('metadata') or {}
    text = row.get('text') or ''
    if not isinstance(text, str) or not text or len(text) > 8_000_000:
        return 'invalid_text'
    if not namespace_ok(sid, row):
        return 'wrong_namespace'
    license_text = str(meta.get('license') or meta.get('oa_license') or '')
    if not OPEN_LICENSE.search(license_text):
        return 'license_evidence_missing'
    if meta.get('language') and meta['language'] not in ('en', 'eng', 'English'):
        return 'language'
    date, date_basis = claimed_date(sid, row)
    year = date_year(date)
    if year and year > 2021:
        return 'post_cutoff'
    if not year and sid != 'gutenberg':
        return 'missing_date'
    if '<html' in text[:300].lower() or '<!doctype' in text[:300].lower():
        return 'needs_html_extraction'
    if sid == 'ubuntu_irc':
        return 'needs_chat_extraction'
    original_id = str(row.get('id', ''))
    url = str(meta.get('url') or meta.get('oa_url') or '')
    if sid == 'gutenberg' and (original_id in exclusions['gutenberg_ids'] or any(t in str(meta.get('title', '')).casefold() for t in exclusions['book_titles'])):
        return 'protected_pg19'
    if sid.startswith('stack_') and 'stackoverflow.com' in url:
        return 'wrong_site'
    doc_id = spec['repo_id'] + ':' + original_id
    cap = 10 if sid == 'gutenberg' else 3 if category in ('scientific', 'professional', 'reference') else 1
    used_doc = db.execute('SELECT count(*) FROM passages WHERE source=? AND doc=?', (sid, doc_id)).fetchone()[0]
    if used_doc >= cap:
        return 'document_cap'
    existing = dict(db.execute('SELECT bin,count(*) FROM passages WHERE source=? GROUP BY bin', (sid,)))
    target = apportion(quota, {str(i): n for i, n in enumerate(LENGTH_WEIGHTS[category])})
    remaining = [max(0, n - existing.get(int(i), 0)) for i, n in target.items()]
    spans = make_passages(text, doc_id, cap - used_doc, remaining)
    old_ranges = [(json.loads(x[0])['raw_start'], json.loads(x[0])['raw_end']) for x in db.execute('SELECT row FROM passages WHERE source=? AND doc=?', (sid, doc_id))]
    retained = 0
    raw_hash = digest(text)
    raw = {'source_id': sid, 'source_dataset': spec['repo_id'], 'source_revision': spec['revision'],
           'source_file': filename, 'source_row': index, 'retrieved_at': now(), 'raw_text_sha256': raw_hash, 'record': row}
    for a, b, wc, bin_id in spans:
        if any(a < end and b > start for start, end in old_ranges):
            continue
        passage = text[a:b]
        result = {'record_id': digest(sid + doc_id + str(a) + str(b)), 'source_id': sid, 'category': category,
                  'text': passage, 'word_count': wc, 'length_bin': bin_id, 'source_dataset': spec['repo_id'],
                  'source_revision': spec['revision'], 'source_file': filename, 'source_row': index,
                  'original_id': original_id, 'source_url': url, 'title': str(meta.get('title') or ''),
                  'author_attribution_json': json.dumps(meta.get('authors') or meta.get('author') or [], ensure_ascii=False),
                  'license_evidence': license_text, 'claimed_original_date': date, 'date_evidence_basis': date_basis,
                  'retrieved_at': raw['retrieved_at'], 'raw_text_sha256': raw_hash, 'passage_sha256': digest(passage),
                  'raw_start': a, 'raw_end': b, 'offset_unit': 'unicode_codepoints',
                  'extraction_method': 'unchanged_contiguous_paragraphs', 'parent_document_id': doc_id,
                  'provisional_family_id': digest(str(meta.get('book_url') or url or doc_id)),
                  'admission_status': 'quarantined_candidate', 'training_eligible': False,
                  'provenance_basis': 'unverified', 'protected_overlap_status': 'not_fully_audited',
                  'reason_codes': ['historical_text_version_unverified', 'record_rights_audit_pending',
                                   'protected_overlap_audit_pending', 'genre_and_extraction_review_pending',
                                   'language_review_pending', 'family_and_author_grouping_pending'],
                  'sampling_seed': 27183, 'pipeline_version': VERSION}
        retained += add(db, result, raw, quota)
    return 'accepted' if retained else 'no_unique_eligible_span'

def scan_source(dbpath, base, sid, spec, category, quota, exclusions):
    db = connect(dbpath)
    rejected = Counter()
    files = list(spec['files'])
    random.Random(27183).shuffle(files)
    # Deterministic spread without the old per-file or 5,000-row hard stop.
    try:
        for filename in files:
            if counts(db).get(sid, 0) >= quota:
                break
            cursor = db.execute('SELECT position,done FROM cursors WHERE source=? AND file=?', (sid, filename)).fetchone()
            if cursor and cursor[1]:
                continue
            start = cursor[0] if cursor else 0
            url = f"https://huggingface.co/datasets/{spec['repo_id']}/resolve/{spec['revision']}/{filename}"
            for attempt in range(3):
                try:
                    with requests.get(url, stream=True, timeout=(20, 90)) as response:
                        response.raise_for_status()
                        with gzip.GzipFile(fileobj=response.raw) as stream:
                            index = start - 1
                            reached_end = True
                            rows = islice(enumerate(stream), start, None)
                            while True:
                                # Read from the network before taking the SQLite write lock.
                                batch = list(islice(rows, 200))
                                if not batch:
                                    break
                                batch_reasons = Counter()
                                with db:
                                    db.execute('BEGIN IMMEDIATE')
                                    for index, line in batch:
                                        row = json.loads(line)
                                        reason = process_row(db, sid, spec, category, quota, row, filename, index, exclusions)
                                        batch_reasons[reason] += 1
                                        if reason == 'accepted' and counts(db).get(sid, 0) >= quota:
                                            reached_end = False
                                            break
                                    # Accepted originals and the last consumed row commit together.
                                    db.execute('INSERT OR REPLACE INTO cursors VALUES (?,?,?,0)', (sid, filename, index + 1))
                                rejected.update(batch_reasons)
                                n = counts(db).get(sid, 0)
                                atomic_json(base / 'progress' / (sid + '.json'), {'source_id': sid, 'count': n, 'target': quota,
                                    'file': filename, 'row': index, 'reasons': dict(rejected), 'updated_at': now()})
                                if not reached_end:
                                    break
                    with db:
                        db.execute('INSERT OR REPLACE INTO cursors VALUES (?,?,?,?)', (sid, filename, index + 1, int(reached_end)))
                    break
                except Exception as exc:
                    atomic_json(base / 'progress' / (sid + '-error.json'), {'source': sid, 'file': filename,
                        'attempt': attempt + 1, 'error': type(exc).__name__ + ': ' + str(exc)[:250], 'updated_at': now()})
                    cursor = db.execute('SELECT position FROM cursors WHERE source=? AND file=?', (sid, filename)).fetchone()
                    start = cursor[0] if cursor else 0
                    time.sleep(2 ** attempt)
        n = counts(db).get(sid, 0)
        atomic_json(base / 'progress' / (sid + '.json'), {'source_id': sid, 'count': n, 'target': quota,
            'state': 'quota_filled' if n == quota else 'available_shards_scanned', 'updated_at': now()})
        print(json.dumps({'source': sid, 'count': n, 'target': quota}), flush=True)
    finally:
        db.close()

def export(db, base, registry, plan):
    out = base / 'intake'
    statuses = []
    for quota in plan['source_quotas']:
        sid = quota['source_id']
        dest = out / 'sources' / sid
        dest.mkdir(parents=True, exist_ok=True)
        docs = set()
        n = 0
        with gzip.open(dest / 'passages.jsonl.gz', 'wt') as stream:
            for (value,) in db.execute('SELECT row FROM passages WHERE source=? ORDER BY id', (sid,)):
                row = json.loads(value)
                stream.write(value + '\n')
                docs.add(row['raw_text_sha256'])
                n += 1
        with gzip.open(dest / 'documents.jsonl.gz', 'wt') as stream:
            for h in sorted(docs):
                stream.write(gzip.decompress(db.execute('SELECT raw FROM documents WHERE hash=?', (h,)).fetchone()[0]).decode() + '\n')
        statuses.append({'source_id': sid, 'candidate_passages': n, 'shortfall': quota['planned_passages'] - n,
                         'state': 'quota_filled' if n == quota['planned_passages'] else 'additional_source_work_required'})
    summary = package(out, registry, plan, statuses)
    summary['status'] = 'complete_quarantined_candidate_pool' if summary['candidate_total'] == 100000 else 'partial_quarantined_candidate_pool'
    atomic_json(out / 'release/collection-summary.json', summary)
    atomic_json(base / 'status.json', summary)
    print(json.dumps({'candidate_total': summary['candidate_total'], 'shortfall': summary['shortfall']}), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    args.base.mkdir(parents=True, exist_ok=True)
    (args.base / 'progress').mkdir(exist_ok=True)
    with (args.base / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = args.base / 'pipeline'
        registry = json.loads((config / 'source-registry.json').read_text())
        plan = json.loads((config / 'sampling-plan.json').read_text())
        assert plan['planned_total'] == 100000
        specs = json.loads((config / 'collection-sources.json').read_text())
        exclusions = json.loads((config / 'protected-exclusions.json').read_text())
        specs['wiki_talk'] = specs['wikipedia']
        quotas = {s['source_id']: s['planned_passages'] for s in plan['source_quotas']}
        dbpath = args.base / 'collection.sqlite3'
        db = connect(dbpath)
        fingerprint = digest(json.dumps(plan, sort_keys=True))
        old = db.execute("SELECT value FROM settings WHERE key='plan_hash'").fetchone()
        if old and old[0] != fingerprint:
            raise RuntimeError('The agreed source quotas changed; refusing to resume')
        with db:
            db.execute("INSERT OR IGNORE INTO settings VALUES ('plan_hash',?)", (fingerprint,))
        import_previous(db, args.previous, quotas)
        print(json.dumps({'seed_total': sum(counts(db).values()), 'started_at': now()}), flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = []
            # IRC and CCCC need dedicated extraction/license evidence before collection.
            for source in plan['source_quotas']:
                sid = source['source_id']
                if sid not in specs or sid in ('ubuntu_irc', 'cccc') or counts(db).get(sid, 0) >= quotas[sid]:
                    continue
                futures.append(pool.submit(scan_source, dbpath, args.base, sid, specs[sid], source['category'], quotas[sid], exclusions))
            for future in futures:
                future.result()
        export(db, args.base, registry, plan)
        db.close()

if __name__ == '__main__':
    main()
