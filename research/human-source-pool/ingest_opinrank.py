"""Original UCI OpinRank archive: whole hotel/car reviews, source-safe parallel intake."""
import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import re
import zipfile
from collect_pool import BINS, LENGTH_WEIGHTS, apportion, atomic_json, digest, now, prose_spans
from expand_pool import connect, counts, add

ARCHIVE_SHA = 'c304a3176a87e801d24bf9c541f6b1bad1148aa6ce734e0725a5762c3b8240c8'
DATASET = 'uci/opinrank-205'

def decode(data):
    try:
        return data.decode('utf-8'), 'utf-8'
    except UnicodeDecodeError:
        # Original files mix UTF-8 and Windows-1252. Preserve bytes in the archive.
        return data.decode('cp1252'), 'cp1252'

def parse_member(name, data):
    text, encoding = decode(data)
    parts = name.split('/')
    if len(parts) != 3 or parts[0] not in ('hotels', 'cars'):
        return
    common = {'member': name, 'domain': parts[0], 'group': parts[1], 'entity': parts[2], 'encoding': encoding}
    if parts[0] == 'hotels':
        for index, line in enumerate(text.splitlines()):
            cols = line.split('\t')
            if len(cols) < 3 or any(c.strip() for c in cols[3:]):
                continue
            yield dict(common, index=index, text=cols[2], date=cols[0].strip(), title=cols[1], author='', original_record=line)
    else:
        for index, match in enumerate(re.finditer(r'<DOC>(.*?)</DOC>', text, re.S)):
            block = match.group(1)
            def field(tag):
                matches = re.findall('<' + tag + r'>(.*?)</' + tag + '>', block, re.S)
                return matches[0] if len(matches) == 1 else ''
            review = field('TEXT')
            if review:
                yield dict(common, index=index, text=review, date=field('DATE').strip(), author=field('AUTHOR'), title='', original_record=match.group(0))

def diversified_rows(archive):
    """Deterministic round-robin across ten cities and three car model years."""
    groups = defaultdict(list)
    for n in archive.namelist():
        parts = n.split('/')
        if len(parts) == 3 and parts[0] in ('hotels', 'cars') and parts[2]:
            groups['/'.join(parts[:2])].append(n)
    def source(names):
        for name in sorted(names, key=lambda n: digest('27183' + n)):
            rows = list(parse_member(name, archive.read(name)))
            for row in sorted(rows, key=lambda r: digest('27183' + name + str(r['index']))):
                yield row
    active = [iter(source(groups[g])) for g in sorted(groups)]
    while active:
        next_active = []
        for it in active:
            try:
                yield next(it); next_active.append(it)
            except StopIteration:
                pass
        active = next_active

def pair(row, manifest):
    text = row['text']
    wc = len(text.split())
    bin_id = next((i for i, (lo, hi) in enumerate(BINS) if lo <= wc <= hi), None)
    # Do not cut short reviews, concatenate different reviews, or use title/favorite fields.
    if bin_id is None or not prose_spans(text):
        return None
    years = re.findall(r'\b(?:19|20)\d{2}\b', row['date'])
    if years and int(years[-1]) > 2021:
        return None
    locator = row['member'] + ':' + str(row['index'])
    doc_id = DATASET + ':' + locator
    raw = {'source_id': 'opinrank', 'source_dataset': DATASET, 'source_revision': ARCHIVE_SHA,
           'source_file': row['member'], 'source_row': row['index'], 'retrieved_at': manifest['retrieved_at'],
           'raw_text_sha256': digest(text), 'record': {'id': locator, 'text': text, 'metadata': row}}
    record = {'record_id': digest('opinrank' + doc_id), 'source_id': 'opinrank', 'category': 'reviews',
      'text': text, 'word_count': wc, 'length_bin': bin_id, 'source_dataset': DATASET,
      'source_revision': ARCHIVE_SHA, 'source_file': row['member'], 'source_row': row['index'],
      'original_id': locator, 'source_url': manifest['publisher'], 'title': row['title'],
      'author_attribution_json': json.dumps([row['author']] if row['author'] else []),
      'license_evidence': manifest['license_evidence'], 'claimed_original_date': row['date'],
      'corpus_release_year': 2011, 'date_evidence_basis': 'original_review_date_in_2011_corpus',
      'retrieved_at': manifest['retrieved_at'], 'raw_text_sha256': digest(text), 'passage_sha256': digest(text),
      'raw_start': 0, 'raw_end': len(text), 'offset_unit': 'unicode_codepoints',
      'extraction_method': 'unchanged_whole_review_field', 'parent_document_id': doc_id,
      'provisional_family_id': digest(DATASET + ':' + row['member']),
      'review_domain': row['domain'], 'review_group': row['group'], 'review_entity': row['entity'],
      'original_split': 'unsplit_original_release', 'admission_status': 'quarantined_candidate',
      'training_eligible': False, 'provenance_basis': 'original_UCI_2011_user_review_release',
      'protected_overlap_status': 'not_fully_audited',
      'reason_codes': ['upstream_review_rights_unverified', 'protected_overlap_audit_pending', 'entity_family_split_pending', 'genre_and_human_provenance_audit_pending'],
      'sampling_seed': 27183, 'pipeline_version': 'opinrank-original-v1'}
    return record, raw

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    base = parser.parse_args().base
    directory = base / 'source-downloads/opinrank'
    manifest = json.loads((directory / 'manifest.json').read_text())
    path = directory / manifest['archive']
    if manifest['sha256'] != ARCHIVE_SHA or hashlib.sha256(path.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise RuntimeError('OpinRank original archive checksum mismatch')
    plan = json.loads((base / 'pipeline/sampling-plan.json').read_text())
    quota = next(s['planned_passages'] for s in plan['source_quotas'] if s['source_id'] == 'opinrank')
    targets = apportion(quota, {str(i): n for i, n in enumerate(LENGTH_WEIGHTS['reviews'])})
    with (base / 'opinrank.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db = connect(base / 'collection.sqlite3'); seen = Counter(); index = 0
        with zipfile.ZipFile(path) as archive:
            for index, row in enumerate(diversified_rows(archive), 1):
                result = pair(row, manifest)
                if not result:
                    seen['quality_or_length'] += 1; continue
                record, raw = result
                # Short atomic write transaction avoids stale concurrent quota checks.
                db.execute('BEGIN IMMEDIATE')
                try:
                    n = db.execute("SELECT count(*) FROM passages WHERE source='opinrank'").fetchone()[0]
                    if n >= quota:
                        db.commit(); break
                    have = db.execute("SELECT count(*) FROM passages WHERE source='opinrank' AND bin=?", (record['length_bin'],)).fetchone()[0]
                    if have < targets[str(record['length_bin'])]:
                        added = add(db, record, raw, quota)
                        seen['accepted' if added else 'duplicate'] += 1
                    else:
                        seen['length_bin_full'] += 1
                    db.commit()
                except Exception:
                    db.rollback(); raise
                if index % 500 == 0:
                    atomic_json(base / 'progress/opinrank.json', {'source_id': 'opinrank', 'count': counts(db).get('opinrank', 0), 'target': quota, 'scanned': index, 'state': 'collecting', 'updated_at': now(), 'reasons': dict(seen)})
        n = counts(db).get('opinrank', 0)
        group_counts = Counter()
        for (payload,) in db.execute("SELECT row FROM passages WHERE source='opinrank'"):
            r = json.loads(payload); group_counts[r['review_domain'] + '/' + r['review_group']] += 1
        db.close()
        status = {'source_id': 'opinrank', 'count': n, 'target': quota, 'scanned': index, 'state': 'quota_filled' if n == quota else 'archive_exhausted', 'updated_at': now(), 'all_quarantined': True, 'groups': dict(group_counts), 'reasons': dict(seen)}
        atomic_json(base / 'progress/opinrank.json', status); print(json.dumps(status), flush=True)

if __name__ == '__main__': main()
