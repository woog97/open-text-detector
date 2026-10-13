"""Original ASAP 2.0 training essays as quarantined candidates, never test data."""
import argparse
from collections import defaultdict, deque, Counter
import csv
import fcntl
import hashlib
import io
import json
from pathlib import Path
import zipfile

from collect_pool import BINS, atomic_json, digest, now
from expand_pool import add, connect, counts

# Exact protected-evaluation matches are recorded in source-exhaustion.json.
# Preserve earlier quarantined rows; exclude these families from new intake.
PROTECTED_PROMPTS = frozenset({'driverless cars', 'facial action coding system'})

def protected_prompt(row):
    return ' '.join(row.get('prompt_name', '').casefold().split()) in PROTECTED_PROMPTS

def balanced_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get('set', '').strip().lower() != 'train':
            continue
        if protected_prompt(row):
            continue
        groups[(row['prompt_name'], row['score'])].append(row)
    queues = {key: deque(sorted(value, key=lambda r: digest('asap2-v1' + r['essay_id']))) for key, value in groups.items()}
    while queues:
        for key in sorted(list(queues)):
            yield queues[key].popleft()
            if not queues[key]:
                del queues[key]

def make_record(row, revision, archive_hash):
    if protected_prompt(row):
        return None
    text = row['full_text']
    wc = len(text.split())
    if not 50 <= wc <= 1500:
        return None
    sid = 'asap2'
    doc_id = 'scrosseye/ASAP_2.0:' + row['essay_id']
    text_hash = digest(text)
    license_text = 'https://creativecommons.org/licenses/by/4.0/'
    url = 'https://github.com/scrosseye/ASAP_2.0/tree/' + revision
    metadata = {'license': license_text, 'url': url, 'title': row['prompt_name'],
                'language': 'en', 'original_columns': row, 'official_split': 'train',
                'archive_sha256': archive_hash, 'collection_date': 'unverified'}
    original = {'source_id': sid, 'source_dataset': 'scrosseye/ASAP_2.0', 'source_revision': revision,
                'source_file': 'ASAP_2_Final_github_train.zip/ASAP_2_Final_github_train.csv',
                'retrieved_at': now(), 'raw_text_sha256': text_hash,
                'record': {'id': row['essay_id'], 'text': text, 'created': None, 'metadata': metadata}}
    record = {'record_id': digest(sid + doc_id), 'source_id': sid, 'category': 'essays', 'text': text,
              'word_count': wc, 'length_bin': next(i for i, (lo, hi) in enumerate(BINS) if lo <= wc <= hi),
              'source_dataset': original['source_dataset'], 'source_revision': revision,
              'source_file': original['source_file'], 'source_row': int(row['_source_row']),
              'original_id': row['essay_id'], 'source_url': url, 'title': row['prompt_name'],
              'author_attribution_json': '[]', 'license_evidence': license_text,
              'claimed_original_date': '', 'retrieved_at': original['retrieved_at'],
              'raw_text_sha256': text_hash, 'passage_sha256': text_hash, 'raw_start': 0,
              'raw_end': len(text), 'offset_unit': 'unicode_codepoints',
              'extraction_method': 'unchanged_original_training_essay', 'parent_document_id': doc_id,
              'provisional_family_id': digest(doc_id), 'prompt_family_id': digest(row['prompt_name'] + row.get('assignment', '')),
              'original_split': 'train', 'admission_status': 'quarantined_candidate', 'training_eligible': False,
              'provenance_basis': 'publisher_described_original_student_training_essay_collection_date_unverified',
              'protected_overlap_status': 'not_fully_audited',
              'reason_codes': ['collection_date_unverified', 'historical_text_version_unverified',
                               'writing_tools_and_human_provenance_audit_pending', 'persuade_overlap_audit_pending',
                               'protected_overlap_audit_pending', 'record_rights_audit_pending',
                               'prompt_and_student_grouping_pending', 'genre_and_extraction_review_pending'],
              'sampling_seed': 27183, 'pipeline_version': 'asap-original-train-v2-protected-families'}
    return record, original

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    folder = args.base / 'source-downloads/asap2'
    manifest = json.loads((folder / 'manifest.json').read_text())
    archive = folder / 'train.zip'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest['sha256']:
        raise RuntimeError('ASAP archive checksum mismatch')
    plan = json.loads((args.base / 'pipeline/sampling-plan.json').read_text())
    quota = next(r['planned_passages'] for r in plan['source_quotas'] if r['source_id'] == 'asap2')
    with (args.base / 'asap2.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with zipfile.ZipFile(archive) as z:
            with z.open('ASAP_2_Final_github_train.csv') as stream:
                rows = [{**row, '_source_row': str(i)} for i, row in enumerate(csv.DictReader(io.TextIOWrapper(stream, encoding='utf-8-sig')))]
        db = connect(args.base / 'collection.sqlite3')
        selected = Counter()
        excluded = Counter(row['prompt_name'] for row in rows if protected_prompt(row))
        for row in balanced_rows(rows):
            if counts(db).get('asap2', 0) >= quota:
                break
            pair = make_record(row, manifest['revision'], manifest['sha256'])
            if pair is None:
                continue
            with db:
                if add(db, *pair, quota):
                    selected[(row['prompt_name'], row['score'])] += 1
        n = counts(db).get('asap2', 0)
        db.close()
        status = {'source_id': 'asap2', 'count': n, 'target': quota, 'state': 'quota_filled' if n == quota else 'shortfall',
                  'new_prompt_score_counts': {str(k): v for k, v in selected.items()}, 'updated_at': now(),
                  'protected_prompt_rows_excluded_from_new_intake': dict(excluded),
                  'all_quarantined': True, 'collection_date_verified': False, 'official_test_split_used': False}
        atomic_json(args.base / 'progress/asap2.json', status)
        print(json.dumps(status))

if __name__ == '__main__':
    main()
