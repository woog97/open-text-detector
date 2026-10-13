"""Original publisher training essays with protected prompt/essay exclusions."""
import argparse
from collections import Counter, defaultdict, deque
import csv
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import time
from collect_pool import BINS, atomic_json, digest, now
from expand_pool import add, connect

SOURCES = {
 'persuade': {'repo': 'scrosseye/persuade_corpus_2.0', 'revision': '67d182ac88ea4a4dda736de859cfdb0bc360ee9b',
             'filename': 'persuade_corpus_2.0_train.csv', 'sha256': 'f61319edd8bf16a982711ea0399fad59c05afaec05cdf0767f16a2c05c467e23',
             'id_column': 'essay_id', 'prompt_column': 'prompt_name', 'split_column': 'competition_set', 'score_column': 'holistic_essay_score'},
 'ellipse': {'repo': 'scrosseye/ELLIPSE-Corpus', 'revision': 'dc3b8f0b3b4332fc9f64302c4ccfc4ed582f4b43',
             'filename': 'ELLIPSE_Final_github_train.csv', 'id_column': 'text_id_kaggle',
             'prompt_column': 'prompt', 'split_column': 'set', 'score_column': 'Overall'},
}
VERSION = 'original-essays-excluded-families-v1'

def normalize(text): return ' '.join(text.casefold().split())

def hash_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(data)
    return h.hexdigest()

def check_manifest(sid, manifest):
    spec = SOURCES[sid]
    if manifest.get('source_id') != sid or manifest.get('repo') != spec['repo'] or manifest.get('revision') != spec['revision']:
        raise ValueError('Unexpected source identity/revision')
    if manifest.get('source_file') != spec['filename'] or manifest.get('split') != 'train':
        raise ValueError('Only the original publisher training file is permitted')
    if 'sha256' in spec and manifest.get('sha256') != spec['sha256']: raise ValueError('Unexpected source hash')

def unique_train_rows(path, sid):
    spec = SOURCES[sid]; seen = {}; ambiguous = set()
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {spec[k] for k in ('id_column', 'prompt_column', 'split_column', 'score_column')} | {'full_text'}
        if not required <= set(reader.fieldnames or []): raise ValueError('Missing required source columns')
        for index, row in enumerate(reader):
            if row[spec['split_column']].strip().casefold() != 'train': continue
            key = row[spec['id_column']]
            if not key: continue
            if key in seen:
                old = seen[key]
                if old['full_text'] != row['full_text'] or old[spec['prompt_column']] != row[spec['prompt_column']]: ambiguous.add(key)
                old['_annotation_count'] += 1
            else: seen[key] = {**row, '_source_row': index, '_annotation_count': 1}
    return [r for k, r in seen.items() if k not in ambiguous], len(ambiguous)

def rejection(row, sid, exclusions):
    spec = SOURCES[sid]
    if row[spec['split_column']].strip().casefold() != 'train': return 'not_train'
    if normalize(row[spec['prompt_column']]) in {normalize(x) for x in exclusions.get('prompt_names', [])}: return 'protected_or_other_source_prompt_family'
    if row[spec['id_column']] in exclusions.get('essay_ids', []): return 'protected_essay_id'
    if digest(normalize(row['full_text'])) in exclusions.get('normalized_text_hashes', []): return 'protected_or_other_source_text'
    if not 50 <= len(row['full_text'].split()) <= 1500: return 'length'
    return None

def balanced_rows(rows, sid):
    spec = SOURCES[sid]; groups = defaultdict(list)
    for row in rows: groups[(row[spec['prompt_column']], row[spec['score_column']])].append(row)
    queues = {key: deque(sorted(value, key=lambda r: digest(VERSION + r[spec['id_column']]))) for key, value in groups.items()}
    while queues:
        for key in sorted(list(queues)):
            yield queues[key].popleft()
            if not queues[key]: del queues[key]

def make_pair(row, sid, manifest):
    spec = SOURCES[sid]; text = row['full_text']; words = len(text.split())
    if row[spec['split_column']].strip().casefold() != 'train' or not 50 <= words <= 1500: return None
    original_id = row[spec['id_column']]; prompt = row[spec['prompt_column']]; doc_id = spec['repo'] + ':' + original_id
    url = 'https://github.com/' + spec['repo'] + '/tree/' + spec['revision']
    raw = {'source_id': sid, 'source_dataset': spec['repo'], 'source_revision': spec['revision'],
           'source_file': manifest['source_file'], 'source_row': row['_source_row'], 'retrieved_at': manifest['retrieved_at'],
           'raw_text_sha256': digest(text), 'record': {'id': original_id, 'text': text,
           'metadata': {'original_columns': row, 'official_split': 'train', 'archive_sha256': manifest['sha256'],
                        'license': 'CC-BY-NC-SA-4.0', 'prompt_name': prompt, 'annotation_count': row['_annotation_count']}}}
    record = {'record_id': digest(sid + ':' + doc_id), 'source_id': sid, 'category': 'essays', 'text': text,
       'word_count': words, 'length_bin': next(i for i, (lo, hi) in enumerate(BINS) if lo <= words <= hi),
       'source_dataset': spec['repo'], 'source_revision': spec['revision'], 'source_file': manifest['source_file'],
       'source_row': row['_source_row'], 'original_id': original_id, 'source_url': url, 'title': prompt,
       'author_attribution_json': '[]', 'license_evidence': 'https://creativecommons.org/licenses/by-nc-sa/4.0/',
       'claimed_original_date': '', 'retrieved_at': manifest['retrieved_at'], 'raw_text_sha256': digest(text),
       'passage_sha256': digest(text), 'raw_start': 0, 'raw_end': len(text), 'offset_unit': 'unicode_codepoints',
       'extraction_method': 'unchanged_original_training_essay', 'parent_document_id': doc_id,
       'provisional_family_id': digest(doc_id), 'prompt_family_id': digest(normalize(prompt)), 'original_split': 'train',
       'admission_status': 'quarantined_candidate', 'training_eligible': False,
       'provenance_basis': 'publisher_original_training_essay_collection_date_unverified',
       'protected_overlap_status': 'known_local_essays_and_prompt_families_excluded_other_audits_pending',
       'reason_codes': ['noncommercial_sharealike_terms', 'collection_date_unverified', 'student_identity_unavailable',
                        'historical_human_provenance_audit_pending', 'remaining_overlap_audit_pending'],
       'sampling_seed': 27183, 'pipeline_version': VERSION}
    return record, raw

def collect(base, sid):
    started = time.monotonic(); folder = base / 'source-downloads' / sid
    manifest = json.loads((folder / 'manifest.json').read_text()); check_manifest(sid, manifest)
    exclusions = json.loads((folder / 'exclusions.json').read_text())
    if not exclusions.get('protected_family_audit_complete'): raise ValueError('Protected family exclusions must be established before collection')
    path = folder / manifest['local_file']
    if hash_file(path) != manifest['sha256']: raise ValueError('Original file checksum mismatch')
    plan = json.loads((base / 'pipeline/sampling-plan.json').read_text())
    quota = next(r['planned_passages'] for r in plan['source_quotas'] if r['source_id'] == sid)
    (base / 'progress').mkdir(exist_ok=True)
    with (base / (sid + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows, ambiguous = unique_train_rows(path, sid)
        rejected = Counter(); eligible = []
        for row in rows:
            why = rejection(row, sid, exclusions)
            if why: rejected[why] += 1
            else: eligible.append(row)
        db = connect(base / 'collection.sqlite3'); initial = db.execute('SELECT count(*) FROM passages WHERE source=?', (sid,)).fetchone()[0]
        n = initial
        try:
            for row in balanced_rows(eligible, sid):
                if n >= quota: break
                pair = make_pair(row, sid, manifest)
                db.execute('BEGIN IMMEDIATE')
                try:
                    if add(db, *pair, quota): n += 1
                    db.commit()
                except BaseException: db.rollback(); raise
            report = {'source_id': sid, 'target': quota, 'count': n, 'new_candidates': n - initial,
               'unique_train_rows': len(rows), 'ambiguous_essay_ids': ambiguous, 'eligible_rows_before_global_dedup': len(eligible),
               'rejections': dict(rejected), 'state': 'quota_filled' if n == quota else 'eligible_source_exhausted',
               'elapsed_seconds': time.monotonic() - started, 'updated_at': now(), 'all_quarantined': True,
               'exclusions_sha256': hash_file(folder / 'exclusions.json'), 'original_file_sha256': manifest['sha256']}
            atomic_json(base / 'progress' / (sid + '.json'), report); print(json.dumps(report), flush=True); return report
        finally: db.close()

def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--base', type=Path, required=True)
    p.add_argument('--source', choices=sorted(SOURCES), required=True)
    args = p.parse_args(); collect(args.base, args.source)

if __name__ == '__main__': main()
