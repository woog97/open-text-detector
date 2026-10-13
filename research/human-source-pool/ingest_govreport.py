"""Pinned GovReport training reports; source-level evidence, automatic row checks."""
import argparse
import fcntl
import hashlib
import json
import re
from pathlib import Path
from collect_pool import BINS, LENGTH_WEIGHTS, apportion, atomic_json, digest, make_passages, now
from expand_pool import connect, counts, add

REPO = 'ccdv/govreport-summarization'
REVISION = '4e21184e01ae8017e2c036e180fe5e541fef60a0'

def check_manifest(m):
    if m['repo_id'] != REPO or m['revision'] != REVISION:
        raise ValueError('Unexpected source revision')
    if m['split'] != 'train' or not re.fullmatch(r'document/train-\d{5}-of-\d{5}\.parquet', m['source_file']):
        raise ValueError('Only original training shards are allowed')

def pairs(row, index, m, remaining):
    check_manifest(m)
    text = row.get('report')
    if not isinstance(text, str) or not text.strip():
        return
    # The mirror does not retain original agency IDs; keep an exact shard/row locator.
    locator = m['source_file'] + ':' + str(index)
    doc_id = REPO + ':' + digest(text)
    spans = make_passages(text, doc_id, 3, remaining.copy())
    if not spans:
        # Some report mirrors collapse paragraph separators. Slice original characters,
        # without rewriting text or using the paired summarization target.
        words = list(re.finditer(r'\S+', text))
        cursor = int(digest(doc_id)[:8], 16) % max(1, len(words) // 2)
        for bin_id in sorted(range(4), key=lambda i: -remaining[i]):
            if remaining[bin_id] <= 0 or len(spans) == 3:
                continue
            lo, hi = BINS[bin_id]
            size = min((lo + hi) // 2, len(words) - cursor)
            if size < lo:
                continue
            a, b = words[cursor].start(), words[cursor + size - 1].end()
            spans.append((a, b, size, bin_id)); cursor += size
    raw = {'source_id': 'govreport', 'source_dataset': REPO, 'source_revision': REVISION,
           'source_file': m['source_file'], 'source_row': index, 'retrieved_at': now(),
           'raw_text_sha256': digest(text), 'record': {'id': locator, 'text': text,
           'metadata': {'original_columns': row, 'official_split': 'train', 'archive_sha256': m['sha256'],
                        'publisher': m['publisher'], 'corpus_release_year': 2021,
                        'original_agency_id': None}}}
    for a, b, wc, bin_id in spans:
        passage = text[a:b]
        record = {'record_id': digest('govreport' + doc_id + str(a) + str(b)),
          'source_id': 'govreport', 'category': 'professional', 'text': passage,
          'word_count': wc, 'length_bin': bin_id, 'source_dataset': REPO, 'source_revision': REVISION,
          'source_file': m['source_file'], 'source_row': index, 'original_id': locator,
          'source_url': 'https://huggingface.co/datasets/' + REPO + '/blob/' + REVISION + '/' + m['source_file'],
          'title': '', 'author_attribution_json': '[]', 'license_evidence': m['license_evidence'],
          'claimed_original_date': '', 'corpus_release_year': 2021,
          'date_evidence_basis': 'corpus_publication_2021_not_individual_report_date',
          'retrieved_at': raw['retrieved_at'], 'raw_text_sha256': digest(text), 'passage_sha256': digest(passage),
          'raw_start': a, 'raw_end': b, 'offset_unit': 'unicode_codepoints',
          'extraction_method': 'unchanged_contiguous_report_span', 'parent_document_id': doc_id,
          'provisional_family_id': digest(doc_id), 'original_split': 'train',
          'admission_status': 'quarantined_candidate', 'training_eligible': False,
          'provenance_basis': 'publisher_described_government_reports_in_2021_corpus',
          'protected_overlap_status': 'not_fully_audited',
          'reason_codes': ['original_agency_identity_mapping_pending', 'record_rights_audit_pending',
                           'protected_overlap_audit_pending', 'genre_and_extraction_review_pending'],
          'sampling_seed': 27183, 'pipeline_version': 'govreport-original-train-v1'}
        yield record, raw

def main():
    import pyarrow.parquet as pq
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--base', type=Path, required=True)
    b = p.parse_args().base; d = b / 'source-downloads/govreport'
    m = json.loads((d / 'manifest.json').read_text()); check_manifest(m)
    path = d / m['source_file']; h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(chunk)
    if h.hexdigest() != m['sha256']: raise RuntimeError('GovReport checksum mismatch')
    plan = json.loads((b / 'pipeline/sampling-plan.json').read_text())
    quota = next(s['planned_passages'] for s in plan['source_quotas'] if s['source_id'] == 'govreport')
    targets = apportion(quota, {str(i): n for i, n in enumerate(LENGTH_WEIGHTS['professional'])})
    with (b / 'govreport.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db = connect(b / 'collection.sqlite3'); index = -1
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                index += 1
                if counts(db).get('govreport', 0) >= quota: break
                bins = dict(db.execute("SELECT bin,count(*) FROM passages WHERE source='govreport' GROUP BY bin"))
                remaining = [max(0, targets[str(i)] - bins.get(i, 0)) for i in range(4)]
                # Existing documents are not resampled on resume.
                doc_id = REPO + ':' + digest(row.get('report') or '')
                if db.execute("SELECT 1 FROM passages WHERE source='govreport' AND doc=?", (doc_id,)).fetchone(): continue
                with db:
                    for record, raw in pairs(row, index, m, remaining): add(db, record, raw, quota)
                if index % 100 == 0:
                    atomic_json(b / 'progress/govreport.json', {'count': counts(db).get('govreport', 0), 'target': quota, 'row': index, 'state': 'collecting', 'updated_at': now()})
            if counts(db).get('govreport', 0) >= quota: break
        n = counts(db).get('govreport', 0); db.close()
        status = {'source_id': 'govreport', 'count': n, 'target': quota, 'state': 'quota_filled' if n == quota else 'shard_exhausted', 'updated_at': now(), 'all_quarantined': True}
        atomic_json(b / 'progress/govreport.json', status); print(json.dumps(status), flush=True)

if __name__ == '__main__': main()
