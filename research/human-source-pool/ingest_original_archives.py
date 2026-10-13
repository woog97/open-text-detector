"""Original train-only IMDb and WritingPrompts archives, one passage per document."""
import argparse
from collections import deque
import fcntl
import hashlib
import io
import itertools
import json
import re
import shutil
import tarfile
import time
from pathlib import Path

from collect_pool import BINS, LENGTH_WEIGHTS, STOPWORDS, apportion, atomic_json, digest, make_passages, now
from expand_pool import add, connect

SOURCES = {
 'imdb': {'url': 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
          'sha256': 'c40f74a18d3b61f90feba1e17730e0d38e8b97c05fde7008942e91923d1658fe',
          'category': 'reviews', 'year': 2011, 'publisher': 'https://nlp.stanford.edu/~amaas/data/sentiment/index.html'},
 'writingprompts': {'url': 'https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz',
          'sha256': '4b8d0415c450fe1ceadec667bdf73c185268bc3d9e1f6ddf64edac0481cb71f4',
          'category': 'creative', 'year': 2018, 'publisher': 'https://github.com/facebookresearch/fairseq/tree/main/examples/stories'},
}
VERSION = 'original-archives-train-v1'

def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def check_manifest(sid, m):
    spec = SOURCES[sid]
    if m.get('source_id') != sid or m.get('url') != spec['url'] or m.get('sha256') != spec['sha256']:
        raise ValueError('Unexpected original archive identity/revision')

def imdb_documents(archive):
    """One gzip scan; stable interleaving of original positive/negative train reviews."""
    groups = {'neg': [], 'pos': []}; urls = {}
    pattern = re.compile(r'aclImdb/train/(neg|pos)/(\d+)_(\d+)\.txt')
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            if not member.isfile(): continue
            match = pattern.fullmatch(member.name)
            urlmatch = re.fullmatch(r'aclImdb/train/urls_(neg|pos)\.txt', member.name)
            if match:
                label, row, rating = match.groups()
                text = tar.extractfile(member).read().decode('utf-8')
                groups[label].append({'text': text, 'source_file': member.name, 'source_row': int(row),
                    'original_id': member.name, 'sentiment': label, 'rating': int(rating)})
            elif urlmatch:
                urls[urlmatch[1]] = tar.extractfile(member).read().decode('utf-8').splitlines()
    queues = {label: deque(sorted(rows, key=lambda r: digest(VERSION + r['original_id']))) for label, rows in groups.items()}
    while any(queues.values()):
        for label in ('neg', 'pos'):
            if not queues[label]: continue
            doc = queues[label].popleft()
            doc['source_url'] = urls.get(label, [])[doc['source_row']] if doc['source_row'] < len(urls.get(label, [])) else ''
            movie = re.search(r'/title/(tt\d+)', doc['source_url'])
            doc['family_key'] = movie.group(1) if movie else doc['original_id']
            doc['family_basis'] = 'imdb_movie_id' if movie else 'review_id_movie_unavailable'
            yield doc

def writingprompts_documents(archive, cache):
    """Extract only two exact train members, preserving targets and paired prompts."""
    cache.mkdir(parents=True, exist_ok=True)
    names = {f'writingPrompts/train.wp_{suffix}': cache / f'train.wp_{suffix}' for suffix in ('source', 'target')}
    receipt = cache / 'extraction.json'
    ready = json.loads(receipt.read_text()) if receipt.exists() else {}
    valid = all(path.exists() and ready.get(path.name) == file_hash(path) for path in names.values())
    if not valid:
        found = set()
        with tarfile.open(archive, 'r|gz') as tar:
            for member in tar:
                if member.name not in names or not member.isfile(): continue
                if member.name in found: raise ValueError('Duplicate train archive member')
                dest = names[member.name]; temporary = dest.with_suffix(dest.suffix + '.partial')
                with temporary.open('wb') as out: shutil.copyfileobj(tar.extractfile(member), out, 4 * 1024 * 1024)
                temporary.replace(dest); found.add(member.name)
        if found != set(names): raise ValueError('Missing original train prompt/target members')
        atomic_json(receipt, {path.name: file_hash(path) for path in names.values()})
    # newline='' avoids changing any original CRLF; line endings are retained in raw text.
    with names['writingPrompts/train.wp_source'].open(encoding='utf-8', newline='') as prompts, names['writingPrompts/train.wp_target'].open(encoding='utf-8', newline='') as targets:
        for row, (prompt, text) in enumerate(itertools.zip_longest(prompts, targets)):
            if prompt is None or text is None: raise ValueError('Mismatched prompt/target line counts')
            yield {'text': text, 'prompt': prompt, 'source_file': 'writingPrompts/train.wp_target',
                'source_row': row, 'original_id': 'train:' + str(row), 'source_url': SOURCES['writingprompts']['url'],
                'family_key': ' '.join(prompt.casefold().split()), 'family_basis': 'normalized_original_prompt'}

def make_pair(sid, doc, manifest, remaining):
    text = doc['text']; words = list(re.finditer(r'\S+', text))
    lexical = re.findall(r'[A-Za-z]+', text)
    if len(words) < 50 or not lexical or sum(w.lower() in STOPWORDS for w in lexical) / len(lexical) < .06:
        return None
    if sum(c.isalpha() for c in text) / max(1, len(text)) < .5: return None
    doc_id = sid + ':' + doc['original_id']
    spans = make_passages(text, doc_id, 1, remaining.copy())
    if not spans:
        # Tokenized WritingPrompts stories and IMDb HTML-break reviews often have
        # no paragraph boundaries. Keep an exact contiguous character window.
        for bin_id in sorted(range(4), key=lambda i: (-remaining[i], i)):
            lo, hi = BINS[bin_id]
            if remaining[bin_id] <= 0 or len(words) < lo: continue
            size = min((lo + hi) // 2, len(words))
            if size < lo: continue
            start = int(digest(doc_id)[:8], 16) % (len(words) - size + 1)
            spans = [(words[start].start(), words[start + size - 1].end(), size, bin_id)]; break
    if not spans: return None
    a, b, wc, bin_id = spans[0]; passage = text[a:b]; spec = SOURCES[sid]
    raw = {'source_id': sid, 'source_dataset': spec['publisher'], 'source_revision': manifest['sha256'],
       'source_file': doc['source_file'], 'source_row': doc['source_row'], 'retrieved_at': manifest['retrieved_at'],
       'raw_text_sha256': digest(text), 'record': {'id': doc['original_id'], 'text': text,
       'metadata': {k: v for k, v in doc.items() if k != 'text'}}}
    raw['record']['metadata'].update(archive_sha256=manifest['sha256'], official_split='train', archive_url=manifest['url'])
    record = {'record_id': digest(doc_id + ':' + str(a) + ':' + str(b)), 'source_id': sid,
       'category': spec['category'], 'text': passage, 'word_count': wc, 'length_bin': bin_id,
       'source_dataset': spec['publisher'], 'source_revision': manifest['sha256'],
       'source_file': doc['source_file'], 'source_row': doc['source_row'], 'original_id': doc['original_id'],
       'source_url': doc['source_url'], 'title': '', 'author_attribution_json': '[]',
       'license_evidence': 'Original research release; underlying text rights unresolved; see source-approvals.json',
       'claimed_original_date': '', 'corpus_release_year': spec['year'],
       'date_evidence_basis': 'original_pre2022_corpus_release_not_individual_document_date',
       'retrieved_at': manifest['retrieved_at'], 'raw_text_sha256': digest(text), 'passage_sha256': digest(passage),
       'raw_start': a, 'raw_end': b, 'offset_unit': 'unicode_codepoints',
       'extraction_method': 'unchanged_contiguous_original_train_span', 'parent_document_id': doc_id,
       'provisional_family_id': digest(sid + ':' + doc['family_key']), 'family_basis': doc['family_basis'],
       'original_split': 'train', 'admission_status': 'quarantined_candidate', 'training_eligible': False,
       'provenance_basis': 'original_publisher_pre2022_training_archive', 'protected_overlap_status': 'not_fully_audited',
       'reason_codes': ['underlying_text_rights_unresolved', 'protected_overlap_audit_pending',
                        'genre_and_extraction_review_pending', 'original_formatting_retained'],
       'sampling_seed': 27183, 'pipeline_version': VERSION}
    if sid == 'writingprompts': record['prompt_family_id'] = record['provisional_family_id']
    return record, raw

def collect(base, sid, max_new=None):
    started = time.monotonic(); folder = base / 'source-downloads' / sid
    m = json.loads((folder / 'manifest.json').read_text()); check_manifest(sid, m)
    archive = folder / 'original.tar.gz'
    if file_hash(archive) != m['sha256']: raise ValueError('Archive checksum mismatch')
    plan = json.loads((base / 'pipeline/sampling-plan.json').read_text())
    quota = next(s['planned_passages'] for s in plan['source_quotas'] if s['source_id'] == sid)
    (base / 'progress').mkdir(exist_ok=True)
    with (base / (sid + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db = connect(base / 'collection.sqlite3')
        n = db.execute('SELECT count(*) FROM passages WHERE source=?', (sid,)).fetchone()[0]; initial = n
        targets = apportion(quota, {str(i): v for i, v in enumerate(LENGTH_WEIGHTS[SOURCES[sid]['category']])})
        bins = dict(db.execute('SELECT bin,count(*) FROM passages WHERE source=? GROUP BY bin', (sid,)))
        cursor_key = VERSION + ':' + m['sha256']
        old = db.execute('SELECT position FROM cursors WHERE source=? AND file=?', (sid, cursor_key)).fetchone()
        last = old[0] if old else -1; scanned = 0
        def status(state):
            elapsed = time.monotonic() - started
            return {'source_id': sid, 'count': n, 'target': quota, 'new_candidates': n - initial,
               'scanned_documents_this_run': scanned, 'cursor': last, 'state': state, 'elapsed_seconds': round(elapsed, 3),
               'new_candidates_per_second': round((n - initial) / max(elapsed, .001), 3),
               'updated_at': now(), 'all_quarantined': True, 'official_test_split_used': False}
        docs = imdb_documents(archive) if sid == 'imdb' else writingprompts_documents(archive, folder / 'train-cache')
        try:
            for index, doc in enumerate(docs):
                if n >= quota or (max_new is not None and n - initial >= max_new): break
                if index <= last: continue
                scanned += 1
                remaining = [max(0, targets[str(i)] - bins.get(i, 0)) for i in range(4)]
                pair = make_pair(sid, doc, m, remaining)
                # Serialize only the brief commit, not parsing or extraction.
                db.execute('BEGIN IMMEDIATE')
                try:
                    if pair and add(db, *pair, quota):
                        n += 1; bin_id = pair[0]['length_bin']; bins[bin_id] = bins.get(bin_id, 0) + 1
                    db.execute('INSERT OR REPLACE INTO cursors VALUES (?,?,?,?)', (sid, cursor_key, index, 0))
                    db.commit()
                except BaseException: db.rollback(); raise
                last = index
                if scanned % 100 == 0: atomic_json(base / 'progress' / (sid + '.json'), status('collecting'))
            state = 'quota_filled' if n == quota else ('paused_at_new_limit' if max_new is not None and n - initial >= max_new else 'archive_exhausted')
            result = status(state); atomic_json(base / 'progress' / (sid + '.json'), result)
            print(json.dumps(result), flush=True); return result
        finally:
            docs.close(); db.close()

def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--base', type=Path, required=True)
    p.add_argument('--source', choices=sorted(SOURCES), required=True)
    p.add_argument('--max-new', type=int, help='Optional bounded benchmark; restart resumes the same cursor')
    args = p.parse_args()
    if args.max_new is not None and args.max_new <= 0: p.error('--max-new must be positive')
    collect(args.base, args.source, args.max_new)

if __name__ == '__main__': main()
