"""Original OANC Slate / ICIC text: pinned archive, exact offsets, source locks."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import re
import time
import xml.etree.ElementTree as ET
import zipfile

from collect_pool import BINS, STOPWORDS, LENGTH_WEIGHTS, apportion, atomic_json, date_year, digest, now
from expand_pool import add, connect, counts

ARCHIVE_URL = 'https://github.com/nancyide/anc-website/releases/download/datasets-2026-09-25/OANC_GrAF.zip'
ARCHIVE_SHA256 = '5a26559a1becba41a527cb674fff4fb9c4fb70b276f60422c4f07a0ef23fd867'
SOURCES = {'oanc_slate': ('/journal/slate/', 'news', 1), 'oanc_icic': ('/letters/icic/', 'professional', 3)}
VERSION = 'oanc-original-text-v1'


def matches(sid, name):
    return (sid in SOURCES and name.startswith('OANC-GrAF/data/written_')
            and SOURCES[sid][0] in name and name.endswith('.txt')
            and '..' not in PurePosixPath(name).parts)


def metadata(header, name):
    root = ET.fromstring(header)
    ns = {'g': 'http://www.xces.org/ns/GrAF/1.0/'}
    source = root.find('.//g:sourceDesc', ns)
    primary = root.find('.//g:primaryData', ns)
    if source is None or primary is None or primary.get('loc') != PurePosixPath(name).name:
        raise ValueError('OANC header does not identify the paired text')
    def values(tag):
        return [' '.join(e.itertext()).strip() for e in source.findall('g:' + tag, ns)]
    date = source.find('g:pubDate', ns)
    return {'title': (values('title') or [''])[0], 'authors': values('author'),
            'publication_date': '' if date is None else date.get('value') or (date.text or '').strip(),
            'header_created': root.get('date.created', ''), 'header_xml': header,
            'original_file_id': PurePosixPath(name).stem}


def logical_paragraphs(text, logical):
    ns = {'g': 'http://www.xces.org/ns/GrAF/1.0/'}
    xid = '{http://www.w3.org/XML/1998/namespace}id'
    root = ET.fromstring(logical)
    regions = {e.get(xid): tuple(map(int,e.get('anchors').split())) for e in root.findall('g:region',ns)}
    nodes = {e.get(xid): e for e in root.findall('g:node',ns)}
    spans = []
    for annotation in root.findall('g:a',ns):
        if annotation.get('label') != 'p': continue
        link = nodes[annotation.get('ref')].find('g:link',ns)
        if link is None: continue
        region = regions[link.get('targets')]
        if len(region) != 2 or not 0 <= region[0] < region[1] <= len(text):
            raise ValueError('Invalid OANC logical paragraph offsets')
        spans.append(region)
    return sorted(set(spans))


def oanc_passages(text, doc_id, cap, remaining, logical=None):
    # OANC has whitespace-only separator lines, often >8 characters wide.
    # Preserve those characters when joining paragraphs instead of treating them
    # as noncontiguous text. Never alter the original string or its offsets.
    paragraphs = []
    boundaries = logical_paragraphs(text,logical) if logical else [m.span() for m in re.finditer(r'\S.*?(?=\n[ \t]*\n|\Z)', text,re.S)]
    for a,b in boundaries:
        while b > a and text[b-1].isspace(): b -= 1
        value = text[a:b]; words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", value)
        if len(words) < 20 or len(words) > 1500: continue
        if sum(w.lower() in STOPWORDS for w in words) / len(words) < .06: continue
        if sum(c.isalpha() for c in value) / max(1,len(value)) < .5: continue
        paragraphs.append((a,b))
    candidates = []
    for index, (a,b) in enumerate(paragraphs):
        for j in range(index, len(paragraphs)):
            if j > index and text[paragraphs[j-1][1]:paragraphs[j][0]].strip(): break
            b = paragraphs[j][1]; wc = len(text[a:b].split())
            if wc > 1500: break
            for bin_id,(lo,hi) in enumerate(BINS):
                if lo <= wc <= hi: candidates.append((a,b,wc,bin_id))
    rng = random.Random(digest('27183' + doc_id)); selected = []
    while len(selected) < cap:
        available = [c for c in candidates if remaining[c[3]] > 0
                     and not any(c[0] < b and c[1] > a for a,b,_,_ in selected)]
        bins = sorted({c[3] for c in available})
        if not bins: break
        chosen = rng.choices(bins, weights=[remaining[i] for i in bins])[0]
        options = [c for c in available if c[3] == chosen]
        # Letters can supply several separate passages; do not consume two
        # usable paragraphs as one when both fit the same short-length bin.
        if cap > 1:
            shortest = min(c[2] for c in options)
            options = [c for c in options if c[2] == shortest]
        candidate = rng.choice(options)
        selected.append(candidate); remaining[chosen] -= 1
    return selected


def pairs(sid, name, text, header, remaining, logical=None):
    if not matches(sid, name):
        raise ValueError('Wrong OANC component or non-text file')
    meta = metadata(header, name)
    year = date_year(meta['publication_date'])
    if not year or year > 2021:
        return
    category, cap = SOURCES[sid][1:]
    doc_id = 'OANC_GrAF:' + name
    spans = oanc_passages(text, doc_id, cap, remaining.copy(), logical)
    retrieved = now()
    raw_hash = digest(text)
    original = {'source_id': sid, 'source_dataset': 'OANC_GrAF',
                'source_revision': ARCHIVE_SHA256, 'source_file': name,
                'retrieved_at': retrieved, 'raw_text_sha256': raw_hash,
                'record': {'id': meta['original_file_id'], 'text': text,
                           'metadata': {**meta, 'archive_url': ARCHIVE_URL,
                                        'archive_sha256': ARCHIVE_SHA256,
                                        'header_sha256': digest(header), 'language': 'en',
                                        'paragraph_annotation_file': name[:-4] + '-logical.xml',
                                        'paragraph_annotation_sha256': digest(logical) if logical else None}}}
    for a, b, wc, bin_id in spans:
        passage = text[a:b]
        yield {'record_id': digest(sid + doc_id + str(a) + str(b)),
               'source_id': sid, 'category': category, 'text': passage,
               'word_count': wc, 'length_bin': bin_id, 'source_dataset': 'OANC_GrAF',
               'source_revision': ARCHIVE_SHA256, 'source_file': name,
               'original_id': meta['original_file_id'], 'source_url': ARCHIVE_URL,
               'title': meta['title'], 'author_attribution_json': json.dumps(meta['authors']),
               'license_evidence': 'OANC unrestricted use and redistribution: https://anc.org/data/oanc/',
               'claimed_original_date': meta['publication_date'],
               'date_evidence_basis': 'OANC_source_header_publication_date_not_download_date',
               'retrieved_at': retrieved, 'raw_text_sha256': raw_hash,
               'passage_sha256': digest(passage), 'raw_start': a, 'raw_end': b,
               'offset_unit': 'unicode_codepoints', 'extraction_method': 'unchanged_contiguous_paragraphs',
               'parent_document_id': doc_id, 'provisional_family_id': digest(doc_id),
               'admission_status': 'quarantined_candidate', 'training_eligible': False,
               'provenance_basis': 'publisher_contributed_pre_2022_OANC_component',
               'protected_overlap_status': 'not_fully_audited',
               'reason_codes': ['protected_overlap_audit_pending', 'genre_and_extraction_review_pending',
                                'correspondence_privacy_audit_pending'] if sid == 'oanc_icic'
                               else ['protected_overlap_audit_pending', 'genre_and_extraction_review_pending'],
               'sampling_seed': 27183, 'pipeline_version': VERSION}, original


def checked_archive(folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest['sha256'] != ARCHIVE_SHA256 or manifest['url'] != ARCHIVE_URL:
        raise ValueError('Unrecognized OANC archive manifest')
    path = folder / 'OANC_GrAF.zip'
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    if h.hexdigest() != ARCHIVE_SHA256:
        raise ValueError('OANC archive checksum mismatch')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--source', choices=sorted(SOURCES), required=True)
    parser.add_argument('--length-weights', help='Explicit source-specific four comma-separated length weights')
    args = parser.parse_args(); base, sid = args.base, args.source
    began = time.monotonic()
    plan = json.loads((base / 'pipeline/sampling-plan.json').read_text())
    quota = next(s['planned_passages'] for s in plan['source_quotas'] if s['source_id'] == sid)
    category = SOURCES[sid][1]
    weights = LENGTH_WEIGHTS[category]
    if args.length_weights:
        weights = [int(x) for x in args.length_weights.split(',')]
        if len(weights) != 4 or min(weights) < 0 or not sum(weights):
            parser.error('--length-weights requires four nonnegative integers with a positive sum')
    targets = apportion(quota, {str(i): n for i, n in enumerate(weights)})
    (base / 'progress').mkdir(exist_ok=True)
    with (base / (sid + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        archive = checked_archive(base / 'source-downloads/oanc')
        db = connect(base / 'collection.sqlite3')
        saved = db.execute("SELECT value FROM settings WHERE key='plan_hash'").fetchone()
        if saved and saved[0] != digest(json.dumps(plan, sort_keys=True)):
            raise ValueError('Plan migration must complete before source ingestion')
        config_key = VERSION + ':' + sid
        config = json.dumps({'quota':quota,'length_weights':weights},sort_keys=True)
        prior = db.execute('SELECT value FROM settings WHERE key=?',(config_key,)).fetchone()
        if prior and prior[0] != config:
            raise ValueError('OANC source quota/length configuration changed; explicit migration needed')
        with db: db.execute('INSERT OR IGNORE INTO settings VALUES (?,?)',(config_key,config))
        before = counts(db).get(sid, 0); scanned = 0; reasons = Counter()
        with zipfile.ZipFile(archive) as z:
            names = sorted((n for n in z.namelist() if matches(sid, n)), key=lambda n: digest('27183' + n))
            for name in names:
                if counts(db).get(sid, 0) >= quota:
                    break
                if db.execute('SELECT done FROM cursors WHERE source=? AND file=?', (sid, name)).fetchone():
                    continue
                text = z.read(name).decode('utf-8-sig')
                header = z.read(name[:-4] + '.anc').decode('utf-8-sig')
                logical = z.read(name[:-4] + '-logical.xml').decode('utf-8-sig')
                # The transaction contains only SQLite work and small passage extraction;
                # network and archive reads happen outside the shared writer lock.
                db.execute('BEGIN IMMEDIATE')
                try:
                    bins = dict(db.execute('SELECT bin,count(*) FROM passages WHERE source=? GROUP BY bin', (sid,)))
                    remaining = [max(0, targets[str(i)] - bins.get(i, 0)) for i in range(4)]
                    doc_id = 'OANC_GrAF:' + name
                    if not db.execute('SELECT 1 FROM passages WHERE source=? AND doc=?', (sid, doc_id)).fetchone():
                        found = list(pairs(sid, name, text, header, remaining, logical))
                        for row, raw in found:
                            if not add(db, row, raw, quota):
                                reasons['duplicate_or_quota'] += 1
                        if not found:
                            reasons['no_eligible_span_in_remaining_bins'] += 1
                    db.execute('INSERT OR REPLACE INTO cursors VALUES (?,?,?,?)', (sid, name, 1, 1))
                    db.commit()
                except BaseException:
                    db.rollback(); raise
                scanned += 1
                if scanned % 50 == 0:
                    atomic_json(base / 'progress' / (sid + '.json'),
                                {'source_id': sid, 'state': 'collecting', 'count': counts(db).get(sid, 0),
                                 'target': quota, 'scanned_this_run': scanned, 'updated_at': now()})
        n = counts(db).get(sid, 0)
        bins = dict(db.execute('SELECT bin,count(*) FROM passages WHERE source=? GROUP BY bin', (sid,)))
        db.close()
        status = {'source_id': sid, 'count': n, 'target': quota,
                  'state': 'quota_filled' if n == quota else 'source_exhausted_with_length_constraints',
                  'length_counts': bins, 'length_targets': targets, 'scanned_this_run': scanned,
                  'length_weights': weights,
                  'new_rows': n - before, 'elapsed_seconds': time.monotonic() - began,
                  'reasons': dict(reasons), 'updated_at': now(), 'all_quarantined': True}
        atomic_json(base / 'progress' / (sid + '.json'), status)
        print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
