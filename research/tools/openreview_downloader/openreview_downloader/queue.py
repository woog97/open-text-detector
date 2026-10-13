"""Persistent single-worker batch queue; no browser credentials or saved passwords."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import io
import json
import logging
from email.utils import parsedate_to_datetime
import os
from pathlib import Path
import sqlite3
import subprocess
import time
from types import SimpleNamespace
import zipfile

from .batching import unpack_batch


def utc():
    return datetime.now(timezone.utc).isoformat()


def credentials(keychain=False, username=None):
    account_file = Path.home() / '.config/open-text-detector/openreview-account.json'
    if not username and not os.environ.get('OPENREVIEW_USERNAME') and account_file.is_file():
        username = json.loads(account_file.read_text())['email']
        keychain = not bool(os.environ.get('OPENREVIEW_PASSWORD'))
    username = username or os.environ.get('OPENREVIEW_USERNAME') or input('OpenReview email: ')
    if keychain:
        result = subprocess.run(
            ['security', 'find-generic-password', '-a', username,
             '-s', 'pangram-openreview', '-w'], capture_output=True, text=True)
        if result.returncode:
            raise SystemExit('OpenReview password unavailable from Keychain. No download request sent.')
        password = result.stdout.removesuffix('\n')
    else:
        password = os.environ.get('OPENREVIEW_PASSWORD') or getpass.getpass('OpenReview password (not saved): ')
    return username, password


def connect(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
    CREATE TABLE IF NOT EXISTS papers (
      id TEXT PRIMARY KEY, group_key TEXT NOT NULL, number TEXT,
      batch_no INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
      file TEXT, bytes INTEGER, sha256 TEXT, pages INTEGER);
    CREATE TABLE IF NOT EXISTS attempts (
      id INTEGER PRIMARY KEY, batch_no INTEGER, ids TEXT NOT NULL,
      status TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
      http_status INTEGER, headers TEXT, body_file TEXT, error TEXT);
    CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
    ''')
    if 'api_version' not in {r['name'] for r in db.execute('PRAGMA table_info(papers)')}:
        db.execute('ALTER TABLE papers ADD COLUMN api_version INTEGER')
        db.commit()
    return db


def seed(db, plan):
    with db:
        for b in plan['batches']:
            for p in b['papers']:
                version = p.get('api_version')
                if version is None:
                    if p.get('venue_id'):
                        raise ValueError('Venue API version has not been verified: ' + p['venue_id'])
                    version = 2  # Minimal local test plans have no venue metadata.
                if version not in (1, 2):
                    raise ValueError('Unsupported API version')
                db.execute('INSERT OR IGNORE INTO papers(id,group_key,number,batch_no) VALUES(?,?,?,?)',
                           (p['forum_id'], b['group'], str(p['submission_number']) if p.get('submission_number') is not None else None, b['batch']))
                db.execute('UPDATE papers SET api_version=? WHERE id=?', (version, p['forum_id']))
                if version == 1:
                    db.execute("UPDATE papers SET status='held_legacy' WHERE id=? AND status='pending'", (p['forum_id'],))


def begin_attempt(db, batch_no, papers):
    """Commit the reservation before sending anything over the network."""
    ids = [p['id'] for p in papers]
    with db:
        for fid in ids:
            row = db.execute('SELECT status FROM papers WHERE id=?', (fid,)).fetchone()
            if row['status'] != 'pending':
                raise ValueError('Paper already reserved or completed: ' + fid)
        cur = db.execute('INSERT INTO attempts(batch_no,ids,status,started_at) VALUES(?,?,?,?)',
                         (batch_no, json.dumps(ids), 'inflight', utc()))
        db.executemany("UPDATE papers SET status='inflight' WHERE id=?", [(fid,) for fid in ids])
    return cur.lastrowid


def local_pdfs(data_root):
    """Discover independently validated downloads, including earlier experiments."""
    for manifest in sorted(data_root.glob('*/manifest.json')):
        obj = json.loads(manifest.read_text())
        records = obj.get('papers', [])
        if not isinstance(records, list):
            continue
        for p in records:
            if p.get('status') in ('downloaded', 'validated') and p.get('file') and p.get('sha256'):
                yield p.get('forum_id'), manifest.parent / p['file'], p
    test = data_root / 'openreview_batch50_test'
    for report in [test / 'browser-results.json', *sorted(test.glob('batch*/results.json'))]:
        if report.exists():
            for p in json.loads(report.read_text()).get('papers', []):
                fid = p['forum_id']
                yield fid, report.parent / p.get('file', fid + '.pdf'), p
    for report in test.glob('updated-downloader-*/validation.json'):
        for p in json.loads(report.read_text()):
            yield p['id'], report.parent / (p['id'] + '.pdf'), p


def reconcile(db, data_root):
    pending = {r['id'] for r in db.execute("SELECT id FROM papers WHERE status NOT IN ('downloaded','remote_verified')")}
    for fid, file, meta in local_pdfs(data_root):
        if fid not in pending or not file.is_file():
            continue
        data = file.read_bytes()
        if data.startswith(b'%PDF-') and hashlib.sha256(data).hexdigest() == meta['sha256']:
            with db:
                db.execute("UPDATE papers SET status='downloaded',file=?,bytes=?,sha256=?,pages=? WHERE id=?",
                           (str(file.resolve()), len(data), meta['sha256'], meta.get('pages'), fid))
            pending.remove(fid)


def finish_body(db, root, attempt):
    """Recover from a fully saved response, including a crash during extraction."""
    from pypdf import PdfReader
    ids = json.loads(attempt['ids'])
    rows = [db.execute('SELECT * FROM papers WHERE id=?', (fid,)).fetchone() for fid in ids]
    notes = [SimpleNamespace(id=p['id'], number=p['number']) for p in rows]
    data = Path(attempt['body_file']).read_bytes()
    files = unpack_batch(data, notes)
    db.execute('CREATE TABLE IF NOT EXISTS invalid_pdf_observations (paper_id TEXT, sha256 TEXT, attempt_id INTEGER, error TEXT, PRIMARY KEY(paper_id,sha256,attempt_id))')
    verified, retry_invalid, quarantined = {}, [], []
    for fid, pdf in files.items():
        sha = hashlib.sha256(pdf).hexdigest()
        try:
            pages = len(PdfReader(io.BytesIO(pdf)).pages)
            if pages < 1:
                raise ValueError('Empty PDF: ' + fid)
        except Exception as exc:
            if isinstance(exc, (OSError, MemoryError)):
                raise
            with db:
                db.execute('INSERT OR IGNORE INTO invalid_pdf_observations VALUES(?,?,?,?)', (fid,sha,attempt['id'],str(exc)))
            observations = db.execute('SELECT count(*) FROM invalid_pdf_observations WHERE paper_id=? AND sha256=?', (fid,sha)).fetchone()[0]
            if observations >= 2:
                folder = root / 'invalid_pdfs';folder.mkdir(exist_ok=True)
                dest = folder / (fid + '-' + sha + '.pdf')
                tmp = dest.with_suffix('.part');tmp.write_bytes(pdf);tmp.replace(dest)
                quarantined.append(fid)
            else:
                retry_invalid.append(fid)
            continue
        verified[fid] = (pdf, pages, sha)
    # No archive-supplied paths are used.
    records = []
    for p in rows:
        if p['id'] not in verified or p['status']=='remote_verified':
            continue
        pdf, pages, sha = verified[p['id']]
        dest = root / 'pdfs' / p['group_key'] / (p['id'] + '.pdf')
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix('.part')
        with tmp.open('wb') as f:
            f.write(pdf)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(dest)
        records.append((str(dest.resolve()), len(pdf), sha, pages, p['id']))
    with db:
        db.executemany("UPDATE papers SET status='downloaded',file=?,bytes=?,sha256=?,pages=? WHERE id=?", records)
        db.executemany("UPDATE papers SET status='unavailable_invalid_pdf' WHERE id=? AND status NOT IN ('downloaded','remote_verified')", [(fid,) for fid in quarantined])
        status = 'response_saved' if retry_invalid else ('partial_recovered' if quarantined else 'completed')
        db.execute("UPDATE attempts SET status=?,finished_at=? WHERE id=?", (status, utc(), attempt['id']))
        if not retry_invalid:
            # A successful response resets the next consecutive-failure backoff.
            db.execute('CREATE TABLE IF NOT EXISTS retry_failures (paper_id TEXT PRIMARY KEY, failures INTEGER NOT NULL)')
            db.execute('DELETE FROM retry_failures')
            db.execute("DELETE FROM settings WHERE key='retry_not_before'")
        db.execute("INSERT OR REPLACE INTO settings VALUES('last_group',?)", (rows[0]['group_key'],))
    if retry_invalid:
        raise ValueError('Unreadable PDFs require isolated retry: ' + ','.join(retry_invalid))
    cleanup_responses(db, root, attempt['id'])


def cleanup_responses(db, root, attempt_id=None):
    """Remove only completed responses whose extracted PDFs still exist."""
    sql = "SELECT * FROM attempts WHERE status='completed' AND body_file IS NOT NULL"
    args = ()
    if attempt_id is not None:
        sql += ' AND id=?'
        args = (attempt_id,)
    removed, freed = 0, 0
    for attempt in db.execute(sql, args).fetchall():
        body = Path(attempt['body_file'])
        expected = root / 'responses' / f"{attempt['id']:06}.bin"
        if body.resolve() != expected.resolve() or not body.is_file():
            continue
        ids = json.loads(attempt['ids'])
        try:
            for fid in ids:
                p = db.execute('SELECT * FROM papers WHERE id=?', (fid,)).fetchone()
                if not p or p['status'] != 'downloaded' or not p['file'] or Path(p['file']).stat().st_size != p['bytes']:
                    break
            else:
                size = body.stat().st_size
                body.unlink()
                removed += 1
                freed += size
        except OSError as exc:
            logging.warning('Retaining response %s: %s', body, exc)
    return {'removed_responses': removed, 'freed_bytes': freed}


def recover(db, root):
    for a in db.execute("SELECT * FROM attempts WHERE status IN ('inflight','response_saved')").fetchall():
        body = root / 'responses' / f"{a['id']:06}.bin"
        if body.exists():
            with db:
                db.execute("UPDATE attempts SET body_file=?,status='response_saved' WHERE id=?", (str(body.resolve()), a['id']))
            a = db.execute('SELECT * FROM attempts WHERE id=?', (a['id'],)).fetchone()
            try:
                finish_body(db, root, a)
                continue
            except Exception as exc:
                error = 'Saved response needs inspection: ' + str(exc)
        else:
            error = 'Interrupted request; server may have processed it. No automatic retry.'
        with db:
            db.execute("UPDATE attempts SET status='uncertain',error=? WHERE id=?", (error, a['id']))
            for fid in json.loads(a['ids']):
                db.execute("UPDATE papers SET status='uncertain' WHERE id=? AND status NOT IN ('downloaded','remote_verified','unavailable_invalid_pdf')", (fid,))


def quota_deadline(db, headers, status):
    """Persist enforced cooldowns even if the response body later fails."""
    value = headers.get('retry-after')
    deadline = 0
    if value:
        try:
            deadline = time.time() + float(value) + 2
        except ValueError:
            deadline = parsedate_to_datetime(value).timestamp() + 2
    elif status == 429 or headers.get('ratelimit-remaining') == '0':
        value = headers.get('ratelimit-reset', '3600')
        deadline = time.time() + float(value) + 2
    if deadline:
        old = db.execute("SELECT value FROM settings WHERE key='not_before'").fetchone()
        with db:
            db.execute("INSERT OR REPLACE INTO settings VALUES('not_before',?)",
                       (str(max(deadline, float(old[0]) if old else 0)),))


def schedule_retry(db, attempt_id, error, max_retries=5):
    """Keep failed responses and count bounded retries durably across restarts."""
    db.execute('CREATE TABLE IF NOT EXISTS retry_failures (paper_id TEXT PRIMARY KEY, failures INTEGER NOT NULL)')
    a = db.execute('SELECT * FROM attempts WHERE id=?', (attempt_id,)).fetchone()
    if a['status'] in ('completed', 'retry_scheduled', 'retry_exhausted'):
        return False
    if a['http_status'] not in (None, 200, 408, 429, 500, 502, 503, 504):
        return False
    headers = json.loads(a['headers'] or '{}')
    quota_deadline(db, headers, a['http_status'])
    ids = [fid for fid in json.loads(a['ids']) if db.execute('SELECT status FROM papers WHERE id=?', (fid,)).fetchone()[0] not in ('downloaded','remote_verified','unavailable_invalid_pdf')]
    if not ids:
        return False
    failures = 1 + max((db.execute('SELECT failures FROM retry_failures WHERE paper_id=?', (fid,)).fetchone() or [0])[0] for fid in ids)
    exhausted = failures > max_retries
    delay = 5 * 2 ** min(failures-1, 4)
    with db:
        db.executemany('INSERT OR REPLACE INTO retry_failures VALUES(?,?)', [(fid, failures) for fid in ids])
        db.execute('UPDATE attempts SET status=?,finished_at=?,error=? WHERE id=?',
                   ('retry_exhausted' if exhausted else 'retry_scheduled', utc(), str(error), attempt_id))
        db.executemany('UPDATE papers SET status=? WHERE id=?', [('retry_exhausted' if exhausted else 'pending', fid) for fid in ids])
        if not exhausted:
            old = db.execute("SELECT value FROM settings WHERE key='retry_not_before'").fetchone()
            db.execute("INSERT OR REPLACE INTO settings VALUES('retry_not_before',?)", (str(max(time.time()+delay, float(old[0]) if old else 0)),))
    print(json.dumps({'attempt':attempt_id,'retry':not exhausted,'failures':failures,'backoff_seconds':None if exhausted else delay}), flush=True)
    return True


def summary(db, root):
    counts = dict(Counter(r['status'] for r in db.execute('SELECT status FROM papers')))
    groups = db.execute('SELECT group_key,status,count(*) n FROM papers GROUP BY group_key,status ORDER BY group_key').fetchall()
    report = {'updated_at': utc(), 'papers': counts,
              'pending_batches': db.execute("SELECT count(DISTINCT batch_no) FROM papers WHERE status='pending'").fetchone()[0],
              'attempts': db.execute('SELECT count(*) FROM attempts').fetchone()[0]}
    text = '# ReviewBench download queue\n\nUpdated: ' + report['updated_at'] + '\n\n' + str(counts) + '\n\n| Conference/year | Status | Papers |\n|---|---|---:|\n'
    text += ''.join(f"| {g['group_key']} | {g['status']} | {g['n']} |\n" for g in groups)
    (root / 'TALLY.md').write_text(text)
    (root / 'status.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def group_order(db):
    groups = [r[0] for r in db.execute('SELECT DISTINCT group_key FROM papers WHERE api_version=2')]
    return sorted((g for g in groups if g.rsplit('/', 1)[1].isdigit()),
                  key=lambda g: (int(g.rsplit('/', 1)[1]), g.split('/')[0]))


def next_batch(db):
    groups = group_order(db)
    cursor = db.execute("SELECT value FROM settings WHERE key='last_group'").fetchone()
    start = (groups.index(cursor['value']) + 1) % len(groups) if cursor and cursor['value'] in groups else 0
    for offset in range(len(groups)):
        group = groups[(start + offset) % len(groups)]
        rows = db.execute("SELECT * FROM papers WHERE group_key=? AND status='pending' AND api_version=2 ORDER BY id", (group,)).fetchall()
        # Track number collisions must go in separate ZIP requests.
        selected, numbers = [], set()
        for paper in rows:
            if paper['number'] not in numbers:
                selected.append(paper); numbers.add(paper['number'])
                if len(selected) == 50: break
        if selected: return selected
    return []


def main(argv=None):
    parser = argparse.ArgumentParser(description='Resume the prepared ReviewBench queue without editing scripts.')
    parser.add_argument('--stop-at-limit', action='store_true', help='Stop at quota exhaustion rather than wait for renewal')
    parser.add_argument('--retry-failures', action='store_true', help='Retry transient and invalid responses up to five times with durable exponential backoff')
    parser.add_argument('--status', action='store_true', help='Offline status; no login or downloads')
    parser.add_argument('--username', help='OpenReview email (otherwise environment or prompt)')
    parser.add_argument('--keychain', action='store_true', help='Read password from macOS Keychain service pangram-openreview')
    parser.add_argument('--max-batches', type=int, default=100, help='Successful batches this run (default 100; 0 means all remaining)')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3] / 'data/reviewbench_download_queue')
    args = parser.parse_args(argv)
    if args.max_batches < 0:
        parser.error('--max-batches must be nonnegative')
    root = args.root.resolve(); root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'runner.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another queue runner is active. No requests sent.')
    db = connect(root / 'queue.sqlite3')
    try:
        plan = json.loads((root / 'plan.json').read_text())
        seed(db, plan); recover(db, root); reconcile(db, root.parent)
        cleanup_responses(db, root)
        print(json.dumps(summary(db, root)), flush=True)
        if args.status:
            return
        if args.retry_failures:
            for a in db.execute("SELECT * FROM attempts WHERE status='uncertain'").fetchall():
                schedule_retry(db, a['id'], a['error'])
        if db.execute("SELECT count(*) FROM papers WHERE status IN ('uncertain','failed')").fetchone()[0]:
            raise SystemExit('A prior request needs inspection. See queue.sqlite3/TALLY.md; no requests sent.')
        if not next_batch(db):
            print('No eligible dated API 2 papers remain. Legacy and undated papers are held aside.'); return
        import openreview
        from requests.adapters import HTTPAdapter
        username, password = credentials(args.keychain, args.username)
        client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net', username=username, password=password)
        password = None
        client.session.mount('https://', HTTPAdapter(max_retries=0))
        completed = 0
        try:
            while args.max_batches == 0 or completed < args.max_batches:
                papers = next_batch(db)
                if not papers: break
                if any(p['number'] is None for p in papers) or len({p['number'] for p in papers}) != len(papers):
                    raise RuntimeError('Missing or duplicate submission numbers; no batch request sent')
                setting = db.execute("SELECT value FROM settings WHERE key='not_before'").fetchone()
                if setting:
                    delay = max(0, float(setting['value']) - time.time())
                    if delay:
                        if args.stop_at_limit:
                            print(f'Quota exhausted; reset in {delay:.0f} seconds. No request sent.', flush=True)
                            break
                        print(f'Quota cooldown: waiting {delay:.0f} seconds. Ctrl-C safely stops.', flush=True)
                        time.sleep(delay)
                retry = db.execute("SELECT value FROM settings WHERE key='retry_not_before'").fetchone()
                if retry:
                    time.sleep(max(0, float(retry[0])-time.time()))
                aid = begin_attempt(db, papers[0]['batch_no'], papers)
                print(f"Request {aid}: batch {papers[0]['batch_no']}, {papers[0]['group_key']}, {len(papers)} PDFs", flush=True)
                params = {'name': 'pdf'}
                params['ids' if len(papers) > 1 else 'id'] = ','.join(p['id'] for p in papers)
                t = time.monotonic()
                try:
                    with client.session.get('https://api2.openreview.net/attachment', params=params,
                                            headers=client.headers, stream=True, timeout=(30, 300)) as response:
                        headers = {k.lower(): v for k, v in response.headers.items() if k.lower().startswith(('ratelimit','retry-after')) or k.lower()=='content-type'}
                        print(json.dumps({'status': response.status_code, 'headers': headers}), flush=True)
                        with db:
                            db.execute('UPDATE attempts SET http_status=?,headers=? WHERE id=?', (response.status_code, json.dumps(headers), aid))
                        quota_deadline(db, headers, response.status_code)
                        if response.status_code != 200:
                            error = response.text[:4000]
                            if response.status_code == 404:
                                with db:
                                    db.execute("UPDATE attempts SET status='rejected',finished_at=?,error=? WHERE id=?", (utc(), error, aid))
                                    db.executemany("UPDATE papers SET status='failed' WHERE id=?", [(p['id'],) for p in papers])
                            if response.status_code == 429:
                                retry = headers.get('retry-after') or headers.get('ratelimit-reset')
                                if retry and retry.isdigit():
                                    with db:
                                        db.execute("INSERT OR REPLACE INTO settings VALUES('not_before',?)", (str(time.time()+int(retry)+2),))
                                        db.execute("UPDATE attempts SET status='rate_limited',finished_at=?,error=? WHERE id=?", (utc(), error, aid))
                                        db.executemany("UPDATE papers SET status='pending' WHERE id=?", [(p['id'],) for p in papers])
                                    if args.stop_at_limit:
                                        print('Rate limit reached; stopping this run.', flush=True)
                                        break
                                    continue
                            raise RuntimeError(f'HTTP {response.status_code}: {error}')
                        body = root / 'responses' / f'{aid:06}.bin'; body.parent.mkdir(exist_ok=True)
                        tmp = body.with_suffix('.part')
                        with tmp.open('wb') as f:
                            for chunk in response.iter_content(1024*1024):
                                if chunk: f.write(chunk)
                            f.flush()
                            os.fsync(f.fileno())
                        tmp.replace(body)
                    with db:
                        db.execute("UPDATE attempts SET status='response_saved',body_file=? WHERE id=?", (str(body), aid))
                    finish_body(db, root, db.execute('SELECT * FROM attempts WHERE id=?', (aid,)).fetchone())
                except Exception as exc:
                    # Local write failures must not burn network quota through retries.
                    from requests.exceptions import RequestException
                    if not args.retry_failures or (isinstance(exc, OSError) and not isinstance(exc, RequestException)):
                        raise
                    if not schedule_retry(db, aid, exc):
                        raise
                    summary(db, root)
                    continue
                completed += 1
                if headers.get('ratelimit-remaining') == '0' and headers.get('ratelimit-reset','').isdigit():
                    with db:
                        db.execute("INSERT OR REPLACE INTO settings VALUES('not_before',?)", (str(time.time()+int(headers['ratelimit-reset'])+2),))
                print(f'Saved {len(papers)} PDFs in {time.monotonic()-t:.1f}s', flush=True)
                summary(db, root)
                if args.stop_at_limit and headers.get('ratelimit-remaining') == '0':
                    print('Quota exhausted; stopping this run.', flush=True)
                    break
        except (Exception, KeyboardInterrupt) as exc:
            recover(db, root)
            print('Stopped: ' + (str(exc) or 'interrupted'), flush=True)
            raise SystemExit(1)
        finally:
            print(json.dumps(summary(db, root)), flush=True)
    finally:
        db.close()
        lock.close()


if __name__ == '__main__':
    logging.getLogger('pypdf').setLevel(logging.ERROR)
    main()
