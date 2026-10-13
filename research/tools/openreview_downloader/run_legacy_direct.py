"""Download the saved legacy catalogue directly, smallest conference-years first."""
import fcntl
import json
import logging
import os
from pathlib import Path
import time
from email import policy
from email.parser import BytesParser
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import requests
from openreview_downloader import queue as q

DATA = Path(__file__).resolve().parents[2] / 'data'
ROOT = DATA / 'openreview_legacy_download_queue'


def pdf_payload(data):
    """Unwrap legacy uploads that retained a MIME form-data envelope."""
    if data.startswith(b'%PDF-'):
        return data
    first_line = data.split(b'\r\n', 1)[0]
    if not first_line.startswith(b'--') or not 3 <= len(first_line) <= 72:
        raise ValueError('Non-PDF response; stopped for inspection')
    boundary = first_line[2:]
    if not all(c in b'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-' for c in boundary):
        raise ValueError('Invalid MIME boundary')
    if not data.rstrip().endswith(b'--' + boundary + b'--'):
        raise ValueError('Incomplete MIME envelope')
    message = BytesParser(policy=policy.default).parsebytes(
        b'Content-Type: multipart/form-data; boundary="' + boundary + b'"\r\nMIME-Version: 1.0\r\n\r\n' + data)
    parts = list(message.iter_parts())
    if message.defects or len(parts) != 1 or parts[0].defects:
        raise ValueError('Unexpected MIME envelope')
    payload = parts[0].get_payload(decode=True)
    if not payload or not payload.startswith(b'%PDF-'):
        raise ValueError('MIME envelope does not contain a PDF')
    return payload


def wait_or_stop(seconds):
    """Allow a clean shutdown even during a long network outage."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if (ROOT / 'STOP_DIRECT').exists():
            return False
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    return not (ROOT / 'STOP_DIRECT').exists()


def fetch_with_recovery(db, paper, session, url, params, headers):
    """Retry safe PDF GETs, including failures while reading response bodies."""
    failures = 0
    while not (ROOT / 'STOP_DIRECT').exists():
        aid = q.begin_attempt(db, paper['batch_no'], [paper])
        print(json.dumps({'at': q.utc(), 'attempt': aid, 'group': paper['group_key'], 'id': paper['id']}), flush=True)
        try:
            response = session.get(url, params=params, headers=headers, timeout=(30, 180))
        except requests.exceptions.SSLError:
            raise  # Certificate failures require inspection, not automatic retries.
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError, requests.exceptions.ContentDecodingError) as exc:
            error, status = type(exc).__name__, None
        else:
            if response.status_code not in (500, 502, 503, 504):
                return response, aid
            # Rate-limit/challenge responses are deliberately returned to the
            # caller, which stops. Retry only temporary server errors here.
            status, error = response.status_code, 'HTTP ' + str(response.status_code)
            if response.headers.get('Retry-After') or response.headers.get('ratelimit-remaining') == '0':
                return response, aid
            response.close()
        failures += 1
        delay = min(300, 7 * 2 ** min(failures - 1, 6))
        with db:
            db.execute("UPDATE attempts SET status='retryable_error',finished_at=?,http_status=?,error=? WHERE id=?", (q.utc(), status, error, aid))
            db.execute("UPDATE papers SET status='pending' WHERE id=?", (paper['id'],))
        q.summary(db, ROOT)
        print(json.dumps({'at': q.utc(), 'event': 'automatic_retry', 'id': paper['id'], 'error': error, 'wait_seconds': delay}), flush=True)
        if not wait_or_stop(delay):
            return None
    return None


def main():
    locks = []
    for path in [ROOT / 'runner.lock', DATA / 'reviewbench_download_queue/runner.lock']:
        lock = path.open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    (ROOT / 'direct.pid').write_text(str(os.getpid()) + '\n')
    db = q.connect(ROOT / 'queue.sqlite3')
    q.recover(db, ROOT)
    rows = db.execute("""SELECT p.* FROM papers p
        JOIN (SELECT group_key, count(*) n FROM papers GROUP BY group_key) g
        ON p.group_key=g.group_key WHERE p.status='pending'
        ORDER BY CASE WHEN p.group_key LIKE 'neurips/%' THEN 0 ELSE 1 END,
                 g.n, p.group_key, p.id""").fetchall()
    plan = [dict(r) for r in db.execute("""SELECT group_key,count(*) papers FROM papers
        GROUP BY group_key ORDER BY CASE WHEN group_key LIKE 'neurips/%' THEN 0 ELSE 1 END,
        papers,group_key""")]
    catalogue = {p['forum_id']: p for p in map(json.loads, (DATA / 'openreview_catalogue_all/papers.jsonl').open())}
    (ROOT / 'direct-plan.json').write_text(json.dumps({'interval_seconds': 7, 'url': 'Direct catalogue PDF or authenticated https://api.openreview.net/pdf?id=<paper_id>', 'groups': plan}, indent=2) + '\n')
    print(json.dumps({'event': 'started', 'pending': len(rows), 'groups': plan}), flush=True)
    session = requests.Session()
    session.mount('https://', requests.adapters.HTTPAdapter(max_retries=0))
    client = None
    try:
        for paper in rows:
            if (ROOT / 'STOP_DIRECT').exists():
                print('Stopped by STOP_DIRECT file.', flush=True)
                break
            source = catalogue[paper['id']].get('pdf') or ''
            parsed = urlparse(source)
            request_session, request_headers, params = session, {}, None
            if parsed.scheme in ('http', 'https') and parsed.netloc not in ('openreview.net', 'api.openreview.net'):
                url = source
                if parsed.netloc == 'arxiv.org' and parsed.path.startswith('/abs/'):
                    url = 'https://arxiv.org/pdf/' + parsed.path.removeprefix('/abs/')
                elif parsed.netloc == 'www.dropbox.com':
                    query = dict(parse_qsl(parsed.query))
                    query['dl'] = '1'
                    url = urlunparse(parsed._replace(query=urlencode(query)))
            else:
                if client is None:
                    import openreview
                    username, password = q.credentials()
                    client = openreview.Client(baseurl='https://api.openreview.net', username=username, password=password)
                    password = None
                    client.session.mount('https://', requests.adapters.HTTPAdapter(max_retries=0))
                url = client.pdf_url
                request_session, request_headers, params = client.session, client.headers, {'id': paper['id']}
            fetched = fetch_with_recovery(db, paper, request_session, url, params, request_headers)
            if fetched is None:
                break
            response, aid = fetched
            with response:
                headers = {k.lower(): v for k, v in response.headers.items() if k.lower().startswith(('ratelimit', 'retry-after')) or k.lower() == 'content-type'}
                with db:
                    db.execute('UPDATE attempts SET http_status=?,headers=? WHERE id=?', (response.status_code, json.dumps(headers), aid))
                if response.status_code != 200:
                    with db:
                        db.execute('UPDATE papers SET status=? WHERE id=?', ('failed' if response.status_code == 404 else 'pending', paper['id']))
                        db.execute("UPDATE attempts SET status='rejected',finished_at=?,error=? WHERE id=?", (q.utc(), 'HTTP ' + str(response.status_code), aid))
                    print(json.dumps({'event': 'http_error', 'status': response.status_code, 'headers': headers}), flush=True)
                    q.summary(db, ROOT)
                    if response.status_code == 404:
                        wait_or_stop(7)
                        continue
                    break
                payload = pdf_payload(response.content)
                body = ROOT / 'responses' / f'{aid:06}.bin'
                body.parent.mkdir(exist_ok=True)
                tmp = body.with_suffix('.part')
                with tmp.open('wb') as f:
                    f.write(payload)
                    f.flush()
                    os.fsync(f.fileno())
                tmp.replace(body)
            with db:
                db.execute("UPDATE attempts SET status='response_saved',body_file=? WHERE id=?", (str(body), aid))
            q.finish_body(db, ROOT, db.execute('SELECT * FROM attempts WHERE id=?', (aid,)).fetchone())
            print(json.dumps({'at': q.utc(), 'event': 'downloaded', 'id': paper['id'], 'status': q.summary(db, ROOT)}), flush=True)
            if headers.get('ratelimit-remaining') == '0':
                print('Server quota exhausted; stopping.', flush=True)
                break
            # Seven seconds after completion: never overlapping downloads.
            wait_or_stop(7)
    finally:
        q.recover(db, ROOT)
        print(json.dumps({'event': 'stopped', 'status': q.summary(db, ROOT)}), flush=True)
        session.close()
        db.close()


if __name__ == '__main__':
    logging.getLogger('pypdf').setLevel(logging.ERROR)
    main()
