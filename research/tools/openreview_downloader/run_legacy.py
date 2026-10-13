"""Resume individual API 1 downloads with durable request accounting."""
import argparse
import fcntl
import hashlib
import io
import json
import logging
from pathlib import Path
import sqlite3
import time
from datetime import datetime
from openreview_downloader import queue as q

DATA = Path(__file__).resolve().parents[2] / 'data'
ROOT = DATA / 'openreview_legacy_download_queue'
STOP = DATA / 'openreview_iclr2027_all/submissions-visible.json'


def skip_existing(db, paper, root=ROOT):
    """Refresh shared progress and adopt a complete local PDF before requesting."""
    current = db.execute('SELECT * FROM papers WHERE id=?', (paper['id'],)).fetchone()
    if current['status'] != 'pending':
        return True
    candidates = [root / 'pdfs' / paper['group_key'] / (paper['id'] + '.pdf')]
    if current['file']:
        candidates.insert(0, Path(current['file']))
    from pypdf import PdfReader
    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
            if not data.startswith(b'%PDF-'):
                continue
            pages = len(PdfReader(io.BytesIO(data)).pages)
            if pages < 1:
                continue
        except Exception:
            continue
        with db:
            db.execute("UPDATE papers SET status='downloaded',file=?,bytes=?,sha256=?,pages=? WHERE id=? AND status='pending'",
                       (str(path.resolve()), len(data), hashlib.sha256(data).hexdigest(), pages, paper['id']))
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-requests', type=int, default=140)
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.max_requests <= 140:
        parser.error('max-requests must be 1–140')
    ROOT.mkdir(exist_ok=True)
    locks = []
    for path in [ROOT / 'runner.lock', DATA / 'reviewbench_download_queue/runner.lock']:
        lock = path.open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            for held in locks:
                held.close()
            print('Another downloader is active; skipping this hourly run without requests.', flush=True)
            return
        locks.append(lock)
    db = q.connect(ROOT / 'queue.sqlite3')
    allowed = {'iclr': {2013, 2014, *range(2017, 2024)}, 'neurips': {2021, 2022}, 'corl': {2021, 2022, 2023}}
    batches = []
    catalogue = DATA / 'openreview_catalogue_all/papers.jsonl'
    for line in catalogue.open():
        paper = json.loads(line)
        conference = paper['conference'].lower()
        if paper.get('api_version') == 1 and int(paper['year']) in allowed.get(conference, set()):
            batches.append({'batch': len(batches) + 1, 'group': f"{conference}/{paper['year']}", 'papers': [paper]})
    q.seed(db, {'batches': batches})
    with db:
        db.execute("UPDATE papers SET status='pending' WHERE status='held_legacy'")
    q.recover(db, ROOT)
    q.reconcile(db, DATA)
    print(json.dumps(q.summary(db, ROOT)), flush=True)
    if args.status:
        return
    if STOP.exists():
        print('Stopped for ICLR 2027 visibility.', flush=True)
        return
    # Count attempts conservatively across persistent download queues.
    def budget_available():
        count = 0
        for path in DATA.glob('*/queue.sqlite3'):
            other = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
            try:
                tables = {r[0] for r in other.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if 'attempts' in tables:
                    for (stamp,) in other.execute('SELECT started_at FROM attempts'):
                        if datetime.fromisoformat(stamp).timestamp() > time.time() - 3600:
                            count += 1
                if 'settings' in tables:
                    row = other.execute("SELECT value FROM settings WHERE key='not_before'").fetchone()
                    if row and float(row[0]) > time.time():
                        return False
            finally:
                other.close()
        return count < 140
    # Hourly pings may occur while the previous run's later requests are
    # still inside a rolling hour. Wait for those slots, preserving pacing.
    def wait_for_slot():
        deadline = time.monotonic() + 600
        while not STOP.exists():
            if budget_available():
                return True
            if time.monotonic() >= deadline:
                print('Quota still unavailable; stopping this run.', flush=True)
                return False
            time.sleep(5)
        return False
    if not wait_for_slot():
        return
    import openreview
    from requests.adapters import HTTPAdapter
    username, password = q.credentials()
    client = openreview.Client(baseurl='https://api.openreview.net', username=username, password=password)
    password = None
    client.session.mount('https://', HTTPAdapter(max_retries=0))
    rows = db.execute("SELECT * FROM papers WHERE status='pending'").fetchall()
    rows.sort(key=lambda p: (int(p['group_key'].rsplit('/', 1)[1]), p['group_key'], p['id']))
    try:
        requested = 0
        for paper in rows:
            if requested >= args.max_requests:
                break
            if skip_existing(db, paper):
                continue
            if STOP.exists() or not wait_for_slot():
                print('Visibility stop or quota limit reached.', flush=True)
                break
            # Recheck after any quota wait: files may have appeared meanwhile.
            q.reconcile(db, DATA)
            if skip_existing(db, paper):
                continue
            aid = q.begin_attempt(db, paper['batch_no'], [paper])
            requested += 1
            print(f"Request {aid}: {paper['group_key']} {paper['id']}", flush=True)
            with client.session.get(client.pdf_url, params={'id': paper['id']}, headers=client.headers, timeout=(30, 180)) as response:
                headers = {k.lower(): v for k, v in response.headers.items() if k.lower().startswith(('ratelimit', 'retry-after')) or k.lower() == 'content-type'}
                print(json.dumps({'status': response.status_code, 'headers': headers}), flush=True)
                with db:
                    db.execute('UPDATE attempts SET http_status=?,headers=? WHERE id=?', (response.status_code, json.dumps(headers), aid))
                    reset = headers.get('retry-after') or headers.get('ratelimit-reset')
                    if response.status_code == 429 or headers.get('ratelimit-remaining') == '0':
                        delay = float(reset) if reset and reset.isdigit() else 3600
                        if delay > 1e9:
                            delay = max(0, delay - time.time())
                        db.execute("INSERT OR REPLACE INTO settings VALUES('not_before',?)", (str(time.time() + delay + 2),))
                    if response.status_code != 200:
                        status = 'pending' if response.status_code == 429 else 'failed'
                        db.execute('UPDATE papers SET status=? WHERE id=?', (status, paper['id']))
                        db.execute("UPDATE attempts SET status='rejected',finished_at=?,error=? WHERE id=?", (q.utc(), f'HTTP {response.status_code}', aid))
                if response.status_code != 200:
                    break
                if not response.content.startswith(b'%PDF-'):
                    raise ValueError('Non-PDF response; stopping for inspection')
                body = ROOT / 'responses' / f'{aid:06}.bin'
                body.parent.mkdir(exist_ok=True)
                tmp = body.with_suffix('.part')
                tmp.write_bytes(response.content)
                tmp.replace(body)
            with db:
                db.execute("UPDATE attempts SET status='response_saved',body_file=? WHERE id=?", (str(body), aid))
            q.finish_body(db, ROOT, db.execute('SELECT * FROM attempts WHERE id=?', (aid,)).fetchone())
            q.summary(db, ROOT)
    finally:
        q.recover(db, ROOT)
        print(json.dumps(q.summary(db, ROOT)), flush=True)


if __name__ == '__main__':
    logging.getLogger('pypdf').setLevel(logging.ERROR)
    main()
