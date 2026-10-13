"""Upload downloaded research PDFs to R2, deduplicated with a conference/year index.

Resumable: remote object sizes/ETags are checked before uploading and after completion.
Uses the existing Wrangler session; never prints credentials or publishes local paths.
"""
import argparse
import concurrent.futures
import email.utils
import hashlib
import json
import subprocess
import threading
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'research/data'
OUT = ROOT / 'app/.sites-runtime/atlas'
CONFIG = Path.home() / 'Library/Preferences/.wrangler/config/default.toml'
BASE = 'https://api.cloudflare.com/client/v4/accounts/6147916993a1cbfa4e3898aaeed27ae3/r2/buckets/pangram-paper-atlas/objects'
refresh_lock = threading.Lock()
last_refresh = time.monotonic()
rate_limit_lock = threading.Lock()
rate_limit_until = 0.0


def refresh(force=False):
    global last_refresh
    with refresh_lock:
        if not force and time.monotonic() - last_refresh < 600:
            return
        proc = subprocess.run([str(ROOT/'app/node_modules/.bin/wrangler'), 'whoami'], cwd=ROOT/'app', capture_output=True)
        if proc.returncode:
            raise RuntimeError('Cloudflare login refresh failed')
        last_refresh = time.monotonic()


def request(url, data=None, content_type=None, raw=False):
    global rate_limit_until
    for attempt in range(20):
        try:
            while True:
                with rate_limit_lock:
                    remaining = rate_limit_until - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(remaining, 10))
            refresh()
            token = tomllib.loads(CONFIG.read_text())['oauth_token']
            headers = {'Authorization': 'Bearer '+token}
            if content_type:
                headers.update({'Content-Type': content_type, 'cf-r2-storage-class': 'Standard'})
            req = urllib.request.Request(url, data=data, headers=headers, method='PUT' if data is not None else 'GET')
            with urllib.request.urlopen(req, timeout=240) as response:
                body = response.read()
            if raw:
                return body
            result = json.loads(body) if body else {}
            if result.get('success') is False:
                raise RuntimeError('R2 rejected request')
            return result
        except Exception as error:
            if isinstance(error, urllib.error.HTTPError) and error.code == 401 and attempt == 0:
                refresh(force=True)
                continue
            if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                retry_after = error.headers.get('Retry-After')
                delay = 120
                if retry_after:
                    try:
                        delay = max(delay, int(retry_after))
                    except ValueError:
                        try:
                            delay = max(delay, (email.utils.parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds())
                        except ValueError:
                            pass
                with rate_limit_lock:
                    rate_limit_until = max(rate_limit_until, time.monotonic() + delay)
                print(f'R2 rate limit; waiting at least {delay:.0f}s', flush=True)
            max_attempts = 20 if isinstance(error, urllib.error.HTTPError) and error.code == 429 else 6
            if isinstance(error, urllib.error.HTTPError) and error.code in (401, 403):
                max_attempts = 1
            if attempt + 1 >= max_attempts:
                raise RuntimeError(f'R2 request failed: {type(error).__name__} {getattr(error, "code", "")}') from None
            time.sleep(min(2**attempt, 30))


def remote_objects():
    found = {}; cursor = None
    while True:
        params = {'per_page': 1000}
        if cursor:
            params['cursor'] = cursor
        page = request(BASE+'?'+urllib.parse.urlencode(params))
        found.update({i['key']: i for i in page['result']})
        info = page.get('result_info', {})
        if not info.get('is_truncated'):
            return found
        cursor = info['cursor']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--remove-local-after-verify', action='store_true',
                        help='Remove local PDF copies only after R2 objects and the archive index are verified')
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error('--workers must be between 1 and 12')
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(DATA.rglob('*.pdf'))
    rows = []; unique = {}
    for n, path in enumerate(paths, 1):
        if not path.resolve().is_relative_to(DATA.resolve()):
            continue
        raw = path.read_bytes()
        if not raw.startswith(b'%PDF-'):
            raise ValueError(f'Not a complete PDF: {path.name}')
        sha = hashlib.sha256(raw).hexdigest(); md5 = hashlib.md5(raw).hexdigest()
        relative = path.relative_to(DATA)
        row = {'source': relative.as_posix(), 'filename': path.name, 'sha256': sha, 'bytes': len(raw), 'key': f'papers/{sha}.pdf'}
        if relative.parts[0] in ('reviewbench_download_queue', 'openreview_legacy_download_queue'):
            row.update(conference=relative.parts[2], year=int(relative.parts[3]), forum_id=path.stem)
        rows.append(row)
        unique.setdefault(row['key'], {'path': path, 'sha256': sha, 'md5': md5, 'bytes': len(raw)})
        if n % 1000 == 0:
            print(f'Hashed {n}/{len(paths)} PDFs', flush=True)
    (OUT/'pdf-upload-inventory.json').write_text(json.dumps(rows))
    remote = remote_objects()
    def matches(key, value):
        obj = remote.get(key, {})
        return obj.get('size') == value['bytes'] and obj.get('etag', '').strip('"') == value['md5']
    pending = [(k,v) for k,v in unique.items() if not matches(k,v)]
    total_bytes = sum(v['bytes'] for _,v in pending)
    print(f'{len(rows)} local PDFs; {len(unique)} unique; {len(pending)} uploads ({total_bytes/1e9:.2f} GB); {len(unique)-len(pending)} already verified in R2', flush=True)
    started = time.monotonic(); uploaded_bytes = 0
    def upload(pair):
        key, item = pair; raw = item['path'].read_bytes()
        if hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('PDF changed during upload')
        request(BASE+'/'+urllib.parse.quote(key, safe='/'), raw, 'application/pdf')
        return item['bytes']
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(upload, pair):pair[0] for pair in pending}
        for n, future in enumerate(concurrent.futures.as_completed(jobs), 1):
            try:
                uploaded_bytes += future.result()
            except Exception as error:
                failures.append({'key': jobs[future], 'error': str(error)})
            if n % 100 == 0 or n == len(pending):
                print(f'Uploaded {n}/{len(pending)}; {uploaded_bytes/1e9:.2f}/{total_bytes/1e9:.2f} GB; {time.monotonic()-started:.0f}s; failures={len(failures)}', flush=True)
    if failures:
        (OUT/'pdf-upload-failures.json').write_text(json.dumps(failures))
        raise RuntimeError(f'{len(failures)} uploads failed; rerun to resume')
    remote = remote_objects()
    missing = [k for k,v in unique.items() if not matches(k,v)]
    if missing:
        raise RuntimeError(f'{len(missing)} remote size/checksum mismatches')
    index_key = 'indexes/downloaded-papers.json'
    previous = request(BASE+'/'+index_key).get('papers', []) if index_key in remote else []
    archive = {(r['source'], r['sha256']): r for r in previous}
    archive.update({(r['source'], r['sha256']): r for r in rows})
    archived_rows = list(archive.values())
    index = {'papers': archived_rows, 'groups': dict(Counter(f"{r['conference']}/{r['year']}" for r in archived_rows if 'conference' in r)), 'unique_pdfs': len({r['key'] for r in archived_rows})}
    raw = json.dumps(index, ensure_ascii=False).encode()
    index_key = 'indexes/downloaded-papers.json'
    request(BASE+'/'+index_key, raw, 'application/json')
    index_remote = remote_objects()[index_key]
    assert index_remote['size'] == len(raw) and index_remote['etag'].strip('"') == hashlib.md5(raw).hexdigest()
    if (DATA/'openreview_iclr2027_all/manifest.jsonl').exists():
        from index_iclr2027_pdfs import main as index_uploaded_pdfs
        index_uploaded_pdfs()
    record = OUT/'uploaded.json'
    done = json.loads(record.read_text()) if record.exists() else {}
    done.update({k:v['sha256'] for k,v in unique.items()})
    temp = record.with_suffix('.tmp'); temp.write_text(json.dumps(done)); temp.replace(record)
    summary = {'local_pdfs': len(rows), 'unique_pdfs': len(unique), 'uploaded': len(pending), 'uploaded_bytes': total_bytes, 'verified': len(unique), 'archive_unique_pdfs': index['unique_pdfs'], 'groups': index['groups'], 'index_key': index_key}
    if args.remove_local_after_verify:
        freed = removed = 0
        for row in rows:
            path = DATA / row['source']
            if not path.is_file() or not path.resolve().is_relative_to(DATA.resolve()):
                continue
            data = path.read_bytes()
            if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
                raise RuntimeError(f'Local PDF changed before cleanup: {row["source"]}')
            if not matches(row['key'], {'bytes': row['bytes'], 'md5': hashlib.md5(data).hexdigest()}):
                raise RuntimeError(f'R2 verification lost before cleanup: {row["key"]}')
            path.unlink()
            freed += len(data)
            removed += 1
            if removed % 1000 == 0:
                print(f'Removed {removed} verified local PDFs; freed {freed/1e9:.2f} GB', flush=True)
        summary.update(removed_local_pdfs=removed, freed_local_bytes=freed)
    (OUT/'pdf-upload-summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
