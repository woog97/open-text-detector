"""Recover known missing public ICLR/NeurIPS PDFs without authentication."""
import hashlib
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from email.parser import BytesParser
from email import policy

ROOT = Path(__file__).resolve().parents[2] / 'data/openreview_pre2023_balanced'


def main():
    missing = json.loads((ROOT / 'missing.json').read_text())
    metadata = json.loads((ROOT / 'public-metadata.json').read_text())
    # Alternate conferences within each year; exhaust newer years first.
    ordered = []
    for year in sorted({int(p['group_key'][-4:]) for p in missing}, reverse=True):
        groups = [[p for p in missing if p['group_key'] == f'{conf}/{year}'] for conf in ('iclr', 'neurips')]
        while any(groups):
            for group in groups:
                if group:
                    ordered.append(group.pop(0))
    results = []
    last_request = 0.0
    opener = urllib.request.build_opener()  # No cookies, login, or credential lookup.
    for paper in ordered:
        fid = paper['id']
        dest = ROOT / 'pdfs' / paper['group_key'] / (fid + '.pdf')
        row = {'id': fid, 'group': paper['group_key'], 'authenticated': False}
        source = metadata.get(fid, {}).get('pdf') or ''
        if dest.exists():
            row['status'] = 'already_saved'
        elif not source:
            row['status'] = 'no_pdf_in_source_metadata'
        else:
            if source.startswith('/pdf/'):
                source = 'https://openreview.net' + source
            elif source.startswith('https://arxiv.org/abs/'):
                source = source.replace('/abs/', '/pdf/', 1)
            elif source.startswith('http://www.stats.ox.ac.uk/'):
                source = source.replace('http:', 'https:', 1)
            if not source.startswith('https://'):
                row['status'] = 'unsupported_source'
            else:
                if shutil.disk_usage(ROOT).free < 5 * 1024**3:
                    raise RuntimeError('Less than 5 GiB free; stopping')
                time.sleep(max(0, 7 - (time.monotonic() - last_request)))
                try:
                    last_request = time.monotonic()
                    with opener.open(source, timeout=60) as response:
                        body = response.read(100 * 1024**2 + 1)
                    if len(body) > 100 * 1024**2:
                        raise ValueError('File exceeds 100 MiB guard')
                    if not body.startswith(b'%PDF-') and body.startswith(b'--'):
                        boundary = body.split(b'\r\n', 1)[0][2:]
                        if len(boundary) > 70 or not body.rstrip().endswith(b'--' + boundary + b'--'):
                            raise ValueError('Invalid MIME boundary')
                        msg = BytesParser(policy=policy.default).parsebytes(b'Content-Type: multipart/form-data; boundary="' + boundary + b'"\r\nMIME-Version: 1.0\r\n\r\n' + body)
                        parts = list(msg.iter_parts())
                        if len(parts) != 1 or msg.defects:
                            raise ValueError('Invalid MIME payload')
                        body = parts[0].get_payload(decode=True)
                    if not body or not body.startswith(b'%PDF-') or b'%%EOF' not in body[-4096:]:
                        raise ValueError('Incomplete or non-PDF response')
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.with_suffix('.part').write_bytes(body)
                    dest.with_suffix('.part').replace(dest)
                    row.update(status='downloaded', bytes=len(body), sha256=hashlib.sha256(body).hexdigest(), file=str(dest))
                except urllib.error.HTTPError as exc:
                    row.update(status='http_error', http_status=exc.code)
                    if exc.code in (403, 429):
                        results.append(row)
                        (ROOT / 'recovery-results.json').write_text(json.dumps(results, indent=2))
                        print(json.dumps(row), flush=True)
                        print('Access or rate limit response; stopped without retry.', flush=True)
                        return
                except Exception as exc:
                    row.update(status='error', error=str(exc))
        results.append(row)
        (ROOT / 'recovery-results.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(row), flush=True)
    print('Recovery pass complete.', flush=True)


if __name__ == '__main__':
    main()
