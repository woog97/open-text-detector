"""Publish the Atlas public catalogue and reader lookup shards as one immutable revision.

The Worker used to build the catalogue on request: list every archive PDF and parse the ~46 MB source catalogue in
each cold isolate, which took ~30 s and ran close to the 128 MB memory limit. This script builds the same result with
the same code (lib/atlas-archive.ts buildPublicCatalogue), uploads a gzip copy plus 256 id -> (PDF hash, detail hash)
shards under atlas-public/public/<revision>/, verifies them, then advances atlas-public/public/current.json.

Rerun after any change to atlas-public/catalogue.json, the PDF archive, or the archive indexes. Until then the Worker
keeps serving the previous revision; PDFs added since are still readable (unclassified) by ID.
"""
import gzip
import hashlib
import json
import subprocess
import tempfile
import time
import urllib.parse
from pathlib import Path
from upload_paper_pdfs import ROOT, BASE, request

SOURCES = ('atlas-public/catalogue.json', 'indexes/downloaded-papers.json', 'indexes/iclr2027-pdfs.json')
CURRENT = 'atlas-public/public/current.json'


def list_objects(prefix):
    found, cursor = [], None
    while True:
        params = {'per_page': 1000, 'prefix': prefix}
        if cursor:
            params['cursor'] = cursor
        page = request(BASE+'?'+urllib.parse.urlencode(params))
        found += page['result']
        info = page.get('result_info', {})
        if not info.get('is_truncated'):
            return found
        cursor = info['cursor']


def put(key, data, content_type='application/json'):
    request(BASE+'/'+key, data, content_type)
    assert hashlib.sha256(request(BASE+'/'+key, raw=True)).digest() == hashlib.sha256(data).digest(), f'Readback mismatch: {key}'


def main(force=False):
    # Source etags first: a body newer than its recorded etag only causes one extra republish later.
    etags = {o['key']: o['etag'] for key in SOURCES for o in list_objects(key) if o['key'] == key}
    pdfs = [{'key': o['key'], 'size': o['size'], 'customMetadata': o.get('custom_metadata') or {}} for o in list_objects('papers/')]
    signature = hashlib.sha256(json.dumps([etags, sorted((p['key'], p['size']) for p in pdfs)]).encode()).hexdigest()
    try:
        live = request(BASE+'/'+CURRENT)
    except RuntimeError:
        live = {}
    if live.get('source_signature') == signature and not force:
        print(f"public catalogue already current: revision {live['revision']}", flush=True); return
    rows = []
    for key in SOURCES[1:]:
        if key in etags:
            rows += request(BASE+'/'+key)['papers']
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp/'source.json').write_bytes(request(BASE+'/'+SOURCES[0], raw=True))
        (tmp/'objects.json').write_text(json.dumps(pdfs))
        (tmp/'rows.json').write_text(json.dumps(rows))
        code = """import fs from 'node:fs';import {buildPublicCatalogue} from './app/lib/atlas-archive.ts';const d=process.argv[1],read=f=>JSON.parse(fs.readFileSync(d+'/'+f,'utf8'));const {catalogue,shards}=buildPublicCatalogue(read('source.json'),read('objects.json'),read('rows.json'));fs.writeFileSync(d+'/catalogue.json',JSON.stringify(catalogue));fs.writeFileSync(d+'/shards.json',JSON.stringify(shards));"""
        subprocess.run(['node', '--experimental-strip-types', '--input-type=module', '-e', code, str(tmp)], cwd=ROOT, check=True)
        body = (tmp/'catalogue.json').read_bytes()
        shards = json.loads((tmp/'shards.json').read_text())
    revision = hashlib.sha256(body).hexdigest()
    base = f'atlas-public/public/{revision}'
    compressed = gzip.compress(body, compresslevel=9, mtime=0)
    put(f'{base}/catalogue.json.gz', compressed)
    for prefix in (f'{i:02x}' for i in range(256)):
        put(f'{base}/reader/{prefix}.json', json.dumps(shards.get(prefix, {}), separators=(',', ':')).encode())
    papers = sum(map(len, shards.values()))
    manifest = dict(revision=revision, catalogue_key=f'{base}/catalogue.json.gz', shards_prefix=f'{base}/reader/',
                    source_signature=signature, source_etags=etags, papers=papers, bytes=len(body),
                    gzip_bytes=len(compressed), published_at=time.time())
    put(CURRENT, json.dumps(manifest).encode())
    print(f'public catalogue published: {papers:,} papers, {len(body)/1e6:.1f} MB ({len(compressed)/1e6:.1f} MB gzip), revision {revision}', flush=True)


if __name__ == '__main__':
    import sys
    main(force='--force' in sys.argv)
