"""Space-side: fetch the selected PDFs from R2 (export Worker bundles), extract them (baseline), clean them
(positioned-clean-v2) and persist everything to the bucket. Runs detached in /tmp/pangram-humancal.

Env: PDF_EXPORT_URL, PDF_EXPORT_TOKEN. Steps are idempotent; rerun after a restart (after --restore of inputs).
Log: /tmp/pangram-humancal/pipeline.log (one JSON line per event).
"""
import concurrent.futures as cf, hashlib, json, os, struct, subprocess, sys, tarfile, time, urllib.request
from pathlib import Path

R = Path('/tmp/pangram-humancal'); PDFS = R / 'source/pdfs'; TOOLS = Path('/tmp/pangram-tools')
PERSIST = Path('/data/workspace/datasets')
SRC_DS, EXT_DS, CLEAN_DS = 'human-cal-2019-2022-20261006-source', 'human-cal-2019-2022-20261006-extraction', 'human-cal-2019-2022-20261006-clean-text-v2'
log = open(R / 'pipeline.log', 'a')
def say(**k): log.write(json.dumps({'t': time.strftime('%H:%M:%S'), **k}) + '\n'); log.flush()


def tools():
    if (TOOLS / 'env/bin/pdftotext').exists() and (TOOLS / 'env/bin/tesseract').exists(): return
    TOOLS.mkdir(exist_ok=True)
    # The Space has neither curl nor bzip2; Python's urllib and tarfile (bz2 module) do both.
    with urllib.request.urlopen('https://micro.mamba.pm/api/micromamba/linux-64/latest', timeout=300) as r:
        (TOOLS / 'micromamba.tar.bz2').write_bytes(r.read())
    with tarfile.open(TOOLS / 'micromamba.tar.bz2', 'r:bz2') as t: t.extract('bin/micromamba', TOOLS)
    subprocess.run([str(TOOLS / 'bin/micromamba'), 'create', '-y', '-q', '-p', str(TOOLS / 'env'), '-r', str(TOOLS / 'root'), '-c', 'conda-forge',
                    'poppler=26.02.0', 'tesseract=5.5.1'], check=True, capture_output=True)


def fetch(keys):
    req = urllib.request.Request(os.environ['PDF_EXPORT_URL'] + '/pack', data=json.dumps(keys).encode(), method='POST',
                                 headers={'Authorization': 'Bearer ' + os.environ['PDF_EXPORT_TOKEN'], 'Content-Type': 'application/json', 'User-Agent': 'open-text-detector-humancal/1.0'})
    got = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for _ in keys:
            n = struct.unpack('>I', r.read(4))[0]; key = r.read(n).decode(); size = struct.unpack('>Q', r.read(8))[0]
            body = r.read(size); sha = key.split('/')[1][:-4]
            if size and hashlib.sha256(body).hexdigest() == sha:
                tmp = PDFS / f'{sha}.pdf.tmp'; tmp.write_bytes(body); tmp.replace(PDFS / f'{sha}.pdf'); got[sha] = size
            else: got[sha] = 0
    return got


def main():
    sel = json.loads((R / 'selection.json').read_text())['papers']
    say(event='start', papers=len(sel))
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pypdf', 'Pillow'], check=True)  # imported by pdf_extraction
    tools(); env = {**os.environ, 'PATH': f"{TOOLS / 'env/bin'}:{os.environ['PATH']}"}
    say(event='tools', pdftotext=subprocess.run(['pdftotext', '-v'], env=env, capture_output=True, text=True).stderr.splitlines()[0])
    PDFS.mkdir(parents=True, exist_ok=True)
    todo = [p['r2_key'] for p in sel if not (PDFS / f"{p['sha256']}.pdf").exists()]
    chunks = [todo[i:i + 100] for i in range(0, len(todo), 100)]
    missing = []
    with cf.ThreadPoolExecutor(4) as ex:
        for i, got in enumerate(ex.map(fetch, chunks)):
            missing += [s for s, n in got.items() if not n]
            say(event='fetched_chunk', chunk=i + 1, of=len(chunks), missing=len(missing))
    import pyarrow as pa, pyarrow.parquet as pq
    rows = [{'id': p['id'], 'title': None, 'conference': p['venue'], 'year': p['year'], 'pdf_sha256': p['sha256'],
             'pdf_path': f"pdfs/{p['sha256']}.pdf", 'positions_path': None, 'text_sha256': None,
             'metadata_json': json.dumps({'venue': p['venue'], 'year': p['year'], 'openreview_id': p['id'], 'r2_key': p['r2_key']}),
             'source_url': f"https://openreview.net/pdf?id={p['id']}"} for p in sel if (PDFS / f"{p['sha256']}.pdf").exists()]
    (R / 'source/data').mkdir(exist_ok=True); pq.write_table(pa.Table.from_pylist(rows), R / 'source/data/train-00000.parquet')
    say(event='source_ready', rows=len(rows), missing=len(missing))
    code = R / 'code'
    w = min(80, os.cpu_count() - 8)
    subprocess.run([sys.executable, str(code / 'scripts/extract_missing_iclr2027.py'), '--source', str(R / 'source'), '--output', str(R / 'extraction'),
                    '--workers', str(w), '--timeout', '180'], env=env, check=True, stdout=open(R / 'extract.log', 'a'), stderr=subprocess.STDOUT)
    comp = json.loads((R / 'extraction/completion.json').read_text())
    say(event='extracted', completion={k: v for k, v in comp.items() if k in ('state', 'states', 'rows')})
    if comp['rows'] < 0.9 * len(rows): raise RuntimeError(f"only {comp['rows']} of {len(rows)} papers extracted")
    subprocess.run([sys.executable, str(code / 'scripts/build_clean_paper_text.py'), '--source', str(R / 'extraction'), '--output', str(R / 'clean'),
                    '--workers', str(w)], env=env, check=True, stdout=open(R / 'clean.log', 'a'), stderr=subprocess.STDOUT)
    say(event='cleaned', completion={k: v for k, v in json.loads((R / 'clean/completion.json').read_text()).items() if k in ('state', 'rows', 'papers_with_flags')})
    # Persist as tar shards (bucket mount: few large files, not thousands of small ones).
    for name, src in [(SRC_DS, R / 'source'), (EXT_DS, R / 'extraction'), (CLEAN_DS, R / 'clean')]:
        dst = PERSIST / name; dst.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in src.rglob('*') if p.is_file() and not p.name.endswith(('.lock', '.tmp')))
        shard, size, n = None, 0, 0
        for f in files:
            if shard is None or size > 1.5e9:
                if shard: shard.close()
                shard = tarfile.open(dst / f'part-{n:03}.tar', 'w'); n += 1; size = 0
            shard.add(f, arcname=str(f.relative_to(src))); size += f.stat().st_size
        if shard: shard.close()
        say(event='persisted', dataset=name, shards=n, files=len(files))
    say(event='done')


if __name__ == '__main__':
    try: main()
    except Exception as e:
        import traceback; say(event='failed', error=repr(e), tb=traceback.format_exc()[-2000:]); raise
