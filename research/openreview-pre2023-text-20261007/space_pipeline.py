"""Space-side: fetch all ICLR 2021/2022 and NeurIPS 2021/2022 PDFs listed in selection.json from R2 (export Worker
bundles), extract them (baseline: positioned text layer), clean them (positioned-clean-v2), persist extraction and clean
outputs (text + position sidecars) to the bucket as tar shards, and write one sampling-ready Parquet dataset
(PAPERS_DS) joining clean text with the selection flags. PDFs are not re-persisted; they stay in R2.
Adapted from research/human-calibration-20261006/space_pipeline.py. Runs detached in /tmp/pangram-pre2023text.

Env: PDF_EXPORT_URL, PDF_EXPORT_TOKEN. Steps are idempotent; rerun after a restart (after --restore of inputs).
Log: /tmp/pangram-pre2023text/pipeline.log (one JSON line per event).
"""
import concurrent.futures as cf, hashlib, json, os, struct, subprocess, sys, tarfile, time, urllib.request
from pathlib import Path

R = Path('/tmp/pangram-pre2023text'); PDFS = R / 'source/pdfs'; TOOLS = Path('/tmp/pangram-tools')
PERSIST = Path('/data/workspace/datasets')
EXT_DS, CLEAN_DS = 'openreview-iclr-neurips-2021-2022-extraction', 'openreview-iclr-neurips-2021-2022-clean-text-v2'
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


def fetch(keys, attempts=4):
    """One /pack bundle; a cut-off stream is retried for the keys not yet saved."""
    got = {}
    for attempt in range(attempts):
        todo = [k for k in keys if k.split('/')[1][:-4] not in got]
        if not todo: break
        try:
            req = urllib.request.Request(os.environ['PDF_EXPORT_URL'] + '/pack', data=json.dumps(todo).encode(), method='POST',
                                         headers={'Authorization': 'Bearer ' + os.environ['PDF_EXPORT_TOKEN'], 'Content-Type': 'application/json', 'User-Agent': 'open-text-detector-pre2023text/1.0'})
            with urllib.request.urlopen(req, timeout=1800) as r:
                for _ in todo:
                    head = r.read(4)
                    if len(head) < 4: raise IOError('stream ended early')
                    n = struct.unpack('>I', head)[0]; key = r.read(n).decode(); size = struct.unpack('>Q', r.read(8))[0]
                    body = r.read(size); sha = key.split('/')[1][:-4]
                    if size and len(body) < size: raise IOError('stream ended early')
                    if size and hashlib.sha256(body).hexdigest() == sha:
                        tmp = PDFS / f'{sha}.pdf.tmp'; tmp.write_bytes(body); tmp.replace(PDFS / f'{sha}.pdf'); got[sha] = size
                    else: got[sha] = 0
        except Exception as e:
            say(event='fetch_retry', attempt=attempt + 1, error=repr(e)[:200], remaining=len(todo)); time.sleep(10 * (attempt + 1))
    for k in keys: got.setdefault(k.split('/')[1][:-4], 0)
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
    for name, src in [(EXT_DS, R / 'extraction'), (CLEAN_DS, R / 'clean')]:
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
    papers(sel)
    say(event='done')


PAPERS_DS = 'openreview-iclr-neurips-2021-2022-papers-v1'


def papers(sel):
    """One row per extracted paper: clean text, metadata, extraction details and the selection flags."""
    import pyarrow as pa, pyarrow.parquet as pq
    flags = {p['sha256']: p for p in sel}
    ext = {}
    for f in sorted((R / 'extraction/data').glob('*.parquet')):
        for r in pq.read_table(f, columns=['pdf_sha256', 'method', 'ocr_pages', 'positions_path']).to_pylist():
            ext[r['pdf_sha256']] = r
    rows = []
    for f in sorted((R / 'clean/data').glob('*.parquet')):
        for r in pq.read_table(f).to_pylist():
            p = flags[r['pdf_sha256']]; e = ext.get(r['pdf_sha256'], {}); st = json.loads(r['stats_json'] or '{}')
            rows.append({'openreview_id': p['id'], 'venue': p['venue'], 'year': p['year'], 'pdf_sha256': p['sha256'], 'r2_key': p['r2_key'],
                         'pages': r['pages'], 'text': r['text'], 'text_sha256': r['text_sha256'], 'words': st.get('retained_words'),
                         'cleanup_version': r['cleanup_version'], 'cleanup_flag_count': r['flag_count'], 'cleanup_stats_json': r['stats_json'],
                         'extraction_method': e.get('method'), 'ocr_pages': e.get('ocr_pages'),
                         'positions_path': f"{EXT_DS}:{e.get('positions_path')}", 'clean_positions_path': f"{CLEAN_DS}:{r['clean_positions_path']}",
                         'calibration_reserved': p['calibration_reserved'], 'in_atlas_catalogue': p['in_atlas_catalogue'],
                         'paired_overlap_sha': p['paired_overlap_sha']})
    rows.sort(key=lambda x: (x['venue'], x['year'], x['openreview_id']))
    dst = PERSIST / PAPERS_DS; (dst / 'data').mkdir(parents=True, exist_ok=True); shards = []
    for i in range(0, len(rows), 1000):
        name = f'data/train-{i // 1000:05d}.parquet'; pq.write_table(pa.Table.from_pylist(rows[i:i + 1000]), dst / name)
        shards.append({'path': name, 'rows': len(rows[i:i + 1000])})
    from collections import Counter
    man = {'rows': len(rows), 'shards': shards, 'by_group': dict(Counter(f"{r['venue']}/{r['year']}" for r in rows)),
           'calibration_reserved': sum(r['calibration_reserved'] for r in rows), 'selected': len(sel),
           'missing_or_failed': len(sel) - len(rows), 'cleanup_version': rows[0]['cleanup_version'] if rows else None,
           'built': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (dst / 'manifest.json').write_text(json.dumps(man, indent=1))
    (dst / 'README.md').write_text(README.format(rows=len(rows), ext=EXT_DS, clean=CLEAN_DS))
    say(event='papers_dataset', **{k: v for k, v in man.items() if k != 'shards'})


README = '''# OpenReview ICLR/NeurIPS 2021-2022 papers, extracted text (v1)

{rows} papers from ICLR 2021, ICLR 2022, NeurIPS 2021 and NeurIPS 2022 (legacy OpenReview downloads, PDFs in R2
bucket `pangram-paper-atlas`), extracted with the baseline (positioned text layer) and cleaned with positioned-clean-v2.
Built by `research/openreview-pre2023-text-20261007/space_pipeline.py`.

- One row per paper: `text` is the cleaned full text; `positions_path` / `clean_positions_path` point into the tar
  shards of `{ext}` and `{clean}` (format `<dataset>:<path inside the shards>`).
- **Not filtered.** Before sampling for training, exclude `calibration_reserved` (human calibration set), and screen with
  `never_train_hit` (research/llm-sentence-edits-20261005/build_llm_edits.py) plus the evaluation-paper checks.
  `in_atlas_catalogue` and `paired_overlap_sha` mark papers already used elsewhere.
- NeurIPS 2021 is excluded by several earlier builders for an unrecorded reason; check before using it.
'''


if __name__ == '__main__':
    try: main()
    except Exception as e:
        import traceback; say(event='failed', error=repr(e), tb=traceback.format_exc()[-2000:]); raise
