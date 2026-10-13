"""Space-side corpus pipeline: every PDF in R2 gets three persisted text versions.

  1. direct extraction  (baseline positioned text layer, `pdf_extraction.extract`): reused where one already exists,
                        extracted here for the rest. Dataset `corpus-direct-extraction-v1` indexes all of them
                        (raw text + where each positions blob lives); new extractions are persisted with it.
  2. cleaned            (`text_cleanup.clean`, positioned-clean-v2) recomputed for every paper: `corpus-clean-text-v2`.
  3. omissions v2       (`omissions`: figures, tables, display equations, garbled inline math, numeric runs and the
                        reference list replaced by markers; readable inline math kept): `corpus-omissions-v2`.

PDFs come from R2 through the export Worker (bundles, never via the Mac) unless already on the Space; fetched copies
are deleted at the end. Resumable: every 500-paper chunk writes its own parts; rerun after a restart skips done chunks.
Env: PDF_EXPORT_URL, PDF_EXPORT_TOKEN. Log: /tmp/pangram-corpus/pipeline.log. Waits for the pre-2023 job to finish.
"""
import concurrent.futures as cf, glob, hashlib, json, os, shutil, signal, struct, subprocess, sys, tarfile, time, urllib.request
from collections import Counter
from pathlib import Path

R = Path('/tmp/pangram-corpus'); D = Path('/data/workspace/corpus-extraction-20261007'); DS = Path('/data/workspace/datasets')
PDFS = R / 'pdfs'; CODE = R / 'code'; VENDOR = '/data/workspace/baseline-omissions-20261005/vendor'
OUT = {'direct': 'corpus-direct-extraction-v1', 'clean': 'corpus-clean-text-v2', 'omit': 'corpus-omissions-v2'}
NEW_EXT = 'corpus-missing-extraction-20261007'
# Existing extraction datasets, highest priority first, with the persisted dataset name that holds each one's blobs.
ROOTS = [(R / 'extraction', NEW_EXT),
         (Path('/tmp/pangram-pre2023text/extraction'), 'openreview-iclr-neurips-2021-2022-extraction'),
         (Path('/tmp/pangram-humancal/extraction'), 'human-cal-2019-2022-20261006-extraction'),
         (DS / 'iclr2027-extraction-backfill-20261005', 'iclr2027-extraction-backfill-20261005'),
         (DS / 'iclr2027-complete-20261004', 'iclr2027-complete-20261004'),
         (DS / 'iclr-2024-2025-openreview-sample-20261005-extraction', 'iclr-2024-2025-openreview-sample-20261005-extraction'),
         (DS / 'paper-text-positions-batch001-20261005', 'paper-text-positions-batch001-20261005')]
CHUNK = 500
log = None
def say(**k): log.write(json.dumps({'t': time.strftime('%H:%M:%S'), **k}) + '\n'); log.flush()


def fetch(keys, attempts=4):
    got = {}
    for attempt in range(attempts):
        todo = [k for k in keys if k[7:-4] not in got]
        if not todo: break
        try:
            req = urllib.request.Request(os.environ['PDF_EXPORT_URL'] + '/pack', data=json.dumps(todo).encode(), method='POST',
                                         headers={'Authorization': 'Bearer ' + os.environ['PDF_EXPORT_TOKEN'], 'Content-Type': 'application/json', 'User-Agent': 'open-text-detector-corpus/1.0'})
            with urllib.request.urlopen(req, timeout=1800) as r:
                for _ in todo:
                    head = r.read(4)
                    if len(head) < 4: raise IOError('stream ended early')
                    n = struct.unpack('>I', head)[0]; key = r.read(n).decode(); size = struct.unpack('>Q', r.read(8))[0]
                    body = r.read(size); sha = key[7:-4]
                    if size and len(body) < size: raise IOError('stream ended early')
                    if size and hashlib.sha256(body).hexdigest() == sha:
                        tmp = PDFS / f'{sha}.pdf.tmp'; tmp.write_bytes(body); tmp.replace(PDFS / f'{sha}.pdf'); got[sha] = size
                    else: got[sha] = 0
        except Exception as e:
            say(event='fetch_retry', attempt=attempt + 1, error=repr(e)[:200], remaining=len(todo)); time.sleep(10 * (attempt + 1))
    for k in keys: got.setdefault(k[7:-4], 0)
    return got


def fetch_all(shas, threads=8):
    PDFS.mkdir(parents=True, exist_ok=True)
    todo = [f'papers/{s}.pdf' for s in shas if not (PDFS / f'{s}.pdf').exists()]
    chunks = [todo[i:i + 100] for i in range(0, len(todo), 100)]; bad = []
    with cf.ThreadPoolExecutor(threads) as ex:
        for i, got in enumerate(ex.map(fetch, chunks)):
            bad += [s for s, n in got.items() if not n]
            if i % 20 == 0 or i + 1 == len(chunks): say(event='fetched', chunk=i + 1, of=len(chunks), failed=len(bad))
    return bad


def canonical(r2):
    """pdf_sha256 -> extraction row (first root in ROOTS order that has a positions blob)."""
    import pyarrow.parquet as pq
    rows = {}
    for root, name in ROOTS:
        for f in sorted(glob.glob(str(root / 'data/*.parquet'))):
            names = set(pq.read_schema(f).names)
            cols = [c for c in ('id', 'forum_id', 'title', 'conference', 'year', 'pdf_sha256', 'text_sha256', 'positions_path', 'blob_sha256', 'pdf_path', 'method') if c in names]
            for r in pq.read_table(f, columns=cols).to_pylist():
                s = r['pdf_sha256']
                if r.get('positions_path') and s in r2 and s not in rows:
                    rows[s] = {**r, 'root': str(root), 'dataset': name}
    return rows


def process(job):
    """One paper: clean v2 + omissions v2. Returns three rows (direct, clean, omit) or an error."""
    sha, row, pdf, out_objects = job
    signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TimeoutError('paper timeout'))); signal.alarm(600)
    try:
        from open_text_detector_backend.result_codec import encode, decode
        from open_text_detector_backend.text_cleanup import clean
        from open_text_detector_backend.omissions import page_features, label_words, classify, render, DEFAULT_POLICY
        import pymupdf
        raw = (Path(row['root']) / row['positions_path']).read_bytes()
        blob = hashlib.sha256(raw).hexdigest()
        if row.get('blob_sha256') and blob != row['blob_sha256']: raise ValueError('source blob checksum mismatch')
        original = decode(raw)
        if original['pdf_sha256'] != sha: raise ValueError('source identity mismatch')
        meta = {'pdf_sha256': sha, 'id': row.get('forum_id') or row.get('id'), 'title': row.get('title'),
                'conference': (row.get('conference') or '').lower() or None, 'year': int(row['year']) if str(row.get('year') or '').isdigit() else None}
        direct = {**meta, 'text': original['text'], 'text_sha256': original['text_sha256'], 'extraction_dataset': row['dataset'],
                  'positions_path': row['positions_path'], 'positions_blob_sha256': blob, 'method': row.get('method') or original.get('method'),
                  'pages': len(original['pages'])}
        result = clean(original); packed = encode(result, level=9)
        side = Path(out_objects) / sha / (result['text_sha256'] + '.pgf'); side.parent.mkdir(parents=True, exist_ok=True)
        tmp = side.with_name(side.name + f'.{os.getpid()}.tmp'); tmp.write_bytes(packed); tmp.replace(side)
        cleanrow = {**meta, 'text': result['text'], 'text_sha256': result['text_sha256'], 'source_text_sha256': result['source_text_sha256'],
                    'cleanup_version': result['format'], 'clean_positions_path': f'objects/{sha}/{side.name}', 'clean_blob_sha256': hashlib.sha256(packed).hexdigest(),
                    'stats_json': json.dumps(result['stats'], sort_keys=True), 'flag_count': len(result['flags'])}
        if not pdf: return sha, direct, cleanrow, None, 'no PDF on the Space: omissions skipped'
        doc = pymupdf.open(pdf); feats = page_features(doc); doc.close()
        labels, body = label_words(original, feats); kinds = classify(result, labels)
        text, omissions = render(result, kinds, DEFAULT_POLICY)
        omitrow = {**meta, 'text': text, 'text_sha256': hashlib.sha256(text.encode()).hexdigest(), 'source_clean_text_sha256': result['text_sha256'],
                   'omission_counts_json': json.dumps(Counter(o['type'] for o in omissions), sort_keys=True),
                   'omissions_json': json.dumps(omissions, ensure_ascii=False), 'body_font': body,
                   'chars_clean': len(result['text']), 'chars_after': len(text)}
        return sha, direct, cleanrow, omitrow, None
    except Exception as e:
        return sha, None, None, None, repr(e)[:300]
    finally:
        signal.alarm(0)


def main():
    global log
    R.mkdir(exist_ok=True); log = open(R / 'pipeline.log', 'a')
    pre = Path('/tmp/pangram-pre2023text/pipeline.log')
    while not (pre.exists() and ('"done"' in pre.read_text() or '"failed"' in pre.read_text().split('"start"')[-1])):
        say(event='waiting_for_pre2023'); time.sleep(120)
    sys.path[:0] = [str(CODE / 'backend'), VENDOR]
    inv = json.loads((D / 'inventory.json').read_text()); r2 = inv['r2']; meta = json.loads((D / 'pdf-metadata.json').read_text())
    say(event='start', r2=len(r2))
    env = {**os.environ, 'PATH': f"/tmp/pangram-tools/env/bin:{os.environ['PATH']}", 'PYTHONPATH': f"{CODE / 'backend'}:{VENDOR}"}
    # Stage 1 for PDFs with no extraction anywhere.
    rows = canonical(r2); missing = sorted(set(r2) - set(rows))
    say(event='canonical', with_extraction=len(rows), missing=len(missing), by_dataset=dict(Counter(r['dataset'] for r in rows.values())))
    if missing and not (R / 'extraction/completion.json').exists():
        bad = fetch_all(missing)
        import pyarrow as pa, pyarrow.parquet as pq
        src = R / 'source'; (src / 'data').mkdir(parents=True, exist_ok=True)
        if not (src / 'pdfs').exists(): (src / 'pdfs').symlink_to(PDFS)
        recs = [{'id': meta.get(s, {}).get('openreview_id') or s, 'title': meta.get(s, {}).get('title'), 'conference': meta.get(s, {}).get('venue'),
                 'year': meta.get(s, {}).get('year'), 'pdf_sha256': s, 'pdf_path': f'pdfs/{s}.pdf', 'positions_path': None, 'text_sha256': None,
                 'metadata_json': json.dumps(meta.get(s, {})), 'source_url': None} for s in missing if (PDFS / f'{s}.pdf').exists()]
        pq.write_table(pa.Table.from_pylist(recs), src / 'data/train-00000.parquet')
        subprocess.run([sys.executable, str(CODE / 'scripts/extract_missing_iclr2027.py'), '--source', str(src), '--output', str(R / 'extraction'),
                        '--workers', str(os.cpu_count() - 8), '--timeout', '180'], env=env, check=True, stdout=open(R / 'extract.log', 'a'), stderr=subprocess.STDOUT)
        say(event='extracted_missing', fetch_failed=len(bad), completion=json.loads((R / 'extraction/completion.json').read_text()).get('states'))
        rows = canonical(r2)
    # PDFs needed for omissions: use Space copies where they exist, fetch the rest.
    def local_pdf(s, r):
        cands = [Path(r['root']) / r['pdf_path'], Path(r['root']).parent / 'source' / r['pdf_path']] if r.get('pdf_path') else []
        for cand in cands + [PDFS / f'{s}.pdf', Path(f'/tmp/pangram-pre2023text/source/pdfs/{s}.pdf'), Path(f'/tmp/pangram-humancal/source/pdfs/{s}.pdf')]:
            if cand.exists(): return cand
        return None
    need = [s for s, r in rows.items() if local_pdf(s, r) is None]
    say(event='pdfs_to_fetch', n=len(need)); bad = fetch_all(need); say(event='pdfs_fetched', failed=len(bad))
    # Stages 2 + 3, chunked and resumable.
    shas = sorted(rows); (D / 'parts').mkdir(parents=True, exist_ok=True); errors = []
    import pyarrow as pa, pyarrow.parquet as pq
    def new_pool(): return cf.ProcessPoolExecutor(os.cpu_count() - 8)
    def kill_pool(pool):
        for pr in list((getattr(pool, '_processes', None) or {}).values()):
            try: pr.kill()
            except Exception: pass
        pool.shutdown(wait=False, cancel_futures=True)
    pool = new_pool()
    try:
        for c in range(0, len(shas), CHUNK):
            part = D / 'parts' / f'{c // CHUNK:05d}'
            if (part / 'done').exists(): continue
            objs = R / 'clean-objects' / part.name; objs.mkdir(parents=True, exist_ok=True)
            jobs = [(s, rows[s], str(local_pdf(s, rows[s]) or ''), str(objs)) for s in shas[c:c + CHUNK]]
            out = {'direct': [], 'clean': [], 'omit': []}; errs = []
            # Per-paper futures: if nothing finishes for STALL seconds (hung workers, seen Oct 7 04:26), kill the pool, record the
            # unfinished papers as stalled and continue with a fresh pool.
            futs = {pool.submit(process, j): j[0] for j in jobs}
            pending = set(futs)
            while pending:
                done_, pending = cf.wait(pending, timeout=900, return_when=cf.FIRST_COMPLETED)
                if not done_:
                    for f in pending: errs.append({'pdf_sha256': futs[f], 'error': 'stalled: worker pool killed'})
                    say(event='pool_stalled', part=part.name, unfinished=len(pending)); kill_pool(pool); pool = new_pool(); break
                for f in done_:
                    try: s_, d, cl, om, err = f.result()
                    except Exception as e: s_, d, cl, om, err = futs[f], None, None, None, repr(e)[:200]
                    if err: errs.append({'pdf_sha256': s_, 'error': err})
                    for k, v in (('direct', d), ('clean', cl), ('omit', om)):
                        if v: out[k].append(v)
            if getattr(pool, '_broken', False):  # a worker died: every pending paper failed; start a fresh pool for the next chunk
                say(event='pool_broken', part=part.name); kill_pool(pool); pool = new_pool()
            part.mkdir(exist_ok=True)
            for k, v in out.items():
                if v: pq.write_table(pa.Table.from_pylist(v), part / f'{k}.parquet')
            with tarfile.open(part / 'clean-objects.tar', 'w') as t: t.add(objs, arcname='objects')
            (part / 'errors.json').write_text(json.dumps(errs)); (part / 'done').write_text('ok'); shutil.rmtree(objs)
            errors += errs; say(event='chunk', part=part.name, papers=len(out['clean']), errors=len(errs), of=-(-len(shas) // CHUNK))
    finally:
        kill_pool(pool)
    finalize(rows, meta)
    shutil.rmtree(PDFS, ignore_errors=True); say(event='done')


def finalize(rows, meta):
    """Assemble the three datasets from the parts (1,000-row shards), with manifests and READMEs."""
    import pyarrow as pa, pyarrow.parquet as pq
    parts = sorted(p for p in (D / 'parts').iterdir() if (p / 'done').exists())
    for kind, name in OUT.items():
        dst = DS / name; (dst / 'data').mkdir(parents=True, exist_ok=True); allrows = []
        for p in parts:
            if (p / f'{kind}.parquet').exists(): allrows += pq.read_table(p / f'{kind}.parquet').to_pylist()
        for r in allrows:
            m = meta.get(r['pdf_sha256'], {})
            r['conference'] = r['conference'] or m.get('venue'); r['year'] = r['year'] or m.get('year')
            r['openreview_id'] = m.get('openreview_id'); r['in_r2'] = True
        allrows.sort(key=lambda r: r['pdf_sha256']); shards = []
        for i in range(0, len(allrows), 1000):
            f = f'data/train-{i // 1000:05d}.parquet'; pq.write_table(pa.Table.from_pylist(allrows[i:i + 1000]), dst / f); shards.append(f)
        if kind == 'clean':
            for p in parts:
                if (dst / f'objects-{p.name}.tar').exists() and (dst / f'objects-{p.name}.tar').stat().st_size == (p / 'clean-objects.tar').stat().st_size: continue
                for attempt in range(3):  # the bucket mount returned EIO once (Oct 7 09:21); retry, then record the part as unreadable
                    try: shutil.copy(p / 'clean-objects.tar', dst / f'objects-{p.name}.tar'); break
                    except OSError as e:
                        if attempt == 2: say(event='objects_copy_failed', part=p.name, error=repr(e)[:120])
                        else: time.sleep(30)
        man = {'rows': len(allrows), 'shards': shards, 'by_venue_year': dict(Counter(f"{r['conference']}/{r['year']}" for r in allrows)),
               'built': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'code_commit': (CODE / 'COMMIT').read_text().strip()}
        if kind == 'direct': man['by_extraction_dataset'] = dict(Counter(r['extraction_dataset'] for r in allrows))
        (dst / 'manifest.json').write_text(json.dumps(man, indent=1)); (dst / 'README.md').write_text(README[kind])
        say(event='dataset', name=name, rows=len(allrows))
    new = R / 'extraction'
    if new.exists():  # persist the new direct extractions (blobs + parquet) as tar shards
        dst = DS / NEW_EXT; dst.mkdir(parents=True, exist_ok=True); files = sorted(p for p in new.rglob('*') if p.is_file()); shard, size, n = None, 0, 0
        for f in files:
            if shard is None or size > 1.5e9:
                if shard: shard.close()
                shard = tarfile.open(dst / f'part-{n:03}.tar', 'w'); n += 1; size = 0
            shard.add(f, arcname=str(f.relative_to(new))); size += f.stat().st_size
        if shard: shard.close()
        say(event='persisted_new_extraction', shards=n, files=len(files))


README = {
    'direct': """# Corpus direct extraction (v1)

One row per PDF in R2 (`pangram-paper-atlas`, key `papers/<pdf_sha256>.pdf`) that has a baseline direct extraction
(poppler positioned text layer, before any cleanup). `text` is the raw extracted text. The positions blob is
`positions_path` inside dataset `extraction_dataset` (blob checksum `positions_blob_sha256`); extractions made by this
pipeline are in `corpus-missing-extraction-20261007`. Built by research/corpus-extraction-20261007/corpus_pipeline.py.
""",
    'clean': """# Corpus clean text (positioned-clean-v2)

Stage 2: `text_cleanup.clean` applied to every paper's direct extraction. Clean position sidecars are in
`objects-*.tar` (paths `objects/<pdf_sha256>/<text_sha256>.pgf`). Join to the other stages on `pdf_sha256`.
""",
    'omit': """# Corpus text with omissions (omissions v2)

Stage 3: the clean v2 text with non-prose content replaced by markers (`omissions.render`, DEFAULT_POLICY): figure
text, tables, display equations, garbled inline math, number runs and the reference list. Readable inline math stays
(policy of 2026-10-06). `omissions_json` lists every omission with its type. Join on `pdf_sha256`.
Known open issues (Oct 6 review): small-caps headings next to tables, pseudocode lines omitted as math, untruled
tables leaking cells.
"""}


if __name__ == '__main__':
    try: main()
    except Exception as e:
        import traceback; say(event='failed', error=repr(e), tb=traceback.format_exc()[-2000:]); raise
