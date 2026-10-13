"""Package locally downloaded OpenReview PDFs as a complete-mirror source dataset for extraction.

Reads <root>/<year>/{accepted,rejected}/*.pdf plus an `ordl --list --format jsonl` listing per
year (<root>/<year>/listing.jsonl) to recover forum ids. Papers whose forum id is in the
exclusion list (already in the clean-text archive) are left out. Writes
<stage>/data/train-00000.parquet (rows: id, title, conference, year, pdf_sha256, pdf_path,
positions_path=None, text_sha256=None, metadata_json, source_url) and tar shards of the PDFs
(<stage>/pdfs-NNN.tar, members pdfs/<sha256>.pdf), then uploads both to the training bucket.
Usage: prepare_openreview_sample.py ROOT STAGE BUCKET_PREFIX --exclude ids.json [--year 2024 ...]
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('stage', type=Path)
    p.add_argument('bucket_prefix')
    p.add_argument('--exclude', type=Path, action='append', default=[])
    p.add_argument('--year', action='append', required=True)
    p.add_argument('--shard-bytes', type=int, default=2 * 1024**3)
    p.add_argument('--no-upload', action='store_true')
    a = p.parse_args()
    import pyarrow as pa, pyarrow.parquet as pq
    skip = {i for f in a.exclude for i in json.loads(f.read_text())}
    rows = []; seen = set(); excluded = 0
    for year in a.year:
        listing = {}
        for line in (a.root / year / 'listing.jsonl').read_text().splitlines():
            r = json.loads(line)
            if r.get('type') == 'paper': listing[Path(r['pdf_path']).name] = r
        for pdf in sorted((a.root / year).glob('*/*.pdf')):
            meta = listing[pdf.name]
            if meta['id'] in skip: excluded += 1; continue
            raw = pdf.read_bytes(); sha = hashlib.sha256(raw).hexdigest()
            if not raw.startswith(b'%PDF') or sha in seen: continue
            seen.add(sha)
            rows.append({'id': meta['id'], 'title': meta.get('title'), 'conference': 'iclr', 'year': int(year),
                         'pdf_sha256': sha, 'pdf_path': f'pdfs/{sha}.pdf', 'positions_path': None, 'text_sha256': None,
                         'metadata_json': json.dumps({'number': meta.get('number'), 'decision': pdf.parent.name,
                                                      'venueid': meta.get('venueid'), 'venue': meta.get('venue'),
                                                      'downloaded_file': f'{year}/{pdf.parent.name}/{pdf.name}'}),
                         'source_url': f"https://openreview.net/forum?id={meta['id']}", '_local': str(pdf)})
    a.stage.mkdir(parents=True, exist_ok=True); (a.stage / 'data').mkdir(exist_ok=True)
    shards = []; tar = None; size = 0
    for r in rows:
        if tar is None or size > a.shard_bytes:
            if tar: tar.close()
            name = f'pdfs-{len(shards):03}.tar'; shards.append(name); tar = tarfile.open(a.stage / name, 'w'); size = 0
        tar.add(r['_local'], arcname=r['pdf_path']); size += Path(r['_local']).stat().st_size
    if tar: tar.close()
    for r in rows: r.pop('_local')
    pq.write_table(pa.Table.from_pylist(rows), a.stage / 'data' / 'train-00000.parquet', compression='zstd')
    manifest = {'rows': len(rows), 'excluded_in_archive': excluded, 'years': {y: sum(r['year'] == int(y) for r in rows) for y in a.year},
                'shards': [{'path': s, 'sha256': hashlib.sha256((a.stage / s).read_bytes()).hexdigest()} for s in shards],
                'parquet_sha256': hashlib.sha256((a.stage / 'data' / 'train-00000.parquet').read_bytes()).hexdigest()}
    (a.stage / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k != 'shards'}, indent=1), flush=True)
    if not a.no_upload:
        from huggingface_hub import HfApi
        files = [a.stage / 'manifest.json', a.stage / 'data' / 'train-00000.parquet', *[a.stage / s for s in shards]]
        HfApi().batch_bucket_files('open-text-detector/training-storage',
                                   add=[(f, a.bucket_prefix + '/' + str(f.relative_to(a.stage))) for f in files])
        print('uploaded', len(files), 'files to', a.bucket_prefix)


if __name__ == '__main__':
    main()
