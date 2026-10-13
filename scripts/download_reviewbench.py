"""Download every pinned ReviewBench file and build a read-only paper catalogue."""
import hashlib
import json
import sqlite3
import zlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from huggingface_hub import HfApi
import pyarrow.parquet as pq
from collect_baselines import download, sha

ROOT = Path(__file__).resolve().parents[1] / 'research/data/reviewbench'
REPO = 'Samarth0710/reviewbench'
REVISION = '7d1b399bd7297318a2d6284b5349326166702530'


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
    def fetch(file):
        path = ROOT / 'original' / file.rfilename
        download(f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{file.rfilename}', path)
        if path.stat().st_size != file.size:
            raise RuntimeError(f'Unexpected size: {file.rfilename}')
        return {'file': file.rfilename, 'bytes': file.size, 'sha256': sha(path)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(fetch, info.siblings))
    target = ROOT / 'catalogue.sqlite3'
    temp = ROOT / 'catalogue.building.sqlite3'
    temp.unlink(missing_ok=True)
    conn = sqlite3.connect(temp)
    conn.execute('CREATE TABLE papers(id TEXT PRIMARY KEY,title TEXT,conference TEXT,year INTEGER,decision TEXT,authors TEXT,abstract TEXT,word_count INTEGER,characters INTEGER,text_hash TEXT,preview TEXT,text BLOB,source_file TEXT)')
    count = 0
    for path in sorted((ROOT / 'original/data').glob('*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            rows = []
            for row in batch.to_pylist():
                text = row.get('markdown') or ''
                pid = row['conference'] + ':' + row['forum_id']
                rows.append((pid, row['title'], row['conference'], row['year'], row.get('decision'), json.dumps(row.get('authors') or []), row.get('abstract') or '', len(text.split()), len(text), hashlib.sha256(text.encode()).hexdigest(), text[:260], zlib.compress(text.encode()), path.name))
            conn.executemany('INSERT INTO papers VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)', rows)
            count += len(rows)
        conn.commit()
        print(f'Indexed {path.name}; {count:,} papers', flush=True)
    conn.execute('CREATE INDEX papers_cohort ON papers(conference,year)')
    conn.execute('CREATE INDEX papers_hash ON papers(text_hash)')
    conn.execute('CREATE INDEX papers_title ON papers(title)')
    conn.commit()
    totals = dict(conn.execute('SELECT conference,count(*) FROM papers GROUP BY conference'))
    missing = conn.execute('SELECT count(*) FROM papers WHERE characters=0').fetchone()[0]
    conn.close()
    temp.replace(target)
    manifest = {'repo':REPO,'revision':REVISION,'downloaded_at':datetime.now(timezone.utc).isoformat(),'files':files,'papers':count,'conferences':totals,'missing_text':missing,'original_bytes':sum(f['bytes'] for f in files),'index_bytes':target.stat().st_size}
    (ROOT / 'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest | {'files':len(files)},indent=2),flush=True)


if __name__ == '__main__':
    main()
