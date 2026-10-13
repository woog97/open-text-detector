"""Default extraction entry point: compressed text and rectangles, never classification.

Usage: backend/.venv/bin/python scripts/extract_positioned.py [PDF-or-directory ...]
Existing PDF hashes are skipped. Defaults to research/data.
"""
import argparse,concurrent.futures,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from open_text_detector_backend.pdf_extraction import extract,save
from open_text_detector_backend.sqlite_runtime import sqlite3
OUT=ROOT/'research/extractions/positioned'
def one(path):
    a=extract(path);p=save(a,OUT/'objects');blob=p.read_bytes()
    return (a['pdf_sha256'],a['text_sha256'],str(p.relative_to(ROOT)),hashlib.sha256(blob).hexdigest(),len(blob),a['mapping']['status'],1.0,'canonical')
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('paths',nargs='*');parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    OUT.mkdir(exist_ok=True);conn=sqlite3.connect(OUT/'index.sqlite3')
    conn.execute('CREATE TABLE IF NOT EXISTS artifacts (pdf_sha256 TEXT,text_sha256 TEXT,path TEXT,blob_sha256 TEXT,bytes INTEGER,mapping_status TEXT,coverage REAL,kind TEXT,PRIMARY KEY(pdf_sha256,text_sha256))');conn.execute('CREATE INDEX IF NOT EXISTS by_text ON artifacts(text_sha256)');conn.commit()
    known={r[0] for r in conn.execute("SELECT pdf_sha256,path FROM artifacts WHERE kind='canonical'") if (ROOT/r[1]).is_file()};selected={};sources={}
    for item in args.paths or [str(ROOT/'research/data')]:
        p=Path(item)
        for f in sorted(p.rglob('*.pdf')) if p.is_dir() else [p]:
            h=hashlib.sha256(f.read_bytes()).hexdigest();sources.setdefault(h,[]).append(str(f.resolve()))
            if h not in known:selected.setdefault(h,str(f.resolve()))
    with (OUT/'discoveries.jsonl').open('a') as log:
        for h,paths in sources.items():log.write(json.dumps({'pdf_sha256':h,'pdfs':paths})+'\n')
    print('New distinct PDFs:',len(selected),flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n,row in enumerate(pool.map(one,selected.values()),1):
            conn.execute('INSERT OR REPLACE INTO artifacts VALUES (?,?,?,?,?,?,?,?)',row);conn.commit()
            if n%100==0:print('Extracted',n,'/',len(selected),flush=True)
    print('Finished; no classifier jobs submitted.',flush=True)
if __name__=='__main__':main()
