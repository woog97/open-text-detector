"""Rebuild saved extractions as verified PGF artifacts; never submit model jobs."""
import argparse,concurrent.futures,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from open_text_detector_backend.pdf_extraction import extract,map_legacy,save,text_hash
from open_text_detector_backend.result_codec import decode
from open_text_detector_backend.sqlite_runtime import sqlite3
OUT=ROOT/'research/extractions/positioned'

def work(row):
    stamp=time.perf_counter();pdf=next((ROOT/p for p in row['pdfs'] if (ROOT/p).is_file()),None)
    if pdf is None:raise ValueError('Missing PDF '+row['pdf_sha256'])
    artifact=extract(pdf)
    assert artifact['pdf_sha256']==row['pdf_sha256'],'PDF version mismatch'
    artifacts=[];aliases=[];done={}
    def persist(a):
        path=save(a,OUT/'objects');b=path.read_bytes()
        item={'pdf_sha256':a['pdf_sha256'],'text_sha256':a['text_sha256'],'path':str(path.relative_to(ROOT)),'blob_sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'mapping_status':a['mapping']['status'],'coverage':a['mapping'].get('coverage',1.0 if a['mapping']['status']=='complete' else 0.0),'kind':a['mapping']['kind']}
        artifacts.append(item);done[a['text_sha256']]=item;return item
    canonical=persist(artifact)
    for source in row['legacy_files']:
        p=ROOT/source
        if not p.exists():continue
        text=p.read_text();h=text_hash(text)
        protected=h in row.get('protected_text_hashes',[])
        target=done.get(h) if protected else canonical
        if target is None:target=persist(map_legacy(artifact,text))
        if protected:assert decode((ROOT/target['path']).read_bytes())['text']==text
        aliases.append({'old_path':source,'old_file_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'original_text_sha256':h,'text_sha256':target['text_sha256'],'replacement':target['path'],'preserve_exact_text':protected})
    return {'pdf_sha256':row['pdf_sha256'],'pdfs':row['pdfs'],'canonical':canonical,'artifacts':artifacts,'aliases':aliases,'seconds':time.perf_counter()-stamp}

def main():
    if (OUT/"cleanup-ledger.json").exists() and json.loads((OUT/"cleanup-ledger.json").read_text()).get('status')=='completed':
        print("Migration already finalized. Use scripts/extract_positioned.py for future PDFs.");return
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    OUT.mkdir(exist_ok=True);source=ROOT/'research/extractions/new-local-pdfs/extraction-index.json'
    inventory=json.loads(source.read_text())['papers'];rows={r['pdf_sha256']:{**r,'legacy_files':{r['text_file']}} for r in inventory}
    for path in (ROOT/'research/classifications').glob('*/selection.json'):
        for r in json.loads(path.read_text()).get('papers',[]):
            if r.get('pdf_sha256') in rows and r.get('text_file'):rows[r['pdf_sha256']]['legacy_files'].add(r['text_file'])
    for r in json.loads((ROOT/'research/extractions/iclr2025-benchmark/results.json').read_text())['papers']:
        if r['sha256'] in rows:rows[r['sha256']]['legacy_files'].add('research/extractions/iclr2025-benchmark/plain/'+r['id']+'.txt')
    hyper=json.loads((ROOT/'research/extractions/hyperdas/positioned-text.json').read_text())
    for name in ['direct-text.txt','layout-text.txt']:
        if hyper['pdf_sha256'] in rows:rows[hyper['pdf_sha256']]['legacy_files'].add('research/extractions/hyperdas/'+name)
    # Preserve even empty native outputs superseded by OCR, so cleanup remains reversible via artifacts.
    for r in rows.values():
        r['legacy_files']=sorted(r['legacy_files'])
        for p in list(r['legacy_files']):
            if p.endswith('.ocr.txt') and (ROOT/p.replace('.ocr.txt','.txt')).exists():r['legacy_files'].append(p.replace('.ocr.txt','.txt'))
    workspace=sqlite3.connect(f'file:{ROOT}/backend/.data/workspace.sqlite3?mode=ro',uri=True)
    protected={text_hash(r[0]) for r in workspace.execute("SELECT text FROM scans WHERE kind='text'")}
    workspace.close()
    (OUT/'protected-text-hashes.json').write_text(json.dumps(sorted(protected)))
    for r in rows.values():r['protected_text_hashes']=list(protected)
    plan=list(rows.values());(OUT/'plan.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='protected_text_hashes'} for r in plan],indent=2))
    log=OUT/'migration.jsonl';finished={}
    if log.exists():
        for line in log.read_text().splitlines():
            r=json.loads(line)
            if all((ROOT/a['path']).is_file() for a in r['artifacts']):finished[r['pdf_sha256']]=r
    pending=[r for r in plan if r['pdf_sha256'] not in finished]
    if args.limit:pending=pending[:args.limit]
    database=OUT/'index.sqlite3';conn=sqlite3.connect(database)
    conn.execute('CREATE TABLE IF NOT EXISTS artifacts (pdf_sha256 TEXT,text_sha256 TEXT,path TEXT,blob_sha256 TEXT,bytes INTEGER,mapping_status TEXT,coverage REAL,kind TEXT,PRIMARY KEY(pdf_sha256,text_sha256))')
    conn.execute('CREATE INDEX IF NOT EXISTS by_text ON artifacts(text_sha256)');conn.commit()
    def record(r):
        for a in r['artifacts']:conn.execute('INSERT OR REPLACE INTO artifacts VALUES (?,?,?,?,?,?,?,?)',tuple(a[k] for k in ('pdf_sha256','text_sha256','path','blob_sha256','bytes','mapping_status','coverage','kind')))
        conn.commit()
    for r in finished.values():record(r)
    print('Already migrated',len(finished),'remaining',len(pending),flush=True)
    start=time.perf_counter()
    with log.open('a') as stream,concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(work,pending,chunksize=1):
            stream.write(json.dumps(r)+'\n');stream.flush();record(r);finished[r['pdf_sha256']]=r
            if len(finished)%100==0 or args.limit:print('Migrated',len(finished),'/',len(plan),'elapsed',round(time.perf_counter()-start),flush=True)
    summary={'papers':len(finished),'expected':len(plan),'artifacts':sum(len(r['artifacts']) for r in finished.values()),'compressed_bytes':sum(a['bytes'] for r in finished.values() for a in r['artifacts']),'aliases':sum(len(r['aliases']) for r in finished.values()),'last_pass_seconds':time.perf_counter()-start}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
if __name__=='__main__':main()
