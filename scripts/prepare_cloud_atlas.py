"""Prepare a curated, read-only export for R2/D1. No model configs or credentials."""
import hashlib, json, shutil, sys, sqlite3
from types import SimpleNamespace
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'app/.sites-runtime/atlas'
out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(root/'backend'))
from open_text_detector_backend.config import Settings
from open_text_detector_backend.result_store import ResultStore
from open_text_detector_backend.pdf_reader import PDFReader
results=ResultStore(Settings(data_dir=root/'backend/.data'))
conn=sqlite3.connect(f"file:{root/'backend/.data/workspace.sqlite3'}?mode=ro",uri=True)
conn.row_factory=sqlite3.Row
class ReadOnlyDatabase:
    def all(self, query, args=()):
        return [dict(row) for row in conn.execute(query,args).fetchall()]
reader=PDFReader(SimpleNamespace(db=ReadOnlyDatabase(),results=results),root/'research/data')
files=reader.files
catalogue=reader.list()['items']
native={}
for selection in sorted((root/'research/classifications').glob('*/selection.json')):
    for item in json.loads(selection.read_text())['papers']:
        if not item.get('scan_id'):continue
        row=conn.execute("SELECT * FROM scans WHERE id=? AND status='completed' AND deleted_at IS NULL",(item['scan_id'],)).fetchone()
        if row is None:continue
        row=dict(row)
        if hashlib.sha256(row['text'].encode()).hexdigest()!=item['text_sha256']:
            raise ValueError('Classification input hash mismatch')
        previous=native.get(item['pdf_sha256'])
        if previous is None or row['created_at'] > previous[1]['created_at']:
            native[item['pdf_sha256']]=(item,row)
print(f"Found {len(native)} completed classifications with recorded PDF/input hashes")
published=json.loads((out/'uploaded.json').read_text()) if (out/'uploaded.json').exists() else {}
exported=0
manifest=[];sql=['CREATE TABLE IF NOT EXISTS papers (id TEXT PRIMARY KEY,title TEXT NOT NULL,filename TEXT NOT NULL,collection TEXT NOT NULL,bytes INTEGER NOT NULL,classified INTEGER NOT NULL,pdf_key TEXT NOT NULL,detail_key TEXT NOT NULL);','CREATE INDEX IF NOT EXISTS paper_collection ON papers(collection);','CREATE INDEX IF NOT EXISTS paper_title ON papers(title);']
def quote(v):return "'"+str(v).replace("'","''")+"'"
for paper in catalogue:
    pid=paper['id']; pdf=files[pid]
    digest=hashlib.sha256(pdf.read_bytes()).hexdigest()
    detail={'id':pid,'version':digest,'reports':[]}
    if digest in native:
        item,row=native[digest]
        result=results.get(row['result_storage']) if row['result_storage'] else json.loads(row['result'])
        model=json.loads(row['model_snapshot'])['model']
        detail['reports']=[{'id':row['id'],'text':row['text'],'model':{'name':model['name']},'result':{'segments':result['segments']}}]
        detail['version_verified']=True
        paper['title']=item['title']
        paper['classified']=True
    elif paper['classified']:
        original=reader.detail(pid)
        detail['reports']=[{'id':r['id'],'text':r['text'],'model':{'name':r['model']['name']},'result':{'segments':r['result']['segments']}} for r in original['reports']]
    if not detail['reports'] and f'results/{pid}.json' not in published:
        continue
    exported+=1
    detail_path=out/f'{pid}.json'
    detail_path.write_text(json.dumps(detail,ensure_ascii=False))
    pdf_key=f'papers/{digest}.pdf';detail_key=f'results/{pid}.json'
    manifest.extend([{'key':pdf_key,'path':str(pdf),'type':'application/pdf'}, {'key':detail_key,'path':str(detail_path),'type':'application/json'}])
    vals=[pid,paper['title'],paper['filename'],paper['collection'],paper['bytes'],int(paper['classified']),pdf_key,detail_key]
    sql.append('INSERT OR REPLACE INTO papers VALUES ('+','.join(quote(v) for v in vals)+');')
(out/'catalogue.sql').write_text('\n'.join(sql))
unique={item['key']:item for item in manifest}
(out/'manifest.json').write_text(json.dumps(list(unique.values())))
assets=out/'assets';assets.mkdir(exist_ok=True)
shutil.copytree(root/'app/public/pdfjs',assets/'pdfjs',dirs_exist_ok=True)
shutil.copy2(root/'app/public/favicon.svg',assets/'favicon.svg')
print(f'{exported} papers, {len(unique)} objects, {sum(Path(i["path"]).stat().st_size for i in unique.values())/1e9:.2f} GB to upload')
