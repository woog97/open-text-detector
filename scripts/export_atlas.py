"""Export only reader-visible fields; never export owner keys or scan configuration."""
import json
import hashlib
import urllib.request
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'app/public/catalogue'
out.mkdir(parents=True, exist_ok=True)
key=(root/'backend/.data/admin.key').read_text().strip()
def api(path):
    req=urllib.request.Request('http://127.0.0.1:8000'+path, headers={'Authorization':'Bearer '+key})
    with urllib.request.urlopen(req) as res:return json.load(res)
catalogue=api('/v1/pdf-reader')
for paper in catalogue['items']:
    pid=paper['id']
    source=Path(paper['filename']).stem
    detail={'id':pid,'version':'openreview-current','reports':[], 'pdfUrl':f'/api/paper-file/{pid}'}
    if paper['classified']:
        original=api('/v1/pdf-reader/'+pid)
        req=urllib.request.Request(f'http://127.0.0.1:8000/v1/pdf-reader/{pid}/file',headers={'Authorization':'Bearer '+key})
        with urllib.request.urlopen(req) as res: data=res.read()
        (out/f'{pid}.pdf').write_bytes(data)
        detail['version']=hashlib.sha256(data).hexdigest()
        detail['pdfUrl']=f'/catalogue/{pid}.pdf'
        detail['reports']=[{'id':r['id'],'text':r['text'],'model':{'name':r['model']['name']},'result':{'segments':r['result']['segments']}} for r in original['reports']]
    (out/f'{pid}.json').write_text(json.dumps(detail,ensure_ascii=False))
    paper['sourceId']=source
(out/'index.json').write_text(json.dumps(catalogue,ensure_ascii=False))
print(f"Exported {len(catalogue['items'])} catalogue entries and reader-only reports.")
