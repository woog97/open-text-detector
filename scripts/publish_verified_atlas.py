"""Publish verified complete-map reports as an atomic, resumable reader snapshot."""
import concurrent.futures, gzip, hashlib, json, sqlite3, sys, time, urllib.request
from pathlib import Path
import pyarrow.parquet as pq
from atlas_scores import summarize
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from open_text_detector_backend.result_codec import decode
OUT=ROOT/'research/exports/atlas-public';OUT.mkdir(parents=True,exist_ok=True)
ACCESS=json.loads(Path('/tmp/pangram-atlas-publish-access.json').read_text())
def sha(raw):return hashlib.sha256(raw).hexdigest()
def atomic(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,separators=(',',':')));tmp.replace(path)
def http(method,key,raw=None):
    for attempt in range(8):
        try:
            req=urllib.request.Request(ACCESS['url']+'/'+key,data=raw,method=method,headers={'Authorization':'Bearer '+ACCESS['token'],'User-Agent':'open-text-detector-atlas-publisher/1.0'})
            with urllib.request.urlopen(req,timeout=120) as response:return response.read()
        except Exception:
            if attempt==7:raise
            time.sleep(min(2**attempt,20))
def put(key,raw):
    ack=json.loads(http('PUT',key,raw));assert ack['size']==len(raw) and ack['etag'].strip('"')==hashlib.md5(raw).hexdigest()

manifest=json.loads((ROOT/'research/exports/classified-r2/manifest.json').read_text())
metadata={r['pdf_sha256']:r for r in pq.read_table(ROOT/'research/exports/paper-text-hf/data/batch-001.parquet',columns=['pdf_sha256','title','conference','year','forum_id']).to_pylist()}
inventory_file=OUT/'pdf-inventory.json'
if not inventory_file.exists():
    from upload_paper_pdfs import request,BASE
    atomic(inventory_file,request(BASE+'/indexes/downloaded-papers.json'))
inventory={r['sha256']:r for r in json.loads(inventory_file.read_text())['papers']}
conn=sqlite3.connect(f'file:{ROOT}/research/extractions/positioned/index.sqlite3?mode=ro',uri=True);conn.row_factory=sqlite3.Row
maps={(r['pdf_sha256'],r['text_sha256']):dict(r) for r in conn.execute("SELECT * FROM artifacts WHERE mapping_status='complete' AND coverage=1")};conn.close()
plan=json.loads((ROOT/'research/exports/classified-r2/upload-plan.json').read_text())
legacy={r['pdf_key']:r['id'] for r in json.loads((OUT/'legacy-ids.json').read_text())[0]['results']}
ledger_path=OUT/'ledger.json';ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {}

def publish(p):
    pdf=p['pdf_sha256'];selected={}
    for r in p['reports']:
        if r['source_map']['mapping_status']!='complete' or r['source_map']['coverage']!=1:continue
        version='v8' if 'v8' in r['model']['name'].lower() else 'v5' if 'v5' in r['model']['name'].lower() else None
        if version and (version not in selected or r.get('created_at','')>selected[version].get('created_at','')):selected[version]=r
    assert 'v5' in selected
    fingerprint=sha(json.dumps({v:r['result']['sha256'] for v,r in selected.items()},sort_keys=True).encode())
    if ledger.get(pdf,{}).get('fingerprint')==fingerprint and all(s.get('policy')==2 for s in ledger[pdf]['item'].get('score_summaries',{}).values()) and ledger[pdf]['item'].get('score_summaries'):
        ledger[pdf]['item']['id']=legacy.get(p['pdf_key'],pdf[:24])
        return pdf,ledger[pdf]
    detail={'id':pdf[:24],'version':pdf,'version_verified':True,'reports':[],'position_maps':{}}
    score_summaries={}
    for version in ['v8','v5']:
        if version not in selected:continue
        r=selected[version];ref=r['result'];key=ref.get('r2_key','')
        if key in plan:path=Path(plan[key]['path'])
        elif key.startswith('classification-runs/vast-complete-20260925/'):
            path=ROOT/'research/classifications/vast-complete-20260925/results'/version/(r['text_sha256']+'.pgf')
        else:path=ROOT/'research/benchmarks/meld-cuda/r2-batch'/(r['text_sha256']+'.pgf')
        raw=path.read_bytes();assert sha(raw)==ref['sha256'];result=decode(raw)
        row=maps[(pdf,r['text_sha256'])];raw=(ROOT/row['path']).read_bytes();assert sha(raw)==r['source_map']['sha256']
        extraction=decode(raw);text=extraction['text'];assert sha(text.encode())==r['text_sha256'] and extraction['pdf_sha256']==pdf
        segments=result['segments'];score_summaries[version]=summarize(segments,text);assert all(0<=s['start']<=s['end']<=len(text) for s in segments)
        detail['reports'].append({'id':r['id'],'text':text,'text_sha256':r['text_sha256'],'model':{'name':'MELD '+version},'result':{'segments':segments}})
        detail['position_maps'][r['text_sha256']]={'pages':extraction['pages'],'rectangles':extraction['rectangles']}
    raw=gzip.compress(json.dumps(detail,separators=(',',':'),ensure_ascii=False).encode(),compresslevel=6,mtime=0)
    key='atlas-public/objects/'+sha(raw)+'.json.gz';put(key,raw)
    m=metadata[pdf];inv=inventory[pdf]
    item={'id':legacy.get(p['pdf_key'],pdf[:24]),'title':m['title'] or p['title'],'filename':inv['filename'],'collection':str(m['conference'])+'/'+str(m['year']),'bytes':inv['bytes'],'classified':True,'models':sorted(selected),'score_summaries':score_summaries,'pdf_key':p['pdf_key'],'detail_key':key}
    return pdf,{'fingerprint':fingerprint,'item':item,'bytes':len(raw)}

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
    for n,(pdf,record) in enumerate(pool.map(publish,manifest['papers']),1):
        ledger[pdf]=record
        if n%100==0 or n==len(manifest['papers']):
            atomic(ledger_path,ledger);print(f'{n}/{len(manifest["papers"])} paper exports verified in R2',flush=True)
items=[ledger[p['pdf_sha256']]['item'] for p in manifest['papers']]
assert len({p['id'] for p in items})==len(items)==14561
counts={v:sum(v in p['models'] for p in items) for v in ['v5','v8']}
catalogue={'published_at':time.time(),'items':items,'models':[{'id':v,'name':'MELD '+v,'available':counts[v],'total':len(items),'complete':counts[v]==len(items)} for v in ['v5','v8']]}
raw=json.dumps(catalogue,separators=(',',':')).encode();put('atlas-public/catalogue.json',raw)
assert http('GET','atlas-public/catalogue.json')==raw
atomic(OUT/'catalogue.json',catalogue)
atomic(OUT/'summary.json',{'published_at':catalogue['published_at'],'paper_count':len(items),'models':counts,'verified':True})
print('Published website snapshot: '+json.dumps(counts),flush=True)
