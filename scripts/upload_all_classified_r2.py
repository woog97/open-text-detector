# The classification-upload Worker is not deployed (none existed on Oct 13, 2026); deploy app/wrangler.classification-upload.json before use.
"""Archive every locally classified PDF, preserving all reports and exact position maps.

Uses a scoped temporary Workers endpoint for immutable uploads, not the rate-limited
Cloudflare management API. Credentials and model secrets never enter the archive.
"""
import concurrent.futures
import hashlib
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from open_text_detector_backend.result_codec import decode, encode
from upload_paper_pdfs import remote_objects

OUT=ROOT/'research/exports/classified-r2'
OUT.mkdir(parents=True,exist_ok=True)
ACCESS=json.loads(Path('/tmp/pangram-classification-upload-access.json').read_text())
REMOTE=remote_objects()
BATCH=ROOT/'research/benchmarks/meld-cuda/r2-batch'
locations=json.loads((BATCH/'artifact-locations.json').read_text())
bundle_plan={x['key']:x for x in json.loads((BATCH/'bundle-plan.json').read_text())}
verified_bundles=set()
uploads={}
reused=set()
papers={}
index=sqlite3.connect(f'file:{ROOT}/research/extractions/positioned/index.sqlite3?mode=ro',uri=True)
index.row_factory=sqlite3.Row
by_text=defaultdict(list)
for row in index.execute('SELECT * FROM artifacts'):
    by_text[row['text_sha256']].append(dict(row))


def sha(raw):return hashlib.sha256(raw).hexdigest()


def location(path, expected, kind, prior=None):
    raw=path.read_bytes()
    assert sha(raw)==expected, str(path)
    decode(raw)  # verify internal PGF checksum and lossless schema before archive
    if prior:
        if 'bundle_key' in prior:
            key=prior['bundle_key']
            if key not in verified_bundles:
                b=bundle_plan[key];data=Path(b['path']).read_bytes();obj=REMOTE[key]
                assert sha(data)==b['sha256']==prior['bundle_sha256']
                assert obj['size']==len(data) and obj['etag'].strip('"')==hashlib.md5(data).hexdigest()
                verified_bundles.add(key)
            reused.add((key,prior['member']))
            return {k:prior[k] for k in ['bundle_key','bundle_sha256','member']} | {'sha256':expected,'bytes':len(raw)}
        if 'r2_key' in prior:
            key=prior['r2_key'];obj=REMOTE.get(key,{})
            if obj.get('size')==len(raw) and obj.get('etag','').strip('"')==hashlib.md5(raw).hexdigest():
                reused.add((key,''))
                return {'r2_key':key,'sha256':expected,'bytes':len(raw)}
    key=f'classified-archive/{kind}/{expected}.pgf'
    uploads[key]={'path':str(path),'sha256':expected,'bytes':len(raw),'md5':hashlib.md5(raw).hexdigest()}
    return {'r2_key':key,'sha256':expected,'bytes':len(raw)}

map_cache={}
def map_ref(row):
    pair=(row['pdf_sha256'],row['text_sha256'])
    if pair not in map_cache:
        key=f'extractions/positioned/{pair[0]}/{pair[1]}.pgf'
        loc=location(ROOT/row['path'],row['blob_sha256'],'extractions',locations.get(key))
        obj=decode((ROOT/row['path']).read_bytes())
        assert obj['pdf_sha256']==pair[0] and obj['text_sha256']==pair[1]==sha(obj['text'].encode())
        map_cache[pair]={'pdf_sha256':pair[0],'text_sha256':pair[1],'mapping_status':row['mapping_status'],
                        'coverage':row['coverage'],'kind':row['kind'],'offset_unit':'unicode_code_points',**loc}
    return map_cache[pair]


def paper(pdf,title):
    if pdf not in papers:
        key=f'papers/{pdf}.pdf'
        assert key in REMOTE, f'PDF absent from R2: {pdf}'
        papers[pdf]={'pdf_sha256':pdf,'pdf_key':key,'title':title,'reports':[]}
    return papers[pdf]


db=sqlite3.connect(f'file:{ROOT}/backend/.data/workspace.sqlite3?mode=ro',uri=True);db.row_factory=sqlite3.Row
ignored=0;reports=0
for n,row in enumerate(db.execute("SELECT id,title,text,result_storage,result,model_snapshot,created_at FROM scans WHERE status='completed' AND deleted_at IS NULL ORDER BY id"),1):
    digest=sha(row['text'].encode());maps=by_text.get(digest,[])
    if not maps:
        ignored+=1;continue
    if row['result_storage']:
        ref=json.loads(row['result_storage']);assert ref['backend']=='local'
        h=ref['sha256'];path=ROOT/'backend/.data/findings'/h[:2]/(h+'.pgf')
        result_loc=location(path,h,'results')
    else:
        raw=encode(json.loads(row['result']),level=3);h=sha(raw);path=OUT/(h+'.pgf');path.write_bytes(raw)
        result_loc=location(path,h,'results')
    model=json.loads(row['model_snapshot'])['model']
    safe_model={k:model[k] for k in ['name','provider','model_id','base_model_id','lower_threshold','upper_threshold'] if k in model}
    for m in maps:
        paper(m['pdf_sha256'],row['title'])['reports'].append({'id':row['id'],'text_sha256':digest,
             'created_at':row['created_at'],'model':safe_model,'result':result_loc,'source_map':map_ref(m)})
    reports+=1
    if n%500==0:print(f'Prepared {n} database records',flush=True)

production=json.loads((ROOT/'research/benchmarks/meld-cuda/production-summary.json').read_text())
result_index={x['text_sha256']:x for x in json.loads((BATCH/'results-index.json').read_text())['items']}
profile=json.loads((ROOT/'models/meld-v5/cuda-runtime-profile.json').read_text())
for item in production['items']:
    published=result_index[item['text_sha256']]
    result_loc=location(BATCH/(item['text_sha256']+'.pgf'),published['sha256'],'results',published)
    m=next(x for x in by_text[item['text_sha256']] if x['pdf_sha256']==item['pdf_sha256'])
    paper(item['pdf_sha256'],'')['reports'].append({'id':'cuda-v5-'+item['text_sha256'],'text_sha256':item['text_sha256'],
        'model':{'name':'MELD v5 CUDA','provider':'meld','revision':profile['revision'],
                 'runtime_profile_sha256':published.get('profile_sha256',json.loads((BATCH/'summary.json').read_text())['profile_sha256'])},
        'result':result_loc,'source_map':map_ref(m)})

manifest={'format':'classified-paper-archive-v1','created_at':time.time(),'paper_count':len(papers),
          'database_reports':reports,'gpu_reports':len(production['items']),'unmapped_database_records_excluded':ignored,
          'papers':list(papers.values()),'cuda_profile':profile}
(OUT/'manifest.json').write_text(json.dumps(manifest,separators=(',',':')))
(OUT/'upload-plan.json').write_text(json.dumps(uploads))
print(json.dumps({'papers':len(papers),'new_objects':len(uploads),'bytes':sum(x['bytes'] for x in uploads.values()),'reused_artifacts':len(reused)}),flush=True)


def http(method,key,raw=None):
    for attempt in range(8):
        req=urllib.request.Request(ACCESS['url']+'/'+key,data=raw,method=method,headers={
            'Authorization':'Bearer '+ACCESS['token'],'Content-Type':'application/octet-stream','User-Agent':'open-text-detector-classified-upload/1.0'})
        try:
            with urllib.request.urlopen(req,timeout=120) as r:return r.read(),dict(r.headers)
        except urllib.error.HTTPError as e:
            if e.code in [401,403]:raise
            if attempt==7:raise
        except (TimeoutError,OSError):
            if attempt==7:raise
        time.sleep(min(2**attempt,30))


def upload(pair):
    key,info=pair;raw=Path(info['path']).read_bytes();assert sha(raw)==info['sha256']
    response,_=http('PUT',key,raw);ack=json.loads(response)
    assert ack['size']==len(raw) and ack['etag'].strip('"')==info['md5'],key
    return key,info

ledger={}
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
    for n,(key,info) in enumerate(pool.map(upload,uploads.items()),1):
        ledger[key]=info
        if n%250==0 or n==len(uploads):
            (OUT/'upload-ledger.json').write_text(json.dumps(ledger));print(f'{n}/{len(uploads)} objects uploaded and ETags verified',flush=True)


def verify(pair):
    key,info=pair;_,headers=http('HEAD',key);headers={k.lower():v for k,v in headers.items()}
    assert int(headers['content-length'])==info['bytes'] and headers['etag'].strip('"')==info['md5'],key
    return key

with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
    for n,_ in enumerate(pool.map(verify,uploads.items()),1):
        if n%1000==0:print(f'Final remote verification {n}/{len(uploads)}',flush=True)
raw=(OUT/'manifest.json').read_bytes();digest=sha(raw);key=f'classified-archive/indexes/{digest}.json'
response,_=http('PUT',key,raw);ack=json.loads(response);assert ack['size']==len(raw) and ack['etag'].strip('"')==hashlib.md5(raw).hexdigest()
retrieved,_=http('GET',key);assert retrieved==raw
summary={'paper_count':len(papers),'database_reports':reports,'gpu_reports':len(production['items']),
         'uploaded_objects':len(uploads),'uploaded_bytes':sum(x['bytes'] for x in uploads.values()),
         'reused_artifacts':len(reused),'index_key':key,'index_sha256':digest,'verified':True,
         'website_published':False,'local_copies_retained':True}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
