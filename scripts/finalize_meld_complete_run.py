# The meld-run Worker was deleted Oct 13, 2026 (as pangram-meld-run); redeploy it from app/wrangler.meld-run.json before use.
"""Verify durable GPU results and merge them into the local and R2 paper archive."""
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'research/classifications/vast-complete-20260925'
ARCHIVE=ROOT/'research/exports/classified-r2'
ACCESS=json.loads(Path('/tmp/pangram-meld-run-access.json').read_text())
rows=[]
for v in ['v5','v8']:
    p=RUN/'metadata'/f'{v}-results.jsonl'
    if p.exists():
        for line in p.read_text().splitlines():
            rows.append(json.loads(line))
canonical={x['pdf_sha256']:x for x in json.loads((RUN/'queue.json').read_text())['v8_papers']}
profiles={v:json.loads((ROOT/f'models/meld-{v}/cuda-runtime-profile.json').read_text()) for v in ['v5','v8']}
profile_sha={v:hashlib.sha256((ROOT/f'models/meld-{v}/cuda-runtime-profile.json').read_bytes()).hexdigest() for v in ['v5','v8']}


def verify(row):
    assert row['source_map']==canonical[row['pdf_sha256']]
    assert row['text_sha256']==row['source_map']['text_sha256']
    assert row['profile_sha256']==profile_sha[row['version']]
    path=RUN/'results'/row['version']/(row['text_sha256']+'.pgf')
    if path.exists():
        cached=path.read_bytes()
        if len(cached)==row['result']['bytes'] and hashlib.sha256(cached).hexdigest()==row['result']['sha256']:
            return row
    key=row['result']['r2_key'];assert key.startswith('classification-runs/vast-complete-20260925/'+row['version']+'/')
    req=urllib.request.Request(ACCESS['url']+'/'+key,headers={'Authorization':'Bearer '+ACCESS['token'],'User-Agent':'open-text-detector-meld-run/1.0'})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req,timeout=60) as response:blob=response.read()
            assert len(blob)==row['result']['bytes'] and hashlib.sha256(blob).hexdigest()==row['result']['sha256']
            break
        except Exception:
            if attempt==4:raise
            time.sleep(2**attempt)
    # Preserve a verified local copy as well as the durable R2 output.
    path=RUN/'results'/row['version']/(row['text_sha256']+'.pgf');path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_bytes(blob);tmp.replace(path)
    return row


with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
    for n,_ in enumerate(pool.map(verify,rows),1):
        if n%250==0 or n==len(rows):print(f'{n}/{len(rows)} results downloaded and SHA256 verified',flush=True)
old=json.loads((ARCHIVE/'manifest.json').read_text())
backup=RUN/'prior-archive-manifest.json'
if not backup.exists():shutil.copy2(ARCHIVE/'manifest.json',backup)
papers={p['pdf_sha256']:p for p in old['papers']}
import pyarrow.parquet as pq

catalogue={r['pdf_sha256']:r for r in pq.read_table(ROOT/'research/exports/paper-text-hf/data/batch-001.parquet',
           columns=['pdf_sha256','title','conference','year','forum_id']).to_pylist()}
run_created=datetime.fromtimestamp(json.loads((RUN/'run-state.json').read_text())['started_at'],timezone.utc).isoformat()
for row in rows:
    pdf=row['pdf_sha256'];v=row['version'];rid='vast-complete-20260925-'+v+'-'+row['text_sha256']
    p=papers.setdefault(pdf,{'pdf_sha256':pdf,'pdf_key':f'papers/{pdf}.pdf',
          **{k:catalogue[pdf][k] for k in ['title','conference','year','forum_id']},'reports':[]})
    existing=next((r for r in p['reports'] if r['id']==rid),None)
    if existing is not None:
        existing.setdefault('created_at',run_created)
        continue
    p['reports'].append({'id':rid,'created_at':run_created,'text_sha256':row['text_sha256'],'model':{'name':f'MELD {v} CUDA','provider':'meld',
                       'revision':profiles[v]['revision'],'runtime_profile_sha256':profile_sha[v]},
                       'result':row['result'],'source_map':row['source_map']})
complete=sum(any(r['source_map']['mapping_status']=='complete' and r['source_map']['coverage']==1 for r in p['reports']) for p in papers.values())
old.update(created_at=time.time(),paper_count=len(papers),papers=list(papers.values()),complete_map_papers=complete,
           gpu_reports=sum('CUDA' in r['model']['name'] for p in papers.values() for r in p['reports']),
           latest_run={'id':'vast-complete-20260925','v5_completed':sum(r['version']=='v5' for r in rows),'v8_completed':sum(r['version']=='v8' for r in rows)},
           hf_dataset=json.loads((RUN/'dataset-receipt.json').read_text()))
raw=json.dumps(old,separators=(',',':')).encode();digest=hashlib.sha256(raw).hexdigest();key='classified-archive/indexes/'+digest+'.json'
sys.path.insert(0,str(ROOT/'scripts'))
from upload_paper_pdfs import BASE, request

subprocess.run([str(ROOT/'app/node_modules/.bin/wrangler'),'whoami'],cwd=ROOT/'app',capture_output=True,check=True)

request(BASE+'/'+key,raw,'application/json')
assert request(BASE+'/'+key)==old, 'Uploaded archive manifest did not match'
temp=ARCHIVE/'manifest.json.tmp';temp.write_bytes(raw);temp.replace(ARCHIVE/'manifest.json')
summary={'paper_count':len(papers),'complete_map_papers':complete,'partial_only_papers':len(papers)-complete,
         'index_key':key,'index_sha256':digest,'verified':True,'new_results_verified':len(rows),'latest_run':old['latest_run'],
         'local_copies_retained':True,'website_published':(ROOT/'research/exports/atlas-public/summary.json').exists(),
         'website_publication_summary':'research/exports/atlas-public/summary.json'}
temp=ARCHIVE/'summary.json.tmp';temp.write_text(json.dumps(summary,indent=2));temp.replace(ARCHIVE/'summary.json')
(RUN/'archive-summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
