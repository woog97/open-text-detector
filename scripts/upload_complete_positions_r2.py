"""Archive all complete position maps, reusing verified R2 objects and bundles."""
import concurrent.futures, hashlib, io, json, sqlite3, sys, tarfile, time
from pathlib import Path
from upload_paper_pdfs import ROOT, BASE, remote_objects, request
sys.path.insert(0,str(ROOT/'backend'))
from open_text_detector_backend.result_codec import decode
OUT=ROOT/'research/exports/complete-positions-r2'
OUT.mkdir(parents=True,exist_ok=True)
def sha(data):return hashlib.sha256(data).hexdigest()
def matches(obj,data):return obj.get('size')==len(data) and obj.get('etag','').strip('"')==hashlib.md5(data).hexdigest()

def main():
    remote=remote_objects()
    db=sqlite3.connect(f'file:{ROOT}/research/extractions/positioned/index.sqlite3?mode=ro',uri=True);db.row_factory=sqlite3.Row
    rows=[dict(r) for r in db.execute("SELECT * FROM artifacts WHERE mapping_status='complete' AND coverage=1 ORDER BY pdf_sha256,text_sha256")]
    batch=ROOT/'research/benchmarks/meld-cuda/r2-batch'
    locations=json.loads((batch/'artifact-locations.json').read_text())
    bundle_paths={b['key']:Path(b['path']) for b in json.loads((batch/'bundle-plan.json').read_text())}
    classified=json.loads((ROOT/'research/exports/classified-r2/manifest.json').read_text())
    prior={r['source_map']['sha256']:r['source_map'] for p in classified['papers'] for r in p['reports']}
    previous=OUT/'bundle-plan.json'
    if previous.exists():bundle_paths.update({b['key']:Path(b['path']) for b in json.loads(previous.read_text())})
    old_index=OUT/'manifest.json'
    if old_index.exists():prior.update({r['sha256']:r for r in json.loads(old_index.read_text())['artifacts']})
    checked_bundles=set();artifacts=[];pending=[];plans=[];group=[];group_size=0
    def flush():
        nonlocal group,group_size
        if not group:return
        temp=OUT/'bundle-building.tar'
        with tarfile.open(temp,'w') as t:
            for ref,raw in group:
                member=ref['sha256']+'.pgf';meta=tarfile.TarInfo(member);meta.size=len(raw);t.addfile(meta,io.BytesIO(raw))
        raw=temp.read_bytes();h=sha(raw);path=OUT/(h+'.tar');temp.replace(path);key=f'extraction-archive/objects/{h}.tar'
        plans.append({'key':key,'path':str(path),'sha256':h,'bytes':len(raw),'md5':hashlib.md5(raw).hexdigest()})
        for ref,_ in group:ref.update(bundle_key=key,bundle_sha256=h,member=ref['sha256']+'.pgf')
        group=[];group_size=0
    for n,row in enumerate(rows,1):
        raw=(ROOT/row['path']).read_bytes();h=sha(raw);assert h==row['blob_sha256']
        obj=decode(raw);assert obj['pdf_sha256']==row['pdf_sha256'] and obj['text_sha256']==row['text_sha256']==sha(obj['text'].encode())
        assert f"papers/{row['pdf_sha256']}.pdf" in remote
        ref={k:row[k] for k in ['pdf_sha256','text_sha256','mapping_status','coverage','kind']}
        ref.update(sha256=h,bytes=len(raw),offset_unit='unicode_code_points')
        logical=f"extractions/positioned/{row['pdf_sha256']}/{row['text_sha256']}.pgf"
        candidates=[prior.get(h,{}),locations.get(logical,{}),{'r2_key':f'classified-archive/extractions/{h}.pgf'},{'r2_key':logical}]
        found=False
        for loc in candidates:
            key=loc.get('r2_key')
            if key and matches(remote.get(key,{}),raw):ref['r2_key']=key;found=True;break
            key=loc.get('bundle_key')
            if key and key in remote and key in bundle_paths:
                p=bundle_paths[key]
                if key not in checked_bundles:
                    data=p.read_bytes();assert sha(data)==loc['bundle_sha256'] and matches(remote[key],data);checked_bundles.add(key)
                with tarfile.open(p) as t:assert sha(t.extractfile(loc['member']).read())==h
                ref.update({k:loc[k] for k in ['bundle_key','bundle_sha256','member']});found=True;break
        if not found:
            if group_size+len(raw)>32*1024*1024:flush()
            group.append((ref,raw));group_size+=len(raw);pending.append(h)
        artifacts.append(ref)
        if n%2000==0:print(f'Validated {n}/{len(rows)} maps; {len(pending)} missing so far',flush=True)
    flush()
    manifest={'format':'complete-position-archive-v1','created_at':time.time(),'paper_count':len({r['pdf_sha256'] for r in rows}),'artifact_count':len(rows),'artifacts':artifacts}
    (OUT/'manifest.json').write_text(json.dumps(manifest,separators=(',',':')))
    saved_plans = {p['key']: p for p in json.loads(previous.read_text())} if previous.exists() else {}
    saved_plans.update({p['key']: p for p in plans})
    (OUT/'bundle-plan.json').write_text(json.dumps(list(saved_plans.values()),indent=2))
    todo=[p for p in plans if remote.get(p['key'],{}).get('size')!=p['bytes'] or remote.get(p['key'],{}).get('etag','').strip('"')!=p['md5']]
    print(json.dumps({'papers':manifest['paper_count'],'complete_maps':len(rows),'reused_maps':len(rows)-len(pending),'missing_maps':len(pending),'bundles_to_upload':len(todo),'upload_bytes':sum(p['bytes'] for p in todo)}),flush=True)
    def upload(p):
        raw=Path(p['path']).read_bytes();assert sha(raw)==p['sha256'];request(BASE+'/'+p['key'],raw,'application/x-tar');return p
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for n,p in enumerate(pool.map(upload,todo),1):print(f'Uploaded bundle {n}/{len(todo)}',flush=True)
    verified=remote_objects()
    for p in plans:assert verified[p['key']]['size']==p['bytes'] and verified[p['key']]['etag'].strip('"')==p['md5']
    for r in artifacts:assert r.get('r2_key',r.get('bundle_key')) in verified
    raw=(OUT/'manifest.json').read_bytes();key=f'extraction-archive/indexes/{sha(raw)}.json'
    request(BASE+'/'+key,raw,'application/json')
    assert request(BASE+'/'+key)==manifest
    summary={'paper_count':manifest['paper_count'],'complete_maps':len(rows),'new_maps':len(pending),'reused_maps':len(rows)-len(pending),'uploaded_bundles':len(todo),'uploaded_bytes':sum(p['bytes'] for p in todo),'index_key':key,'index_sha256':sha(raw),'verified':True,'local_copies_retained':True,'website_published':False}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
