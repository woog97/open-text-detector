# The meld-run Worker was deleted Oct 13, 2026 (as pangram-meld-run); redeploy it from app/wrangler.meld-run.json before use.
"""Pinned HF text -> complete R2-map-aligned MELD results, v5 first then v8."""
import concurrent.futures
import hashlib
import json
import os
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

os.environ.setdefault('RAYON_NUM_THREADS','8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from open_text_detector_backend.result_codec import decode, encode

RUN=ROOT/'research/classifications/vast-complete-20260925'
META=RUN/'metadata';META.mkdir(parents=True,exist_ok=True)
access=json.loads((ROOT/'r2-run-access.json').read_text())
queue=json.loads((RUN/'queue.json').read_text())
state=json.loads((ROOT/'run-state.json').read_text())
STOP=state['classify_until']


def digest(raw):return hashlib.sha256(raw).hexdigest()


def request(method,key,raw=None):
    for attempt in range(5):
        try:
            req=urllib.request.Request(access['url']+'/'+key,data=raw,method=method,headers={
                'Authorization':'Bearer '+access['token'],'User-Agent':'open-text-detector-meld-run/1.0'})
            with urllib.request.urlopen(req,timeout=45) as res:return res.read()
        except Exception:
            if attempt==4:raise
            time.sleep(min(2**attempt,8))


def save_upload(version,item,result):
    result.update(text_sha256=item['text_sha256'],pdf_sha256=item['pdf_sha256'],source_maps=[item],
                  cuda_profile_sha256=profiles[version]['sha256'])
    blob=encode(result,level=3)
    verified=decode(blob)
    assert verified['text_sha256']==item['text_sha256'] and verified['source_maps']==[item]
    path=RUN/'results'/version/(item['text_sha256']+'.pgf');path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_bytes(blob);tmp.replace(path)
    key=access['prefix']+'/'+version+'/'+item['text_sha256']+'.pgf'
    ack=json.loads(request('PUT',key,blob))
    assert ack['size']==len(blob) and ack['etag'].strip('"')==hashlib.md5(blob).hexdigest()
    return {'version':version,'pdf_sha256':item['pdf_sha256'],'text_sha256':item['text_sha256'],
            'result':{'r2_key':key,'sha256':digest(blob),'bytes':len(blob)},'source_map':item,
            'profile_sha256':profiles[version]['sha256'],'label':result['label']}


if __name__=='__main__':
    import pyarrow.parquet as pq
    import torch
    from open_text_detector_backend.providers.meld_cuda import MeldCudaRunner
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    hf=json.loads((ROOT/'hf-access.json').read_text());parquet=ROOT/'papers.parquet'
    assert digest(parquet.read_bytes())==hf['sha256']
    texts={}
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=128,columns=['pdf_sha256','text_sha256','text']):
        for row in batch.to_pylist():
            assert digest(row['text'].encode())==row['text_sha256']
            texts[row['pdf_sha256']]=row
    for item in queue['v8_papers']:
        assert item['mapping_status']=='complete' and item['coverage']==1
        assert texts[item['pdf_sha256']]['text_sha256']==item['text_sha256']
    print(json.dumps({'validated_dataset_papers':len(texts),'v5_jobs':len(queue['v5_papers'])}),flush=True)
    profiles={}
    for version in ['v5','v8']:
        p=ROOT/f'models/meld-{version}/cuda-runtime-profile.json';raw=p.read_bytes()
        profiles[version]={'sha256':digest(raw),'data':json.loads(raw)}
        for file,expected in profiles[version]['data']['source_sha256'].items():
            assert digest((ROOT/'backend/open_text_detector_backend/providers'/file).read_bytes())==expected
    completed={'v5':0,'v8':0};done={'v5':set(),'v8':set()};started=time.time()
    for version in ['v5','v8']:
        expected={item['text_sha256']:item for item in queue[version+'_papers']}
        logfile=META/(version+'-results.jsonl')
        if logfile.exists():
            for line in logfile.read_text().splitlines():
                row=json.loads(line);sha=row['text_sha256']
                assert row['version']==version and row['source_map']==expected[sha]
                assert row['pdf_sha256']==expected[sha]['pdf_sha256']
                assert row['profile_sha256']==profiles[version]['sha256']
                blob=(RUN/'results'/version/(sha+'.pgf')).read_bytes()
                assert len(blob)==row['result']['bytes'] and digest(blob)==row['result']['sha256']
                assert sha not in done[version]
                done[version].add(sha)
        completed[version]=len(done[version])
    print(json.dumps({'resuming':completed}),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as writer:
        for version in ['v5','v8']:
            jobs=[item for item in queue[version+'_papers'] if item['text_sha256'] not in done[version]]
            if not jobs:continue
            if time.time()>STOP-120:break
            runner=MeldCudaRunner(ROOT/f'models/meld-{version}',**profiles[version]['data']['settings'])
            print(json.dumps({'loading':version,'time':time.time()}),flush=True)
            runner.warmup()
            pending=deque()
            log=(META/(version+'-results.jsonl')).open('a')
            def drain(pending=pending, log=log, version=version):
                row=pending.popleft().result();log.write(json.dumps(row,separators=(',',':'))+'\n');log.flush()
                completed[version]+=1
                if completed[version]%100==0:
                    status={'completed':completed.copy(),'elapsed_seconds':time.time()-started,'time':time.time()}
                    (META/'status.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
            for offset in range(0,len(jobs),10):
                if time.time()>=STOP:break
                group=jobs[offset:offset+10];inputs=[texts[x['pdf_sha256']]['text'] for x in group]
                results=runner.predict_many(inputs)
                for item,text,result in zip(group,inputs,results,strict=True):
                    for segment in result.get('segments',[]):
                        assert 0<=segment['start']<=segment['end']<=len(text)
                    pending.append(writer.submit(save_upload,version,item,result))
                while len(pending)>=30:drain()
            while pending:drain()
            log.close();del runner;torch.cuda.empty_cache()
            print(json.dumps({'finished':version,'completed':completed[version],'requested':len(queue[version+'_papers'])}),flush=True)
            if version=='v5' and completed['v5']!=len(queue['v5_papers']):break
    summary={'completed':completed,'v5_requested':len(queue['v5_papers']),'v8_requested':len(queue['v8_papers']),
             'finished_at':time.time(),'elapsed_seconds':time.time()-started,'results_verified_in_r2':True}
    (META/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
