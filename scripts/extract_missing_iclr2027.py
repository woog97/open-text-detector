"""Backfill positioned extractions for complete-mirror rows that have none.

Run on the training Space with poppler/tesseract matching the original
pipeline. Reuses pdf_extraction.extract and the original verification and
page-level OCR repair rules (process_new_positioned_papers.verify_one,
repair_iclr2027_extractions.repair). The source dataset is read-only; output
is an additive dataset with content-addressed sidecars (objects/<sha256>.pgf).
Resumable: completed papers are recorded in receipts.jsonl. No model inference.
"""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from open_text_detector_backend.pdf_extraction import extract, text_hash
from open_text_detector_backend.result_codec import encode, decode


TIMEOUT=180  # seconds per poppler call; --timeout raises it for pathological PDFs


def digest(raw):return hashlib.sha256(raw).hexdigest()


def atomic(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    temp.replace(path)


def plain_text(pdf,page=None):
    pages=['-f',str(page),'-l',str(page)] if page else []
    return subprocess.run(['pdftotext',*pages,'-enc','UTF-8',str(pdf),'-'],capture_output=True,check=True,timeout=TIMEOUT).stdout.decode('utf-8')


def letters(value):return collections.Counter(re.sub(r'\s','',value))


def verify(obj,pdf,pdf_sha256):
    """Same checks and thresholds as process_new_positioned_papers.verify_one."""
    text=obj['text']
    assert text_hash(text)==obj['text_sha256'] and obj['pdf_sha256']==pdf_sha256
    assert obj['geometry_version']==2 and obj['mapping']['status']=='complete'
    pages={p['page']:p for p in obj['pages']}
    previous=0;clipped=0;bypage=collections.Counter()
    for box in obj['rectangles']:
        a,b=box['start'],box['end'];page=pages[box['page']]
        assert previous<=a<b<=len(text) and not text[previous:a].strip() and text[a:b].strip()
        assert all(math.isfinite(box[k]) for k in ['x0','y0','x1','y1'])
        assert box['x0']<=box['x1'] and box['y0']<=box['y1'] and page['width']>0 and page['height']>0
        if box['x0']<-2 or box['y0']<-2 or box['x1']>page['width']+2 or box['y1']>page['height']+2:clipped+=1
        previous=b;bypage[box['page']]+=1
    assert not text[previous:].strip()
    ocr_pages=set(obj.get('ocr_pages',[]))
    if ocr_pages:
        plain=''.join(plain_text(pdf,p) for p in pages if p not in ocr_pages)
        comparison=' '.join(text[r['start']:r['end']] for r in obj['rectangles'] if r['page'] not in ocr_pages)
    else:plain,comparison=plain_text(pdf),text
    raw,new=letters(plain),letters(comparison)
    missing=sum((raw-new).values())/max(1,sum(raw.values()))
    replacements=text.count('�')/max(1,len(text))
    issues=[]
    if missing>.02:issues.append('plain_text_content_difference')
    if replacements>.005:issues.append('font_decoding_replacements')
    if len(text.split())<50:issues.append('too_little_text')
    if clipped/max(1,len(obj['rectangles']))>.01:issues.append('many_out_of_page_rectangles')
    if any(bypage[p]<20 for p in ocr_pages):issues.append('ocr_page_sparse')
    return {'rectangles':len(obj['rectangles']),'pages':len(pages),'empty_pages':[p for p in pages if not bypage[p]],
            'out_of_page_boxes':clipped,'plain_text_missing_character_fraction':missing,
            'replacement_character_fraction':replacements,'ocr_pages':sorted(ocr_pages),
            'ocr_words_per_page':{p:bypage[p] for p in sorted(ocr_pages)},'issues':issues}


def repair_pages(obj,pdf):
    """Same page selection as repair_iclr2027_extractions.repair."""
    selected=[];diagnostics=[]
    for page in obj['pages']:
        n=page['page'];native=' '.join(obj['text'][b['start']:b['end']] for b in obj['rectangles'] if b['page']==n)
        plain=plain_text(pdf,n)
        difference=sum((letters(plain)-letters(native)).values())/max(1,sum(letters(plain).values()))
        replacements=native.count('�')/max(1,len(native))
        if difference>.02 or replacements>.005:selected.append(n)
        diagnostics.append({'page':n,'plain_text_difference':difference,'replacement_fraction':replacements})
    return selected,diagnostics


def one(job):
    source,output,row=job;source=Path(source);output=Path(output)
    pdf=source/row['pdf_path'];started=time.time()
    if digest(pdf.read_bytes())!=row['pdf_sha256']:raise ValueError('Source PDF checksum mismatch')
    artifact=None;error=None
    try:artifact=extract(pdf,timeout=TIMEOUT)
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    repair=None
    if artifact is not None:
        report=verify(artifact,pdf,row['pdf_sha256'])
        if set(report['issues'])&{'plain_text_content_difference','font_decoding_replacements'}:
            selected,diagnostics=repair_pages(artifact,pdf)
            if selected:
                repair={'ocr_pages':selected,'diagnostics':diagnostics,'prior_text_sha256':artifact['text_sha256'],'prior_issues':report['issues']}
                artifact=extract(pdf,ocr_pages=selected,timeout=TIMEOUT);report=verify(artifact,pdf,row['pdf_sha256'])
    else:
        # No native words at all: extract() OCRs every page itself only when
        # poppler returns no words; a hard failure here is recorded, not hidden.
        return {'id':row['id'],'pdf_sha256':row['pdf_sha256'],'state':'failed','error':error,'seconds':time.time()-started}
    blob=encode(artifact,level=9)
    if decode(blob)!=artifact:raise ValueError('Extraction roundtrip failed')
    blob_sha256=digest(blob);path=output/'objects'/f'{blob_sha256}.pgf'
    if path.exists():
        if path.read_bytes()!=blob:raise ValueError('Existing sidecar differs')
    else:atomic(path,blob)
    return {'id':row['id'],'pdf_sha256':row['pdf_sha256'],'state':'extracted','text_sha256':artifact['text_sha256'],
            'blob_sha256':blob_sha256,'bytes':len(blob),'positions_path':f'objects/{blob_sha256}.pgf',
            'method':artifact['method'],'ocr_pages':artifact.get('ocr_pages',[]),'pages':len(artifact['pages']),
            'words':len(artifact['text'].split()),'verification':report,'repair':repair,'seconds':time.time()-started}


def tools():
    return {name:subprocess.run([name,'-v' if name.startswith('pdf') else '--version'],capture_output=True,text=True).stdout.splitlines()[:1]+
            subprocess.run([name,'-v' if name.startswith('pdf') else '--version'],capture_output=True,text=True).stderr.splitlines()[:1]
            for name in ['pdftotext','pdftoppm','tesseract']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--timeout',type=int,default=180)
    p.add_argument('--reproduce',type=int,default=0,help='Re-extract N rows that already have extractions and compare text hashes; writes nothing.')
    args=p.parse_args()
    global TIMEOUT
    TIMEOUT=args.timeout
    if not 1<=args.workers<=(os.cpu_count() or 1):p.error('Too many workers')
    import pyarrow as pa
    import pyarrow.parquet as pq
    source=args.source.resolve();output=args.output.resolve()
    if output==source or output.is_relative_to(source):p.error('Output must be outside the source dataset')
    shards=sorted((source/'data').glob('train-*.parquet'))
    columns=['id','title','conference','year','pdf_sha256','pdf_path','positions_path','text_sha256','metadata_json','source_url']
    rows=[r for s in shards for r in pq.read_table(s,columns=columns).to_pylist()]
    if args.reproduce:
        import random
        random.seed(0);sample=random.sample([r for r in rows if r['positions_path']],args.reproduce)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results=list(pool.map(reproduce,[(str(source),r) for r in sample]))
        print(json.dumps({'reproduced':sum(r['match'] for r in results),'of':len(results),'mismatches':[r for r in results if not r['match']]}),flush=True)
        return
    todo=[r for r in rows if not r['positions_path']]
    output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(output/'build.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipts=output/'receipts.jsonl';done={}
    if receipts.exists():
        for line in receipts.read_text().splitlines():
            r=json.loads(line);done[r['id']]=r
    provenance={'source_dataset':source.name,'source_shards_sha256':digest(b''.join(digest(s.read_bytes()).encode() for s in shards)),
                'rows_without_extraction':len(todo),'tools':tools(),'code_sha256':digest(Path(__file__).read_bytes())}
    atomic(output/'provenance.json',json.dumps(provenance,indent=2).encode())
    pending=[r for r in todo if r['id'] not in done or done[r['id']]['state']!='extracted']
    print(json.dumps({'to_extract':len(pending),'already_done':len(todo)-len(pending)}),flush=True)
    started=time.time();n=0
    with ProcessPoolExecutor(max_workers=args.workers) as pool,receipts.open('a') as log:
        futures={pool.submit(one,(str(source),str(output),r)):r for r in pending}
        for future in as_completed(futures):
            row=futures[future]
            try:result=future.result()
            except Exception as exc:result={'id':row['id'],'pdf_sha256':row['pdf_sha256'],'state':'failed','error':f'{type(exc).__name__}: {exc}'}
            done[row['id']]=result;log.write(json.dumps(result)+'\n');log.flush();os.fsync(log.fileno());n+=1
            if n%50==0 or n==len(pending):
                states=collections.Counter(r['state'] for r in done.values())
                status={'state':'extracting','completed':n,'pending':len(pending),'states':dict(states),'elapsed_seconds':time.time()-started}
                atomic(output/'status.json',json.dumps(status).encode());print(json.dumps(status),flush=True)
    # Publish one Parquet table in the complete-mirror schema, re-verifying every sidecar.
    out=[]
    for r in todo:
        result=done.get(r['id'])
        if not result or result['state']!='extracted':continue
        raw=(output/result['positions_path']).read_bytes()
        if digest(raw)!=result['blob_sha256']:raise ValueError('Sidecar changed after extraction')
        obj=decode(raw)
        if text_hash(obj['text'])!=result['text_sha256'] or obj['pdf_sha256']!=r['pdf_sha256']:raise ValueError('Sidecar identity mismatch')
        out.append({'id':r['id'],'title':r['title'],'conference':r['conference'],'year':r['year'],'text':obj['text'],
                    'pdf_sha256':r['pdf_sha256'],'text_sha256':result['text_sha256'],
                    'pdf_path':f'../{source.name}/{r["pdf_path"]}','positions_path':result['positions_path'],
                    'extraction_available':True,'metadata_json':r['metadata_json'],'source_url':r['source_url'],
                    'method':result['method'],'ocr_pages':result['ocr_pages'],
                    'verification_json':json.dumps({'verification':result['verification'],'repair':result['repair']})})
    table=pa.Table.from_pylist(out);path=output/'data'/'train-00000.parquet';path.parent.mkdir(exist_ok=True)
    temp=path.with_suffix('.tmp');pq.write_table(table,temp,compression='zstd')
    if pq.read_table(temp).to_pylist()!=out:raise ValueError('Parquet readback mismatch')
    temp.replace(path)
    states=collections.Counter(done[r['id']]['state'] for r in todo if r['id'] in done)
    flagged=[{'id':x['id'],'issues':done[x['id']]['verification']['issues']} for x in out if done[x['id']]['verification']['issues']]
    completion={**provenance,'state':'complete' if states.get('extracted',0)==len(todo) else 'partial','states':dict(states),
                'rows':len(out),'parquet_sha256':digest(path.read_bytes()),'remaining_issues':flagged,
                'failures':[done[r['id']] for r in todo if r['id'] in done and done[r['id']]['state']!='extracted'],
                'source_modified':False,'model_inference_runs':0}
    atomic(output/'completion.json',json.dumps(completion,indent=2).encode())
    print(json.dumps({k:v for k,v in completion.items() if k not in ('remaining_issues','failures')}),flush=True)


def reproduce(job):
    source,row=job;source=Path(source)
    existing=decode((source/row['positions_path']).read_bytes())
    obj=extract(source/row['pdf_path'],ocr_pages=existing.get('ocr_pages') or None)
    return {'id':row['id'],'match':obj['text_sha256']==row['text_sha256'],'existing_method':existing['method'],
            'existing_ocr_pages':existing.get('ocr_pages',[]),'native_text_sha256':obj['text_sha256']}


if __name__=='__main__':main()
