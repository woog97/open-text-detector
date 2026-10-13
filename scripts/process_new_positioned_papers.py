"""Resumable extraction in verified 500-paper batches, R2 cleanup, and MELD submission."""
import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from open_text_detector_backend.pdf_extraction import extract, save, text_hash
from open_text_detector_backend.result_codec import decode
from open_text_detector_backend.sqlite_runtime import sqlite3

STORE = ROOT / 'research/extractions/positioned'
RUN = ROOT / 'research/classifications/new-positioned-meld-20260924'
MANIFEST = RUN / 'selection.json'

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)

def api(route, body=None, key=None):
    headers = {'Authorization': 'Bearer ' + (ROOT / 'backend/.data/admin.key').read_text().strip(), 'Content-Type': 'application/json'}
    if key: headers['Idempotency-Key'] = key
    request = urllib.request.Request('http://127.0.0.1:8000' + route, headers=headers,
        data=json.dumps(body).encode() if body is not None else None)
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)

def discover():
    if MANIFEST.exists(): return json.loads(MANIFEST.read_text())
    conn = sqlite3.connect(STORE / 'index.sqlite3')
    known = {r[0] for r in conn.execute("SELECT pdf_sha256,path FROM artifacts WHERE kind='canonical'") if (ROOT/r[1]).is_file()}
    conn.close()
    catalogue = sqlite3.connect(f'file:{ROOT}/research/data/reviewbench/catalogue.sqlite3?mode=ro', uri=True)
    papers = {}; count = 0
    for pdf in sorted((ROOT / 'research/data').rglob('*.pdf')):
        if pdf.is_symlink(): continue
        stat = pdf.stat(); raw = pdf.read_bytes()
        if pdf.stat().st_mtime_ns != stat.st_mtime_ns: raise RuntimeError(f'PDF is still being written: {pdf}')
        sha = hashlib.sha256(raw).hexdigest()
        if sha in known: continue
        relative = str(pdf.relative_to(ROOT))
        if sha in papers:
            papers[sha]['pdfs'].append(relative); continue
        title = pdf.stem; conference = pdf.parent.parent.name; year = pdf.parent.name
        meta = catalogue.execute('SELECT title FROM papers WHERE id=?', (conference + ':' + pdf.stem,)).fetchone()
        if meta: title = meta[0]
        papers[sha] = {'pdf_sha256':sha, 'pdf':relative, 'pdfs':[relative], 'pdf_bytes':len(raw),
            'pdf_md5':hashlib.md5(raw).hexdigest(), 'r2_key':f'papers/{sha}.pdf',
            'title':title, 'forum_id':pdf.stem, 'group':conference+'/'+year}
        count += 1
        if count % 1000 == 0: print('Discovered new PDFs:', count, flush=True)
    catalogue.close()
    manifest = {'created_at':time.time(), 'population':len(papers), 'selection':'New PDF hashes absent from canonical extraction index at discovery; exact full canonical text with word rectangles.',
        'preprocessing':'Positioned native extraction, OCR fallback for papers without native words; no classifier text cleanup or truncation.',
        'batch_size':500, 'papers':list(papers.values()), 'verified_batches':[]}
    write(MANIFEST, manifest)
    print('Frozen new distinct PDF snapshot:', len(papers), flush=True)
    return manifest

def extract_one(paper):
    artifact = extract(ROOT/paper['pdf'], ocr_pages=paper.get('ocr_pages'))
    assert artifact['pdf_sha256'] == paper['pdf_sha256'], 'Source changed'
    path = save(artifact, STORE/'objects')
    blob = path.read_bytes()
    return {'text_file':str(path.relative_to(ROOT)), 'text_sha256':artifact['text_sha256'],
        'blob_sha256':hashlib.sha256(blob).hexdigest(), 'bytes':len(blob),
        'characters':len(artifact['text']), 'words':len(artifact['text'].split()),
        'pages':len(artifact['pages']), 'method':artifact['method'], 'ocr_pages':artifact.get('ocr_pages',[])}

def verify_one(paper):
    blob = (ROOT/paper['text_file']).read_bytes(); obj = decode(blob); text = obj['text']
    assert hashlib.sha256(blob).hexdigest() == paper['blob_sha256']
    assert text_hash(text) == paper['text_sha256'] == obj['text_sha256']
    assert obj['pdf_sha256'] == paper['pdf_sha256']
    assert obj['geometry_version'] == 2 and obj['mapping']['status'] == 'complete'
    assert hashlib.sha256((ROOT/paper['pdf']).read_bytes()).hexdigest() == paper['pdf_sha256']
    pages = {p['page']:p for p in obj['pages']}
    previous = 0; clipped = 0; bypage = collections.Counter()
    for box in obj['rectangles']:
        a,b = box['start'],box['end']; page = pages[box['page']]
        assert previous <= a < b <= len(text)
        assert not text[previous:a].strip(), 'Unmapped non-whitespace text'
        assert text[a:b].strip(), 'Empty rectangle'
        assert all(math.isfinite(box[k]) for k in ['x0','y0','x1','y1'])
        assert box['x0'] <= box['x1'] and box['y0'] <= box['y1']
        assert page['width'] > 0 and page['height'] > 0
        if box['x0'] < -2 or box['y0'] < -2 or box['x1'] > page['width']+2 or box['y1'] > page['height']+2: clipped += 1
        previous = b; bypage[box['page']] += 1
    assert not text[previous:].strip()
    # A separately generated plain-text extraction checks for text lost by the XML path.
    plain = subprocess.run(['pdftotext','-enc','UTF-8',str(ROOT/paper['pdf']),'-'], capture_output=True, check=True, timeout=180).stdout.decode('utf-8')
    comparison_text=text
    ocr_pages=set(obj.get('ocr_pages',[]))
    if ocr_pages:
        plain=''.join(subprocess.run(['pdftotext','-f',str(page),'-l',str(page),'-enc','UTF-8',str(ROOT/paper['pdf']),'-'],capture_output=True,check=True,timeout=180).stdout.decode('utf-8') for page in pages if page not in ocr_pages)
        comparison_text=' '.join(text[r['start']:r['end']] for r in obj['rectangles'] if r['page'] not in ocr_pages)
    def letters(value): return collections.Counter(re.sub(r'\s', '', value))
    rawchars, newchars = letters(plain), letters(comparison_text)
    missing = sum((rawchars-newchars).values()) / max(1, sum(rawchars.values()))
    replacements = text.count('\ufffd') / max(1,len(text))
    report = {'pdf_sha256':paper['pdf_sha256'], 'rectangles':len(obj['rectangles']), 'pages':len(pages),
        'empty_pages':[p for p in pages if not bypage[p]], 'out_of_page_boxes':clipped,
        'plain_text_missing_character_fraction':missing, 'replacement_character_fraction':replacements,
        'ocr_pages':sorted(ocr_pages), 'ocr_words_per_page':{page:bypage[page] for page in sorted(ocr_pages)}}
    issues = []
    if missing > .02: issues.append('plain_text_content_difference')
    if replacements > .005: issues.append('font_decoding_replacements')
    if len(text.split()) < 50: issues.append('too_little_text')
    if clipped / max(1,len(obj['rectangles'])) > .01: issues.append('many_out_of_page_rectangles')
    if any(bypage[page]<20 for page in ocr_pages): issues.append('ocr_page_sparse')
    report['issues'] = issues
    return report

def verify_checked(paper):
    try: return verify_one(paper)
    except Exception as exc:
        return {'pdf_sha256':paper['pdf_sha256'],'pdf':paper['pdf'],
            'issues':['verification_failure'],'error':type(exc).__name__+': '+str(exc)}

def extraction(manifest, only_one=False):
    index = sqlite3.connect(STORE/'index.sqlite3')
    for start in range(0, len(manifest['papers']), 500):
        number = start//500+1
        if number in manifest['verified_batches']: continue
        batch = manifest['papers'][start:start+500]
        pending = [p for p in batch if not p.get('text_file') or not (ROOT/p['text_file']).exists()]
        print(f'Batch {number}: extracting {len(pending)} remaining of {len(batch)}', flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
            jobs = {pool.submit(extract_one,p):p for p in pending}
            for n, future in enumerate(concurrent.futures.as_completed(jobs),1):
                paper = jobs[future]
                try: paper.update(future.result()); paper.pop('extraction_error',None)
                except Exception as exc:
                    paper['extraction_error'] = str(exc); write(MANIFEST, manifest); raise
                index.execute('INSERT OR REPLACE INTO artifacts VALUES (?,?,?,?,?,?,?,?)',
                    (paper['pdf_sha256'],paper['text_sha256'],paper['text_file'],paper['blob_sha256'],paper['bytes'],'complete',1.0,'canonical'))
                index.commit(); write(MANIFEST, manifest)
                if n % 100 == 0: print(f'Batch {number}: extracted {n}/{len(pending)}', flush=True)
        print(f'Batch {number}: verifying every compressed artifact, text offset, rectangle and source', flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
            reports = list(pool.map(verify_checked,batch))
        write(RUN/f'verification-{number:03}.json', reports)
        flagged = [r for r in reports if r['issues']]
        if flagged:
            # Font-damaged pages must be rendered and OCRed rather than accepting garbled input.
            repairs=[]
            for report in flagged:
                paper=next(p for p in batch if p['pdf_sha256']==report['pdf_sha256'])
                if paper.get('ocr_pages') or set(report['issues'])-{'plain_text_content_difference','font_decoding_replacements'}: continue
                affected=[]
                for page in range(1,paper['pages']+1):
                    raw=subprocess.run(['pdftotext','-f',str(page),'-l',str(page),'-enc','UTF-8',str(ROOT/paper['pdf']),'-'],capture_output=True,check=True,timeout=180).stdout.decode('utf-8')
                    bad=len(re.findall(r'[\x00-\x08\x0e-\x1f\ufffd]',raw))
                    if bad>20: affected.append(page)
                if not affected: continue
                print(f'Repairing font-damaged pages {affected}: {paper["pdf"]}',flush=True)
                repairs.append((paper,{**paper,'ocr_pages':affected}))
            if repairs:
                with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
                    jobs={pool.submit(extract_one,request):paper for paper,request in repairs}
                    for future in concurrent.futures.as_completed(jobs):
                        paper=jobs[future];old_path=paper['text_file']
                        paper.update(future.result())
                        paper.setdefault('superseded_extractions',[]).append(old_path)
                        index.execute("DELETE FROM artifacts WHERE pdf_sha256=? AND kind='canonical'",(paper['pdf_sha256'],))
                        index.execute('INSERT OR REPLACE INTO artifacts VALUES (?,?,?,?,?,?,?,?)',
                            (paper['pdf_sha256'],paper['text_sha256'],paper['text_file'],paper['blob_sha256'],paper['bytes'],'complete',1.0,'canonical'))
                        index.commit();write(MANIFEST,manifest)
                with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
                    reports=list(pool.map(verify_checked,batch))
                write(RUN/f'verification-{number:03}.json',reports)
                flagged=[r for r in reports if r['issues']]
        if flagged:
            print(json.dumps({'batch':number,'issues':flagged}),flush=True)
            raise RuntimeError(f'Batch {number} needs review before proceeding')
        manifest['verified_batches'].append(number); write(MANIFEST, manifest)
        for paper in batch:
            for old in paper.get('superseded_extractions',[]):
                if old!=paper['text_file']: (ROOT/old).unlink(missing_ok=True)
        print(f'Batch {number} VERIFIED ({len(batch)} papers)', flush=True)
        if only_one: break
    index.close()

def cleanup(manifest):
    assert len(manifest['verified_batches']) == math.ceil(len(manifest['papers'])/500), 'Extraction verification incomplete'
    import upload_paper_pdfs as uploader
    remote = uploader.remote_objects()
    verified = []; pending = []
    for paper in manifest['papers']:
        obj = remote.get(paper['r2_key'],{})
        if obj.get('size') != paper['pdf_bytes'] or obj.get('etag','').strip('"') != paper['pdf_md5']:
            pending.append(paper['pdf_sha256']); continue
        assert hashlib.sha256((ROOT/paper['text_file']).read_bytes()).hexdigest() == paper['blob_sha256']
        for relative in paper['pdfs']:
            path = ROOT/relative
            if path.exists():
                assert path.resolve().is_relative_to((ROOT/'research/data').resolve()) and not path.is_symlink()
                assert hashlib.sha256(path.read_bytes()).hexdigest() == paper['pdf_sha256']
        paper['r2_verified_at'] = time.time(); verified.append(paper)
    write(MANIFEST,manifest)
    # Save the complete verification decision before unlinking any source.
    write(RUN/'cleanup.json', {'checked_at':time.time(),'verified':[p['pdf_sha256'] for p in verified], 'pending':pending})
    for paper in verified:
        for relative in paper['pdfs']:
            path = ROOT/relative
            if path.exists():
                assert hashlib.sha256(path.read_bytes()).hexdigest() == paper['pdf_sha256']
                path.unlink()
        paper['local_deleted'] = True
    write(MANIFEST,manifest)
    print(json.dumps({'r2_verified':len(verified),'waiting_for_r2':len(pending),'deleted_bytes':sum(p['pdf_bytes']*len(p['pdfs']) for p in verified)}),flush=True)
    return not pending

def classifier_parts(obj):
    """Partition oversized inputs at word boundaries, preserving every character and box."""
    text=obj['text'];count=max(math.ceil(len(text)/400000),math.ceil(len(text.split())/80000))
    boundaries=[0]
    for i in range(1,count):
        target=len(text)*i//count
        boundaries.append(next(r['start'] for r in obj['rectangles'] if r['start']>=target))
    boundaries.append(len(text));parts=[]
    for start,end in zip(boundaries,boundaries[1:]):
        part={**obj,'text':text[start:end],'text_sha256':text_hash(text[start:end]),
            'parent_text_sha256':obj['text_sha256'],'parent_start':start,'parent_end':end,
            'rectangles':[{**r,'start':r['start']-start,'end':r['end']-start} for r in obj['rectangles'] if start<=r['start']<end]}
        assert 50<=len(part['text'].split())<=100000 and len(part['text'])<=500000
        assert all(0<=r['start']<r['end']<=len(part['text']) for r in part['rectangles'])
        parts.append(part)
    assert ''.join(p['text'] for p in parts)==text
    restored=[{**r,'start':r['start']+p['parent_start'],'end':r['end']+p['parent_start']} for p in parts for r in p['rectangles']]
    assert restored==obj['rectangles']
    return parts

def scan_ids(paper):
    return [p['scan_id'] for p in paper.get('classification_parts',[paper]) if p.get('scan_id')]

def fully_submitted(paper):
    return all(p.get('scan_id') for p in paper.get('classification_parts',[paper]))

def submit(manifest):
    assert all(p.get('local_deleted') and p.get('r2_verified_at') for p in manifest['papers']), 'Cleanup not finished'
    models = [m for m in api('/v1/models')['items'] if m['provider']=='meld' and m['enabled']]
    assert len(models)==1
    model = models[0]
    if manifest.get('model_id'): assert manifest['model_id'] == model['id']
    manifest.update(model_id=model['id'], model_name=model['name']); write(MANIFEST,manifest)
    # Use the application's own validated, idempotent queue service locally. This
    # avoids thousands of HTTP requests while retaining its transaction semantics.
    from open_text_detector_backend.config import Settings
    from open_text_detector_backend.db import Database
    from open_text_detector_backend.schemas import ScanRequest
    from open_text_detector_backend.service import ScanService
    settings=Settings(_env_file=ROOT/'backend/.env',data_dir=ROOT/'backend/.data')
    database=Database(settings.data_dir);service=ScanService(database,settings)
    for n,paper in enumerate(manifest['papers'],1):
        if fully_submitted(paper): continue
        obj = decode((ROOT/paper['text_file']).read_bytes()); text = obj['text']
        assert text_hash(text)==paper['text_sha256']
        if len(text)>500000 or len(text.split())>100000:
            parts=classifier_parts(obj)
            if not paper.get('classification_parts'):
                paper['classification_parts']=[]
                with sqlite3.connect(STORE/'index.sqlite3') as index:
                    for part in parts:
                        path=save(part,STORE/'objects');blob=path.read_bytes();digest=hashlib.sha256(blob).hexdigest()
                        index.execute('INSERT OR REPLACE INTO artifacts VALUES (?,?,?,?,?,?,?,?)',
                            (part['pdf_sha256'],part['text_sha256'],str(path.relative_to(ROOT)),digest,len(blob),'complete',1.0,'classifier_part'))
                        paper['classification_parts'].append({'text_file':str(path.relative_to(ROOT)),'text_sha256':part['text_sha256'],
                            'parent_start':part['parent_start'],'parent_end':part['parent_end'],'blob_sha256':digest})
                write(MANIFEST,manifest)
            for number,(part,entry) in enumerate(zip(parts,paper['classification_parts']),1):
                assert part['text_sha256']==entry['text_sha256']
                assert decode((ROOT/entry['text_file']).read_bytes())==part
                if entry.get('scan_id'):continue
                body={'text':part['text'],'title':paper['title'][:175]+f' [part {number}/{len(parts)}]','model_id':model['id'],'check_plagiarism':False}
                key='bootstrap:positioned:'+model['id']+':'+paper['pdf_sha256']+f':part{number}'
                result=service.insert([service.prepare(ScanRequest(**body),idempotency=key)])[0]
                assert text_hash(result['text'])==entry['text_sha256']
                database.audit('scan.submitted','bootstrap',result['id'])
                entry.update(scan_id=result['id'],status_at_submission=result['status']);write(MANIFEST,manifest)
            print(f'Queued complete oversized paper in {len(parts)} mapped parts: {paper["title"]}',flush=True)
            continue
        assert 50 <= len(text.split()) <= 100000 and len(text)<=500000, 'Paper exceeds classifier limits; do not truncate'
        body = {'text':text,'title':paper['title'][:200],'model_id':model['id'],'check_plagiarism':False}
        key = 'bootstrap:positioned:'+model['id']+':'+paper['pdf_sha256']
        result=service.insert([service.prepare(ScanRequest(**body),idempotency=key)])[0]
        database.audit('scan.submitted','bootstrap',result['id'])
        assert text_hash(result['text'])==paper['text_sha256']
        paper.update(scan_id=result['id'],status_at_submission=result['status']); write(MANIFEST,manifest)
        if n%100==0: print(f'MELD queued {n}/{len(manifest["papers"])}',flush=True)
    print('All new papers queued for MELD.',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['discover','extract','cleanup','submit','finish'])
    parser.add_argument('--one-batch',action='store_true')
    args=parser.parse_args(); RUN.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (RUN/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=discover()
        if args.stage=='extract': extraction(manifest,args.one_batch)
        elif args.stage=='cleanup':
            if not cleanup(manifest): sys.exit(3)
        elif args.stage=='submit': submit(manifest)
        elif args.stage=='finish':
            # Completed deletion is a durable checkpoint. Resuming queue submission
            # requires no new remote read when every source was already verified/deleted.
            while not all(p.get('local_deleted') and p.get('r2_verified_at') for p in manifest['papers']):
                try:
                    if cleanup(manifest): break
                except RuntimeError as exc:
                    if 'R2 request failed' not in str(exc): raise
                    print('R2 temporarily unavailable; preserving local PDFs and retrying:',str(exc),flush=True)
                for _ in range(5):time.sleep(60)
            submit(manifest)

if __name__=='__main__': main()
