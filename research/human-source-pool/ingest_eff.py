"""Archived EFF Deeplinks, faithful HTML extraction into isolated Space staging."""
import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urlsplit, urlunsplit
import requests
from bs4 import BeautifulSoup
from collect_pool import LENGTH_WEIGHTS, apportion, atomic_json, digest, make_passages, now
from expand_pool import add, connect

VERSION='eff-commoncrawl-2021-v2'
CRAWL='CC-MAIN-2021-49'
LICENSE='https://creativecommons.org/licenses/by/4.0/'

class Excluded(ValueError):pass

def normalized_url(url):
    p=urlsplit(url)
    if p.hostname not in ('eff.org','www.eff.org'):raise Excluded('not_eff')
    return urlunsplit(('https','www.eff.org',p.path.rstrip('/'),'',''))

def unpack_warc(data,index):
    decoded=gzip.decompress(data);head,_,tail=decoded.partition(b'\r\n\r\n')
    headers=dict(line.split(': ',1) for line in head.decode('utf-8').split('\r\n')[1:] if ': ' in line)
    block=tail[:int(headers['Content-Length'])];http,sep,body=block.partition(b'\r\n\r\n')
    if not sep or not http.startswith(b'HTTP/1.1 200') and not http.startswith(b'HTTP/1.0 200'):raise Excluded('not_http200')
    if headers.get('WARC-Type')!='response' or headers.get('WARC-Date','9999')>='2022-01-01':raise Excluded('not_pre2022_response')
    if normalized_url(headers['WARC-Target-URI'])!=normalized_url(index['url']):raise Excluded('capture_url_mismatch')
    actual=base64.b32encode(hashlib.sha1(body).digest()).decode().rstrip('=')
    if actual!=index['digest']:raise Excluded('archive_payload_digest_mismatch')
    return body,headers

def extract(body,index,warc):
    soup=BeautifulSoup(body,'html.parser');lang=(soup.html.get('lang','') if soup.html else '').lower()
    if lang not in ('en','en-us','en-gb'):raise Excluded('not_english')
    article=soup.select_one('article.node--blog--full')
    if article is None:raise Excluded('not_full_deeplinks_article')
    region=article.select_one('.field--name-body')
    if region is None:raise Excluded('missing_article_body')
    date=soup.find('meta',attrs={'property':'article:published_time'})
    if date is None or not re.match(r'^20\d\d-\d\d-\d\d',date.get('content','')) or date['content']>='2022-01-01':raise Excluded('missing_or_post2021_date')
    canonical=soup.select_one('link[rel=canonical]');url=normalized_url(canonical['href'] if canonical else index['url'])
    if not re.match(r'https://www.eff.org/deeplinks/20(?:0\d|1\d|2[01])/\d\d/[^/]+',url):raise Excluded('not_historical_deeplinks_url')
    byline=soup.select_one('.pane-eff-author');authors=[];byline_text=''
    if byline:
        name_region=byline.select_one('.byline') or byline
        byline_text=re.sub(r'^By\s+','',name_region.get_text(' ',strip=True),flags=re.I)
        linked={anchor.get_text(' ',strip=True):anchor['href'] for anchor in name_region.select('a[href]')}
        for name in re.split(r',\s*|\s+and\s+|\s*&\s*',byline_text):
            name=name.strip()
            if not name:continue
            href=linked.get(name,'')
            authors.append({'name':name,'url':'https://www.eff.org'+href if href.startswith('/') else href})
    if not authors:raise Excluded('missing_original_author_byline')
    full=region.get_text(' ',strip=True)
    if re.search(r'(?i)(originally (?:published|appeared)|republished (?:with|from)|reprinted (?:with|from))',full):raise Excluded('external_republication_notice')
    paragraphs=[]
    for p in region.find_all('p'):
        if p.find(['q','iframe','img','script','form']):continue
        lineage=[p]
        for ancestor in p.parents:
            if ancestor is region:break
            lineage.append(ancestor)
        if any(a.name in ['blockquote','figure','aside','form','nav','header','footer'] for a in lineage):continue
        if any(re.search(r'(?i)(share|donat|subscribe|newsletter|caption|attribution|copyright|license|action-widget)', ' '.join(a.get('class',[]))) for a in lineage if getattr(a,'attrs',None)):continue
        # get_text with empty separator preserves the source's inline-node spacing;
        # no spelling, punctuation, or model-based prose cleanup is applied.
        text=p.get_text('',strip=False)
        if text.strip():paragraphs.append(text)
    text='\n\n'.join(paragraphs)
    if len(text.split())<50:raise Excluded('too_short_after_extraction')
    title=soup.select_one('h1.page-title') or soup.select_one('h1')
    return {'text':text,'url':url,'title':title.get_text(' ',strip=True) if title else '',
            'authors':authors,'byline_text':byline_text,'published':date['content'],'capture_date':warc['WARC-Date'],
            'article_modified':(soup.find('meta',attrs={'property':'article:modified_time'}) or {}).get('content','')}

def pairs(doc,index,warc_bytes,body,warc,remaining):
    text=doc['text'];doc_id=doc['url'];crawl_name=index['filename'].split('/')[1];source_file=index['filename']+':'+index['offset']+':'+index['length']
    raw={'source_id':'eff','source_dataset':'Common Crawl/'+crawl_name,'source_revision':index['digest'],'source_file':source_file,
         'retrieved_at':now(),'raw_text_sha256':digest(text),'record':{'id':doc_id,'text':text,'metadata':{
         'source_html':body.decode('utf-8'),'html_sha256':hashlib.sha256(body).hexdigest(),
         'warc_gzip_sha256':hashlib.sha256(warc_bytes).hexdigest(),'archive_index':index,'warc_headers':warc,
         'authors':doc['authors'],'byline_text':doc['byline_text'],'published':doc['published'],'capture_date':doc['capture_date'],
         'extraction_version':VERSION,'license_basis':'EFF original-material policy; source-approvals.json review snapshot'}}}
    for a,b,w,bin_id in make_passages(text,doc_id,3,remaining.copy()):
        passage=text[a:b]
        yield {'record_id':digest('eff'+doc_id+str(a)+str(b)),'source_id':'eff','category':'general_web','text':passage,
          'word_count':w,'length_bin':bin_id,'source_dataset':'Common Crawl/'+crawl_name,'source_revision':index['digest'],
          'source_file':source_file,'source_row':int(index['offset']),'original_id':doc_id,'source_url':doc_id,'title':doc['title'],
          'author_attribution_json':json.dumps(doc['authors'],ensure_ascii=False),'source_byline':doc['byline_text'],'license_evidence':LICENSE,
          'claimed_original_date':doc['published'],'archive_capture_date':doc['capture_date'],'date_evidence_basis':'2021_WARC_capture_with_verified_payload_digest',
          'retrieved_at':raw['retrieved_at'],'raw_text_sha256':digest(text),'passage_sha256':digest(passage),'raw_start':a,'raw_end':b,
          'offset_unit':'unicode_codepoints','extraction_method':'unchanged_contiguous_span_of_faithfully_decoded_archived_HTML_paragraphs',
          'parent_document_id':doc_id,'provisional_family_id':digest(doc_id),'admission_status':'quarantined_candidate','training_eligible':False,
          'provenance_basis':'official_EFF_archived_pre2022_original_Deeplinks_with_staff_byline','protected_overlap_status':'not_fully_audited',
          'reason_codes':['remaining_rights_and_inline_quotation_audit_pending','author_family_grouping_pending','genre_and_extraction_review_pending','protected_overlap_audit_pending'],
          'sampling_seed':27183,'pipeline_version':VERSION},raw

def fetch_capture(base,index):
    dest=base/'warc'/(''.join(c for c in index['digest'] if c.isalnum())+'.warc.gz');dest.parent.mkdir(exist_ok=True)
    if dest.exists():data=dest.read_bytes()
    else:
        a,n=int(index['offset']),int(index['length']);url='https://data.commoncrawl.org/'+index['filename']
        with requests.get(url,headers={'Range':f'bytes={a}-{a+n-1}'},stream=True,timeout=(20,90)) as response:
            if response.status_code!=206:raise Excluded('warc_range_http_'+str(response.status_code))
            data=b''.join(response.iter_content(1024*1024))
        if len(data)!=n:raise Excluded('warc_range_length_mismatch')
        temp=dest.with_suffix('.partial');temp.write_bytes(data);temp.replace(dest)
    body,warc=unpack_warc(data,index);doc=extract(body,index,warc)
    return doc,data,body,warc

def collect(base,quota):
    started=time.monotonic();(base/'progress').mkdir(exist_ok=True)
    indices={}
    for path in sorted(base.glob('cc-index-20*.jsonl')):
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row.get('languages') not in ('eng',None) or row['timestamp']>='20220101':continue
            url=normalized_url(row['url'])
            if url not in indices or row['timestamp']<indices[url]['timestamp']:indices[url]=row
    rows=[indices[k] for k in sorted(indices,key=lambda k:digest(VERSION+k))]
    with (base/'eff.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);db=connect(base/'collection.sqlite3')
        initial=db.execute("SELECT count(*) FROM passages WHERE source='eff'").fetchone()[0];n=initial;reasons=Counter();scanned=0
        targets=apportion(quota,{str(i):v for i,v in enumerate(LENGTH_WEIGHTS['general_web'])})
        with ThreadPoolExecutor(max_workers=4) as pool:
            for start in range(0,len(rows),8):
                if n>=quota:break
                batch=rows[start:start+8];futures=[pool.submit(fetch_capture,base,r) for r in batch]
                for index,future in zip(batch,futures):
                    scanned+=1
                    try:doc,data,body,warc=future.result()
                    except Exception as exc:reasons[type(exc).__name__+':'+str(exc)[:100]]+=1;continue
                    if n>=quota:continue
                    if db.execute("SELECT 1 FROM passages WHERE source='eff' AND doc=?",(doc['url'],)).fetchone():continue
                    bins=dict(db.execute("SELECT bin,count(*) FROM passages WHERE source='eff' GROUP BY bin"));remaining=[max(0,targets[str(i)]-bins.get(i,0)) for i in range(4)]
                    results=list(pairs(doc,index,data,body,warc,remaining))
                    if not results:reasons['no_matching_length_passages']+=1;continue
                    db.execute('BEGIN IMMEDIATE')
                    try:
                        for record,raw in results:
                            if add(db,record,raw,quota):n+=1
                        db.commit()
                    except BaseException:db.rollback();raise
                status={'source_id':'eff','count':n,'target':quota,'scanned':scanned,'index_unique_urls':len(rows),'rejections':dict(reasons),'state':'collecting','updated_at':now()}
                atomic_json(base/'progress/eff.json',status);print(json.dumps(status),flush=True)
        status.update(state='quota_filled' if n==quota else 'current_archive_inventory_exhausted',new_candidates=n-initial,elapsed_seconds=time.monotonic()-started,all_quarantined=True,global_merge_pending=True)
        atomic_json(base/'progress/eff.json',status);package=base/'accepted-pairs.jsonl.gz';verified=0
        with gzip.open(package,'wt',encoding='utf-8') as out:
            for (saved,) in db.execute("SELECT row FROM passages WHERE source='eff' ORDER BY id"):
                row=json.loads(saved);raw=json.loads(gzip.decompress(db.execute('SELECT raw FROM documents WHERE hash=?',(row['raw_text_sha256'],)).fetchone()[0]));text=raw['record']['text']
                assert digest(text)==row['raw_text_sha256'] and text[row['raw_start']:row['raw_end']]==row['text'] and digest(row['text'])==row['passage_sha256']
                out.write(json.dumps({'row':row,'raw':raw},ensure_ascii=False)+'\n');verified+=1
        db.close();status.update(validated_rows=verified,package=str(package),package_bytes=package.stat().st_size,package_sha256=hashlib.sha256(package.read_bytes()).hexdigest())
        atomic_json(base/'package-manifest.json',status);print(json.dumps(status),flush=True);return status

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--quota',type=int,default=821);a=p.parse_args()
    if not str(a.base).startswith('/tmp/'):p.error('Use isolated Space /tmp staging')
    collect(a.base,a.quota)
