"""Stage official Board-authored Fed speeches; independent from the production DB."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urljoin,urlparse

import requests
from bs4 import BeautifulSoup
from collect_pool import LENGTH_WEIGHTS, apportion, atomic_json, digest, make_passages, now
from expand_pool import add,connect

VERSION='institutional-official-fed-v1'
HOST='https://www.federalreserve.gov'
SPEECH=re.compile(r'^/newsevents/speech/[a-z]+((?:19|20)\d{6})[a-z]\.htm$')
BOARD_ROLE=re.compile(r'^(?:Governor\b|Chair(?:man|woman)?\b|Vice Chair(?:man|woman)?\b)',re.I)


def speech_date(url):
    parsed=urlparse(url)
    if parsed.scheme!='https' or parsed.netloc!='www.federalreserve.gov':return None
    match=SPEECH.fullmatch(parsed.path)
    if not match:return None
    value=match[1]
    if int(value[:4])>2021:return None
    from datetime import datetime
    try:return datetime.strptime(value,'%Y%m%d').date().isoformat()
    except ValueError:return None


def speech_links(html,index_url):
    soup=BeautifulSoup(html,'html.parser')
    return sorted({urljoin(index_url,a['href']) for a in soup.select('a[href]')
                   if speech_date(urljoin(index_url,a['href']))})


def extract_fed(html,url):
    date=speech_date(url)
    if not date:raise ValueError('Not a pre-2022 official Fed speech URL')
    soup=BeautifulSoup(html,'html.parser');article=soup.select_one('#article')
    if article is None:raise ValueError('Official article container missing')
    speaker=article.select_one('.speaker');title=article.select_one('.title')
    author=speaker.get_text(' ',strip=True) if speaker else ''
    if not BOARD_ROLE.search(author):raise ValueError('Board-author role not established')
    bodies=[d for d in article.select('div.col-md-8') if 'heading' not in d.get('class',[]) and 'panel' not in d.get('class',[])]
    if not bodies:raise ValueError('Speech prose container missing')
    body=max(bodies,key=lambda d:len(d.get_text()))
    # Remove attributed material and non-prose at the DOM level. Preserve the
    # complete HTML separately and record exact offsets into derived plain text.
    for node in body.select('script,style,table,figure,blockquote,.footnotes,.panel'):
        node.decompose()
    paragraphs=[]
    for p in body.find_all(['p','h4','h5']):
        text=p.get_text(' ',strip=True)
        if not text:continue
        # Inline footnote *references* occur in ordinary speech paragraphs too;
        # only the actual footnote section's return link terminates prose.
        if re.search(r'Return to text\s*$',text):
            break
        if p.find_parent(['blockquote','table']):continue
        # Long quoted third-party passages are omitted rather than labeled as
        # Board authorship. Short economic terms in quotes are retained.
        quotes=re.findall(r'[“"]([^”"]+)[”"]',text)
        if any(len(q.split())>40 for q in quotes):continue
        paragraphs.append(text)
    text='\n\n'.join(paragraphs)
    if len(text.split())<50:raise ValueError('Speech lacks sufficient Board prose')
    return {'text':text,'date':date,'author':author,'title':title.get_text(' ',strip=True) if title else '',
            'extraction':'HTML speech body paragraphs; inline whitespace canonicalized; quotes/tables/footnotes excluded'}


def pairs(html,url,remaining):
    extracted=extract_fed(html,url);text=extracted['text'];html_hash=digest(html);raw_hash=digest(text)
    doc_id=url;retrieved=now();source_file='source-downloads/fed/'+digest(url)+'.html'
    raw={'source_id':'fed','source_dataset':'federalreserve.gov/board-speeches','source_revision':html_hash,
         'source_file':source_file,'retrieved_at':retrieved,'raw_text_sha256':raw_hash,
         'record':{'id':url,'text':text,'metadata':{'html':html,'html_sha256':html_hash,'url':url,
                    'title':extracted['title'],'author':extracted['author'],'date':extracted['date'],
                    'extraction':extracted['extraction'],'license':'US government Board-authored prose; public domain unless otherwise indicated',
                    'license_evidence':'https://www.federalreserve.gov/disclaimer.htm'}}}
    for a,b,wc,bin_id in make_passages(text,doc_id,3,remaining.copy()):
        value=text[a:b]
        row={'record_id':digest('fed'+doc_id+str(a)+str(b)),'source_id':'fed','category':'professional','text':value,
             'word_count':wc,'length_bin':bin_id,'source_dataset':'federalreserve.gov/board-speeches','source_revision':html_hash,
             'source_file':source_file,'original_id':url,'source_url':url,'title':extracted['title'],
             'author_attribution_json':json.dumps([extracted['author']]),
             'license_evidence':'US government Board-authored prose: https://www.federalreserve.gov/disclaimer.htm',
             'claimed_original_date':extracted['date'],'date_evidence_basis':'official dated speech URL',
             'retrieved_at':retrieved,'raw_text_sha256':raw_hash,'passage_sha256':digest(value),'raw_start':a,'raw_end':b,
             'offset_unit':'unicode_codepoints_in_preserved_extracted_text','extraction_method':'contiguous_derived_official_HTML_paragraphs',
             'parent_document_id':doc_id,'provisional_family_id':digest(doc_id),'admission_status':'quarantined_candidate',
             'training_eligible':False,'provenance_basis':'official_Board_role_attributed_pre_2022_speech',
             'protected_overlap_status':'not_fully_audited','reason_codes':['protected_overlap_audit_pending',
             'inline_quotation_and_extraction_review_pending','historical_HTML_version_unverified'],
             'sampling_seed':27183,'pipeline_version':VERSION}
        yield row,raw


def fetch(folder,url):
    path=folder/(digest(url)+'.html');manifest=path.with_suffix('.json')
    if path.exists() and manifest.exists():
        saved=json.loads(manifest.read_text());body=path.read_bytes()
        if saved['url']!=url or hashlib.sha256(body).hexdigest()!=saved['sha256']:raise ValueError('Saved HTML checksum mismatch')
        return body.decode('utf-8-sig'),saved
    r=requests.get(url,timeout=(15,45),headers={'User-Agent':'OpenTextDetector-research-corpus/1.0'})
    r.raise_for_status()
    if urlparse(r.url).hostname!='www.federalreserve.gov':raise ValueError('Unexpected redirect domain')
    body=r.content;text=body.decode('utf-8-sig')
    path.write_bytes(body)
    value={'url':url,'resolved_url':r.url,'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body),'retrieved_at':now()}
    atomic_json(manifest,value)
    return text,value


def stage(base,quota=776,workers=4):
    """Writes ONLY base/stage.sqlite3; merging production is a separate operation."""
    folder=base/'source-downloads/fed';folder.mkdir(parents=True,exist_ok=True);(base/'progress').mkdir(exist_ok=True)
    began=time.monotonic();errors=[]
    with (base/'fed.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        db=connect(base/'stage.sqlite3');config=json.dumps({'source':'fed','quota':quota,'version':VERSION},sort_keys=True)
        saved=db.execute("SELECT value FROM settings WHERE key='institutional_stage'").fetchone()
        if saved and saved[0]!=config:raise ValueError('Stage configuration changed')
        with db:db.execute("INSERT OR IGNORE INTO settings VALUES ('institutional_stage',?)",(config,))
        target=apportion(quota,{str(i):v for i,v in enumerate(LENGTH_WEIGHTS['professional'])})
        links=set()
        for year in range(2021,2009,-1):
            url=HOST+'/newsevents/speech/'+str(year)+'-speeches.htm'
            try:html,_=fetch(folder,url);links.update(speech_links(html,url))
            except Exception as exc:errors.append({'url':url,'error':str(exc)[:180]})
        ordered=sorted(links,key=lambda u:digest('27183'+u));count=db.execute("SELECT count(*) FROM passages WHERE source='fed'").fetchone()[0]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0,len(ordered),workers):
                if count>=quota:break
                urls=[u for u in ordered[start:start+workers] if not db.execute("SELECT 1 FROM cursors WHERE source='fed' AND file=? AND done=1",(u,)).fetchone()]
                futures=[pool.submit(fetch,folder,u) for u in urls]
                for url,future in zip(urls,futures):
                    if count>=quota:break
                    try:
                        html,manifest=future.result();bins=dict(db.execute("SELECT bin,count(*) FROM passages WHERE source='fed' GROUP BY bin"))
                        remaining=[max(0,target[str(i)]-bins.get(i,0)) for i in range(4)]
                        results=list(pairs(html,url,remaining))
                        db.execute('BEGIN IMMEDIATE')
                        try:
                            for row,raw in results:
                                if add(db,row,raw,quota):count+=1
                            db.execute("INSERT OR REPLACE INTO cursors VALUES ('fed',?,1,1)",(url,));db.commit()
                        except BaseException:db.rollback();raise
                    except Exception as exc:errors.append({'url':url,'error':str(exc)[:180]})
                atomic_json(base/'progress/fed.json',{'source':'fed','state':'staging','count':count,'target':quota,
                            'discovered_documents':len(links),'processed_position':start,'errors':errors[-5:],'updated_at':now()})
        out=base/'staged/fed';out.mkdir(parents=True,exist_ok=True)
        with gzip.open(out/'passages.jsonl.gz','wt') as f:
            for row, in db.execute("SELECT row FROM passages WHERE source='fed' ORDER BY id"):f.write(row+'\n')
        with gzip.open(out/'documents.jsonl.gz','wt') as f:
            for raw, in db.execute('SELECT raw FROM documents'):f.write(gzip.decompress(raw).decode()+'\n')
        bins=dict(db.execute("SELECT bin,count(*) FROM passages WHERE source='fed' GROUP BY bin"));db.close()
        status={'source_id':'fed','state':'staged_quota_filled' if count==quota else 'staged_shortfall','count':count,'target':quota,
                'length_counts':bins,'elapsed_seconds':time.monotonic()-began,'errors':errors,'updated_at':now(),
                'production_pool_modified':False,'global_dedup_against_production_pending':True,'all_quarantined':True}
        atomic_json(base/'progress/fed.json',status);print(json.dumps(status),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--source',choices=['fed'],default='fed')
    p.add_argument('--quota',type=int,default=776);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();stage(args.base,args.quota,args.workers)
