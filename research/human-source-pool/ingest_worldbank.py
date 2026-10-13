"""World Bank official text mirrors, per-work CC BY grants; isolated staging."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import time
from urllib.parse import urlparse,urlunparse

import requests
from collect_pool import BINS,LENGTH_WEIGHTS,STOPWORDS,apportion,atomic_json,digest,now
from expand_pool import add,connect

VERSION='worldbank-official-text-ccby-v1'
API='https://search.worldbank.org/api/v3/wds'
HOSTS={'documents.worldbank.org','documents1.worldbank.org'}
PARAMS={'format':'json','rows':200,'lang_exact':'English','majdocty_exact':'Publications',
        'srt':'docdt','order':'desc','enddate':'2021-12-31','strdate':'2012-01-01',
        'fl':'id,docdt,lang,authr,pdfurl,txturl,dois,isbn,repnb,guid,chronical_docm_id,disclstat,versiontyp'}


def eligible_doc(doc):
    return (isinstance(doc,dict) and doc.get('id') and doc.get('lang')=='English'
            and re.match(r'^20(?:0\d|1\d|2[01])-\d\d-\d\d',str(doc.get('docdt','')))
            and doc.get('txturl') and doc.get('disclstat','Disclosed')=='Disclosed')


def text_url(doc):
    value=urlparse(doc['txturl'])
    if value.hostname not in HOSTS or not re.fullmatch(r'/curated/en/[0-9]+/text/[^/]+\.txt',value.path):
        raise ValueError('Unexpected official text URL')
    return urlunparse(('https','documents1.worldbank.org',value.path,'','',''))


def license_grant(text):
    # A license URL cited in a reference is not a license for this work. Require
    # the publication's own explicit grant near its front matter.
    head=text[:30000]
    claim=re.search(r'\b(?:This|The)\s+(?:work|publication|material)\s+is\s+(?:made\s+)?(?:available|licensed|distributed)\s+under\b',head,re.I)
    if not claim:return None
    end=min(len(head),claim.end()+1100);grant=head[claim.start():end]
    compact=' '.join(grant.split())
    if re.search(r'creativecommons\.org/licenses/by-(?:nc|nd)|CC\s+BY[ -](?:NC|ND)|noncommercial|non-commercial|no.?derivatives',compact,re.I):return None
    match=re.search(r'(?:https?://\s*)?creativecommons\.org/licenses/\s*by/\s*([234]\.0)(?:/\s*(igo))?/?',compact,re.I)
    if not match:return None
    return {'url':'https://creativecommons.org/licenses/by/'+match[1]+('/igo' if match[2] else '')+'/',
            'raw_start':claim.start(),'raw_end':end,'verbatim_context':grant,
            'basis':'explicit_per_work_front_matter_license_grant'}


def spans(text,doc_id,remaining,grant):
    paragraphs=[]
    for match in re.finditer(r'\S.*?(?=\r?\n[ \t]*\r?\n|\f|\Z)',text,re.S):
        a,b=match.span();value=text[a:b];words=re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?",value)
        if a<grant['raw_end'] or len(words)<25 or len(words)>1500:continue
        if sum(w.lower() in STOPWORDS for w in words)/len(words)<.07:continue
        if sum(c.isalpha() for c in value)/max(1,len(value))<.55:continue
        if re.search(r'creative commons|copyright|all rights reserved|ISBN|rights and permissions|third-party content|creativecommons|queries on rights|\.\s*\.\s*\.',value,re.I):continue
        if re.search(r'(?im)^\s*(?:references|bibliography|acknowledgments|contents)\s*$',value):continue
        paragraphs.append((a,b))
    rng=random.Random(digest('27183'+doc_id));order=list(range(len(paragraphs)));rng.shuffle(order)
    selected=[]
    for start in order:
        if len(selected)==3:break
        a,b=paragraphs[start]
        if any(a<end and b>begin for begin,end,_,_ in selected):continue
        available=[i for i in range(4) if remaining[i]>0]
        if not available:break
        chosen=rng.choices(available,weights=[remaining[i] for i in available])[0];lo,hi=BINS[chosen]
        wc=len(text[a:b].split());end=start
        while wc<lo and end+1<len(paragraphs):
            na,nb=paragraphs[end+1];gap=text[b:na]
            # A report paragraph may continue over a publisher page break.
            # Retain the *entire* original gap, including any short running
            # header/page number, rather than silently joining edited text.
            page_gap='\f' in gap and len(gap.split())<=12 and not re.search(r'copyright|creativecommons|https?://|references|bibliography',gap,re.I)
            if (gap.strip() and not page_gap) or len(text[a:nb].split())>hi:break
            end+=1;b=nb;wc=len(text[a:b].split())
        if not lo<=wc<=hi or any(a<q and b>p for p,q,_,_ in selected):continue
        selected.append((a,b,wc,chosen));remaining[chosen]-=1
    return selected


def pairs(doc,text,manifest,remaining):
    if not eligible_doc(doc):return
    grant=license_grant(text)
    if not grant:return
    doc_id='worldbank:'+str(doc['id']);raw_hash=digest(text);retrieved=manifest['retrieved_at']
    family=str(doc.get('dois') or doc.get('chronical_docm_id') or doc.get('guid') or doc_id)
    authors=doc.get('authors') or doc.get('authr') or []
    raw={'source_id':'worldbank','source_dataset':'World Bank Documents & Reports','source_revision':manifest['sha256'],
         'source_file':'source-downloads/worldbank/'+str(doc['id'])+'.txt','retrieved_at':retrieved,
         'raw_text_sha256':raw_hash,'record':{'id':str(doc['id']),'text':text,
         'metadata':{'official_document_metadata':doc,'download_manifest':manifest,'license_grant':grant}}}
    for a,b,wc,bin_id in spans(text,doc_id,remaining.copy(),grant):
        value=text[a:b]
        yield {'record_id':digest(doc_id+str(a)+str(b)),'source_id':'worldbank','category':'professional','text':value,
               'word_count':wc,'length_bin':bin_id,'source_dataset':raw['source_dataset'],'source_revision':manifest['sha256'],
               'source_file':raw['source_file'],'original_id':str(doc['id']),'source_url':str(doc.get('url') or manifest['url']),
               'title':doc.get('display_title',''),'author_attribution_json':json.dumps(authors,ensure_ascii=False),
               'license_evidence':grant['url'],'claimed_original_date':doc['docdt'],'retrieved_at':retrieved,
               'raw_text_sha256':raw_hash,'passage_sha256':digest(value),'raw_start':a,'raw_end':b,
               'offset_unit':'unicode_codepoints','extraction_method':'unchanged_official_text_mirror_contiguous_paragraphs',
               'parent_document_id':doc_id,'provisional_family_id':digest(family),'admission_status':'quarantined_candidate',
               'training_eligible':False,'provenance_basis':'official_dated_publication_with_explicit_CC_BY_work_license',
               'protected_overlap_status':'not_fully_audited','reason_codes':['protected_overlap_audit_pending',
               'publisher_text_extraction_and_third_party_content_audit_pending','historical_text_version_unverified']
               +(['publisher_page_break_inside_contiguous_span'] if '\f' in value else []),
               'sampling_seed':27183,'pipeline_version':VERSION},raw


def retrieve(folder,doc):
    url=text_url(doc);path=folder/(str(doc['id'])+'.txt');metadata=path.with_suffix('.json')
    if path.exists() and metadata.exists():
        value=json.loads(metadata.read_text());body=path.read_bytes()
        if value['url']!=url or hashlib.sha256(body).hexdigest()!=value['sha256']:raise ValueError('Saved World Bank original changed')
    else:
        response=requests.get(url,timeout=(15,75));response.raise_for_status()
        if urlparse(response.url).hostname not in HOSTS:raise ValueError('Unexpected text redirect')
        body=response.content
        if len(body)>20_000_000:raise ValueError('Text unusually large; needs dedicated review')
        path.write_bytes(body)
        value={'url':url,'resolved_url':response.url,'sha256':hashlib.sha256(body).hexdigest(),
               'bytes':len(body),'retrieved_at':now(),'official_document_metadata':doc}
        atomic_json(metadata,value)
    # Do not silently replace undecodable characters or corrupt original text.
    text=body.decode('utf-8-sig')
    if '<html' in text[:500].lower():raise ValueError('Text endpoint returned HTML')
    return text,value


def discover(folder):
    documents={};offset=0
    while True:
        params={**PARAMS,'os':offset};path=folder/('index-'+str(offset)+'.json')
        if path.exists():result=json.loads(path.read_text())
        else:
            response=requests.get(API,params=params,timeout=(15,60));response.raise_for_status();result=response.json()
            atomic_json(path,result);atomic_json(path.with_suffix('.request.json'),{'url':response.url,'sha256':digest(json.dumps(result,sort_keys=True)),'retrieved_at':now()})
        for item in result.get('documents',{}).values():
            if eligible_doc(item):documents[str(item['id'])]=item
        offset+=PARAMS['rows']
        if offset>=int(result.get('total',0)):break
    return sorted(documents.values(),key=lambda d:digest('27183'+str(d['id'])))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--quota',type=int,default=776);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();base=args.base;quota=args.quota;folder=base/'source-downloads/worldbank';folder.mkdir(parents=True,exist_ok=True)
    (base/'progress').mkdir(exist_ok=True);began=time.monotonic();reasons=Counter();errors=[]
    with (base/'worldbank.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        db=connect(base/'stage.sqlite3');config=json.dumps({'version':VERSION,'quota':quota},sort_keys=True)
        saved=db.execute("SELECT value FROM settings WHERE key='worldbank_stage'").fetchone()
        if saved and saved[0]!=config:raise ValueError('Stage config changed')
        with db:db.execute("INSERT OR IGNORE INTO settings VALUES ('worldbank_stage',?)",(config,))
        target=apportion(quota,{str(i):v for i,v in enumerate(LENGTH_WEIGHTS['professional'])})
        docs=discover(folder);count=db.execute("SELECT count(*) FROM passages WHERE source='worldbank'").fetchone()[0]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for start in range(0,len(docs),args.workers):
                if count>=quota:break
                batch=[d for d in docs[start:start+args.workers]
                       if not db.execute("SELECT 1 FROM cursors WHERE source='worldbank' AND file=? AND done=1",('license-v2:'+str(d['id']),)).fetchone()
                       and not db.execute("SELECT 1 FROM passages WHERE source='worldbank' AND doc=?",('worldbank:'+str(d['id']),)).fetchone()]
                futures=[pool.submit(retrieve,folder,d) for d in batch]
                for doc,future in zip(batch,futures):
                    if count>=quota:break
                    try:
                        text,manifest=future.result()
                        bins=dict(db.execute("SELECT bin,count(*) FROM passages WHERE source='worldbank' GROUP BY bin"));remaining=[max(0,target[str(i)]-bins.get(i,0)) for i in range(4)]
                        result=list(pairs(doc,text,manifest,remaining))
                        if not result:reasons['no_license_or_eligible_span']+=1
                        db.execute('BEGIN IMMEDIATE')
                        try:
                            for row,raw in result:
                                if add(db,row,raw,quota):count+=1
                            db.execute("INSERT OR REPLACE INTO cursors VALUES ('worldbank',?,1,1)",('license-v2:'+str(doc['id']),));db.commit()
                        except BaseException:db.rollback();raise
                    except Exception as exc:errors.append({'id':str(doc['id']),'error':str(exc)[:180]})
                atomic_json(base/'progress/worldbank.json',{'source_id':'worldbank','state':'staging','count':count,'target':quota,'position':start,
                            'discovered':len(docs),'reasons':dict(reasons),'errors':errors[-5:],'updated_at':now()})
        out=base/'staged/worldbank';out.mkdir(parents=True,exist_ok=True)
        with gzip.open(out/'passages.jsonl.gz','wt') as stream:
            for row, in db.execute("SELECT row FROM passages WHERE source='worldbank' ORDER BY id"):stream.write(row+'\n')
        with gzip.open(out/'documents.jsonl.gz','wt') as stream:
            for raw, in db.execute('SELECT raw FROM documents'):stream.write(gzip.decompress(raw).decode()+'\n')
        status={'source_id':'worldbank','state':'staged_quota_filled' if count==quota else 'staged_shortfall','count':count,'target':quota,
                'length_counts':dict(db.execute("SELECT bin,count(*) FROM passages WHERE source='worldbank' GROUP BY bin")),
                'elapsed_seconds':time.monotonic()-began,'reasons':dict(reasons),'errors':errors,'all_quarantined':True,
                'production_pool_modified':False,'global_dedup_against_production_pending':True,'updated_at':now()}
        db.close();atomic_json(base/'progress/worldbank.json',status);print(json.dumps(status),flush=True)


if __name__=='__main__':main()
