"""Stage diverse original 2018 Amazon reviews from bounded full-file gzip prefixes."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import fcntl
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time

import requests
from collect_pool import BINS,LENGTH_WEIGHTS,STOPWORDS,apportion,atomic_json,digest,now
from expand_pool import add,connect

VERSION='amazon2018-original-prefix-v1'
ORIGIN='https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/index.html'
ROOT='https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/'
CATEGORIES=['Books','Electronics','Home_and_Kitchen','Clothing_Shoes_and_Jewelry','All_Beauty',
            'Sports_and_Outdoors','Toys_and_Games','Office_Products','Pet_Supplies','Movies_and_TV']


class PrefixLimit(Exception):pass


class CapturingReader:
    def __init__(self,source,dest,limit):self.source=source;self.dest=dest;self.limit=limit;self.count=0;self.sha=hashlib.sha256()
    def read(self,n=-1):
        if self.count>=self.limit:raise PrefixLimit('compressed prefix bound reached')
        n=min(n if n>=0 else 65536,self.limit-self.count)
        value=self.source.read(n);self.dest.write(value);self.sha.update(value);self.count+=len(value);return value


def make_pair(row,category,index,line,offset,archive):
    text=row.get('reviewText');user=str(row.get('reviewerID') or '');product=str(row.get('asin') or '')
    if category not in CATEGORIES or not isinstance(text,str) or not user or not product:return None
    try:
        rating=float(row.get('overall'));date=datetime.fromtimestamp(int(row['unixReviewTime']),timezone.utc).date().isoformat()
    except (ValueError,KeyError,TypeError,OverflowError,OSError):return None
    if rating not in (1,2,3,4,5) or not '1996-01-01'<=date<='2018-10-31':return None
    wc=len(text.split());words=re.findall(r'[A-Za-z]+',text)
    if not 50<=wc<=1500 or len(words)<35:return None
    if sum(w.lower() in STOPWORDS for w in words)/len(words)<.06:return None
    if sum(c.isalpha() for c in text)/max(1,len(text))<.55 or re.search(r'<(?:html|div|style|script)\b',text,re.I):return None
    bin_id=next(i for i,(lo,hi) in enumerate(BINS) if lo<=wc<=hi)
    review_id=digest(user+'|'+product+'|'+str(row['unixReviewTime'])+'|'+text)
    doc_id='amazon2018:'+review_id;raw_hash=digest(text);revision=archive['etag'] or archive['last_modified'] or archive['url']
    raw={'source_id':'amazon2018','source_dataset':'UCSD Amazon Review Data 2018','source_revision':revision,
         'source_file':'categoryFiles/'+category+'.json.gz','source_row':index,'retrieved_at':archive['retrieved_at'],
         'raw_text_sha256':raw_hash,'record':{'id':review_id,'text':text,'metadata':{'original_review':row,
         'original_json_line':line.decode('utf-8'),'original_line_sha256':hashlib.sha256(line).hexdigest(),
         'uncompressed_line_byte_offset':offset,'product_category':category,'archive':archive}}}
    record={'record_id':digest(doc_id),'source_id':'amazon2018','category':'reviews','text':text,'word_count':wc,
            'length_bin':bin_id,'source_dataset':raw['source_dataset'],'source_revision':revision,'source_file':raw['source_file'],
            'source_row':index,'original_id':review_id,'source_url':archive['url'],'title':str(row.get('summary') or ''),
            'author_attribution_json':json.dumps([user]),'license_evidence':'Original 2018 research release; underlying review rights unresolved: '+ORIGIN,
            'claimed_original_date':date,'retrieved_at':archive['retrieved_at'],'raw_text_sha256':raw_hash,'passage_sha256':raw_hash,
            'raw_start':0,'raw_end':len(text),'offset_unit':'unicode_codepoints','extraction_method':'unchanged_complete_customer_review',
            'parent_document_id':doc_id,'provisional_family_id':digest('amazon-product:'+product),
            'reviewer_family_id':digest('amazon-reviewer:'+user),'product_family_id':digest('amazon-product:'+product),
            'product_id':product,'reviewer_id':user,'product_category':category,'rating':int(rating),'verified_purchase':bool(row.get('verified')),
            'admission_status':'quarantined_candidate','training_eligible':False,'commercial_training_cleared':False,
            'provenance_basis':'publisher_original_2018_full_category_review_release','protected_overlap_status':'not_fully_audited',
            'reason_codes':['underlying_review_rights_unresolved','fake_incentivized_and_templated_review_audit_pending',
                            'protected_overlap_audit_pending','reviewer_product_grouping_required','archive_prefix_sampling_bias'],
            'sampling_seed':27183,'pipeline_version':VERSION}
    return record,raw


class Gate:
    def __init__(self,db,quota):
        self.lock=threading.RLock();self.quota=quota;self.minimum=min(250,quota//len(CATEGORIES));self.maximum=math.ceil(quota*.15)
        self.rating_minimum=min(20,self.minimum//5);self.targets=apportion(quota,{str(i):v for i,v in enumerate(LENGTH_WEIGHTS['reviews'])})
        self.count=0;self.bins=Counter();self.categories=Counter();self.ratings=Counter();self.products=Counter();self.users=set()
        for encoded, in db.execute("SELECT row FROM passages WHERE source='amazon2018'"):
            self.remember(json.loads(encoded))
    def remember(self,row):
        self.count+=1;self.bins[row['length_bin']]+=1;self.categories[row['product_category']]+=1
        self.ratings[(row['product_category'],row['rating'])]+=1;self.products[row['product_id']]+=1;self.users.add(row['reviewer_id'])
    def accept(self,db,pair,phase):
        row,raw=pair;c=row['product_category'];rating=row['rating']
        with self.lock:
            category_cap=self.minimum if phase==1 else self.maximum
            reserved=sum(max(0,self.minimum-self.categories[other]) for other in CATEGORIES if other!=c)
            rating_reserved=sum(max(0,self.rating_minimum-self.ratings[(c,other)]) for other in range(1,6) if other!=rating)
            effective_cap=self.minimum if self.categories[c]<self.minimum else category_cap
            if (self.count>=self.quota-reserved or self.categories[c]>=category_cap
                or self.categories[c]>=effective_cap-rating_reserved
                or self.ratings[(c,rating)]>=math.ceil(self.maximum*.6)
                or self.bins[row['length_bin']]>=self.targets[str(row['length_bin'])]
                or self.products[row['product_id']]>=3 or row['reviewer_id'] in self.users):return False
            db.execute('BEGIN IMMEDIATE')
            try:
                inserted=add(db,row,raw,self.quota);db.commit()
            except BaseException:db.rollback();raise
            if inserted:self.remember(row)
            return inserted
    def status(self):
        with self.lock:return {'count':self.count,'length_counts':dict(self.bins),'product_category_counts':dict(self.categories),
            'rating_counts':{c:{str(r):self.ratings[(c,r)] for r in range(1,6)} for c in CATEGORIES},
            'products':len(self.products),'reviewers':len(self.users),'max_reviews_per_product':max(self.products.values(),default=0)}


def scan_category(base,category,phase,gate,max_bytes,max_lines):
    with gate.lock:
        if gate.count>=gate.quota or gate.categories[category]>=(gate.minimum if phase==1 else gate.maximum):
            return {'category':category,'phase':phase,'rows_scanned':0,'state':'already_at_cap'}
    db=connect(base/'stage.sqlite3');folder=base/'source-downloads/amazon2018';url=ROOT+category+'.json.gz'
    run=int(time.time()*1000);name=category+'-phase'+str(phase)+'-'+str(run)+'.prefix.gz';path=folder/name
    response=None;capture=None;complete=False;scanned=0;offset=0;state='bounded_prefix_complete'
    try:
        response=requests.get(url,stream=True,timeout=(20,90));response.raise_for_status()
        archive={'url':url,'etag':response.headers.get('ETag',''),'last_modified':response.headers.get('Last-Modified',''),
                 'retrieved_at':now(),'prefix_file':'source-downloads/amazon2018/'+name,'gzip_complete':False,
                 'original_release':'2018','selection':'full category review archive, not five-core'}
        with path.open('wb') as out:
            capture=CapturingReader(response.raw,out,max_bytes)
            with gzip.GzipFile(fileobj=capture) as stream:
                for index in range(max_lines):
                    with gate.lock:
                        cap=gate.minimum if phase==1 else gate.maximum
                        if gate.count>=gate.quota or gate.categories[category]>=cap:state='quota_or_category_cap';break
                    line=stream.readline()
                    if not line:complete=True;state='archive_exhausted';break
                    scanned+=1
                    try:row=json.loads(line)
                    except (ValueError,UnicodeDecodeError):offset+=len(line);continue
                    pair=make_pair(row,category,index,line,offset,archive);offset+=len(line)
                    if pair:gate.accept(db,pair,phase)
                    if scanned%2000==0:
                        atomic_json(base/'progress'/('amazon-'+category+'.json'),{'source_id':'amazon2018','product_category':category,
                          'phase':phase,'rows_scanned':scanned,'compressed_bytes':capture.count,'state':'scanning','updated_at':now()})
    except PrefixLimit:state='compressed_prefix_limit'
    except Exception as exc:state='error:'+type(exc).__name__+':'+str(exc)[:180]
    finally:
        if response is not None:response.close()
        db.close()
        if capture:
            manifest={**archive,'gzip_complete':complete,'compressed_prefix_bytes':capture.count,'compressed_prefix_sha256':capture.sha.hexdigest(),
                      'rows_scanned':scanned,'uncompressed_bytes_scanned':offset,'state':state}
            atomic_json(path.with_suffix('.manifest.json'),manifest)
        atomic_json(base/'progress'/('amazon-'+category+'.json'),{'source_id':'amazon2018','product_category':category,'phase':phase,
          'rows_scanned':scanned,'state':state,'updated_at':now()})
    return {'category':category,'phase':phase,'rows_scanned':scanned,'state':state}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--quota',type=int,default=5306)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--max-prefix-mb',type=int,default=128);p.add_argument('--max-lines',type=int,default=400000)
    args=p.parse_args();base=args.base;(base/'source-downloads/amazon2018').mkdir(parents=True,exist_ok=True);(base/'progress').mkdir(exist_ok=True)
    began=time.monotonic()
    with (base/'amazon2018.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        db=connect(base/'stage.sqlite3');config=json.dumps({'version':VERSION,'quota':args.quota,'categories':CATEGORIES},sort_keys=True)
        prior=db.execute("SELECT value FROM settings WHERE key='amazon2018_stage'").fetchone()
        if prior and prior[0]!=config:raise ValueError('Stage config changed')
        with db:db.execute("INSERT OR IGNORE INTO settings VALUES ('amazon2018_stage',?)",(config,))
        gate=Gate(db,args.quota);db.close();results=[]
        for phase in (1,2):
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                jobs=[pool.submit(scan_category,base,c,phase,gate,args.max_prefix_mb*1024*1024,args.max_lines) for c in CATEGORIES]
                for job in jobs:
                    results.append(job.result());atomic_json(base/'progress/amazon2018.json',{'state':'staging','source_id':'amazon2018','target':args.quota,**gate.status(),'updated_at':now()})
        db=connect(base/'stage.sqlite3');out=base/'staged/amazon2018';out.mkdir(parents=True,exist_ok=True)
        with gzip.open(out/'passages.jsonl.gz','wt') as stream:
            for row, in db.execute("SELECT row FROM passages WHERE source='amazon2018' ORDER BY id"):stream.write(row+'\n')
        with gzip.open(out/'documents.jsonl.gz','wt') as stream:
            for raw, in db.execute('SELECT raw FROM documents'):stream.write(gzip.decompress(raw).decode()+'\n')
        db.close();status={'source_id':'amazon2018','state':'staged_quota_filled' if gate.count==args.quota else 'staged_bounded_shortfall',
            'target':args.quota,**gate.status(),'scans':results,'elapsed_seconds':time.monotonic()-began,'updated_at':now(),
            'all_quarantined':True,'production_pool_modified':False,'global_dedup_against_production_pending':True}
        atomic_json(base/'progress/amazon2018.json',status);print(json.dumps(status),flush=True)


if __name__=='__main__':main()
