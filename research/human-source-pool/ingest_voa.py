"""MOT v1.0 VOA originals: pre-2022 retrieval, explicit VOA byline, no wire mentions."""
import argparse
from collections import Counter
from datetime import datetime
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import tarfile
import time
from urllib.parse import urlsplit, urlunsplit
from collect_pool import BINS, LENGTH_WEIGHTS, apportion, atomic_json, digest, make_passages, now
from expand_pool import add, connect

VERSION='voa-mot-v1-originals-2'
ARCHIVE_SHA256='a6f96fbdae475a6d3d340dd7c32dc1eab2ed6735fb429de0f8add7c097f077d1'
WIRE=re.compile(r'\b(?:Reuters|Associated Press|Agence France.Presse|AFP|AP)\b',re.I)
BYLINES={'voa','voa news','voice of america'}
class Excluded(ValueError):pass

def historical_date(value,required=True):
    if not value and not required:return ''
    if not isinstance(value,str):raise Excluded('missing_date')
    try:d=datetime.fromisoformat(value.replace('Z','+00:00'))
    except ValueError:raise Excluded('invalid_date')
    if d.year<2000 or d.year>=2022:raise Excluded('not_pre2022_date')
    return value

def extract(d,staff=None):
    staff=staff or {}
    if d.get('content_type')!='article' or d.get('site_language')!='eng' or d.get('predicted_language')!='eng':raise Excluded('not_english_article')
    capture=historical_date(d.get('time_retrieved'))
    published=historical_date(d.get('time_published'),False);historical_date(d.get('time_modified'),False)
    p=urlsplit(d.get('url',''))
    if p.hostname not in ('voanews.com','www.voanews.com') or not p.path or p.path=='/':raise Excluded('not_main_voa_article_url')
    authors=d.get('authors')
    if not isinstance(authors,list) or not authors or any(not isinstance(a,str) or a.strip().lower() not in BYLINES and a.strip().lower() not in staff for a in authors):raise Excluded('not_verified_voa_only_byline')
    paragraphs=d.get('paragraphs')
    if not isinstance(paragraphs,list) or not paragraphs or any(not isinstance(v,str) for v in paragraphs):raise Excluded('invalid_paragraphs')
    text='\n\n'.join(paragraphs)
    # Conservative full-document filter also excludes mixed VOA/wire reports and
    # mere wire mentions; no external report is relabeled as a VOA original.
    if WIRE.search(text+' '+str(d.get('title',''))):raise Excluded('wire_mention_or_credit')
    if re.search(r'(?i)(originally (?:published|appeared)|republished (?:with|from)|reprinted (?:with|from))',text):raise Excluded('external_republication_notice')
    if len(text.split())<50:raise Excluded('too_short')
    return dict(text=text,url=urlunsplit(('https','www.voanews.com',p.path.rstrip('/'),'','')),authors=authors,capture=capture,published=published,byline_evidence=[staff[a.strip().lower()] for a in authors if a.strip().lower() in staff])

def pairs(d,doc,member,remaining):
    text=doc['text'];url=doc['url'];retrieved=now()
    raw={'source_id':'voa','source_dataset':'bltlab/mot','source_revision':'v1.0:'+ARCHIVE_SHA256,'source_file':'eng_voanews.tgz:'+member,
         'retrieved_at':retrieved,'raw_text_sha256':digest(text),'record':{'id':url,'text':text,'metadata':{'original_mot_record':d,'archive_sha256':ARCHIVE_SHA256,'member':member,'extraction_version':VERSION,'staff_byline_evidence':doc['byline_evidence']}}}
    words=len(text.split())
    full_bin=next((i for i,(lo,hi) in enumerate(BINS) if lo<=words<=hi and remaining[i]>0),None)
    spans=[(0,len(text),words,full_bin)] if full_bin is not None else make_passages(text,url,3,remaining.copy())
    for a,b,w,bin_id in spans:
        passage=text[a:b]
        yield {'record_id':digest('voa'+url+str(a)+str(b)),'source_id':'voa','category':'news','text':passage,'word_count':w,'length_bin':bin_id,
          'source_dataset':'bltlab/mot','source_revision':'v1.0:'+ARCHIVE_SHA256,'source_file':'eng_voanews.tgz:'+member,'source_row':0,
          'original_id':url,'source_url':url,'title':d.get('title',''),'source_byline':'; '.join(doc['authors']),
          'author_attribution_json':json.dumps(doc['authors'],ensure_ascii=False),'topic_section':d.get('section') or '',
          'license_evidence':'https://www.voanews.com/p/5338.html','collection_license':'CC-BY-4.0','original_rights_basis':'VOA-produced original material public domain; explicit VOA or verified staff byline; all wire mentions excluded','staff_profile_evidence_json':json.dumps(doc['byline_evidence'],ensure_ascii=False),
          'claimed_original_date':doc['published'],'archive_capture_date':doc['capture'],'date_evidence_basis':'MOT_original_crawler_time_retrieved_pre2022_in_pinned_release',
          'retrieved_at':retrieved,'raw_text_sha256':digest(text),'passage_sha256':digest(passage),'raw_start':a,'raw_end':b,'offset_unit':'unicode_codepoints',
          'extraction_method':'unchanged_span_of_MOT_paragraphs_joined_with_double_newlines','parent_document_id':url,'provisional_family_id':digest(url),
          'admission_status':'quarantined_candidate','training_eligible':False,'provenance_basis':'VOA_explicit_byline_pre2022_crawl_original_MOT_release',
          'protected_overlap_status':'not_fully_audited','reason_codes':['protected_overlap_and_LR_Sum_split_audit_pending','quoted_material_rights_and_near_duplicate_audit_pending','MOT_extraction_fidelity_audit_pending'],
          'sampling_seed':27183,'pipeline_version':VERSION},raw

def collect(base,quota):
    started=time.monotonic();archive=base/'eng_voanews.tgz';(base/'progress').mkdir(exist_ok=True)
    h=hashlib.sha256()
    with archive.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    if h.hexdigest()!=ARCHIVE_SHA256:raise ValueError('pinned_archive_sha256_mismatch')
    staff={}
    if (base/'staff-byline-evidence.json').exists():
        for item in json.loads((base/'staff-byline-evidence.json').read_text()):
            if not item.get('verified'):continue
            assert urlsplit(item['url']).hostname=='www.voanews.com' and '/author/' in urlsplit(item['url']).path
            assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
            assert item['name']==item['h1'] and 'Reporter bio' in item['title']
            staff[item['name'].strip().lower()]=item
    with (base/'voa.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);db=connect(base/'collection.sqlite3');rejected=Counter();eligible=[];scanned=0
        # Hash-order sampling spans archive order, dates and news sections.
        cache=base/'named-articles.jsonl.gz';cache_manifest=base/'named-articles-manifest.json'
        def records():
            if cache.exists() and cache_manifest.exists():
                meta=json.loads(cache_manifest.read_text())
                assert meta['source_archive_sha256']==ARCHIVE_SHA256
                ch=hashlib.sha256()
                with cache.open('rb') as f:
                    for chunk in iter(lambda:f.read(1024*1024),b''):ch.update(chunk)
                assert ch.hexdigest()==meta['cache_sha256'],'cache_hash_mismatch'
                with gzip.open(cache,'rt',encoding='utf8') as f:
                    for line in f:
                        item=json.loads(line);yield item['member'],item['record']
            else:
                with tarfile.open(archive,'r|gz') as tf:
                    for member in tf:
                        if member.isfile() and '/article/' in member.name and member.name.endswith('.json'):
                            yield member.name,json.load(tf.extractfile(member))
        for member,d in records():
            scanned+=1
            try:doc=extract(d,staff)
            except Excluded as exc:rejected[str(exc)]+=1;continue
            eligible.append((digest(VERSION+doc['url']),member,d,doc))
        eligible.sort(key=lambda x:x[0]);initial=db.execute("SELECT count(*) FROM passages WHERE source='voa'").fetchone()[0];n=initial
        targets=apportion(quota,{str(i):v for i,v in enumerate(LENGTH_WEIGHTS['news'])})
        for _,member,d,doc in eligible:
            if n>=quota:break
            if db.execute("SELECT 1 FROM passages WHERE source='voa' AND doc=?",(doc['url'],)).fetchone():continue
            bins=dict(db.execute("SELECT bin,count(*) FROM passages WHERE source='voa' GROUP BY bin"));remaining=[max(0,targets[str(i)]-bins.get(i,0)) for i in range(4)]
            result=list(pairs(d,doc,member,remaining))
            if not result:rejected['no_matching_length_passages']+=1;continue
            with db:
                for row,raw in result:
                    if add(db,row,raw,quota):n+=1
        status={'source_id':'voa','count':n,'target':quota,'scanned_articles':scanned,'eligible_original_documents':len(eligible),'rejections':dict(rejected),'state':'quota_filled' if n==quota else 'current_release_subset_exhausted','new_candidates':n-initial,'updated_at':now(),'all_quarantined':True,'global_merge_pending':True}
        atomic_json(base/'progress/voa.json',status);package=base/'accepted-pairs.jsonl.gz';verified=0;sections=Counter();parents=set()
        with gzip.open(package,'wt',encoding='utf8') as out:
            for (saved,) in db.execute("SELECT row FROM passages WHERE source='voa' ORDER BY id"):
                row=json.loads(saved);raw=json.loads(gzip.decompress(db.execute('SELECT raw FROM documents WHERE hash=?',(row['raw_text_sha256'],)).fetchone()[0]));text=raw['record']['text']
                assert digest(text)==row['raw_text_sha256'] and text[row['raw_start']:row['raw_end']]==row['text'] and digest(row['text'])==row['passage_sha256']
                extract(raw['record']['metadata']['original_mot_record'],staff);out.write(json.dumps({'row':row,'raw':raw},ensure_ascii=False)+'\n');verified+=1;parents.add(row['parent_document_id']);sections[row['topic_section']]+=1
        integrity=db.execute('PRAGMA integrity_check').fetchone()[0];assert integrity=='ok';bins=dict(db.execute("SELECT bin,count(*) FROM passages WHERE source='voa' GROUP BY bin"));db.close()
        status.update(length_bin_counts=bins,verified_named_staff_count=len(staff),used_named_article_cache=cache.exists() and cache_manifest.exists(),validated_rows=verified,parent_documents=len(parents),topic_section_counts=dict(sections),integrity_check=integrity,package=str(package),package_bytes=package.stat().st_size,package_sha256=hashlib.sha256(package.read_bytes()).hexdigest(),elapsed_seconds=time.monotonic()-started)
        atomic_json(base/'package-manifest.json',status);print(json.dumps(status),flush=True);return status

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--quota',type=int,default=1652);a=p.parse_args()
    if not str(a.base).startswith('/tmp/'):p.error('Use isolated Space /tmp staging only')
    collect(a.base,a.quota)
