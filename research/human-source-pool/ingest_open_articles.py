"""Pinned ACL-owned 2016–2021 original paper passages, with known local families excluded."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import re
from collect_pool import BINS,LENGTH_WEIGHTS,STOPWORDS,apportion,atomic_json,digest,make_passages,now
from expand_pool import connect,add

REPO='WINGNUS/ACL-OCL'
REVISION='300ee6c5e5629d042bfc07cbc406e2f330b53659'
SHA='7e652ac4c80759723e56d760ad8ee52673359ab33aa77be1613d117d1f3ec7c5'
VERSION='acl-owned-original-v1'

def normalized_title(value):return re.sub(r'[^a-z0-9]+','',str(value or '').casefold())

def eligible(row,exclusions):
    aid=row.get('acl_id') or ''
    if str(row.get('year')) not in {str(y) for y in range(2016,2022)}:return 'outside_years'
    if row.get('publisher')!='Association for Computational Linguistics':return 'third_party_publisher'
    # Restrict to major ACL-owned proceedings, not third-party anthology holdings.
    if not (re.fullmatch(r'[PDNE](?:16|17|18|19)-[1-9]\d{3}',aid) or re.fullmatch(r'202[01]\.(?:acl|emnlp|naacl|eacl)-(?:main|long|short|demos|srw)\.[1-9]\d*',aid)):
        return 'outside_owned_proceedings'
    if aid in exclusions['ids'] or normalized_title(row.get('title')) in exclusions['titles']:return 'protected_family'
    if row.get('language') not in (None,'','English','en'):return 'language'
    return None

def pairs(row,index,manifest,remaining):
    text=row.get('full_text') or ''
    words=re.findall(r'[A-Za-z]+',text)
    if len(words)<50 or sum(w.lower() in STOPWORDS for w in words)/len(words)<.06:return
    aid=row['acl_id'];doc_id='acl:'+aid
    # Retain full extracted source, but avoid selecting its references and acknowledgments.
    stop=re.search(r'(?im)^\s*(?:references|bibliography|acknowledg(?:e)?ments)\s*$',text)
    body=text[:stop.start()] if stop else text
    spans=make_passages(body,doc_id,3,remaining.copy())
    if not spans:
        tokens=list(re.finditer(r'\S+',body));cursor=0
        for bin_id in sorted(range(4),key=lambda i:-remaining[i]):
            if remaining[bin_id]<=0 or len(spans)>=3:continue
            lo,hi=BINS[bin_id];size=min((lo+hi)//2,len(tokens)-cursor)
            if size<lo:continue
            a,b=tokens[cursor].start(),tokens[cursor+size-1].end();cursor+=size
            spans.append((a,b,size,bin_id))
    raw={'source_id':'acl','source_dataset':REPO,'source_revision':REVISION,'source_file':manifest['source_file'],
         'source_row':index,'retrieved_at':manifest['retrieved_at'],'raw_text_sha256':digest(text),
         'record':{'id':aid,'text':text,'metadata':{k:v for k,v in row.items() if k!='full_text'}}}
    for a,b,wc,bin_id in spans:
        passage=text[a:b]
        record={'record_id':digest(doc_id+':'+str(a)+':'+str(b)),'source_id':'acl','category':'scientific','text':passage,
          'word_count':wc,'length_bin':bin_id,'source_dataset':REPO,'source_revision':REVISION,
          'source_file':manifest['source_file'],'source_row':index,'original_id':aid,'source_url':'https://aclanthology.org/'+aid+'/',
          'title':row.get('title') or '', 'author_attribution_json':json.dumps(row.get('author') or ''),
          'license_evidence':manifest['license_evidence'],'claimed_original_date':str(row['year']),
          'date_evidence_basis':'pinned_anthology_bibliography_publication_year','retrieved_at':manifest['retrieved_at'],
          'raw_text_sha256':digest(text),'passage_sha256':digest(passage),'raw_start':a,'raw_end':b,'offset_unit':'unicode_codepoints',
          'extraction_method':'unchanged_span_of_original_paper_grobid_extraction','parent_document_id':doc_id,
          'provisional_family_id':digest(doc_id),'doi':row.get('doi') or '', 'original_split':'unsplit_original_corpus',
          'admission_status':'quarantined_candidate','training_eligible':False,'provenance_basis':'ACL_owned_pre2022_original_paper',
          'protected_overlap_status':'known_local_manifest_families_excluded_comprehensive_audit_pending',
          'reason_codes':['mirror_corpus_noncommercial_restriction','near_duplicate_and_protected_overlap_audit_pending','grobid_extraction_review_pending'],
          'sampling_seed':27183,'pipeline_version':VERSION}
        yield record,raw

def main():
    import pyarrow.parquet as pq
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--source',choices=['acl'],default='acl');a=p.parse_args();b=a.base
    d=b/'source-downloads/acl';m=json.loads((d/'manifest.json').read_text())
    if m['repo_id']!=REPO or m['revision']!=REVISION or m['sha256']!=SHA:raise RuntimeError('Unexpected ACL source identity')
    path=d/m['source_file'];h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    if h.hexdigest()!=SHA:raise RuntimeError('ACL source checksum mismatch')
    excluded=json.loads((b/'pipeline/acl-protected-families.json').read_text())
    exclusions={'ids':set(excluded['acl_ids']),'titles':set(excluded['normalized_titles'])}
    plan=json.loads((b/'pipeline/sampling-plan.json').read_text())
    quota=next(s['planned_passages'] for s in plan['source_quotas'] if s['source_id']=='acl')
    targets=apportion(quota,{str(i):n for i,n in enumerate(LENGTH_WEIGHTS['scientific'])})
    (b/'progress').mkdir(exist_ok=True)
    with (b/'acl.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);db=connect(b/'collection.sqlite3')
        count=lambda:db.execute("SELECT count(*) FROM passages WHERE source='acl'").fetchone()[0]
        old=db.execute("SELECT position FROM cursors WHERE source='acl' AND file=?",(VERSION,)).fetchone()
        last=old[0] if old else -1;reasons=Counter();index=-1
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                index+=1
                if index<=last:continue
                if count()>=quota:break
                reason=eligible(row,exclusions)
                if reason:reasons[reason]+=1;continue
                if db.execute("SELECT 1 FROM passages WHERE source='acl' AND doc=?",('acl:'+row['acl_id'],)).fetchone():continue
                bins=dict(db.execute("SELECT bin,count(*) FROM passages WHERE source='acl' GROUP BY bin"))
                remaining=[max(0,targets[str(i)]-bins.get(i,0)) for i in range(4)]
                prepared=list(pairs(row,index,m,remaining))
                db.execute('BEGIN IMMEDIATE')
                try:
                    for r,raw in prepared:
                        if add(db,r,raw,quota):reasons['accepted']+=1
                    db.execute('INSERT OR REPLACE INTO cursors VALUES (?,?,?,?)',('acl',VERSION,index,0));db.commit()
                except BaseException:db.rollback();raise
                if index%100==0:atomic_json(b/'progress/acl.json',{'source_id':'acl','count':count(),'target':quota,'row':index,'state':'collecting','updated_at':now(),'reasons':dict(reasons)})
            if count()>=quota:break
        n=count();db.close();result={'source_id':'acl','count':n,'target':quota,'row':index,'state':'quota_filled' if n==quota else 'archive_exhausted','updated_at':now(),'all_quarantined':True,'reasons':dict(reasons)}
        atomic_json(b/'progress/acl.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
