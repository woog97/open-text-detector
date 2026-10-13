"""Bounded, resumable remote EFF archive expansion; isolated quarantined stage only."""
import sys,os,json,time,re,hashlib,base64,sqlite3,gzip,fcntl
from pathlib import Path
from collections import Counter
from email.utils import parsedate_to_datetime
from datetime import datetime,timezone
import requests
from bs4 import BeautifulSoup
sys.path.insert(0,'/tmp/pangram-human-active-20261002/pipeline')
from ingest_eff import extract,normalized_url,Excluded,LICENSE
from collect_pool import digest,make_passages,atomic_json,now
from expand_pool import connect,add
BASE=Path('/tmp/pangram-human-active-20261002/scaling-survey/eff')
VERSION='eff-wayback-bounded-20261002-v1'
def main():
 BASE.mkdir(parents=True,exist_ok=True)
 lock=(BASE/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 db=connect(BASE/'collection.sqlite3')
 prod=sqlite3.connect('file:/tmp/pangram-human-active-20261002/collection.sqlite3?mode=ro',uri=True)
 used={r[0] for r in prod.execute("SELECT DISTINCT doc FROM passages WHERE source='eff'")};norms={r[0] for r in prod.execute('SELECT norm FROM passages')};prod.close()
 reasons=Counter();errors=0;attempts=0;verified=0;start=time.monotonic()
 status={'source':'eff','state':'discovering','updated_at':now(),'existing_production_count':415,'bounded_max_captures':1200,'originals_preserved':True,'training_admitted':0}
 def save(**kw):
  status.update(kw);status.update(updated_at=now(),attempted_captures=attempts,verified_captures=verified,rejections=dict(reasons),new_candidates=db.execute('SELECT count(*) FROM passages').fetchone()[0],elapsed_seconds=time.monotonic()-start);atomic_json(BASE/'progress.json',status)
 save()
 inventory={}
 seed=Path('/tmp/pangram-eff-staging-20261002/access-refresh-20261002/wayback2020.response')
 queries=[('2020-12',None)]+[(f'{y}-{m:02}',{'url':f'www.eff.org/deeplinks/{y}/{m:02}/*','output':'json','filter':['statuscode:200','mimetype:text/html'],'to':'20211231','collapse':'urlkey','fl':'timestamp,original,mimetype,statuscode,digest','limit':'1000'}) for y in (2021,2020,2019) for m in range(12,0,-1) if (y,m)!=(2020,12)]
 for key,params in queries:
  f=BASE/('cdx-'+key+'.json')
  try:
   if not f.exists():
    if params is None: f.write_bytes(seed.read_bytes())
    else:
     time.sleep(3);resp=requests.get('https://web.archive.org/cdx/search/cdx',params=params,timeout=(15,40));resp.raise_for_status();rows=resp.json();f.write_text(json.dumps(rows))
   rows=json.loads(f.read_text())
   for values in rows[1:]:
    r=dict(zip(rows[0],values));u=normalized_url(r['original'])
    if re.fullmatch(r'https://www.eff.org/deeplinks/20(?:0\d|1\d|2[01])/\d\d/[^/]+',u) and r['timestamp']<'20220101' and u not in used:inventory.setdefault(u,r)
   errors=0
  except Exception as e:reasons['discovery_'+type(e).__name__]+=1;errors+=1
  save(discovery_queries_finished=queries.index((key,params))+1,unique_uncollected_urls=len(inventory))
  # Process a bounded discovery wave before fetching more index months.
  if len(inventory)>=1200 or errors>=3:break
 atomic_json(BASE/'inventory.json',{'rows':list(inventory.values()),'production_documents_excluded':len(used)})
 errors=0
 for u,index in sorted(inventory.items(),key=lambda x:digest(x[0]))[:1200]:
  dest=BASE/'captures'/digest(u);dest.mkdir(parents=True,exist_ok=True)
  if (dest/'receipt.json').exists():continue
  attempts+=1
  try:
   if time.monotonic()-start>4*3600:save(state='bounded_runtime_complete');break
   time.sleep(3)
   r=requests.get('https://web.archive.org/web/'+index['timestamp']+'id_/'+index['original'],timeout=(15,45))
   (dest/'capture.html').write_bytes(r.content);r.raise_for_status()
   assert base64.b32encode(hashlib.sha1(r.content).digest()).decode().rstrip('=')==index['digest'],'cdx_payload_digest_mismatch'
   md=parsedate_to_datetime(r.headers['Memento-Datetime']);assert md.year<2022,'memento_post_cutoff'
   assert md.strftime('%Y%m%d%H%M%S')==index['timestamp'],'memento_timestamp_mismatch'
   soup=BeautifulSoup(r.content,'html.parser');licenses=[a['href'] for a in soup.select('a[href]') if 'creativecommons.org/licenses/by/4.0' in a['href']]
   if not licenses:raise Excluded('historical_ccby4_link_missing')
   doc=extract(r.content,{'url':index['original']},{'WARC-Date':md.isoformat()})
   if doc['url']!=u:raise Excluded('canonical_capture_url_mismatch')
   if not all('/about/staff/' in a['url'] for a in doc['authors']):raise Excluded('staff_byline_not_verified')
   bodyregion=soup.select_one('article.node--blog--full .field--name-body')
   if re.search(r'(?i)(all rights reserved|used (?:with|by) permission|copyright\s+\d)',bodyregion.get_text(' ',strip=True)):raise Excluded('article_specific_rights_exception')
   text=doc['text'];raw={'source_id':'eff','source_dataset':'Internet Archive Wayback/EFF Deeplinks','source_revision':index['digest'],'source_file':str(dest/'capture.html'),'retrieved_at':now(),'raw_text_sha256':digest(text),'record':{'id':u,'text':text,'metadata':{'archive_index':index,'memento_datetime':r.headers['Memento-Datetime'],'capture_sha256':hashlib.sha256(r.content).hexdigest(),'source_html':r.content.decode('utf-8'),'authors':doc['authors'],'published':doc['published'],'extraction_version':VERSION,'license_links':licenses}}}
   added=0
   with db:
    for a,b,w,bi in make_passages(text,u,3,[10000]*4):
     passage=text[a:b]
     if digest(' '.join(passage.casefold().split())) in norms:continue
     row={'record_id':digest('eff'+u+str(a)+str(b)),'source_id':'eff','category':'general_web','text':passage,'word_count':w,'length_bin':bi,'source_dataset':raw['source_dataset'],'source_revision':index['digest'],'source_file':raw['source_file'],'source_row':0,'original_id':u,'source_url':u,'title':doc['title'],'author_attribution_json':json.dumps(doc['authors']),'source_byline':doc['byline_text'],'license_evidence':LICENSE,'claimed_original_date':doc['published'],'archive_capture_date':md.isoformat(),'date_evidence_basis':'pre2022_Wayback_CDX_SHA1_and_exact_Memento_timestamp_verified','retrieved_at':raw['retrieved_at'],'raw_text_sha256':digest(text),'passage_sha256':digest(passage),'raw_start':a,'raw_end':b,'offset_unit':'unicode_codepoints','extraction_method':'unchanged_contiguous_span_of_faithfully_decoded_archived_HTML_paragraphs','parent_document_id':u,'provisional_family_id':digest(u),'admission_status':'quarantined_candidate','training_eligible':False,'provenance_basis':'official_EFF_archived_pre2022_original_Deeplinks_with_staff_byline','protected_overlap_status':'not_fully_audited','reason_codes':['remaining_rights_and_inline_quotation_audit_pending','author_family_grouping_pending','genre_and_extraction_review_pending','protected_overlap_audit_pending'],'sampling_seed':27183,'pipeline_version':VERSION}
     assert text[a:b]==row['text'];added+=add(db,row,raw,10000)
   verified+=1;errors=0
   atomic_json(dest/'receipt.json',{'state':'verified_original_capture','index':index,'sha256':hashlib.sha256(r.content).hexdigest(),'memento_datetime':r.headers['Memento-Datetime'],'new_candidates':added})
  except (Excluded,AssertionError) as e:
   reasons[str(e)]+=1;errors=0;atomic_json(dest/'receipt.json',{'state':'excluded','reason':str(e),'index':index})
  except Exception as e:
   reasons[type(e).__name__]+=1;errors+=1;atomic_json(dest/'error.json',{'error':type(e).__name__,'at':now()})
  save(state='acquiring')
  if errors>=3:save(state='backoff_after_three_errors');break
 else:save(state='bounded_inventory_complete')
 db.execute('PRAGMA wal_checkpoint(FULL)');assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
 save(database_integrity='ok',source_capacity_assured=False,global_merge_pending=True)
if __name__=='__main__':main()
