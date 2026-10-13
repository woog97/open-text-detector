"""Recover original English SCP tale captures from the pinned filtered CCCC corpus."""
import argparse,base64,fcntl,gzip,hashlib,json,re,sqlite3,time
from collections import Counter
from datetime import datetime
from pathlib import Path
from collect_pool import BINS,LENGTH_WEIGHTS,apportion,atomic_json,digest,now
from expand_pool import add,connect
from ingest_scp import select_spans,REVISION as INDEX_REVISION
from scan_scp_cccc import slug,REVISION,INDEX_SHA
VERSION='scp-cccc-historical-tales-v1'
FORBIDDEN=re.compile(r'(?im)^\s*(?:special containment procedures|object class|containment class|disruption class|risk class|item\s*#|character sheet|game mechanics|ability scores?|hit points|stat block)\s*[:：]')
FOOTER=re.compile(r'(?im)^\s*(?:unless otherwise stated,? the content|page revision:|powered by wikidot|footnotes\s*$|comments\s*$|discussion\s*$|licensing / citation\s*$)')
class Excluded(ValueError):pass

def date(value):
 if not isinstance(value,str):raise Excluded('missing_historical_date')
 try:d=datetime.fromisoformat(value.replace('Z','+00:00'))
 except ValueError:raise Excluded('unparseable_historical_date')
 if not 2000<=d.year<2022:raise Excluded('not_pre2022')
 return value[:19]
def eligible(w,index):
 row=w['record'];meta=row.get('metadata') or {};key=slug(meta.get('warc_url') or meta.get('url') or '')
 if not key or key!=w['slug'] or key not in index:raise Excluded('not_exact_known_tale_url')
 page=index[key]
 if page.get('domain')!='scp-wiki.wikidot.com' or 'tale' not in page.get('tags',[]):raise Excluded('not_english_tale')
 if set(page.get('tags',[]))&{'rpg','game','goi-format','_template'}:raise Excluded('non_narrative_current_tag')
 if not page.get('page_id') or not page.get('creator') or page['creator'].lower() in ('deleted','unknown'):raise Excluded('missing_identity_or_creator')
 captured=date(meta.get('warc_date'));created=date(row.get('created'));page_created=date(page.get('created_at'))
 if page_created>captured or created>captured:raise Excluded('creation_after_capture')
 source=re.fullmatch(r'cccc_CC-MAIN-(\d{4})-\d{2}',row.get('source',''))
 if not source or int(source[1])>=2022:raise Excluded('unverified_crawl_snapshot')
 text=row.get('text')
 if not isinstance(text,str) or len(text.split())<50:raise Excluded('not_text_narrative')
 if FORBIDDEN.search(text):raise Excluded('containment_or_rpg_format')
 if meta.get('content_type')!='text/html':raise Excluded('not_html_capture')
 raw_line=base64.b64decode(w['upstream_json_line_base64'],validate=True)
 if hashlib.sha256(raw_line).hexdigest()!=w['upstream_json_line_sha256'] or json.loads(raw_line)!=row:raise Excluded('upstream_original_mismatch')
 return page,captured

def pairs(w,index,remaining):
 page,captured=eligible(w,index);row=w['record'];text=row['text'];doc='scp:'+str(page['page_id']);end=FOOTER.search(text);span_text=text[:end.start()] if end else text
 credits=[{'name':page['creator'],'role':'creator_from_current_pinned_metadata_not_verified_archived_byline'}]
 credits.extend({'name':a,'role':'pre_capture_revision_contributor_recorded_in_current_metadata'} for a in sorted({h['author'] for h in page.get('history',[]) if h.get('author') and h['author'] not in ('deleted',page['creator']) and str(h.get('date','9999'))[:19]<=captured}))
 raw={'source_id':'scp','source_dataset':'common-pile/cccc_filtered','source_revision':REVISION,'source_file':w['source_file'],'source_row':w['source_row'],'retrieved_at':now(),'raw_text_sha256':digest(text),'record':{'id':row.get('id'),'text':text,'metadata':{'original_cccc_wrapper':w,'current_scp_page_metadata':page,'scp_index_revision':INDEX_REVISION,'scp_index_sha256':INDEX_SHA,'attribution_basis':'current_pinned_metadata_association_not_independently_verified_historical_credit','publisher_license_audit_sha256':'6e374b44eec11d647b3517be7b2aaf41d9076fc6dfd85ca003c519339ddde487'}}}
 for a,b,words,bin_id in select_spans(span_text,doc,remaining,capacity=3):
  passage=text[a:b]
  yield {'record_id':digest(VERSION+doc+digest(text)+str(a)+str(b)),'source_id':'scp','category':'creative','text':passage,'word_count':words,'length_bin':bin_id,
    'source_dataset':'common-pile/cccc_filtered','source_revision':REVISION,'source_file':w['source_file'],'source_row':w['source_row'],'original_id':str(page['page_id']),
    'source_url':'https://scp-wiki.wikidot.com/'+w['slug'],'archived_source_url':row['metadata']['warc_url'],'title':page.get('title',''),
    'author_attribution_json':json.dumps(credits,ensure_ascii=False),'author_attribution_basis':'current_pinned_page_metadata_association_not_verified_archived_byline',
    'license_evidence':'https://creativecommons.org/licenses/by-sa/3.0/','license_provenance':'official_SCP_licensing_guide_and_publisher_exact_host_audit; version_not_in_upstream_row',
    'claimed_original_date':page['created_at'],'archive_capture_date':captured,'date_evidence_basis':'pinned_CCCC_pre2022_snapshot_and_warc_capture_dates; current_page_history_not_used_as_capture_text',
    'retrieved_at':raw['retrieved_at'],'raw_text_sha256':digest(text),'passage_sha256':digest(passage),'raw_start':a,'raw_end':b,'offset_unit':'unicode_codepoints',
    'extraction_method':'unchanged_contiguous_narrative_span_of_original_archived_CCCC_text','parent_document_id':doc,'provisional_family_id':digest(doc),'canon_hub_ids_json':json.dumps(page.get('hubs',[])),
    'admission_status':'quarantined_candidate','training_eligible':False,'provenance_basis':'historical_official_SCP_domain_capture_matched_to_pinned_English_tale_page_metadata',
    'protected_overlap_status':'not_fully_audited','reason_codes':['current_creator_metadata_association_requires_historical_credit_audit','archived_extraction_and_genre_review_pending','canon_and_author_family_grouping_pending','protected_overlap_and_near_duplicate_audit_pending'],
    'sampling_seed':27183,'pipeline_version':VERSION},raw

def collect(base,old_stage,quota):
 start=time.monotonic();index_bytes=(base/'tales-index.json').read_bytes();assert hashlib.sha256(index_bytes).hexdigest()==INDEX_SHA;index=json.loads(index_bytes)
 assert json.loads((base/'scan-progress.json').read_text())['state']=='bounded_cached_inventory_scanned'
 (base/'progress').mkdir(exist_ok=True)
 with (base/'scp.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);db=connect(base/'collection.sqlite3')
  old_package=old_stage/'accepted-pairs.jsonl.gz';assert hashlib.sha256(old_package.read_bytes()).hexdigest()=='a032fc4949a8702672d0c1736d5b537968d3fc2c08a83550d0d433b671e5385a'
  with gzip.open(old_package,'rt') as stream,db:
   for line in stream:
    item=json.loads(line);add(db,item['row'],item['raw'],quota)
  initial=db.execute("SELECT count(*) FROM passages WHERE source='scp'").fetchone()[0];n=initial;targets=apportion(quota,{str(i):v for i,v in enumerate(LENGTH_WEIGHTS['creative'])});reasons=Counter();inventory=[];scanned=0
  for receipt_path in sorted((base/'scan').glob('*.receipt.json')):
   receipt=json.loads(receipt_path.read_text());file=Path(receipt['output'])
   with file.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['output_sha256']
   with gzip.open(file,'rt') as stream:
    for line in stream:
     w=json.loads(line);assert w['archive_sha256']==receipt['archive_sha256'] and w['source_revision']==REVISION;inventory.append(w)
  inventory.sort(key=lambda w:(str(w['record'].get('metadata',{}).get('warc_date','9999')),digest(w['slug'])))
  for w in inventory:
   scanned+=1
   if n>=quota:break
   try:page,_=eligible(w,index)
   except Excluded as e:reasons[str(e)]+=1;continue
   doc='scp:'+str(page['page_id'])
   if db.execute("SELECT 1 FROM passages WHERE source='scp' AND doc=?",(doc,)).fetchone():reasons['already_selected_parent']+=1;continue
   bins=dict(db.execute("SELECT bin,count(*) FROM passages WHERE source='scp' GROUP BY bin"));remaining=[max(0,targets[str(i)]-bins.get(i,0)) for i in range(4)]
   results=list(pairs(w,index,remaining))
   if not results:reasons['no_qualifying_narrative_spans']+=1;continue
   with db:
    for row,raw in results:
     if add(db,row,raw,quota):n+=1
  package=base/'accepted-pairs-new.jsonl.gz';verified=0;parents=set()
  with gzip.open(package,'wt',encoding='utf8') as out:
   for (saved,) in db.execute("SELECT row FROM passages WHERE source='scp' ORDER BY id"):
    row=json.loads(saved)
    if row.get('pipeline_version')!=VERSION:continue
    raw=json.loads(gzip.decompress(db.execute('SELECT raw FROM documents WHERE hash=?',(row['raw_text_sha256'],)).fetchone()[0]));text=raw['record']['text'];assert digest(text)==row['raw_text_sha256'] and digest(row['text'])==row['passage_sha256'] and text[row['raw_start']:row['raw_end']]==row['text'];eligible(raw['record']['metadata']['original_cccc_wrapper'],index)
    out.write(json.dumps({'row':row,'raw':raw},ensure_ascii=False)+'\n');verified+=1;parents.add(row['parent_document_id'])
  assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok';bins=dict(db.execute("SELECT bin,count(*) FROM passages WHERE source='scp' GROUP BY bin"));db.close()
  status={'source_id':'scp','total_with_existing_106':n,'quota':quota,'new_package_candidates':verified,'new_parent_documents':len(parents),'shortfall':quota-n,'capture_rows_considered':scanned,'bounded_inventory_rows':len(inventory),'rejections':dict(reasons),'length_bins_including_baseline':bins,'all_quarantined':True,'global_merge_pending':True,'package':str(package),'package_sha256':hashlib.sha256(package.read_bytes()).hexdigest(),'package_bytes':package.stat().st_size,'elapsed_seconds':time.monotonic()-start,'state':'quota_filled' if n==quota else 'bounded_cached_historical_inventory_exhausted'}
  atomic_json(base/'package-manifest.json',status);atomic_json(base/'progress/scp.json',status);print(json.dumps(status),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--old-stage',type=Path,required=True);p.add_argument('--quota',type=int,default=4444);a=p.parse_args();collect(a.base,a.old_stage,a.quota)
