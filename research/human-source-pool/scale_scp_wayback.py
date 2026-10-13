"""Independent remote SCP historical-capture extraction and bounded acquisition."""
import os
os.environ['HF_HOME']='/tmp/pangram-finish-hf-cache'
os.environ['HF_XET_CACHE']='/tmp/pangram-finish-hf-cache/xet'
import base64, collections, gzip, hashlib, json, shutil, sqlite3, sys, time
from pathlib import Path
from email.utils import parsedate_to_datetime
from bs4 import BeautifulSoup
from collect_pool import atomic_json,digest,now
from expand_pool import add,connect
from ingest_scp import select_spans
from ingest_scp_cccc import FORBIDDEN
BASE=Path('/tmp/pangram-scp-scale-20261002')
OLD=Path('/tmp/pangram-scp-wayback-20261002')
PRODUCTION=Path('/tmp/pangram-human-active-20261002/collection.sqlite3')
VERSION='scp-wayback-narrative-paragraphs-v1'

def extract(html,receipt):
 assert hashlib.sha256(html).hexdigest()==receipt['sha256']
 assert base64.b32encode(hashlib.sha1(html).digest()).decode().rstrip('=')==receipt['cdx']['digest']
 assert receipt['cdx']['timestamp']<'20220101' and parsedate_to_datetime(receipt['memento_datetime']).year<2022
 page=receipt['current_metadata']
 assert page['domain']=='scp-wiki.wikidot.com' and 'tale' in page['tags']
 assert not set(page['tags'])&{'rpg','game','goi-format','_template'}
 assert page.get('creator') and page['creator'].lower() not in ('deleted','unknown') and page.get('page_id')
 assert str(page['created_at'])[:19].replace('-','').replace(':','').replace('T','')<=receipt['cdx']['timestamp']
 soup=BeautifulSoup(html,'html.parser')
 assert any('creativecommons.org/licenses/by-sa/3.0' in a.get('href','') for a in soup.select('a[href]'))
 region=soup.select_one('#page-content');assert region is not None
 for node in region.select('script,style,table,nav,footer,.licensebox,.licensebox2,.footnotes-footer,.footnote-footer,.page-rate-widget-box,.creditRate,.credit-rate,.info-container,.image-caption,.scp-image-caption,.collapsible-block-link'):
  node.decompose()
 # Preserve paragraph order; no joining across removed content. Only paragraph prose is eligible.
 paragraphs=[p.get_text(' ',strip=True) for p in region.find_all('p') if not p.find_parent('p')]
 text='\n\n'.join(paragraphs)
 assert not FORBIDDEN.search(text)
 return text

def collect():
 baseline=sqlite3.connect('file:'+str(PRODUCTION)+'?mode=ro',uri=True)
 used={x[0] for x in baseline.execute("SELECT DISTINCT doc FROM passages WHERE source='scp'")}
 norms={x[0] for x in baseline.execute('SELECT norm FROM passages')};baseline.close()
 db=connect(BASE/'collection.sqlite3');reasons=collections.Counter();captures=0
 for path in sorted(BASE.glob('*/receipt.json')):
  r=json.loads(path.read_text())
  if r['state']!='verified_original_capture':reasons[r['state']]+=1;continue
  captures+=1;page=r['current_metadata'];doc='scp:'+str(page['page_id'])
  if doc in used or db.execute('SELECT 1 FROM passages WHERE doc=?',(doc,)).fetchone():reasons['existing_parent']+=1;continue
  try:text=extract((path.parent/'capture.html').read_bytes(),r)
  except (AssertionError,KeyError,ValueError):reasons['capture_or_genre_validation_rejected']+=1;continue
  raw={'source_id':'scp','source_dataset':'Internet Archive historical official SCP captures','source_revision':r['cdx']['timestamp'],'source_file':str(path.parent/'capture.html'),'retrieved_at':now(),'raw_text_sha256':digest(text),'record':{'id':str(page['page_id']),'text':text,'metadata':{'receipt':r,'extraction_method':VERSION,'original_html_base64':base64.b64encode((path.parent/'capture.html').read_bytes()).decode()}}}
  spans=select_spans(text,doc,[44440]*4,capacity=3)
  if not spans:reasons['no_qualifying_paragraph_spans']+=1
  with db:
   for a,b,words,bin_id in spans:
    passage=text[a:b]
    if digest(' '.join(passage.casefold().split())) in norms:reasons['baseline_exact_duplicate']+=1;continue
    row={'record_id':digest(VERSION+doc+digest(text)+str(a)+str(b)),'source_id':'scp','category':'creative','text':passage,'word_count':words,'length_bin':bin_id,'source_dataset':raw['source_dataset'],'source_revision':raw['source_revision'],'source_file':raw['source_file'],'source_row':0,'original_id':str(page['page_id']),'source_url':'https://scp-wiki.wikidot.com/'+r['slug'],'archived_source_url':'https://web.archive.org/web/'+r['cdx']['timestamp']+'id_/'+r['cdx']['original'],'title':page.get('title',''),'author_attribution_json':json.dumps([{'name':page['creator'],'role':'creator_from_current_pinned_metadata_not_verified_archived_byline'}]),'author_attribution_basis':'current_pinned_page_metadata_association_not_verified_archived_byline','license_evidence':'https://creativecommons.org/licenses/by-sa/3.0/','license_provenance':'link_in_digest_verified_archived_HTML','claimed_original_date':page['created_at'],'archive_capture_date':r['memento_datetime'],'date_evidence_basis':'CDX_digest_and_Memento_pre2022_capture','retrieved_at':raw['retrieved_at'],'raw_text_sha256':digest(text),'passage_sha256':digest(passage),'raw_start':a,'raw_end':b,'offset_unit':'unicode_codepoints','extraction_method':'unchanged_contiguous_span_of_deterministic_HTML_paragraph_extraction','parent_document_id':doc,'provisional_family_id':digest(doc),'canon_hub_ids_json':json.dumps(page.get('hubs',[])),'admission_status':'quarantined_candidate','training_eligible':False,'provenance_basis':'digest_verified_historical_official_SCP_capture','protected_overlap_status':'not_fully_audited','reason_codes':['current_creator_metadata_requires_historical_credit_audit','historical_HTML_extraction_and_genre_review_pending','canon_author_family_and_protected_near_duplicate_audit_pending'],'sampling_seed':27183,'pipeline_version':VERSION}
    add(db,row,raw,44440)
 count=db.execute('SELECT count(*) FROM passages').fetchone()[0];bins=dict(db.execute('SELECT bin,count(*) FROM passages GROUP BY bin'))
 package=BASE/'accepted-pairs.jsonl.gz'
 with gzip.open(package,'wt') as out:
  for saved, in db.execute('SELECT row FROM passages ORDER BY id'):
   row=json.loads(saved);raw=json.loads(gzip.decompress(db.execute('SELECT raw FROM documents WHERE hash=?',(row['raw_text_sha256'],)).fetchone()[0]));m=raw['record']['metadata'];text=extract(base64.b64decode(m['original_html_base64']),m['receipt']);assert text==raw['record']['text'] and digest(text)==row['raw_text_sha256'] and text[row['raw_start']:row['raw_end']]==row['text'] and digest(row['text'])==row['passage_sha256'];out.write(json.dumps({'row':row,'raw':raw})+'\n')
 assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok';db.close()
 result={'source_id':'scp','updated_at':now(),'verified_captures':captures,'new_candidates':count,'length_bins':bins,'rejections':dict(reasons),'all_quarantined':True,'global_merge_pending':True,'baseline_parent_and_exact_duplicate_excluded':True,'per_parent_cap':3,'package_sha256':hashlib.sha256(package.read_bytes()).hexdigest(),'package':str(package)}
 atomic_json(BASE/'extraction-status.json',result);print(json.dumps(result),flush=True)
 return result

def persist():
 from huggingface_hub import HfApi,get_token
 api=HfApi(token=os.environ.get('HF_TOKEN') or get_token());bucket='open-text-detector/training-storage';prefix='workspace/human-source-mix-v2-recovered-20261002/scaling-survey/scp/'
 paths=[BASE/'accepted-pairs.jsonl.gz',BASE/'extraction-status.json',BASE/'inventory.json',BASE/'worker-status.json']
 api.batch_bucket_files(bucket,add=[(p,prefix+p.name) for p in paths if p.exists()])
 objects=list(api.get_bucket_paths_info(bucket,[prefix+'accepted-pairs.jsonl.gz']));assert objects[0].size==(BASE/'accepted-pairs.jsonl.gz').stat().st_size
 atomic_json(BASE/'durability.json',{'updated_at':now(),'bucket':bucket,'prefix':prefix,'bytes':objects[0].size,'verification':'SDK upload and committed size; full GET checksum pending'})

def main():
 import acquire_scp_wayback as acquire
 BASE.mkdir(exist_ok=True)
 for path in OLD.iterdir():
  if path.is_dir() and not (BASE/path.name).exists():shutil.copytree(path,BASE/path.name)
 acquire.OUT=BASE
 atomic_json(BASE/'worker-status.json',{'state':'extracting_existing_verified_captures','updated_at':now()})
 collect()
 for cycle in range(3):
  atomic_json(BASE/'worker-status.json',{'state':'bounded_acquisition','cycle':cycle+1,'max_cycles':3,'updated_at':now()})
  acquire.main();collect()
  try:persist()
  except Exception as e:atomic_json(BASE/'durability-error.json',{'type':type(e).__name__,'updated_at':now()})
  state=json.loads((BASE/'progress.json').read_text())['state']
  if state=='bounded_inventory_complete':break
  time.sleep(60*(2**cycle))
 atomic_json(BASE/'worker-status.json',{'state':'bounded_worker_complete','acquisition_state':state,'updated_at':now(),'further_expansion_requires_inventory_and_access_review':True})
 try:persist()
 except Exception as e:atomic_json(BASE/'durability-error.json',{'type':type(e).__name__,'updated_at':now()})
if __name__=='__main__':main()
