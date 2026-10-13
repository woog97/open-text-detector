"""Isolated CCCC personal-blog intake from the publisher's audited filtered source.

No inferred CC license string is injected into upstream metadata. Every accepted
record retains the exact UTF-8 input line and the pinned audit that justified
candidate collection. This does not certify training admission.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl,gzip,hashlib,json,os,random,re,sqlite3,sys,time
from pathlib import Path
from urllib.parse import urlparse,urlunparse,unquote
import requests
from collect_pool import BINS,LENGTH_WEIGHTS,apportion,atomic_json,digest,make_passages,now,STOPWORDS
from expand_pool import add,connect

REPO='common-pile/cccc_filtered'
REVISION='03a3de5713a0bb23267d26724346508af0f25327'
VERSION='cccc-audited-personal-blogs-v1'
AUDIT_SHA='6e374b44eec11d647b3517be7b2aaf41d9076fc6dfd85ca003c519339ddde487'
FORBIDDEN_PATH=re.compile(r'/(?:tag|tags|category|categories|archive|archives|search|feed|comments|page|news|wiki|questions|answers|reviews?|journals?|papers?)(?:/|$)',re.I)
# These two explicitly dated article routes use an "archives" directory.
ARCHIVE_ARTICLE_HOSTS={'windley.com','malvasiabianca.org','paraesthesia.com'}
REVIEW_HEAD=re.compile(r'\b(?:book|film|movie|product|restaurant|hotel|game)\s+review\b|\babstract\s*[:\n]|\bspecial containment procedures\b',re.I)
BOILER=re.compile(r'creative commons|all rights reserved|leave a (?:reply|comment)|subscribe|share this|posted by|posted on|privacy policy|related posts|comments? (?:on|off)|^\s*\d+\s+comments?\s*$',re.I)
COMMENT_CUTOFF=re.compile(r'(?im)^\s*(?:comments|\d+ comments|leave a reply|leave a comment|post a comment)\s*[:!]?\s*$')
DATE_PATH=re.compile(r'/(19\d{2}|20[01]\d|202[01])/(0?[1-9]|1[0-2])(?:/(0?[1-9]|[12]\d|3[01]))?(?=/|[._-])')


def sha_file(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()


def host(url):
 try:
  p=urlparse(url);port=p.port
 except ValueError:return None
 if p.scheme not in ('http','https') or p.username or p.password or port not in (None,80,443):return None
 return p.hostname.lower() if p.hostname else None


def load_evidence(base):
 e=base/'evidence';mapping=json.loads((e/'cccc-domain-map.json').read_text());audit=e/'urls_to_keep.txt'
 assert sha_file(audit)==AUDIT_SHA==mapping['audit_list_sha256']
 hosts=set()
 for line in audit.read_text().splitlines():
  if line.strip():hosts.update([line.strip(),line.strip()[4:] if line.startswith('www.') else 'www.'+line.strip()])
 selected=set(mapping['canonical_hosts']);assert selected<={h.removeprefix('www.') for h in hosts}
 manifest=json.loads((e/'manifest.json').read_text())
 for item in manifest['files']:assert sha_file(e/item['file'])==item['sha256']
 return mapping,hosts,manifest


def route(row,mapping,audited):
 meta=row.get('metadata') or {};url=meta.get('warc_url') or '';h=host(url)
 if h not in audited:return 'not_publisher_audited_host',None
 canonical=h.removeprefix('www.')
 if canonical not in mapping['canonical_hosts']:return 'other_source_or_genre',None
 if meta.get('content_type')!='text/html':return 'not_html',None
 dates=[]
 for field in [meta.get('warc_date'),row.get('created')]:
  try:date=datetime.fromisoformat(str(field).replace('Z','+00:00'))
  except (TypeError,ValueError):return 'missing_capture_date',None
  if not 1990<=date.year<2022:return 'post2021_capture',None
  dates.append(date)
 source_match=re.fullmatch(r'cccc_CC-MAIN-(\d{4})-\d{2}',str(row.get('source','')))
 if not source_match or int(source_match[1])>=2022:return 'post2021_or_unknown_snapshot',None
 parsed=urlparse(url);path=unquote(parsed.path)
 if parsed.query:return 'query_page',None
 required=mapping.get('required_path_prefixes',{}).get(canonical)
 if required and not any(path.startswith(prefix) for prefix in required):return 'outside_reviewed_blog_route',None
 route_path=path
 if canonical in ARCHIVE_ARTICLE_HOSTS:route_path=re.sub(r'^/(?:archive|archives)/','/',path)
 if FORBIDDEN_PATH.search(route_path):return 'other_genre_or_listing_path',None
 match=DATE_PATH.search(route_path)
 # Date must be followed by a real post identifier, not a monthly index.
 if not match or not route_path[match.end():].strip('/'):return 'not_dated_individual_post',None
 year,month,day=[int(v) if v else None for v in match.groups()]
 try:post_date=datetime(year,month,day or 1).date()
 except ValueError:return 'invalid_post_date',None
 if post_date>min(d.date() for d in dates):return 'postdate_after_capture',None
 text=row.get('text')
 if not isinstance(text,str) or not 50<=len(text.split())<=100000:return 'invalid_text_length',None
 if REVIEW_HEAD.search(text[:800]) or re.search(r'/(?:reviews?|review-of)[/-]',path,re.I):return 'review_or_scientific_genre',None
 normalized=urlunparse(('https',canonical,parsed.path.rstrip('/'),'','',''))
 return None,{'host':canonical,'url':url,'canonical_url':normalized,'post_date':post_date.isoformat() if day else f'{year:04d}-{month:02d}','post_date_granularity':'day' if day else 'month','capture_date':meta['warc_date']}



def long_sentence_span(text):
 """Retain a sentence-bounded span inside one oversized prose paragraph."""
 for para in re.finditer(r'\S[^\n]*(?:\n(?!\s*\n)[^\n]*)*',text):
  value=para.group()
  if len(value.split())<=1500:continue
  boundaries=[0]+[m.end() for m in re.finditer(r'[.!?][\"”’\x27)]*\s+',value)]+[len(value)]
  for start_index,a in enumerate(boundaries[:-1]):
   for b in boundaries[start_index+1:]:
    span=value[a:b];words=span.split();n=len(words)
    if n>1500:break
    if n<1000:continue
    alpha=re.findall(r'[A-Za-z]+',span)
    if BOILER.search(span) or len(re.findall(r'https?://',span))>3 or span.count('|')>5:continue
    if '<html' in span.lower() or '<div' in span.lower():continue
    if not alpha or sum(w.lower() in STOPWORDS for w in alpha)/len(alpha)<.06:continue
    if sum(c.isalpha() for c in span)/len(span)<.5:continue
    return para.start()+a,para.start()+b,n,3
 return None


def make_pair(row,line,source_file,source_row,archive_sha,mapping,audited,remaining):
 reason,info=route(row,mapping,audited)
 if reason:return reason,None
 text=row['text'];cut=COMMENT_CUTOFF.search(text);body=text[:cut.start()] if cut else text
 doc_id='cccc:'+info['canonical_url']
 # Candidate passages are unchanged spans, strictly before an explicit comment section.
 selected=None
 available=[i for i,n in enumerate(remaining) if n>0]
 if not available:return 'length_bins_full',None
 first=random.Random(digest(doc_id)).choices(available,weights=[remaining[i] for i in available])[0]
 order=[first]+[i for i in sorted(available,key=lambda i:(-remaining[i],i)) if i!=first]
 for chosen_bin in order:
  candidates=make_passages(body,doc_id,8,[remaining[i] if i==chosen_bin else 0 for i in range(4)])
  for a,b,w,bin_id in candidates:
   if BOILER.search(text[a:b]):continue
   selected=(a,b,w,bin_id);break
  if selected:break
 fallback=False
 if selected is None and remaining[3]>0 and mapping.get('oversized_paragraph_sentence_spans'):
  selected=long_sentence_span(body);fallback=selected is not None
 if selected is None:return 'no_clean_prose_span',None
 a,b,w,bin_id=selected;value=text[a:b];audit_url='https://raw.githubusercontent.com/r-three/common-pile/'+mapping['github_audit_revision']+'/sources/cccc/scripts/urls_to_keep.txt'
 raw={'source_id':'cccc','source_dataset':REPO,'source_revision':REVISION,'source_file':source_file,'source_row':source_row,
      'archive_sha256':archive_sha,'retrieved_at':now(),'raw_text_sha256':digest(text),'record':row,
      'upstream_json_line_utf8':line.decode('utf-8'),'upstream_json_line_sha256':hashlib.sha256(line).hexdigest(),
      'license_evidence':{'basis':'publisher_manually_audited_text_domain','audit_url':audit_url,'audit_list_sha256':AUDIT_SHA,'host':info['host'],'metadata_license_present':'license' in row['metadata'],'specific_license_version':None}}
 result={'record_id':digest(doc_id+str(a)+str(b)),'source_id':'cccc','category':'general_web','text':value,'word_count':w,'length_bin':bin_id,
         'source_dataset':REPO,'source_revision':REVISION,'source_file':source_file,'source_row':source_row,'original_id':str(row['id']),
         'source_url':info['url'],'title':'','author_attribution_json':json.dumps({'publisher_host':info['host'],'individual_author':'not_supplied_by_upstream'}),
         'license_evidence':'Publisher domain text-license audit: '+audit_url+' ; exact license/version absent from upstream metadata',
         'license_evidence_basis':'publisher_domain_audit_not_document_license_metadata','license_version_status':'not_supplied_unverified',
         'claimed_original_date':info['post_date'],'date_evidence_basis':'dated_individual_post_url_'+info['post_date_granularity'],
         'historical_capture_date':info['capture_date'],'retrieved_at':raw['retrieved_at'],'raw_text_sha256':digest(text),'passage_sha256':digest(value),
         'raw_start':a,'raw_end':b,'offset_unit':'unicode_codepoints','extraction_method':'unchanged_sentence_span_in_oversized_paragraph_before_comments' if fallback else 'unchanged_paragraph_span_before_explicit_comments',
         'parent_document_id':doc_id,'provisional_family_id':digest(info['canonical_url']),'publisher_domain':info['host'],
         'admission_status':'quarantined_candidate','training_eligible':False,'provenance_basis':'pre2022_WARC_capture_and_publisher_text_domain_audit',
         'protected_overlap_status':'not_fully_audited','reason_codes':['exact_per_document_license_and_version_not_supplied','independent_historical_domain_license_audit_pending',
          'genre_and_extraction_review_pending','author_and_third_party_quotation_review_pending','protected_overlap_audit_pending','author_family_grouping_pending'],
         'sampling_seed':27183,'pipeline_version':VERSION}
 return None,(result,raw)


def _download_once(base,filename):
 folder=base/'source-downloads/cccc';folder.mkdir(parents=True,exist_ok=True);target=folder/filename;manifest=target.with_suffix(target.suffix+'.manifest.json')
 if target.exists() and manifest.exists():
  m=json.loads(manifest.read_text());assert m['revision']==REVISION and sha_file(target)==m['sha256'];return target,m
 url=f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{filename}';part=target.with_name(target.name+'.part-'+str(time.time_ns()));h=hashlib.sha256();size=0
 with requests.get(url,stream=True,timeout=(15,90)) as response:
  response.raise_for_status()
  with part.open('xb') as out:
   for chunk in response.iter_content(4*1024*1024):out.write(chunk);h.update(chunk);size+=len(chunk)
 if target.exists():raise RuntimeError('Unmanifested original requires reconciliation: '+str(target))
 part.rename(target);m={'repo':REPO,'revision':REVISION,'file':filename,'url':url,'bytes':size,'sha256':h.hexdigest(),'retrieved_at':now()};atomic_json(manifest,m)
 return target,m


def download(base,filename):
 for attempt in range(4):
  try:return _download_once(base,filename)
  except (requests.RequestException,TimeoutError) as exc:
   (base/'progress').mkdir(exist_ok=True)
   atomic_json(base/'progress'/('download-'+filename+'.json'),{'file':filename,'attempt':attempt+1,'error_type':type(exc).__name__,'state':'retrying' if attempt<3 else 'failed','updated_at':now()})
   if attempt==3:raise
   time.sleep(2**attempt)

def export_stage(base,db):
 folder=base/'staged/cccc';folder.mkdir(parents=True,exist_ok=True);hashes=set()
 with gzip.open(folder/'passages.jsonl.gz','wt') as out:
  for saved, in db.execute("SELECT row FROM passages WHERE source='cccc' ORDER BY id"):
   out.write(saved+'\n');hashes.add(json.loads(saved)['raw_text_sha256'])
 with gzip.open(folder/'documents.jsonl.gz','wt') as out:
  for h in sorted(hashes):out.write(gzip.decompress(db.execute('SELECT raw FROM documents WHERE hash=?',(h,)).fetchone()[0]).decode()+'\n')
 aux=base/'auxiliary';aux.mkdir(exist_ok=True)
 with gzip.open(aux/'scidev-candidates.jsonl.gz','wb') as out:
  for raw, in db.execute('SELECT raw FROM auxiliary_scidev ORDER BY id'):out.write(gzip.decompress(raw)+b'\n')
 return {'files':{str(p.relative_to(base)):{'sha256':sha_file(p),'bytes':p.stat().st_size} for p in [folder/'passages.jsonl.gz',folder/'documents.jsonl.gz',aux/'scidev-candidates.jsonl.gz']}}


def collect(base,quota=5335,max_shards=None,token=None,new_domains=None,cache_only=False,cursor_namespace="cccc"):
 if new_domains:
  assert cursor_namespace!="cccc" and cache_only, "Refinement must use preserved archives and separate cursors"
 mapping,audited,evidence=load_evidence(base);assert mapping['per_site_cap']==410
 (base/'progress').mkdir(exist_ok=True)
 with (base/'cccc.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);db=connect(base/'stage.sqlite3')
  db.executescript('CREATE TABLE IF NOT EXISTS cccc_selection(id TEXT PRIMARY KEY,host TEXT,url TEXT UNIQUE); CREATE TABLE IF NOT EXISTS auxiliary_scidev(id TEXT PRIMARY KEY,raw BLOB);')
  config={'version':VERSION,'quota':quota,'revision':REVISION,'domain_map_sha256':sha_file(base/'evidence/cccc-domain-map.json'),'audit_sha256':AUDIT_SHA}
  if new_domains:config.update(new_domains=sorted(new_domains),cache_only=True,cursor_namespace=cursor_namespace)
  old=db.execute("SELECT value FROM settings WHERE key='cccc_config'").fetchone()
  assert old is None or json.loads(old[0])==config,'Explicit migration required for changed plan'
  with db:db.execute("INSERT OR IGNORE INTO settings VALUES ('cccc_config',?)",(json.dumps(config,sort_keys=True),))
  listing=json.loads((base/'filtered-repo-info.json').read_text());assert listing['sha']==REVISION;files=sorted(x['rfilename'] for x in listing['siblings'] if x['rfilename'].endswith('.json.gz'))
  random.Random(27183).shuffle(files)
  if max_shards is not None:files=files[:max_shards]
  targets=apportion(quota,{str(i):v for i,v in enumerate(LENGTH_WEIGHTS['general_web'])});counts=dict(db.execute("SELECT bin,count(*) FROM passages WHERE source='cccc' GROUP BY bin"));remaining=[targets[str(i)]-counts.get(i,0) for i in range(4)];n=sum(counts.values());domains=Counter(dict(db.execute('SELECT host,count(*) FROM cccc_selection GROUP BY host')));seen_urls={r[0] for r in db.execute('SELECT url FROM cccc_selection')};reasons=Counter();total_scanned=0;began=time.monotonic()
  pending=[f for f in files if not (db.execute("SELECT done FROM cursors WHERE source=? AND file=?",(cursor_namespace,f)).fetchone() or (0,))[0]]
  def get_file(filename):
   if cache_only:
    path=base/"source-downloads/cccc"/filename;mp=path.with_suffix(path.suffix+".manifest.json");assert path.exists() and mp.exists(), "Refinement cannot download uncached archives"
   return download(base,filename)
  def checkpoint():
   package=export_stage(base,db);atomic_json(base/'package-manifest.json',{'source_id':'cccc','count':n,'target':quota,'domain_counts':dict(domains),'all_quarantined':True,**package,'updated_at':now()})
   if not token:return
   from huggingface_hub import HfApi
   from huggingface_hub.utils import disable_progress_bars
   disable_progress_bars();api=HfApi(token=token);key='workspace/human-source-mix-v2-recovered-20261002/stages/cccc/';snapshot=base/'checkpoints'/('stage-'+str(time.time_ns())+'.sqlite3');snapshot.parent.mkdir(exist_ok=True)
   out=sqlite3.connect(snapshot);db.backup(out);out.execute('PRAGMA journal_mode=DELETE');out.close();assert sqlite3.connect(snapshot).execute('PRAGMA quick_check').fetchone()==('ok',)
   selected=[base/'package-manifest.json',*list((base/'staged/cccc').glob('*.gz')),*list((base/'auxiliary').glob('*.gz')),*list((base/'evidence').glob('*')),*list((base/'pipeline').glob('*.py')),*list((base/'progress').glob('*.json')),*list((base/'source-downloads/cccc').glob('*.manifest.json'))]
   api.batch_bucket_files('open-text-detector/training-storage',add=[(p,key+str(p.relative_to(base))) for p in selected]+[(snapshot,key+'checkpoints/'+snapshot.name)])
   receipt={'count':n,'database':str(snapshot),'database_sha256':sha_file(snapshot),'bucket_key':key+'checkpoints/'+snapshot.name,'updated_at':now(),'verification':'SDK validated uploads; raw accepted input lines retained in DB/documents exports'};atomic_json(base/'durable-checkpoint.json',receipt);api.batch_bucket_files('open-text-detector/training-storage',add=[(base/'durable-checkpoint.json',key+'durable-checkpoint.json')])
  with ThreadPoolExecutor(max_workers=2) as pool:
   futures={i:pool.submit(get_file,f) for i,f in enumerate(pending[:2])}
   for i,filename in enumerate(pending):
    if n>=quota:break
    target,m=futures.pop(i).result()
    if i+2<len(pending):futures[i+2]=pool.submit(get_file,pending[i+2])
    cursor=db.execute("SELECT position FROM cursors WHERE source=? AND file=?",(cursor_namespace,filename)).fetchone();start=cursor[0] if cursor else 0;batch=[];aux=[];last_index=start-1;finished=True
    def flush(position,done=False):
     nonlocal n,batch,aux
     with db:
      for record,raw in batch:
       if n>=quota:break
       h=record['publisher_domain'];url=record['parent_document_id'][5:];bin_id=record['length_bin']
       if domains[h]>=410 or url in seen_urls or remaining[bin_id]<=0:continue
       if add(db,record,raw,quota):
        db.execute('INSERT INTO cccc_selection VALUES (?,?,?)',(record['record_id'],h,url));n+=1;domains[h]+=1;remaining[bin_id]-=1;seen_urls.add(url)
      for value in aux:db.execute('INSERT OR IGNORE INTO auxiliary_scidev VALUES (?,?)',(value['record']['id'],gzip.compress(json.dumps(value,ensure_ascii=False).encode())))
      db.execute('INSERT OR REPLACE INTO cursors VALUES (?,?,?,?)',(cursor_namespace,filename,position,int(done)))
     batch=[];aux=[]
    with gzip.open(target,'rb') as source:
     for index,line in enumerate(source):
      last_index=index
      if index<start:continue
      total_scanned+=1
      if new_domains and not any(h.encode() in line for h in new_domains):
       if (index+1)%1000==0:flush(index+1)
       continue
      row=json.loads(line);meta=row.get('metadata') or {};h=host(meta.get('warc_url',''))
      if h in ('scidev.net','www.scidev.net') and str(meta.get('warc_date',''))[:4].isdigit() and int(meta['warc_date'][:4])<2022:
       aux.append({'source_dataset':REPO,'source_revision':REVISION,'source_file':filename,'source_row':index,'archive_sha256':m['sha256'],'upstream_json_line_sha256':hashlib.sha256(line).hexdigest(),'record':row})
      if new_domains and h.removeprefix('www.') not in new_domains:continue
      reason,info=route(row,mapping,audited)
      if reason:reasons[reason]+=1
      elif domains[info['host']]>=410:reasons['domain_cap']+=1
      elif info['canonical_url'] in seen_urls:reasons['canonical_url_seen']+=1
      else:
       reason,pair=make_pair(row,line,filename,index,m['sha256'],mapping,audited,remaining)
       if pair:batch.append(pair)
       else:reasons[reason]+=1
      if (index+1)%1000==0:
       flush(index+1);atomic_json(base/'progress/cccc.json',{'state':'collecting','count':n,'target':quota,'domain_counts':dict(domains),'file':filename,'source_row':index,'scanned_this_run':total_scanned,'reasons':dict(reasons),'elapsed_seconds':round(time.monotonic()-began,2),'updated_at':now()})
       if n>=quota:finished=False;break
    flush(last_index+1,done=finished);checkpoint();print(json.dumps({'file':filename,'count':n,'target':quota,'domain_counts':dict(domains),'scanned_this_run':total_scanned,'updated_at':now()}),flush=True)
  state='quota_filled' if n==quota else 'bounded_scan_finished' if max_shards is not None else 'available_filtered_shards_scanned'
  atomic_json(base/'progress/cccc.json',{'state':state,'count':n,'target':quota,'domain_counts':dict(domains),'remaining_bins':remaining,'reasons':dict(reasons),'scanned_this_run':total_scanned,'updated_at':now()});checkpoint();db.close()

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--quota',type=int,default=5335);p.add_argument('--max-shards',type=int);p.add_argument('--durable-token-stdin',action='store_true');p.add_argument('--new-domains',nargs='+');p.add_argument('--cache-only',action='store_true');p.add_argument('--cursor-namespace',default='cccc');a=p.parse_args()
 if str(a.base).startswith('/data/'):p.error('Use healthy Space local staging plus direct bucket durability')
 collect(a.base,a.quota,a.max_shards,sys.stdin.readline().strip() if a.durable_token_stdin else None,a.new_domains,a.cache_only,a.cursor_namespace)
