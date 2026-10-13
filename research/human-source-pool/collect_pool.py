"""Collect a private, quarantined candidate pool; never certify training readiness.

Run on the training Space. No models, inference, or paid services are used.
Every selected document is retained; missing source quotas remain explicit.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
import random
import re
from pathlib import Path
import threading
from urllib.parse import urlparse

import requests
from build_plan import apportion

VERSION = 'candidate-intake-v1'
SEED = 27183
BINS = [(50,149),(150,399),(400,999),(1000,1500)]
LENGTH_WEIGHTS = {'creative':[10,25,45,20],'scientific':[10,35,40,15],'reference':[10,35,40,15],
 'reviews':[35,45,18,2],'social':[40,40,18,2],'general_web':[20,40,30,10],
 'news':[15,40,35,10],'essays':[10,35,45,10],'professional':[15,35,35,15]}
STOPWORDS=set('the and of to in a is that for with as was on are it be this from by an or at not have'.split())
OPEN_LICENSE = re.compile(r'public domain|creativecommons.org/licenses/(?:by|by-sa)/|creativecommons.org/publicdomain|^CCBY$|^CC0$|^CC-BY(?:-SA)?(?:-\d\.\d)?$',re.I)


def now():return datetime.now(timezone.utc).isoformat()
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def atomic_json(path, obj):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def date_year(value):
 if not value:return None
 found=re.findall(r'\b(?:19|20)\d{2}\b',str(value))
 return int(found[-1]) if found else None

def prose_spans(text):
 """Offsets into untouched source text; paragraphs only, no rewriting."""
 spans=[]
 for m in re.finditer(r'\S[^\n]*(?:\n(?!\s*\n)[^\n]*)*',text):
  s=m.group();words=re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?",s)
  if len(words)<20 or len(words)>1500:continue
  if len(re.findall(r'https?://',s))>3 or s.count('|')>5 or '<html' in s.lower() or '<div' in s.lower():continue
  if sum(w.lower() in STOPWORDS for w in words)/len(words)<.06:continue
  if sum(c.isalpha() for c in s)/max(len(s),1)<.5:continue
  if re.search(r'(?im)^\s*(references|bibliography|acknowledg(?:e)?ments)\s*$',s):continue
  spans.append((m.start(),m.end(),len(s.split())))
 return spans

def make_passages(text, doc_id, capacity, remaining):
 spans=prose_spans(text);rng=random.Random(digest(str(SEED)+doc_id));order=list(range(len(spans)));rng.shuffle(order)
 selected=[];occupied=[]
 for index in order:
  if len(selected)>=capacity:break
  a,b,w=spans[index]
  # Favor an underfilled length bin, deterministically varying choices per document.
  available=[i for i,n in enumerate(remaining) if n>0]
  if not available:break
  chosen=rng.choices(available,weights=[remaining[i] for i in available])[0]
  lo,hi=BINS[chosen];j=index
  while w<lo and j+1<len(spans):
   na,nb,nw=spans[j+1]
   if na-b>8 or w+nw>hi:break
   j+=1;b=nb;w=len(text[a:b].split())
  if not lo<=w<=hi:continue
  if any(a<q and b>p for p,q in occupied):continue
  selected.append((a,b,w,chosen));occupied.append((a,b));remaining[chosen]-=1
 return selected


def collect_source(sid,spec,source,quota,out,exclusions,max_files,max_rows):
 dest=out/'sources'/sid;dest.mkdir(parents=True,exist_ok=True)
 status_path=dest/'status.json'
 config={'version':VERSION,'source_id':sid,'quota':quota,'revision':spec['revision'],'seed':SEED,'max_files':max_files,'max_rows':max_rows}
 fingerprint=digest(json.dumps(config,sort_keys=True))
 if status_path.exists():
  old=json.loads(status_path.read_text())
  if old.get('fingerprint')!=fingerprint:raise ValueError(f'Existing {sid} has different configuration; choose a new output directory')
  if old.get('finished'):return old
  # Do not truncate a partial collection under an ambiguous resume.
  raise RuntimeError(f'{sid}: partial output exists; inspect before resuming')
 stats=dict(config,fingerprint=fingerprint,started_at=now(),candidate_passages=0,scanned_documents=0,retained_documents=0,rejections={},errors=[],files_attempted=[],finished=False)
 atomic_json(status_path,stats)
 remaining=list(apportion(quota,{str(i):v for i,v in enumerate(LENGTH_WEIGHTS[source['category']])}).values())
 rejected=Counter();seen=set();files=list(spec['files']);random.Random(SEED).shuffle(files)
 # Spread a bounded pass across source shards; do not drain one shard to fill the source.
 selected_files=files[:max_files];per_file=max(1,(quota+len(selected_files)-1)//max(len(selected_files),1))
 with gzip.open(dest/'documents.jsonl.gz','wt',encoding='utf8') as docs,gzip.open(dest/'passages.jsonl.gz','wt',encoding='utf8') as passages:
  for filename in selected_files:
   if stats['candidate_passages']>=quota:break
   stats['files_attempted'].append(filename);file_count=0
   url=f"https://huggingface.co/datasets/{spec['repo_id']}/resolve/{spec['revision']}/{filename}"
   try:
    with requests.get(url,stream=True,timeout=(15,60)) as response:
     response.raise_for_status()
     with gzip.GzipFile(fileobj=response.raw) as rows:
      for row_index,line in enumerate(rows):
       if row_index>=max_rows or file_count>=per_file or stats['candidate_passages']>=quota:break
       stats['scanned_documents']+=1
       try:row=json.loads(line)
       except Exception:rejected['malformed_json']+=1;continue
       meta=row.get('metadata') or {};text=row.get('text') or ''
       if not isinstance(text,str) or len(text)>8_000_000:rejected['invalid_or_oversize_text']+=1;continue
       original_id=str(row.get('id',''));source_url=str(meta.get('url') or meta.get('oa_url') or '')
       license_text=str(meta.get('license') or meta.get('oa_license') or '')
       if not OPEN_LICENSE.search(license_text):rejected['no_allowed_record_license']+=1;continue
       if meta.get('language') and meta['language'] not in ['en','eng','English']:rejected['language']+=1;continue
       year=date_year(row.get('created') or meta.get('year'))
       if year and year>2021:rejected['post_cutoff_date']+=1;continue
       if not year and sid!='gutenberg':rejected['missing_date']+=1;continue
       if '<!doctype html' in text[:200].lower() or '<html' in text[:200].lower():rejected['html_requires_source_extractor']+=1;continue
       if sid=='gutenberg' and (original_id in exclusions['gutenberg_ids'] or any(t in str(meta.get('title','')).casefold() for t in exclusions['book_titles'])):rejected['protected_pg19']+=1;continue
       if sid.startswith('stack_') and 'stackoverflow.com' in source_url:rejected['wrong_site']+=1;continue
       if sid=='ubuntu_irc':rejected['chat_requires_bot_and_turn_audit']+=1;continue
       raw_hash=digest(text);doc_id=f"{spec['repo_id']}:{original_id}"
       if raw_hash in seen:rejected['duplicate_document']+=1;continue
       cap=10 if sid=='gutenberg' else 3 if source['category'] in ['scientific','professional','reference'] else 1
       cap=min(cap,quota-stats['candidate_passages'],per_file-file_count)
       spans=make_passages(text,doc_id,cap,remaining)
       if not spans:rejected['no_eligible_paragraph_or_length_bin']+=1;continue
       seen.add(raw_hash)
       raw_record={'source_id':sid,'source_dataset':spec['repo_id'],'source_revision':spec['revision'],'source_file':filename,'source_row':row_index,'retrieved_at':now(),'raw_text_sha256':raw_hash,'record':row}
       docs.write(json.dumps(raw_record,ensure_ascii=False)+'\n');stats['retained_documents']+=1
       group_basis=meta.get('book_url') or source_url or doc_id
       reasons=['historical_text_version_unverified','record_rights_audit_pending','protected_overlap_audit_pending','genre_and_extraction_review_pending','language_review_pending','family_and_author_grouping_pending']
       for a,b,w,bin_id in spans:
        passage=text[a:b];ph=digest(passage)
        record={'record_id':digest(sid+doc_id+str(a)+str(b)),'source_id':sid,'category':source['category'],'text':passage,'word_count':w,'length_bin':bin_id,
         'source_dataset':spec['repo_id'],'source_revision':spec['revision'],'source_file':filename,'source_row':row_index,'original_id':original_id,'source_url':source_url,
         'title':str(meta.get('title') or ''),'author_attribution_json':json.dumps(meta.get('authors') or meta.get('author') or [],ensure_ascii=False),
         'license_evidence':license_text,'claimed_original_date':str(row.get('created') or meta.get('year') or ''),'retrieved_at':raw_record['retrieved_at'],
         'raw_text_sha256':raw_hash,'passage_sha256':ph,'raw_start':a,'raw_end':b,'offset_unit':'unicode_codepoints','extraction_method':'unchanged_contiguous_paragraphs',
         'parent_document_id':doc_id,'provisional_family_id':digest(str(group_basis)),'admission_status':'quarantined_candidate','training_eligible':False,
         'provenance_basis':'unverified','protected_overlap_status':'not_fully_audited','reason_codes':reasons,'sampling_seed':SEED,'pipeline_version':VERSION}
        passages.write(json.dumps(record,ensure_ascii=False)+'\n');stats['candidate_passages']+=1;file_count+=1
       if stats['scanned_documents']%100==0:
        stats['rejections']=dict(rejected);atomic_json(status_path,stats)
   except Exception as exc:stats['errors'].append({'file':filename,'type':type(exc).__name__,'message':str(exc)[:240]})
   stats['rejections']=dict(rejected);atomic_json(status_path,stats)
 stats.update(finished=True,finished_at=now(),shortfall=quota-stats['candidate_passages'],length_bin_shortfall=remaining)
 atomic_json(status_path,stats)
 return stats


def package(out,registry,plan,statuses):
 import pyarrow as pa
 import pyarrow.parquet as pq
 release=out/'release';(release/'data').mkdir(parents=True,exist_ok=True);(release/'provenance').mkdir(exist_ok=True)
 seen=set();counts=Counter();hashes=[];duplicate_removed=Counter()
 for status in statuses:
  sid=status['source_id'];path=out/'sources'/sid/'passages.jsonl.gz'
  if not path.exists():continue
  records=[]
  with gzip.open(path,'rt') as stream:
   for line in stream:
    row=json.loads(line)
    key=digest(' '.join(row['text'].casefold().split()))
    if key in seen:duplicate_removed[sid]+=1;continue
    seen.add(key);records.append(row)
  if records:
   dest=release/'data'/f'{sid}.parquet';pq.write_table(pa.Table.from_pylist(records),dest,compression='zstd')
   counts[sid]=len(records);hashes.append({'path':str(dest.relative_to(release)),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'bytes':dest.stat().st_size,'rows':len(records)})
 # Every source shard must expose the same columns to Hugging Face's Parquet
 # reader. Preserve source-specific metadata as nullable fields in the union.
 if hashes:
  schema=pa.unify_schemas([pq.read_schema(release/a['path']) for a in hashes],promote_options='permissive')
  for artifact in hashes:
   dest=release/artifact['path'];table=pq.read_table(dest)
   arrays=[table[f.name].cast(f.type) if f.name in table.column_names else pa.nulls(table.num_rows,type=f.type) for f in schema]
   pq.write_table(pa.Table.from_arrays(arrays,schema=schema),dest,compression='zstd')
   artifact.update(sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),bytes=dest.stat().st_size)
 source_rows=[]
 for quota in plan['source_quotas']:
  sid=quota['source_id'];status=next(s for s in statuses if s['source_id']==sid)
  source_rows.append(dict(source_id=sid,category=quota['category'],planned_passages=quota['planned_passages'],candidate_passages=counts[sid],admitted_passages=0,
   shortfall=quota['planned_passages']-counts[sid],state=status.get('state','collected_candidates'),rights_evidence_status=quota['rights_evidence_status'],exact_duplicates_removed=duplicate_removed[sid]))
 summary={'status':'partial_quarantined_candidate_pool','planned_total':plan['planned_total'],'candidate_total':sum(counts.values()),'admitted_total':0,'shortfall':plan['planned_total']-sum(counts.values()),
  'sources':source_rows,'artifacts':hashes,'finished_at':now(),'all_candidates_require_review':True,'no_train_test_split_assigned':True}
 target_total=plan['planned_total'];actual_total=sum(counts.values());category_targets=Counter();category_actual=Counter()
 def breakdown(key,target,actual):
  target_share=target/target_total;actual_share=actual/actual_total if actual_total else 0
  return {'id':key,'target':target,'actual':actual,'shortfall':target-actual,'target_share':target_share,'actual_share':actual_share,'difference_percentage_points':100*(actual_share-target_share)}
 for row in source_rows:
  category_targets[row['category']]+=row['planned_passages'];category_actual[row['category']]+=row['candidate_passages']
 atomic_json(release/'distribution-report.json',{'target_total':target_total,'actual_total':actual_total,
  'policy':'Targets remain fixed; source shortfalls are recorded without reallocating quotas.',
  'sources':[breakdown(r['source_id'],r['planned_passages'],r['candidate_passages']) for r in source_rows],
  'categories':[breakdown(k,n,category_actual[k]) for k,n in category_targets.items()]})
 atomic_json(release/'collection-summary.json',summary);atomic_json(release/'source-registry.json',registry);atomic_json(release/'sampling-plan.json',plan)
 atomic_json(release/'provenance'/'collection-status.json',statuses)
 if hashes:
  header='---\nlanguage:\n- en\npretty_name: Human Source Mix — Quarantined Candidates\nconfigs:\n- config_name: candidates\n  data_files:\n  - split: candidate\n    path: data/*.parquet\n---\n'
 else:header=''
 table='\n'.join(f"| {r['source_id']} | {r['planned_passages']:,} | {r['candidate_passages']:,} | {r['shortfall']:,} | {r['state']} |" for r in source_rows)
 card=header+f'''# Human source mix: candidate intake

**Not a training-ready release.** This private repository contains {summary['candidate_total']:,} real candidate passages toward a {plan['planned_total']:,}-passage plan. **Zero passages are admitted.** It is incomplete and must not be described as the full Pangram distribution or as verified human-authored data.

The nine category proportions follow the researched plan; source-level allocations are our proposals. The fixed 2021 cutoff is applied to claimed dates as an intake filter only. An old publication/creation date does not verify the age of these captured bytes. All records remain quarantined for historical-version, rights, language, genre, extraction, grouping, and protected-evaluation audits. No training labels or train/test split are supplied.

This bounded first collection samples a deterministic set of pinned source shards, then scans bounded prefixes. It is not a uniform sample of entire source repositories. Document and source caps and four length bins limit intake. Exact normalized-passage duplicates are removed across collected sources. Near-duplicate, source-family and comprehensive benchmark exclusion audits remain unfinished. Known PG-19 test IDs/titles are excluded from Gutenberg intake, which is not a comprehensive contamination check. Chat and raw HTML require dedicated extraction before collection. No source substitution, quota renormalization, model inference, or AI generation is performed.

Use `candidates` / `candidate` to inspect the texts. Do not train on this configuration without completing admission checks. Missing sources and shortfalls remain explicit:

| Source | Planned | Candidates | Shortfall | State |
|---|---:|---:|---:|---|
{table}

Each record includes original identity, source URL, pinned repository revision and shard, raw-text hash, exact passage offsets, recorded attribution and license evidence, and review reasons. Original retained documents are under `raw/`; their metadata and text are preserved as received. Record license evidence is not a new license grant: original applicable terms remain in force. This mixed candidate collection has no blanket relicensing.

Source repositories are linked in `collection-sources.json`; curation evidence and the detailed admission specification are included under `design/`. Source-level open-subset labels are not document audit results. Build scripts are included under `pipeline/` for reproducibility. SHA-256 hashes and row counts for Parquet shards are in `collection-summary.json`.
'''
 (release/'README.md').write_text(card)
 return summary


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--config-dir',type=Path,default=Path(__file__).resolve().parent)
 parser.add_argument('--out',type=Path,required=True)
 parser.add_argument('--workers',type=int,default=3)
 parser.add_argument('--max-files',type=int,default=6)
 parser.add_argument('--max-rows',type=int,default=5000)
 args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 import fcntl
 lock=(args.out/'collection.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 registry=json.loads((args.config_dir/'source-registry.json').read_text());plan=json.loads((args.config_dir/'sampling-plan.json').read_text());specs=json.loads((args.config_dir/'collection-sources.json').read_text())
 exclusions=json.loads((args.config_dir/'protected-exclusions.json').read_text());sources={s['id']:s for s in registry['sources']};statuses=[];futures={}
 with ThreadPoolExecutor(max_workers=args.workers) as pool:
  for quota in plan['source_quotas']:
   sid=quota['source_id'];q=quota['planned_passages'];source=sources[sid]
   if source['rights_evidence_status']!='open_subset':statuses.append({'source_id':sid,'state':'held_rights_unresolved_or_restricted','candidate_passages':0,'shortfall':q});continue
   if sid not in specs:statuses.append({'source_id':sid,'state':'source_adapter_not_yet_verified','candidate_passages':0,'shortfall':q});continue
   futures[pool.submit(collect_source,sid,specs[sid],source,q,args.out,exclusions,args.max_files,args.max_rows)]=sid
  for future in as_completed(futures):
   status=future.result();statuses.append(status);print(json.dumps({'source':status['source_id'],'candidates':status['candidate_passages'],'shortfall':status['shortfall']}),flush=True)
   atomic_json(args.out/'progress.json',{'completed_sources':len(statuses),'candidate_total':sum(s.get('candidate_passages',0) for s in statuses),'updated_at':now(),'sources':statuses})
 summary=package(args.out,registry,plan,statuses);atomic_json(args.out/'status.json',summary);print(json.dumps({'complete':True,'candidates':summary['candidate_total'],'shortfall':summary['shortfall']}),flush=True)

if __name__=='__main__':main()
