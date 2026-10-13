"""Pinned English SCP tales, pre-2022 complete recorded histories, remote staging only."""
import argparse
from collections import Counter
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import re
import time
from collect_pool import BINS, LENGTH_WEIGHTS, apportion, atomic_json, digest, now, prose_spans
from expand_pool import add, connect

REVISION = 'abc3303e16fb5ebf8761f5d240f1b1a071527fc0'
REPO = 'scp-data/scp-api'
VERSION = 'scp-pinned-history-tales-v1'
MARKUP = re.compile(r'\[\[|\]\]|<[^>]+>|(?m:^\s*[#*|>])|(?i:license.?box|image attribution|image source|author.?s note|special containment procedures|object class:)')

def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def eligibility(row):
    if row.get('domain') != 'scp-wiki.wikidot.com' or 'tale' not in row.get('tags', []): return 'not_english_tale'
    if not row.get('page_id') or not row.get('creator') or row.get('creator') == 'deleted': return 'missing_identity_or_creator'
    history = row.get('history') or []
    if not history or any(not re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d', str(h.get('date',''))) for h in history): return 'missing_revision_evidence'
    if max(h['date'] for h in history) >= '2022-01-01': return 'post2021_revision'
    if not row.get('created_at') or row['created_at'] >= '2022-01-01': return 'post2021_creation'
    return None

def select_spans(text, doc_id, remaining, capacity=3):
    spans = [(a,b,w) for a,b,w in prose_spans(text) if not MARKUP.search(text[a:b])]
    order = sorted(range(len(spans)), key=lambda i:digest(doc_id+':'+str(i)))
    chosen=[];used=[];left=remaining.copy()
    for index in order:
        if len(chosen)>=capacity: break
        a,b,w=spans[index]
        for bin_id in sorted(range(4),key=lambda i:(-left[i],i)):
            if left[bin_id]<=0: continue
            lo,hi=BINS[bin_id];end=b;count=w;j=index
            while count<lo and j+1<len(spans):
                na,nb,nw=spans[j+1]
                if na-end>8 or len(text[a:nb].split())>hi: break
                end=nb;count=len(text[a:end].split());j+=1
            if not lo<=count<=hi or any(a<y and end>x for x,y in used): continue
            chosen.append((a,end,count,bin_id));used.append((a,end));left[bin_id]-=1;break
    return chosen

def pairs(row, source_file, manifest, remaining, source_row=0):
    if eligibility(row): return
    text=row.get('raw_source')
    if not isinstance(text,str) or not text.strip(): return
    doc_id='scp:'+str(row['page_id']);spans=select_spans(text,doc_id,remaining)
    contributors=[{'name':row['creator'],'role':'page_creator'}]
    contributors.extend({'name':name,'role':'revision_contributor'} for name in sorted({h['author'] for h in row['history'] if h.get('author') and h['author'] not in ('deleted',row['creator'])}))
    raw={'source_id':'scp','source_dataset':REPO,'source_revision':REVISION,'source_file':source_file,
         'retrieved_at':manifest['retrieved_at'],'raw_text_sha256':digest(text),
         'record':{'id':str(row['page_id']),'text':text,'metadata':row}}
    for a,b,w,bin_id in spans:
        passage=text[a:b]
        yield {'record_id':digest(doc_id+':'+str(a)+':'+str(b)),'source_id':'scp','category':'creative','text':passage,
          'word_count':w,'length_bin':bin_id,'source_dataset':REPO,'source_revision':REVISION,'source_file':source_file,
          'source_row':source_row,'original_id':str(row['page_id']),'source_url':row['url'],'title':row['title'],
          'author_attribution_json':json.dumps(contributors,ensure_ascii=False),'license_evidence':'https://creativecommons.org/licenses/by-sa/3.0/',
          'claimed_original_date':row['created_at'],'latest_recorded_revision_date':max(h['date'] for h in row['history']),
          'date_evidence_basis':'pinned_mirror_full_recorded_revision_history_latest_before_2022',
          'retrieved_at':manifest['retrieved_at'],'raw_text_sha256':digest(text),'passage_sha256':digest(passage),
          'raw_start':a,'raw_end':b,'offset_unit':'unicode_codepoints','extraction_method':'unchanged_plain_narrative_spans_from_original_wikitext',
          'parent_document_id':doc_id,'provisional_family_id':digest(doc_id),'canon_hub_ids_json':json.dumps(row.get('hubs',[])),
          'admission_status':'quarantined_candidate','training_eligible':False,'provenance_basis':'english_tale_tag_and_pinned_revision_history',
          'protected_overlap_status':'not_fully_audited','reason_codes':['historical_mirror_revision_history_requires_independent_audit',
             'creator_and_revision_attribution_retained_final_author_credit_review_pending','genre_and_extraction_review_pending',
             'canon_and_author_family_grouping_pending','protected_overlap_audit_pending'],
          'sampling_seed':27183,'pipeline_version':VERSION},raw

def collect(base, quota):
    started=time.monotonic();m=json.loads((base/'source-manifest.json').read_text())
    assert m['revision']==REVISION and m['repo']==REPO and m['source_id']=='scp'
    assert sha_file(base/'tales-index.json')==m['index_sha256']
    (base/'progress').mkdir(exist_ok=True)
    with (base/'scp.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        db=connect(base/'collection.sqlite3');initial=db.execute("SELECT count(*) FROM passages WHERE source='scp'").fetchone()[0]
        targets=apportion(quota,{str(i):v for i,v in enumerate(LENGTH_WEIGHTS['creative'])});n=initial;reasons=Counter();documents=set()
        for info in m['files']:
            file=base/info['file'];assert sha_file(file)==info['sha256']
            data=json.loads(file.read_text())
            for source_row,key in enumerate(sorted(data,key=lambda k:digest(VERSION+k))):
                if n>=quota:break
                row=data[key];reason=eligibility(row)
                if reason:reasons[reason]+=1;continue
                if db.execute("SELECT 1 FROM passages WHERE source='scp' AND doc=?",('scp:'+str(row['page_id']),)).fetchone():continue
                bins=dict(db.execute("SELECT bin,count(*) FROM passages WHERE source='scp' GROUP BY bin"));remaining=[max(0,targets[str(i)]-bins.get(i,0)) for i in range(4)]
                candidates=list(pairs(row,info['file'],m,remaining,source_row))
                if not candidates:reasons['no_plain_narrative_passages']+=1;continue
                db.execute('BEGIN IMMEDIATE')
                try:
                    for record,raw in candidates:
                        if add(db,record,raw,quota):n+=1;documents.add(record['parent_document_id'])
                    db.commit()
                except BaseException:db.rollback();raise
            if n>=quota:break
        status={'source_id':'scp','target':quota,'count':n,'new_candidates':n-initial,'new_parent_documents':len(documents),
           'state':'quota_filled' if n==quota else 'eligible_mirror_subset_exhausted','rejections':dict(reasons),
           'all_quarantined':True,'isolated_staging':True,'global_merge_pending':True,'elapsed_seconds':time.monotonic()-started,'updated_at':now()}
        atomic_json(base/'progress/scp.json',status)
        package=base/'accepted-pairs.jsonl.gz';verified=0
        with gzip.open(package,'wt',encoding='utf-8') as out:
            for (saved,) in db.execute("SELECT row FROM passages WHERE source='scp' ORDER BY id"):
                row=json.loads(saved);raw=json.loads(gzip.decompress(db.execute('SELECT raw FROM documents WHERE hash=?',(row['raw_text_sha256'],)).fetchone()[0]))
                text=raw['record']['text'];assert digest(text)==row['raw_text_sha256'];assert text[row['raw_start']:row['raw_end']]==row['text'];assert digest(row['text'])==row['passage_sha256']
                out.write(json.dumps({'row':row,'raw':raw},ensure_ascii=False)+'\n');verified+=1
        db.close();status.update(validated_rows=verified,package=str(package),package_bytes=package.stat().st_size,package_sha256=sha_file(package))
        atomic_json(base/'package-manifest.json',status);print(json.dumps(status),flush=True);return status

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--quota',type=int,required=True)
    args=p.parse_args()
    if str(args.base).startswith('/data/'):p.error('This adapter run is isolated temporary staging only')
    collect(args.base,args.quota)

if __name__=='__main__':main()
