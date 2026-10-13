"""Recover overlooked faithful prose windows from already audited source originals."""
import argparse
from collections import Counter,defaultdict
import fcntl,gzip,hashlib,json,random,re,sqlite3,time
from pathlib import Path
from collect_pool import BINS,LENGTH_WEIGHTS,STOPWORDS,OPEN_LICENSE,apportion,atomic_json,digest,now,date_year
from expand_pool import add,connect,namespace_ok,claimed_date
VERSION='cached-prose-window-recovery-v2'
SPECS={'pressbooks':('reference',3979,3),'wikivoyage':('general_web',1231,1)}
BARRIER=re.compile(r'creativecommons|all rights reserved|copyright\s*(?:©|\d)|\bcookie\s+(?:policy|settings)|\bprivacy policy\b|\bterms of use\b|back to top|skip to (?:main|content)|^\s*(?:figure|table)\s+\d|\{\{|\}\}|</?(?:html|div|script|style)\b',re.I)
END_SECTION=re.compile(r'^\s*(?:references|bibliography|footnotes|external links|works cited)\s*[:.]?\s*$',re.I)

def line_quality(value):
    words=re.findall('[A-Za-z]+',value);wc=len(value.split())
    if (BARRIER.search(value) or value.count('|')>2 or '\t' in value or value.count('http')>1
        or sum(c.isalpha() for c in value)/max(1,len(value))<.45):return 'barrier'
    if len(words)>=8 and sum(w.lower() in STOPWORDS for w in words)/len(words)>=.06:return 'prose'
    if wc<=12 and not re.search(r'https?://|www\.|@|[=<>]|\d{3,}',value):return 'heading'
    return 'barrier'

def prose_regions(text):
    regions=[];begin=None;end=None;gap_words=0;gap_lines=0
    for m in re.finditer(r'[^\r\n]+',text):
        value=m[0].strip()
        if not value:continue
        if END_SECTION.match(value):
            if begin is not None:regions.append((begin,end))
            break
        kind=line_quality(value)
        if kind=='prose':
            if begin is None:begin=m.start()
            elif gap_words>30 or gap_lines>5:regions.append((begin,end));begin=m.start()
            end=m.end();gap_words=0;gap_lines=0
        elif kind=='heading' and begin is not None:gap_words+=len(value.split());gap_lines+=1
        else:
            if begin is not None:regions.append((begin,end))
            begin=end=None;gap_words=gap_lines=0
    else:
        if begin is not None:regions.append((begin,end))
    return regions

def split_occupied(regions,occupied):
    for begin,end in regions:
        parts=[(begin,end)]
        for a,b in occupied:
            next_parts=[]
            for x,y in parts:
                if b<=x or a>=y:next_parts.append((x,y));continue
                if x<a:next_parts.append((x,a))
                if b<y:next_parts.append((b,y))
            parts=next_parts
        yield from parts

def clean_span(text):
    words=re.findall('[A-Za-z]+',text)
    return len(words)>=35 and sum(w.lower() in STOPWORDS for w in words)/len(words)>=.06 and sum(c.isalpha() for c in text)/max(1,len(text))>=.5 and text.count('http')<=2 and not BARRIER.search(text)

def select_windows(text,doc,cap,remaining,occupied):
    regions=list(split_occupied(prose_regions(text),occupied));rng=random.Random(digest('27183'+doc));rng.shuffle(regions);chosen=[]
    for begin,end in regions:
        if len(chosen)>=cap:break
        boundary={begin,end}
        for m in re.finditer(r'[.!?][\"\u201d\u2019\x27)]*\s+|\r?\n+',text[begin:end]):boundary.add(begin+m.end())
        bounds=sorted(boundary);pos=0
        while pos<len(bounds)-1 and len(chosen)<cap:
            options=[]
            for k in range(pos+1,len(bounds)):
                a,b=bounds[pos],bounds[k];wc=len(text[a:b].split())
                if wc>1500:break
                bi=next((i for i,(lo,hi) in enumerate(BINS) if lo<=wc<=hi and remaining[i]>0),None)
                if bi is not None and clean_span(text[a:b]):options.append((a,b,wc,bi,k))
            if not options:pos+=1;continue
            # Rare long coherent regions must not be consumed as arbitrary short
            # prefixes while the longer bins remain underfilled.
            bi=max(v[3] for v in options);opts=[v for v in options if v[3]==bi]
            a,b,wc,bi,k=rng.choice(opts);chosen.append((a,b,wc,bi));remaining[bi]-=1;pos=k
    return chosen

def eligible(sid,row):
    meta=row.get('metadata') or {};text=row.get('text')
    if not isinstance(text,str) or not text or len(text)>8000000:return False
    if not namespace_ok(sid,row):return False
    if not OPEN_LICENSE.search(str(meta.get('license') or meta.get('oa_license') or '')):return False
    if meta.get('language') and meta['language'] not in ('en','eng','English'):return False
    year=date_year(claimed_date(sid,row)[0])
    return bool(year and year<=2021 and not re.search(r'<!doctype|<html',text[:300],re.I))

def pair(sid,spec,wrapper,a,b,wc,bi):
    original=wrapper['record'];text=original['text'];meta=original.get('metadata') or {};doc=spec['source_dataset']+':'+str(original.get('id',''))
    url=str(meta.get('url') or meta.get('oa_url') or '');date,basis=claimed_date(sid,original);category=SPECS[sid][0];raw_hash=digest(text)
    archive=next(m for m in spec['files'] if m['filename']==wrapper['source_file'])
    raw={'source_id':sid,'source_dataset':spec['source_dataset'],'source_revision':spec['revision'],'source_file':wrapper['source_file'],
         'source_row':wrapper['source_row'],'retrieved_at':now(),'raw_text_sha256':raw_hash,'record':original,
         'recovery_provenance':{'original_archive_sha256':archive['sha256'],'original_archive_bytes':archive['bytes'],'audit_source':'preserved_complete_pinned_archive'}}
    value=text[a:b]
    row={'record_id':digest(sid+doc+str(a)+str(b)),'source_id':sid,'category':category,'text':value,'word_count':wc,'length_bin':bi,
         'source_dataset':spec['source_dataset'],'source_revision':spec['revision'],'source_file':wrapper['source_file'],'source_row':wrapper['source_row'],
         'original_id':str(original.get('id','')),'source_url':url,'title':str(meta.get('title') or ''),'author_attribution_json':json.dumps(meta.get('authors') or meta.get('author') or []),
         'license_evidence':str(meta.get('license') or meta.get('oa_license') or ''),'claimed_original_date':date,'date_evidence_basis':basis,
         'retrieved_at':raw['retrieved_at'],'raw_text_sha256':raw_hash,'passage_sha256':digest(value),'raw_start':a,'raw_end':b,'offset_unit':'unicode_codepoints',
         'extraction_method':'unchanged_contiguous_sentence_window_with_short_interior_headings','parent_document_id':doc,
         'provisional_family_id':digest(str(meta.get('book_url') or url or doc)),'admission_status':'quarantined_candidate','training_eligible':False,
         'provenance_basis':'same_pinned_original_as_audited_pool','protected_overlap_status':'not_fully_audited','reason_codes':['historical_text_version_unverified',
          'record_rights_audit_pending','protected_overlap_audit_pending','genre_and_extraction_review_pending','language_review_pending','family_and_author_grouping_pending'],
         'sampling_seed':27183,'pipeline_version':VERSION}
    return row,raw

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',choices=SPECS,required=True);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--audit-base',type=Path,required=True);p.add_argument('--production-db',type=Path,required=True);args=p.parse_args();sid=args.source
    category,quota,cap=SPECS[sid];base=args.base;base.mkdir(parents=True,exist_ok=True);(base/'progress').mkdir(exist_ok=True);began=time.monotonic()
    with (base/(sid+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prod=sqlite3.connect('file:'+str(args.production_db)+'?mode=ro',uri=True);prod.execute('BEGIN')
        baseline=[json.loads(x[0]) for x in prod.execute('SELECT row FROM passages WHERE source=?',(sid,))];seen={x[0] for x in prod.execute('SELECT norm FROM passages')};original_hashes={x[0] for x in prod.execute('SELECT hash FROM documents')};prod.rollback();prod.close()
        path=base/'baseline.json.gz'
        if path.exists():
            old=json.loads(gzip.decompress(path.read_bytes()));assert old==baseline,'Production source baseline changed; coordinate new recovery stage'
        else:path.write_bytes(gzip.compress(json.dumps(baseline).encode()))
        spec=json.loads((args.audit_base/sid/'status.json').read_text());assert spec['state']=='audit_complete'
        atomic_json(base/'source-audit.json',spec);db=connect(base/'stage.sqlite3');ranges=defaultdict(list);counts=Counter();bins=Counter();hashes=defaultdict(set)
        for row in baseline+[json.loads(v[0]) for v in db.execute('SELECT row FROM passages WHERE source=?',(sid,))]:
            doc=row['parent_document_id'];ranges[doc].append((row['raw_start'],row['raw_end']));counts[doc]+=1;hashes[doc].add(row['raw_text_sha256']);bins[row['length_bin']]+=1;seen.add(digest(' '.join(row['text'].casefold().split())))
        targets=apportion(quota,{str(i):v for i,v in enumerate(LENGTH_WEIGHTS[category])});remaining=[targets[str(i)]-bins[i] for i in range(4)];reasons=Counter();scanned=0;seen_input_docs=set()
        with gzip.open(args.audit_base/sid/'eligible-originals.jsonl.gz','rt') as f:
            for index,line in enumerate(f):
                if not any(remaining):break
                scanned+=1;w=json.loads(line);original=w['record']
                if not eligible(sid,original):reasons['eligibility_rejected']+=1;continue
                doc=spec['source_dataset']+':'+str(original.get('id',''));text=original['text']
                if counts[doc] or doc in seen_input_docs:reasons['previously_sampled_document']+=1;continue
                seen_input_docs.add(doc)
                if digest(text) in original_hashes:reasons['duplicate_original_text']+=1;continue
                original_hashes.add(digest(text))
                if hashes[doc] and digest(text) not in hashes[doc]:reasons['different_text_same_document']+=1;continue
                windows=select_windows(text,doc,cap-counts[doc],remaining.copy(),ranges[doc])
                for a,b,wc,bi in windows:
                    norm=digest(' '.join(text[a:b].casefold().split()))
                    if norm in seen:reasons['global_duplicate']+=1;continue
                    row,raw=pair(sid,spec,w,a,b,wc,bi);db.execute('BEGIN IMMEDIATE')
                    try:inserted=add(db,row,raw,quota-len(baseline));db.commit()
                    except BaseException:db.rollback();raise
                    if inserted:
                        ranges[doc].append((a,b));counts[doc]+=1;hashes[doc].add(digest(text));bins[bi]+=1;remaining[bi]-=1;seen.add(norm)
                if not windows:reasons['no_quality_window']+=1
                if index%100==0:atomic_json(base/'progress'/f'{sid}.json',{'state':'recovering_cached_originals','source':sid,'baseline_count':len(baseline),'new_count':sum(bins.values())-len(baseline),'source_total':sum(bins.values()),'remaining_bins':remaining,'scanned':scanned,'updated_at':now()})
        out=base/'staged'/sid;out.mkdir(parents=True,exist_ok=True)
        for name,query in [('passages','SELECT row FROM passages'),('documents','SELECT raw FROM documents')]:
            with gzip.open(out/(name+'.jsonl.gz'),'wt') as f:
                for value, in db.execute(query):f.write((gzip.decompress(value).decode() if name=='documents' else value)+'\n')
        status={'state':'source_quota_filled' if not any(remaining) else 'available_cached_window_shortfall','source':sid,'baseline_count':len(baseline),
                'new_count':sum(bins.values())-len(baseline),'source_total':sum(bins.values()),'target':quota,'length_counts':dict(bins),'remaining_bins':remaining,
                'scanned_eligible_originals':scanned,'reasons':dict(reasons),'elapsed_seconds':time.monotonic()-began,'all_quarantined':True,'production_modified':False,'updated_at':now()}
        atomic_json(base/'progress'/f'{sid}.json',status);print(json.dumps(status));db.close()

if __name__=='__main__':main()
