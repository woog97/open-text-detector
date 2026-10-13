"""Historical SciDev captures, corroborated by explicit current article notices."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
import fcntl,gzip,hashlib,json,random,re,sqlite3,time,threading
from pathlib import Path
from urllib.parse import urlparse,urlunparse
import requests
from bs4 import BeautifulSoup
from collect_pool import BINS,LENGTH_WEIGHTS,STOPWORDS,apportion,atomic_json,digest,now
from expand_pool import add,connect
VERSION='scidev-historical-explicit-republish-v1'
TOKEN=re.compile(r'\w+',re.UNICODE)
HOSTS={'www.scidev.net','scidev.net'}

class PublisherPacer:
    """One shared rate limit across all verification workers."""
    def __init__(self,base):
        self.path=base/'progress/publisher-backoff.json';self.lock=threading.Lock();self.next_request=0;self.blocked_until=0;self.consecutive_limits=0
        if self.path.exists():self.blocked_until=json.loads(self.path.read_text()).get('retry_at_unix',0)
    def before(self):
        while True:
            with self.lock:
                if self.consecutive_limits>=3:raise RuntimeError('publisher rate-limit circuit open')
                wait=max(self.next_request,self.blocked_until)-time.time()
                if wait<=0:self.next_request=time.time()+2;return
            time.sleep(min(wait,1))
    def limited(self,response):
        from email.utils import parsedate_to_datetime
        value=response.headers.get('Retry-After','');delay=300
        try:delay=max(delay,float(value))
        except ValueError:
            try:delay=max(delay,parsedate_to_datetime(value).timestamp()-time.time())
            except (ValueError,TypeError):pass
        with self.lock:
            self.consecutive_limits+=1;self.blocked_until=time.time()+delay
            atomic_json(self.path,{'http_status':response.status_code,'retry_after':value,'retry_at_unix':self.blocked_until,
                                  'consecutive_limits':self.consecutive_limits,'updated_at':now()})
    def success(self):
        with self.lock:self.consecutive_limits=0

def source_url(wrapper):
    record=wrapper['record'];meta=record['metadata'];date=str(meta.get('warc_date',''))
    if not re.match(r'^20(?:0\d|1\d|2[01])-',date):return None
    u=urlparse(meta.get('warc_url',''))
    if u.hostname not in HOSTS or not u.path.startswith('/global/') or not any('/'+kind+'/' in u.path for kind in ('news','features','editorials','opinions','analysis')):return None
    if not u.path.rstrip('/').split('/')[-1]:return None
    return urlunparse(('https','www.scidev.net',u.path,'','',''))

def slug(url):return urlparse(url).path.rstrip('/').split('/')[-1].removesuffix('.html').strip('-')

def current_evidence(html,url):
    soup=BeautifulSoup(html,'html.parser');popup=soup.select_one('.republish-popup')
    if popup is None:raise ValueError('no article-specific republish popup')
    notice=BeautifulSoup(str(popup),'html.parser')
    for node in notice.select('textarea'):node.decompose()
    notice_text=notice.get_text(' ',strip=True)
    if not re.search(r'republish this article.*?creative commons attribution license',notice_text,re.I|re.S):
        raise ValueError('no explicit article republishing grant')
    area=popup.select_one('textarea.form-control')
    if area is None:raise ValueError('no article republish HTML')
    article=BeautifulSoup(area.get_text(),'html.parser');byline=article.find('h4',string=re.compile(r'^\s*By:',re.I))
    if byline is None:
        byline=next((n for n in article.find_all('h4') if re.match(r'^\s*By:',n.get_text())),None)
    author=re.sub(r'^\s*By:\s*','',byline.get_text(' ',strip=True)) if byline else ''
    if not author or author.lower() in {'editor','admin','scidev.net'}:raise ValueError('no named publisher article byline')
    if re.search(r'reuters|associated press|agence france|\bAFP\b',author,re.I):raise ValueError('external wire author')
    body=article.select_one('#article-body')
    if body is None:raise ValueError('no publisher article body')
    for n in body.select('script,style,figure,figcaption,img,iframe,table'):n.decompose()
    prose=body.get_text(' ',strip=True)
    if re.search(r'(?:originally published|republished|reproduced) (?:by|from|in|on)\s+(?!SciDev)',prose,re.I):
        raise ValueError('third-party republication indicator')
    dates=[]
    for script in soup.select('script[type="application/ld+json"]'):
        try:value=json.loads(script.string or script.get_text())
        except ValueError:continue
        stack=[value]
        while stack:
            item=stack.pop()
            if isinstance(item,list):stack.extend(item)
            elif isinstance(item,dict):
                if item.get('@type') in ('WebPage','Article','NewsArticle') and item.get('datePublished'):dates.append(str(item['datePublished']))
                stack.extend(v for v in item.values() if isinstance(v,(list,dict)))
    date=next((d for d in dates if re.match(r'^(?:19\d\d|20(?:0\d|1\d|2[01]))-',d)),None)
    if date is None:raise ValueError('no pre2022 article publication date')
    links=[a.get('href') for a in notice.select('a[href]')]
    title=article.find('h1')
    return {'author':author,'author_basis':'current_publisher_republish_byline','license_label':'Creative Commons Attribution; version unspecified in article notice',
            'license_notice':notice_text,'license_links':links,'current_article_body':prose,'claimed_publication_date':date,
            'title':title.get_text(' ',strip=True) if title else '', 'evidence_url':url,
            'current_page_is_historical_capture':False}

def matching_regions(historical,current):
    h=list(TOKEN.finditer(historical));c=list(TOKEN.finditer(current))
    if not h or not c:return [],{}
    ht=[m[0].casefold() for m in h];ct=[m[0].casefold() for m in c]
    blocks=[b for b in SequenceMatcher(None,ct,ht,autojunk=False).get_matching_blocks() if b.size>=8]
    matched=sum(b.size for b in blocks);coverage=matched/len(ct)
    evidence={'matched_tokens':matched,'current_body_tokens':len(ct),'current_body_match_fraction':coverage,
              'method':'monotonic_normalized_exact_token_blocks_min8'}
    if matched<120 or coverage<.5:return [],evidence
    merged=[]
    for b in blocks:
        if merged and b.b-merged[-1][1]<=3 and b.a-merged[-1][3]<=3:
            merged[-1][1]=b.b+b.size;merged[-1][3]=b.a+b.size
        else:merged.append([b.b,b.b+b.size,b.a,b.a+b.size])
    return [(h[a].start(),h[b-1].end()) for a,b,_,_ in merged if b-a>=20],evidence

def select_spans(text,regions,doc,remaining):
    rng=random.Random(digest('27183'+doc));regions=regions.copy();rng.shuffle(regions);selected=[]
    for begin,end in regions:
        if len(selected)>=3:break
        # Sentence ends preserve original character offsets and contiguous text.
        boundaries=[begin]+[begin+m.end() for m in re.finditer(r'[.!?][\"\u201d\u2019\x27)]*\s+',text[begin:end])]+[end]
        pos=0
        while pos<len(boundaries)-1 and len(selected)<3:
            options=[]
            for k in range(pos+1,len(boundaries)):
                a,b=boundaries[pos],boundaries[k];wc=len(text[a:b].split())
                if wc>1500:break
                for bin_id,(lo,hi) in enumerate(BINS):
                    if remaining[bin_id]>0 and lo<=wc<=hi:options.append((a,b,wc,bin_id,k))
            if not options:break
            bins=sorted({v[3] for v in options});choice=3 if 3 in bins else rng.choices(bins,weights=[remaining[i] for i in bins])[0]
            a,b,wc,bi,k=rng.choice([v for v in options if v[3]==choice]);selected.append((a,b,wc,bi));remaining[bi]-=1;pos=k
    return selected

def pairs(wrapper,evidence,manifest,remaining):
    url=source_url(wrapper)
    if not url:return
    original=wrapper['record'];text=original['text'];regions,overlap=matching_regions(text,evidence['current_article_body'])
    doc='scidev:'+slug(url);raw_hash=digest(text)
    raw={'source_id':'scidev','source_dataset':wrapper['source_dataset'],'source_revision':wrapper['source_revision'],
         'source_file':wrapper['source_file'],'source_row':wrapper['source_row'],'raw_text_sha256':raw_hash,'retrieved_at':manifest['retrieved_at'],
         'record':{'id':original['id'],'text':text,'metadata':{'historical_capture_wrapper':wrapper,
             'current_publisher_evidence':evidence,'current_html_manifest':manifest,'historical_current_overlap':overlap}}}
    for a,b,wc,bi in select_spans(text,regions,doc,remaining.copy()):
        value=text[a:b];words=re.findall('[A-Za-z]+',value)
        if len(words)<35 or sum(w.lower() in STOPWORDS for w in words)/len(words)<.06:continue
        yield {'record_id':digest(doc+str(a)+str(b)),'source_id':'scidev','category':'news','text':value,'word_count':wc,'length_bin':bi,
            'source_dataset':wrapper['source_dataset'],'source_revision':wrapper['source_revision'],'source_file':wrapper['source_file'],
            'source_row':wrapper['source_row'],'original_id':original['id'],'source_url':url,'title':evidence['title'],
            'author_attribution_json':json.dumps([evidence['author']]),'license_evidence':evidence['license_label']+'; '+manifest['url'],
            'claimed_original_date':evidence['claimed_publication_date'],'historical_capture_date':original['metadata']['warc_date'],
            'retrieved_at':manifest['retrieved_at'],'raw_text_sha256':raw_hash,'passage_sha256':digest(value),'raw_start':a,'raw_end':b,
            'offset_unit':'unicode_codepoints','extraction_method':'unchanged_historical_capture_spans_matching_current_publisher_article',
            'parent_document_id':doc,'provisional_family_id':digest(doc),'admission_status':'quarantined_candidate','training_eligible':False,
            'commercial_training_cleared':False,'provenance_basis':'pre2022_CCCC_capture_with_current_article_notice_and_byline_corroboration',
            'protected_overlap_status':'not_fully_audited','reason_codes':['protected_overlap_audit_pending','license_version_unspecified_article_notice',
               'publisher_policy_CC_BY_2_vs_3_discrepancy','byline_and_notice_corroboration_is_current_not_historical','quoted_material_audit_pending'],
            'sampling_seed':27183,'pipeline_version':VERSION},raw

def retrieve(base,wrapper,pacer):
    url=source_url(wrapper);folder=base/'source-downloads/current-html';key=digest(slug(url));path=folder/(key+'.html');mp=folder/(key+'.json')
    if path.exists() and mp.exists():
        body=path.read_bytes();manifest=json.loads(mp.read_text());assert hashlib.sha256(body).hexdigest()==manifest['sha256']
    else:
        for attempt in range(3):
            pacer.before();r=requests.get(url,timeout=(12,40))
            if r.status_code==429:
                pacer.limited(r);r.close();continue
            r.raise_for_status();pacer.success();break
        else:raise RuntimeError('publisher rate-limit retry bound reached')
        if urlparse(r.url).hostname not in HOSTS or slug(r.url)!=slug(url):raise ValueError('unexpected article redirect')
        body=r.content
        if len(body)>3000000:raise ValueError('unexpected HTML size')
        path.write_bytes(body);manifest={'url':url,'resolved_url':r.url,'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body),
                                       'path':str(path.relative_to(base)),'retrieved_at':now(),'role':'current_corroboration_only'}
        atomic_json(mp,manifest)
    return current_evidence(body,manifest['resolved_url']),manifest

def snapshot_candidates(base,aux):
    db=sqlite3.connect('file:'+str(aux)+'?mode=ro',uri=True);db.execute('BEGIN');values=list(db.execute('SELECT id,raw FROM auxiliary_scidev'));db.rollback();db.close()
    path=base/'source-downloads'/('historical-wrappers-'+str(int(time.time()))+'.jsonl.gz');wrappers=[];seen=set()
    with gzip.open(path,'wt') as out:
        for identity,raw in values:
            w=json.loads(gzip.decompress(raw));url=source_url(w)
            if not url:continue
            out.write(json.dumps(w,ensure_ascii=False)+'\n')
            if slug(url) in seen:continue
            seen.add(slug(url));wrappers.append(w)
    atomic_json(path.with_suffix('.manifest.json'),{'upstream_auxiliary':str(aux),'rows_read':len(values),'unique_news_articles':len(wrappers),
             'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'retrieved_at':now()})
    return sorted(wrappers,key=lambda w:digest('27183'+source_url(w)))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--aux-db',type=Path,required=True)
    p.add_argument('--quota',type=int,default=991);p.add_argument('--workers',type=int,default=4);args=p.parse_args();base=args.base
    (base/'source-downloads/current-html').mkdir(parents=True,exist_ok=True);(base/'progress').mkdir(exist_ok=True);began=time.monotonic()
    with (base/'scidev.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);db=connect(base/'stage.sqlite3');config=json.dumps({'version':VERSION,'quota':args.quota},sort_keys=True)
        old=db.execute("SELECT value FROM settings WHERE key='scidev_stage'").fetchone()
        if old and old[0]!=config:raise ValueError('stage config changed')
        with db:db.execute("INSERT OR IGNORE INTO settings VALUES ('scidev_stage',?)",(config,))
        candidates=snapshot_candidates(base,args.aux_db);targets=apportion(args.quota,{str(i):v for i,v in enumerate(LENGTH_WEIGHTS['news'])});reasons=Counter();errors=[];pacer=PublisherPacer(base)
        count=db.execute("SELECT count(*) FROM passages WHERE source='scidev'").fetchone()[0]
        initial_bins=dict(db.execute("SELECT bin,count(*) FROM passages WHERE source='scidev' GROUP BY bin"))
        if all(initial_bins.get(i,0)>=targets[str(i)] for i in range(3)):
            candidates=[w for w in candidates if len(w['record']['text'].split())>=1000]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for start in range(0,len(candidates),args.workers):
                if count>=args.quota:break
                batch=[w for w in candidates[start:start+args.workers] if not db.execute("SELECT 1 FROM cursors WHERE source='scidev' AND file=? AND done=1",(slug(source_url(w)),)).fetchone()]
                if pacer.consecutive_limits>=3:break
                futures=[pool.submit(retrieve,base,w,pacer) for w in batch]
                for w,future in zip(batch,futures):
                    try:
                        evidence,manifest=future.result();bins=dict(db.execute("SELECT bin,count(*) FROM passages WHERE source='scidev' GROUP BY bin"));remaining=[max(0,targets[str(i)]-bins.get(i,0)) for i in range(4)]
                        results=list(pairs(w,evidence,manifest,remaining));reasons['matched_articles' if results else 'no_eligible_spans']+=1
                        db.execute('BEGIN IMMEDIATE')
                        try:
                            for row,raw in results:
                                if add(db,row,raw,args.quota):count+=1
                            db.execute("INSERT OR REPLACE INTO cursors VALUES ('scidev',?,1,1)",(slug(source_url(w)),));db.commit()
                        except BaseException:db.rollback();raise
                    except Exception as exc:errors.append({'url':source_url(w),'error':str(exc)[:180]})
                atomic_json(base/'progress/scidev.json',{'state':'staging','count':count,'target':args.quota,'position':start,'discovered':len(candidates),'length_counts':dict(db.execute("SELECT bin,count(*) FROM passages WHERE source='scidev' GROUP BY bin")),'reasons':dict(reasons),'errors':errors[-3:],'updated_at':now()})
        out=base/'staged/scidev';out.mkdir(parents=True,exist_ok=True)
        for name,query in [('passages',"SELECT row FROM passages WHERE source='scidev' ORDER BY id"),('documents','SELECT raw FROM documents')]:
            with gzip.open(out/(name+'.jsonl.gz'),'wt') as stream:
                for value, in db.execute(query):stream.write((gzip.decompress(value).decode() if name=='documents' else value)+'\n')
        status={'state':'staged_quota_filled' if count==args.quota else ('staged_publisher_rate_limited' if pacer.consecutive_limits>=3 else 'staged_available_capture_shortfall'),'count':count,'target':args.quota,
                'length_counts':dict(db.execute("SELECT bin,count(*) FROM passages WHERE source='scidev' GROUP BY bin")),'discovered':len(candidates),
                'reasons':dict(reasons),'errors':errors,'elapsed_seconds':time.monotonic()-began,'all_quarantined':True,'global_production_dedup_pending':True,'updated_at':now()}
        atomic_json(base/'progress/scidev.json',status);print(json.dumps(status));db.close()

if __name__=='__main__':main()
