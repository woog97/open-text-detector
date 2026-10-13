"""Pinned pre2022 official revisions for untouched Wikivoyage articles."""
import argparse,gzip,json,hashlib,re,sys,time,sqlite3,fcntl,threading
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
from collect_pool import digest,now,atomic_json
from expand_pool import connect,add
from recover_cached_spans import select_windows
API='https://en.wikivoyage.org/w/api.php';CUTOFF='2021-12-31T23:59:59Z';VERSION='wikivoyage-official-history-v1'

def title_key(title):return str(title).replace('_',' ').strip().casefold()

def extract_wikitext(text):
    import mwparserfromhell
    code=mwparserfromhell.parse(text)
    # Templates are deliberately not expanded using current versions. Images,
    # references, tables and listings are not treated as authored travel prose.
    for node in list(code.filter_templates(recursive=False)):
        try:code.remove(node)
        except ValueError:pass
    for node in list(code.filter_tags(recursive=False)):
        if str(node.tag).casefold() in {'ref','references','table','gallery','imagemap','mapframe','maplink','timeline','math','source','syntaxhighlight'}:
            try:code.remove(node)
            except ValueError:pass
    for node in list(code.filter_wikilinks(recursive=False)):
        if re.match(r'^\s*(?:file|image|category):',str(node.title),re.I):
            try:code.remove(node)
            except ValueError:pass
    return code.strip_code(normalize=True,collapse=False)

class Fetcher:
    def __init__(self,base):
        self.base=base;self.lock=threading.Lock();self.next=0;self.limits=0;self.backoff=base/'progress/api-backoff.json'
        if self.backoff.exists():self.next=json.loads(self.backoff.read_text()).get('retry_at_unix',0)
    def get(self,title):
        import requests
        path=self.base/'original-revisions'/(digest(title)+'.json');meta=path.with_suffix('.manifest.json')
        if path.exists() and meta.exists():
            body=path.read_bytes();m=json.loads(meta.read_text());assert hashlib.sha256(body).hexdigest()==m['sha256'];return json.loads(body),m
        params={'action':'query','prop':'revisions','titles':title,'rvstart':CUTOFF,'rvdir':'older','rvlimit':1,'rvprop':'ids|timestamp|user|sha1|content','rvslots':'main','format':'json','formatversion':2}
        for attempt in range(3):
            with self.lock:
                if self.limits>=3:raise RuntimeError('API rate-limit circuit open')
                wait=self.next-time.time()
                if wait>0:time.sleep(wait)
                self.next=time.time()+5
                r=requests.get(API,params=params,headers={'User-Agent':'OpenTextDetectorResearch/1.0 (historical travel text provenance audit)'},timeout=(15,60))
                if r.status_code==429:
                    from email.utils import parsedate_to_datetime
                    retry=r.headers.get('Retry-After','');delay=300
                    try:delay=max(delay,float(retry))
                    except ValueError:
                        try:delay=max(delay,parsedate_to_datetime(retry).timestamp()-time.time())
                        except (ValueError,TypeError):pass
                    self.limits+=1;self.next=time.time()+delay;atomic_json(self.backoff,{'http_status':429,'retry_after':retry,'retry_at_unix':self.next,'consecutive_limits':self.limits});r.close();continue
                r.raise_for_status();self.limits=0;v=r.json();break
        else:raise RuntimeError('API rate-limit retry bound reached')
        if 'error' in v:raise ValueError(v['error'].get('code'))
        body=r.content;path.write_bytes(body);m={'url':r.url,'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body),'path':str(path.relative_to(self.base)),'retrieved_at':now()};atomic_json(meta,m);return v,m

def make_pair(value,manifest,remaining):
    pages=value.get('query',{}).get('pages',[])
    if len(pages)!=1:return None
    p=pages[0]
    if p.get('ns')!=0 or not p.get('revisions'):return None
    rev=p['revisions'][0];slot=rev.get('slots',{}).get('main',{});wiki=slot.get('content','')
    if rev.get('timestamp','Z')>CUTOFF or slot.get('contentmodel')!='wikitext' or re.match(r'^\s*#redirect',wiki,re.I):return None
    text=extract_wikitext(wiki);doc='wikivoyage:pageid:'+str(p['pageid']);windows=select_windows(text,doc,1,[0,0,0,remaining],[])
    if not windows:return None
    a,b,wc,bi=windows[0];raw_hash=digest(text);url='https://en.wikivoyage.org/w/index.php?oldid='+str(rev['revid']);history='https://en.wikivoyage.org/w/index.php?title='+quote(p['title'])+'&action=history'
    raw={'source_id':'wikivoyage','source_dataset':'en.wikivoyage.org historical revision API','source_revision':str(rev['revid']),'source_file':manifest['path'],'source_row':0,'retrieved_at':manifest['retrieved_at'],'raw_text_sha256':raw_hash,
         'record':{'id':str(p['pageid']),'text':text,'metadata':{'namespace':'0','title':p['title'],'url':url,'license':'https://creativecommons.org/licenses/by-sa/3.0/','revision':rev,'original_api_response':value,'api_manifest':manifest,'extraction':'mwparserfromhell0.7.2 strip_code with templates/images/references/tables removed; no live template expansion','revision_history_attribution':history}}}
    row={'record_id':digest(doc+str(a)+str(b)),'source_id':'wikivoyage','category':'general_web','text':text[a:b],'word_count':wc,'length_bin':bi,'source_dataset':raw['source_dataset'],'source_revision':raw['source_revision'],'source_file':raw['source_file'],'source_row':0,'original_id':str(p['pageid']),'source_url':url,'title':p['title'],'author_attribution_json':json.dumps({'collective':'Wikivoyage contributors','revision_history':history,'revision_editor_not_sole_author':rev.get('user')}),'license_evidence':'https://creativecommons.org/licenses/by-sa/3.0/; historical Wikivoyage:Copyleft revision4065344','claimed_original_date':rev['timestamp'],'date_evidence_basis':'official_frozen_revision_timestamp','retrieved_at':manifest['retrieved_at'],'raw_text_sha256':raw_hash,'passage_sha256':digest(text[a:b]),'raw_start':a,'raw_end':b,'offset_unit':'unicode_codepoints','extraction_method':'contiguous_window_of_deterministic_historical_wikitext_extraction','parent_document_id':doc,'provisional_family_id':digest('wikivoyage:'+title_key(p['title'])),'admission_status':'quarantined_candidate','training_eligible':False,'provenance_basis':'official_pre2022_revision_and_historical_license_policy','protected_overlap_status':'not_fully_audited','reason_codes':['protected_overlap_audit_pending','wikitext_extraction_review_pending','contributor_attribution_via_revision_history'],'sampling_seed':27183,'pipeline_version':VERSION}
    return row,raw

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--vendor',type=Path,required=True);p.add_argument('--production-db',type=Path,required=True);p.add_argument('--audit-base',type=Path,required=True);args=p.parse_args();sys.path.insert(0,str(args.vendor));base=args.base;(base/'original-revisions').mkdir(parents=True,exist_ok=True);(base/'progress').mkdir(exist_ok=True)
    with (base/'wikivoyage.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prod=sqlite3.connect('file:'+str(args.production_db)+'?mode=ro',uri=True);prod.execute('BEGIN');baseline=[json.loads(r[0]) for r in prod.execute("SELECT row FROM passages WHERE source='wikivoyage'")];norms={r[0] for r in prod.execute('SELECT norm FROM passages')};prod.rollback();prod.close()
        used={title_key(r['title']) for r in baseline};needed=1231-len(baseline);assert 0<=needed<=110
        db=connect(base/'stage.sqlite3');count=db.execute('SELECT count(*) FROM passages').fetchone()[0]
        for v, in db.execute('SELECT row FROM passages'):used.add(title_key(json.loads(v)['title']))
        (base/'baseline.json.gz').write_bytes(gzip.compress(json.dumps(baseline).encode()));status=json.loads((args.audit_base/'wikivoyage/status.json').read_text());candidates={}
        for info in status['files']:
            with gzip.open(args.audit_base/'wikivoyage'/info['filename'],'rt') as f:
                for line in f:
                    r=json.loads(line);m=r.get('metadata') or {};title=m.get('title','');key=title_key(title)
                    if str(m.get('namespace'))=='0' and key not in used and ':' not in title and len(r.get('text','').split())>=1800:candidates[key]=title
        titles=sorted(candidates.values(),key=lambda t:digest('27183'+t));fetcher=Fetcher(base);errors=[];reasons=Counter();began=time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as pool:
            for start in range(0,len(titles),2):
                if count>=needed:break
                if fetcher.limits>=3:break
                batch=titles[start:start+2];futures=[pool.submit(fetcher.get,t) for t in batch]
                for title,future in zip(batch,futures):
                    if count>=needed:break
                    try:
                        value,manifest=future.result();pair=make_pair(value,manifest,needed-count)
                        if not pair:reasons['no_eligible_long_historical_prose']+=1;continue
                        row,raw=pair;key=title_key(row['title']);norm=digest(' '.join(row['text'].casefold().split()))
                        if key in used or norm in norms:reasons['duplicate_page_or_text']+=1;continue
                        with db:inserted=add(db,row,raw,needed)
                        if inserted:count+=1;used.add(key);norms.add(norm)
                    except Exception as exc:errors.append({'title':title,'error':str(exc)[:180]})
                atomic_json(base/'progress/wikivoyage.json',{'state':'staging','count':count,'target_new_rows':needed,'baseline':len(baseline),'position':start,'candidates':len(titles),'reasons':dict(reasons),'errors':errors[-3:],'updated_at':now()})
        out=base/'staged/wikivoyage';out.mkdir(parents=True,exist_ok=True)
        for name,query in [('passages','SELECT row FROM passages'),('documents','SELECT raw FROM documents')]:
            with gzip.open(out/(name+'.jsonl.gz'),'wt') as f:
                for value, in db.execute(query):f.write((gzip.decompress(value).decode() if name=='documents' else value)+'\n')
        result={'state':'staged_quota_filled' if count==needed else 'bounded_candidate_shortfall','count':count,'target_new_rows':needed,'baseline':len(baseline),'source_total':len(baseline)+count,'reasons':dict(reasons),'errors':errors,'elapsed_seconds':time.monotonic()-began,'updated_at':now()};atomic_json(base/'progress/wikivoyage.json',result);print(json.dumps(result));db.close()

if __name__=='__main__':main()
