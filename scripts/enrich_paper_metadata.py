"""Fetch public OpenReview discovery metadata, then publish it independently of PDF ingestion."""
import argparse, collections, concurrent.futures, hashlib, json, re, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/metadata-coverage-20261003/enrichment'
CAT=ROOT/'research/data/openreview_catalogue_all'
sys.path.insert(0,str(CAT))

def clean(value):
    return value.strip() if isinstance(value,str) else ''

def normalize(note):
    if 'everyone' not in note.get('readers',[]): return None
    content={k:(v.get('value') if isinstance(v,dict) else v) for k,v in note.get('content',{}).items() if not isinstance(v,dict) or 'readers' not in v or 'everyone' in v['readers']}
    def text(key): return clean(content.get(key))
    tags=[];areas=[];tldr=''
    for key,value in content.items():
        name=key.lower()
        if name.replace(';','').replace('_','')=='tldr': tldr=clean(value)
        if 'keyword' in name or name in ('subject_areas','research_area','primary_area','please_choose_the_closest_area_that_your_submission_falls_into'):
            values=value if isinstance(value,list) else (re.split(r'[,;\n]',value) if 'keyword' in name else [value]) if isinstance(value,str) else []
            values=[clean(v) for v in values if clean(v)]
            (tags if 'keyword' in name else areas).extend(values)
    tags=list(dict.fromkeys(tags+areas))
    return dict(id=note['id'],forum_id=note.get('forum') or note['id'],number=note.get('number'),title=text('title'),keywords=tags,primary_area=areas[0] if areas else '',tldr=tldr,abstract_preview=text('abstract')[:420],updated=note.get('mdate') or note.get('tmdate'),detail=dict(abstract=text('abstract'),bibtex=text('_bibtex'),license=text('license'),venue=text('venue')))

def identities(live,catalogue):
    titles=collections.defaultdict(list)
    def key(title):return re.sub(r'[^a-z0-9]','',title.lower())
    for n in catalogue.values():titles[(f"{n['conference']}/{n['year']}",key(n['title']))].append(n['id'])
    result={}
    for p in live:
        fid=p.get('forum_id') or p['filename'].removesuffix('.pdf')
        matches=titles.get((p['collection'],key(p['title'])),[])
        if fid not in catalogue and len(matches)==1:fid=matches[0]
        result[p['id']]=fid
    return result

def fetch():
    from discover import session
    OUT.mkdir(parents=True,exist_ok=True);cache=OUT/'queries-v2';cache.mkdir(exist_ok=True)
    live=json.loads((OUT.parent/'live-papers.json').read_text())['items']
    catalogue={p['id']:p for p in map(json.loads,(CAT/'papers.jsonl').open())}
    ids=identities(live,catalogue)
    wanted={ids[p['id']] for p in live if p['collection'] not in ('iclr/2027','Other uploads')}
    (OUT/'identities.json').write_text(json.dumps(ids))
    venues={v for fid in wanted for v in catalogue.get(fid,{}).get('source_venues',[])}
    queries=set()
    for venue in venues:
        report=json.loads((CAT/'venues'/venue/'report.json').read_text());version=report['api_version']
        for q in report['queries']:
            if 'invitation' in q: queries.add((version,'invitation',q['invitation']))
            elif q.get('api_count') and q['api_count']!=q.get('matched_count'): queries.add((version,'content.venueid',q['status_venue_id']))
    s=session();found={}
    def get(version,params):
        host='api2.openreview.net' if version==2 else 'api.openreview.net'
        for attempt in range(6):
            r=s.get('https://'+host+'/notes',params=params,timeout=(20,180))
            if r.status_code not in (429,500,502,503,504):r.raise_for_status();return r.json()
            time.sleep(min(60,2**attempt*2))
        r.raise_for_status()
    for version,key,value in sorted(queries):
        path=cache/(hashlib.sha256(json.dumps([version,key,value]).encode()).hexdigest()+'.json')
        if path.exists(): normalized=json.loads(path.read_text())
        else:
            params={key:value,'select':'id,forum,number,readers,mdate,tmdate,content','limit':1000,'offset':0}
            normalized=[]
            while True:
                data=get(version,params);notes=data.get('notes',[])
                for n in notes:
                    if n['id'] in wanted:
                        row=normalize(n)
                        if row:normalized.append(row)
                params['offset']+=len(notes)
                if params['offset']>=data.get('count',0) or len(notes)<1000:break
                if not notes:raise RuntimeError('Incomplete metadata pagination')
                time.sleep(.2)
            path.write_text(json.dumps(normalized,ensure_ascii=False))
        for row in normalized:
            if (row.get('updated') or 0)>=(found.get(row['id'],{}).get('updated') or 0):found[row['id']]=row
        print(value,len(normalized),'matched',len(found),'/',len(wanted),flush=True)
    # Covers changed invitations and individual legacy records.
    for fid in sorted(wanted-found.keys()):
        path=cache/(fid+'.json')
        if path.exists():rows=json.loads(path.read_text())
        else:
            if fid not in catalogue:
                print('Unresolved source identity:',fid,flush=True);continue
            data=get(catalogue[fid].get('api_version',2),{'id':fid})
            rows=[row for n in data.get('notes',[]) if (row:=normalize(n))]
            path.write_text(json.dumps(rows,ensure_ascii=False));time.sleep(.2)
        for row in rows:found[row['id']]=row
    (OUT/'notes.json').write_text(json.dumps(found,ensure_ascii=False))
    print('Public metadata:',len(found),'of',len(wanted),flush=True)

def publish():
    from upload_paper_pdfs import BASE,request
    from publish_browse_snapshot import main
    notes=json.loads((OUT/'notes.json').read_text())
    manifest=request(BASE+'/indexes/browse/current.json');live=request(BASE+'/'+manifest['index_key'])['items']
    catalogue={p['id']:p for p in map(json.loads,(CAT/'papers.jsonl').open())};ids=identities(live,catalogue)
    pdf_ids={p['id']:p['pdf_key'][7:31] for p in request(BASE+'/atlas-public/catalogue.json')['items']}
    rows=[];shards=collections.defaultdict(dict);counts=collections.defaultdict(collections.Counter)
    for p in live:
        n=notes.get(ids[p['id']])
        if n:
            row={k:v for k,v in n.items() if k not in ('id','detail')};canonical=pdf_ids.get(p['id'],p['id']);row.update(id=canonical,collection=p['collection'])
            rows.append(row);shards[canonical[:2]][canonical]=n['detail']
        else:row=p
        c=counts[p['collection']];c['papers']+=1;c['tags']+=bool(row.get('keywords') or row.get('primary_area'));c['tldr']+=bool(clean(row.get('tldr')))
    def put(item):
        key,value=item;request(BASE+'/'+key,json.dumps(value,ensure_ascii=False,separators=(',',':')).encode(),'application/json');assert request(BASE+'/'+key)==value
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(put,[(f'indexes/paper-details/{prefix}.json',value) for prefix,value in shards.items()]))
    put(('indexes/paper-discovery.json',{'papers':rows}))
    (OUT/'coverage.json').write_text(json.dumps(counts,indent=2))
    main(enrich=True)
    print(json.dumps(counts,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['fetch','publish']);args=parser.parse_args()
    fetch() if args.action=='fetch' else publish()
