"""Fetch source-level evidence once for every source in the approved plan."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
import hashlib,json,re
import requests

ROOT=Path(__file__).resolve().parent
class Text(HTMLParser):
    def __init__(self):super().__init__();self.parts=[];self.skip=0
    def handle_starttag(self,tag,attrs):
        if tag in ('script','style'):self.skip+=1
    def handle_endtag(self,tag):
        if tag in ('script','style'):self.skip=max(0,self.skip-1)
    def handle_data(self,s):
        if not self.skip and s.strip():self.parts.append(s.strip())

def main():
    plan=json.loads((ROOT/'sampling-plan.json').read_text());ids={x['source_id'] for x in plan['source_quotas']}
    sources=[s for s in json.loads((ROOT/'source-registry.json').read_text())['sources'] if s['id'] in ids]
    assert len(sources)==len(ids)
    folder=ROOT/'evidence'/'source-review-20261002';folder.mkdir(exist_ok=True)
    urls=sorted({url for s in sources for url in s['rights_evidence_urls']})
    def fetch(url):
        key=hashlib.sha256(url.encode()).hexdigest()[:16];rec={'url':url,'retrieved_at':datetime.now(timezone.utc).isoformat()}
        try:
            with requests.get(url,timeout=(15,35),stream=True,headers={'User-Agent':'OpenTextDetector-source-research/1.0'}) as r:
                rec.update(http_status=r.status_code,final_url=r.url,content_type=r.headers.get('Content-Type',''))
                r.raise_for_status();parts=[];n=0
                for part in r.iter_content(65536):
                    n+=len(part)
                    if n>5_000_000:raise ValueError('Evidence exceeds metadata size limit')
                    parts.append(part)
                body=b''.join(parts)
            ext='.pdf' if body.startswith(b'%PDF') else '.txt'
            path=folder/(key+ext);path.write_bytes(body)
            rec.update(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(body).hexdigest(),bytes=len(body),status='retrieved')
            if ext!='.pdf':
                value=body.decode('utf-8',errors='replace')
                if '<html' in value.lower() or '<!doctype' in value.lower():
                    parser=Text();parser.feed(value);value='\n'.join(parser.parts)
                clean=folder/(key+'.readable.txt');clean.write_text(value);rec['readable_path']=str(clean.relative_to(ROOT))
        except Exception as e:rec.update(status='unavailable',error=str(e)[:220])
        return rec
    with ThreadPoolExecutor(max_workers=6) as pool:records=list(pool.map(fetch,urls))
    out={'reviewed_source_ids':sorted(ids),'evidence':records}
    (folder/'manifest.json').write_text(json.dumps(out,indent=2))
    byurl={r['url']:r for r in records}
    for s in sources:
        print(s['id'],json.dumps([{'url':u,'status':byurl[u]['status'],'path':byurl[u].get('readable_path',byurl[u].get('path'))} for u in s['rights_evidence_urls']]))
if __name__=='__main__':main()
