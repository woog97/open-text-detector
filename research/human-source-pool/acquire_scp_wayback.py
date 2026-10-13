"""Bounded remote-only historical SCP acquisition; never admits text."""
import requests,json,time,hashlib,base64,sqlite3
from pathlib import Path
from bs4 import BeautifulSoup
from email.utils import parsedate_to_datetime
ROOT=Path('/tmp/pangram-scp-cccc-staging-20261002');OUT=Path('/tmp/pangram-scp-wayback-20261002')
def main():
 OUT.mkdir(exist_ok=True);idx=json.loads((ROOT/'tales-index.json').read_text());db=sqlite3.connect('file:'+str(ROOT/'collection.sqlite3')+'?mode=ro',uri=True);used={x[0] for x in db.execute('SELECT DISTINCT doc FROM passages')};db.close()
 selected=sorted([k for k,v in idx.items() if v.get('domain')=='scp-wiki.wikidot.com' and 'tale' in v.get('tags',[]) and v.get('creator') not in (None,'deleted','unknown') and str(v.get('created_at','9999'))<'2022' and 'scp:'+str(v.get('page_id')) not in used],key=lambda k:hashlib.sha256(k.encode()).hexdigest())[:250]
 (OUT/'inventory.json').write_text(json.dumps({'slugs':selected,'scope':'250 deterministic unused pre2022-created pinned English tales','source_index_sha256':hashlib.sha256((ROOT/'tales-index.json').read_bytes()).hexdigest()}));done=good=errors=0
 for slug in selected:
  p=OUT/hashlib.sha256(slug.encode()).hexdigest()[:24];p.mkdir(exist_ok=True)
  if (p/'receipt.json').exists():continue
  try:
   time.sleep(2);r=requests.get('https://web.archive.org/cdx/search/cdx',params={'url':'scp-wiki.wikidot.com/'+slug,'output':'json','filter':'statuscode:200','from':idx[slug]['created_at'][:10].replace('-',''),'to':'20211231','collapse':'digest','fl':'timestamp,original,mimetype,statuscode,digest','limit':'-1'},timeout=(15,45));(p/'cdx.response').write_bytes(r.content);r.raise_for_status();rows=r.json()
   if len(rows)<2:receipt={'slug':slug,'state':'no_capture_in_bounded_query'}
   else:
    item=dict(zip(rows[0],rows[-1]));assert item['timestamp']<'20220101' and item['timestamp']>=idx[slug]['created_at'][:19].replace('-','').replace(':','').replace('T','');time.sleep(2)
    cap=requests.get('https://web.archive.org/web/'+item['timestamp']+'id_/'+item['original'],timeout=(15,45));(p/'capture.html').write_bytes(cap.content);cap.raise_for_status();assert base64.b32encode(hashlib.sha1(cap.content).digest()).decode().rstrip('=')==item['digest'];md=parsedate_to_datetime(cap.headers['Memento-Datetime']);assert md.year<2022
    soup=BeautifulSoup(cap.content,'html.parser');region=soup.select_one('#page-content');assert region is not None;licenses=sorted({a['href'] for a in soup.select('a[href]') if 'creativecommons.org/licenses/by-sa/3.0' in a['href']});assert licenses
    receipt={'slug':slug,'state':'verified_original_capture','cdx':item,'memento_datetime':cap.headers['Memento-Datetime'],'sha256':hashlib.sha256(cap.content).hexdigest(),'bytes':len(cap.content),'licenses':licenses,'current_metadata':idx[slug],'current_credit_not_independently_historical':True,'candidate_collection_pending':True};good+=1
   (p/'receipt.json').write_text(json.dumps(receipt,indent=2));errors=0
  except Exception as e:
   errors+=1;(p/'error.json').write_text(json.dumps({'slug':slug,'error':type(e).__name__,'time':time.time()}))
  done+=1;(OUT/'progress.json').write_text(json.dumps({'attempted':done,'verified_captures':good,'consecutive_errors':errors,'target_urls':len(selected),'state':'acquiring'}))
  if errors>=3:
   (OUT/'progress.json').write_text(json.dumps({'attempted':done,'verified_captures':good,'state':'backoff_after_three_errors','resume_requires_error_audit':True}));break
 else:(OUT/'progress.json').write_text(json.dumps({'attempted':done,'verified_captures':good,'state':'bounded_inventory_complete'}))
if __name__=='__main__':main()
