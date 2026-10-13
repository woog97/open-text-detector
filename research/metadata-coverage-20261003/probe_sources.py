import json,time,urllib.request,urllib.parse
from pathlib import Path
out=Path(__file__).parent;root=out.parents[1];papers=json.loads((out/'live-papers.json').read_text())['items'];meta={p['id']:p for p in map(json.loads,(root/'research/data/openreview_catalogue_all/papers.jsonl').open())};samples={}
for p in papers:
 if p['collection'] not in ['Other uploads','iclr/2027']:samples.setdefault(p['collection'],p)
import sys
sys.path.insert(0,str(root/"research/data/openreview_catalogue_all"))
from discover import session
client=session()
rows=[]
for collection,p in sorted(samples.items()):
 fid=p.get('forum_id') or p['filename'].removesuffix('.pdf');version=meta.get(fid,{}).get('api_version',2);base='https://api.openreview.net' if version==1 else 'https://api2.openreview.net';url=base+'/notes?'+urllib.parse.urlencode({'id':fid})
 try:
  r=client.get(url,timeout=25);r.raise_for_status();data=r.json()
  notes=data.get('notes',[]);n=notes[0] if notes else {};fields={}
  for k,v in n.get('content',{}).items():
   public='everyone' in n.get('readers',[]) and (not isinstance(v,dict) or 'readers' not in v or 'everyone' in v['readers'])
   if public:fields[k]=v.get('value') if isinstance(v,dict) else v
  row={'collection':collection,'forum_id':fid,'api_version':version,'url':url,'public':bool(fields),'abstract':bool(fields.get('abstract')),'tldr_fields':[k for k in fields if k.lower().replace(';','').replace('_','')=='tldr' and fields[k]],'tag_fields':[k for k in fields if any(s in k.lower() for s in ['keyword','subject','area']) and fields[k]]}
  (out/(collection.replace('/','-')+'-sample.json')).write_text(json.dumps(data))
 except Exception as e:row={'collection':collection,'forum_id':fid,'url':url,'error':type(e).__name__}
 rows.append(row);print(json.dumps(row),flush=True);time.sleep(.25)
(out/'source-samples.json').write_text(json.dumps(rows,indent=2))
