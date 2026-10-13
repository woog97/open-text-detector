from pathlib import Path
import gzip,json,collections
R=Path('/data/workspace/paper-diversity-v1');res={}
for name in ['objective-document-control-v1','gradtex-document10-v1']:
 counts={}
 for p in sorted((R/name/'prepared').glob('stage2*.gz')):
  counter=collections.Counter()
  for r in map(json.loads,gzip.decompress(p.read_bytes()).splitlines()):
   if r.get('supervision')=='document_only':counter['external_'+str(r['document_label'])]+=1;continue
   regions=r['regions'];known=all(z['label'] in (0,1) for z in regions);end=0
   for a,b in sorted((z['start'],z['end']) for z in regions):
    if r['text'][end:a].strip():known=False
    end=max(end,b)
   if r['text'][end:].strip():known=False
   counter['paper_'+(str(int(any(z['label']==1 for z in regions))) if known else 'unknown')]+=1
  counts[p.name]=dict(counter)
 res[name]=counts
print(json.dumps(res))
(R/'gradtex-document-objective/document-label-counts.json').write_text(json.dumps(res,indent=2))
for n in ['objective-document-control-v1','gradtex-document10-v1']:
 p=R/(n+'-status.json');print(n,p.read_text() if p.exists() else 'queued')
