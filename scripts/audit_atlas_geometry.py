"""Read-only corpus audit of published extraction maps and renderer assumptions."""
import collections,concurrent.futures,json,math,sqlite3,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from open_text_detector_backend.result_codec import decode
OUT=ROOT/'research/exports/atlas-reader-audit';OUT.mkdir(exist_ok=True)
cat=json.loads((ROOT/'research/exports/atlas-public/catalogue.json').read_text())['items']
by_pdf={p['pdf_key'].split('/')[-1].split('.')[0]:p for p in cat}
c=sqlite3.connect(f'file:{ROOT}/research/extractions/positioned/index.sqlite3?mode=ro',uri=True)
rows=[r for r in c.execute('SELECT pdf_sha256,text_sha256,path FROM artifacts') if r[0] in by_pdf]
def audit(row):
 pdf,textsha,path=row
 try:
  d=decode((ROOT/path).read_bytes());text=d['text'];rects=d['rectangles'];pages=d['pages']; issues=collections.Counter();counts=collections.Counter()
  page_ids={p['page'] for p in pages};last_end=-1
  for p in pages:
   if p['rotation']:issues['rotated_pages']+=1
   if not all(math.isfinite(v) for v in p['crop_box']+p['media_box']):issues['nonfinite_page']+=1
  for r in sorted(rects,key=lambda r:r['start']):
   counts[r['page']]+=1
   if r['end']<last_end:issues['nonmonotonic_ends']+=1
   last_end=r['end']
   if not(0<=r['start']<=r['end']<=len(text)):issues['invalid_offsets']+=1
   if r['page'] not in page_ids:issues['missing_page']+=1
   if not all(math.isfinite(r[k]) for k in ['x0','x1','y0','y1']):issues['nonfinite_rect']+=1
   if r['x1']<=r['x0'] or r['y1']<=r['y0']:issues['empty_rect']+=1
  return {'id':by_pdf[pdf]['id'],'text_sha256':textsha,'pages':len(pages),'rectangles':len(rects),'text_characters':len(text),'densest_page':max(counts,key=counts.get,default=1),'max_page_rectangles':max(counts.values(),default=0),'issues':dict(issues)}
 except Exception as e:return {'id':by_pdf[pdf]['id'],'error':type(e).__name__+': '+str(e)}
results=[];start=time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for n,result in enumerate(pool.map(audit,rows),1):
  results.append(result)
  if n%2000==0:print(f'Audited {n}/{len(rows)} extraction maps',flush=True)
counts=collections.Counter()
for r in results:counts.update(r.get('issues',{}))
summary={'maps':len(results),'papers':len({r['id'] for r in results}),'seconds':round(time.monotonic()-start,1),'issues':dict(counts),'errors':[r for r in results if 'error' in r],'densest':sorted([r for r in results if 'error' not in r],key=lambda r:r['max_page_rectangles'],reverse=True)[:20]}
(OUT/'geometry.json').write_text(json.dumps({'summary':summary,'maps':results}))
print(json.dumps(summary),flush=True)
