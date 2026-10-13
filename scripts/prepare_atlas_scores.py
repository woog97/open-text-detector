"""Prepare catalogue-only score summaries from the published report fingerprints."""
import hashlib,json,sys,sqlite3
import pyarrow.parquet as pq
from pathlib import Path
from atlas_scores import summarize
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from open_text_detector_backend.result_codec import decode
OUT=ROOT/'research/exports/atlas-public'
ledger=json.loads((OUT/'ledger.json').read_text())
manifest=json.loads((ROOT/'research/exports/classified-r2/manifest.json').read_text())
plan=json.loads((ROOT/'research/exports/classified-r2/upload-plan.json').read_text())
conn=sqlite3.connect(f"file:{ROOT}/research/extractions/positioned/index.sqlite3?mode=ro",uri=True)
patch={}
cached=json.loads((OUT/'score-patch.json').read_text()) if (OUT/'score-patch.json').exists() else {}
by_pdf={p['pdf_sha256']:p for p in manifest['papers']}
def sources():
 for batch in pq.ParquetFile(ROOT/'research/exports/paper-text-hf/data/batch-001.parquet').iter_batches(batch_size=32,columns=['pdf_sha256','text_sha256','text']):
  yield from batch.to_pylist()
for n,source in enumerate(sources(),1):
 p=by_pdf[source['pdf_sha256']]
 pdf=p['pdf_sha256'];selected={}
 for r in p['reports']:
  if r['source_map']['mapping_status']!='complete' or r['source_map']['coverage']!=1:continue
  version='v8' if 'v8' in r['model']['name'].lower() else 'v5' if 'v5' in r['model']['name'].lower() else None
  if version and (version not in selected or r.get('created_at','')>selected[version].get('created_at','')):selected[version]=r
 selected={v:r for v,r in selected.items() if v in ledger[pdf]['item']['models']}
 fingerprint=hashlib.sha256(json.dumps({v:r['result']['sha256'] for v,r in selected.items()},sort_keys=True).encode()).hexdigest()
 assert ledger[pdf]['fingerprint']==fingerprint, 'Report snapshot changed; rerun after publication'
 item=ledger[pdf]['item'];prior=cached.get(item['id'])
 if prior and all(s.get('policy')==2 for s in prior['score_summaries'].values()) and prior['fingerprint']==fingerprint and prior['detail_key']==item['detail_key']:
  patch[item['id']]=prior
  continue
 stats={}
 for version,r in selected.items():
  ref=r['result'];key=ref.get('r2_key','')
  if key in plan:path=Path(plan[key]['path'])
  elif key.startswith('classification-runs/vast-complete-20260925/'):path=ROOT/'research/classifications/vast-complete-20260925/results'/version/(r['text_sha256']+'.pgf')
  else:path=ROOT/'research/benchmarks/meld-cuda/r2-batch'/(r['text_sha256']+'.pgf')
  raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256']
  text=source['text']
  if source['text_sha256']!=r['text_sha256']:
   row=conn.execute('SELECT path FROM artifacts WHERE pdf_sha256=? AND text_sha256=?',(pdf,r['text_sha256'])).fetchone()
   assert row, 'Missing matching extraction'
   text=decode((ROOT/row[0]).read_bytes())['text']
  assert hashlib.sha256(text.encode()).hexdigest()==r['text_sha256']
  stats[version]=summarize(decode(raw)['segments'],text)
 item=ledger[pdf]['item'];patch[item['id']]={'pdf_sha256':pdf,'detail_key':item['detail_key'],'fingerprint':fingerprint,'score_summaries':stats}
 if n%2000==0:print(f'Summarized {n}/{len(manifest["papers"])} papers',flush=True)
(OUT/'score-patch.json').write_text(json.dumps(patch,separators=(',',':')))
cat=json.loads((OUT/'catalogue.json').read_text())
# Preview uses the captured ledger snapshot; publication rechecks the live catalogue.
cat['items']=[dict(entry['item']) for entry in ledger.values()]
for item in cat['items']:
 p=patch[item['id']];assert item['detail_key']==p['detail_key'];item['score_summaries']=p['score_summaries']
for model in cat.get('models',[]):
 model['available']=sum(model['id'] in item['models'] for item in cat['items'])
 model['complete']=model['available']==len(cat['items'])
(OUT/'catalogue-with-scores.json').write_text(json.dumps(cat,separators=(',',':')))
print('Prepared summaries:',len(patch),flush=True)
