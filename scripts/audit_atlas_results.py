"""Audit classification offsets/scores against renderer validation across saved results."""
import concurrent.futures,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from open_text_detector_backend.result_codec import decode
OUT=ROOT/'research/exports/atlas-reader-audit'
lengths={r['text_sha256']:r['text_characters'] for r in json.loads((OUT/'geometry.json').read_text())['maps'] if 'error' not in r}
manifest=json.loads((ROOT/'research/exports/classified-r2/manifest.json').read_text());plan=json.loads((ROOT/'research/exports/classified-r2/upload-plan.json').read_text())
reports={r['result']['sha256']:r for p in manifest['papers'] for r in p['reports'] if r['text_sha256'] in lengths}
def audit(pair):
 sha,r=pair;key=r['result'].get('r2_key','');version='v8' if 'v8' in r['model']['name'].lower() else 'v5'
 try:
  if key in plan:path=Path(plan[key]['path'])
  elif key.startswith('classification-runs/vast-complete-20260925/'):path=ROOT/'research/classifications/vast-complete-20260925/results'/version/(r['text_sha256']+'.pgf')
  else:path=ROOT/'research/benchmarks/meld-cuda/r2-batch'/(r['text_sha256']+'.pgf')
  segments=decode(path.read_bytes())['segments'];end=0
  for p in segments:
   assert isinstance(p['start'],int) and isinstance(p['end'],int) and end<=p['start']<=p['end']<=lengths[r['text_sha256']], 'Invalid/overlapping offsets'
   assert math.isfinite(p['score']) and 0<=p['score']<=1 and isinstance(p['label'],str), 'Invalid score/label'
   end=p['end']
  return {'segments':len(segments)}
 except Exception as e:return {'sha':sha,'error':type(e).__name__+': '+str(e)}
errors=[];segments=0;start=time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for n,r in enumerate(pool.map(audit,reports.items()),1):
  if 'error' in r:errors.append(r)
  else:segments+=r['segments']
  if n%5000==0:print(f'Audited {n}/{len(reports)} classification results',flush=True)
summary={'results':len(reports),'segments':segments,'errors':errors,'seconds':round(time.monotonic()-start,1)}
(OUT/'results.json').write_text(json.dumps(summary));print(json.dumps(summary),flush=True)
