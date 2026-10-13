from pathlib import Path
import json,subprocess,sys,time
root=Path(__file__).resolve().parent;out=root/'profiling-v2';out.mkdir(exist_ok=True)
for stage in [1,2]:
 for micro,checkpoint in [(1,0),(2,0),(4,0),(8,0),(16,0),(8,1)]:
  name=f'stage{stage}-micro{micro}-checkpoint{checkpoint}'
  if (out/(name+'.json')).exists():continue
  (out/'driver-status.json').write_text(json.dumps({'state':'profiling','case':name,'time':time.time()}))
  with (out/(name+'.log')).open('w') as log:
   code=subprocess.call([sys.executable,str(root/'profile_lora_v2_case.py'),str(stage),str(micro),str(checkpoint)],stdout=log,stderr=subprocess.STDOUT)
  if code and not (out/(name+'.json')).exists():
   (out/(name+'.json')).write_text(json.dumps({'case':name,'status':'process_failed','returncode':code}))
(out/'driver-status.json').write_text(json.dumps({'state':'measurements_complete','time':time.time()}))
