from pathlib import Path
import subprocess,json,sys,time
r=Path(__file__).resolve().parent;p=r/'profiling-v2';o=r/'learning-check-v2';o.mkdir(exist_ok=True)
while True:
 s=json.loads((p/'driver-status.json').read_text())
 if s['state']=='measurements_complete':break
 time.sleep(15)
selected={}
for stage in [1,2]:
 cases=[json.loads(x.read_text()) for x in p.glob(f'stage{stage}-micro*-checkpoint0.json')]
 cases=[x for x in cases if x.get('bf16_verified') and x.get('finite_gradients') and x.get('frozen_verified') and x['peak_allocated_bytes']<.70*x['gpu_total_bytes']]
 assert cases,'No safe isolated-kernel candidate'
 selected[stage]=min(cases,key=lambda x:x['seconds_per_32_examples'])['micro_batch']
(o/'selected.json').write_text(json.dumps(selected,indent=2))
for stage in [1,2]:
 for fast,micro in [(0,1),(1,selected[stage]),(2,selected[stage])]:
  name=f'stage{stage}-'+('fast-bf16' if fast==2 else 'fast' if fast else 'baseline')+f'-micro{micro}'
  if (o/(name+'.json')).exists():continue
  (o/'status.json').write_text(json.dumps({'state':'running','case':name,'time':time.time()}))
  with (o/(name+'.log')).open('w') as log:code=subprocess.call([sys.executable,str(r/'learning_check_v2.py'),str(fast),str(stage),str(micro)],stdout=log,stderr=subprocess.STDOUT)
  if code:
   (o/'status.json').write_text(json.dumps({'state':'failed','case':name,'returncode':code}));sys.exit(code)
(o/'status.json').write_text(json.dumps({'state':'complete','time':time.time()}))
