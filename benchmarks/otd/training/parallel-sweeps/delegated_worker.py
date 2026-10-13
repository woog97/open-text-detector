import runner as r
import json,time,traceback,sys

def state(**kw):
 r.save(r.ROOT/'gpu3-status.json',dict(time=time.time(),**kw));print(json.dumps(kw),flush=True)
r.status=state
try:
 plan=json.loads((r.ROOT/'plan.json').read_text());spec=next(s for s in plan['trials'] if s['name']==plan['gpu3_trial'])
 root=r.prepare(spec)
 if not (root/'READY.json').exists():r.command(root,'preflight',['preflight.py','--models',spec['flow']])
 r.command(root,'training',['train.py','--flow',spec['flow'],'--output',str(root/'run')])
 state(state='complete',trial=spec['name'])
except Exception as e:state(state='failed',error=str(e));traceback.print_exc();sys.exit(1)
