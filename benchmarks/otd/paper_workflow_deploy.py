"""Run frozen calibration/test generation and deterministic exports; persist lifecycle."""
import json, subprocess, sys
from pathlib import Path
import paper_workflow_eval as w

def main():
 assert (w.OUT/'main-freeze.json').exists(),'Main generation requires a reviewed pilot and frozen protocol'
 results=[]
 try:
  for split in ['calibration','test']:
   w.save('deployment-status.json',{'at_utc':w.b.now(),'state':'running','stage':split,'completed_stages':results})
   with (w.OUT/(split+'.log')).open('a') as log:
    p=subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('paper_workflow_eval.py')),split],stdout=log,stderr=subprocess.STDOUT)
   results.append({'stage':split,'exit_code':p.returncode})
   if p.returncode:raise RuntimeError(split+' generation has failures; inspect its status and ledger before resuming')
 finally:
  with (w.OUT/'export.log').open('a') as log:
   export=subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('paper_workflow_finish.py'))],stdout=log,stderr=subprocess.STDOUT)
  summary=json.loads((w.OUT/'summary.json').read_text()) if (w.OUT/'summary.json').exists() else {}
  complete=summary.get('complete',False) and export.returncode==0
  w.save('deployment-status.json',{'at_utc':w.b.now(),'state':'complete' if complete else 'needs_attention','stages':results,'export_exit_code':export.returncode,'generated_targets':summary.get('generated_targets'),'planned_targets':1323})
if __name__=='__main__':main()
