"""Complete the authorized 1500-paper expansion and publish validated private data."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research/data/paper-gap10000-v3-luna-20260930'
RUN=[sys.executable,'-u',str(ROOT/'benchmarks/otd/paper_scale10000_run.py')]
HF='/tmp/pangram-paper-hf-env/bin/python'
STATUS=OUT/'pipeline-status.json'
def read(p):return json.loads(p.read_text())
def status(stage,**extra):
 value={'stage':stage,'at_epoch':time.time(),**extra};STATUS.write_text(json.dumps(value,indent=2));print(json.dumps(value),flush=True)
def command(args,log):
 with (OUT/log).open('a') as f:return subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
def require(args,log):
 if command(args,log):raise RuntimeError('Failed '+log+'; inspect preserved log')
try:
 status('acquiring_sources')
 while not (OUT/'source-selection-summary.json').exists():
  progress=OUT/'acquisition-status.json'
  if progress.exists() and time.time()-progress.stat().st_mtime>1800:raise RuntimeError('Source acquisition stopped making progress')
  time.sleep(15)
 status('generation_and_evaluation')
 for retry in range(5):
  if command(RUN+['run'],'generation.log')==0:break
  files=[OUT/(s+'-status.json') for s in ['outline','writer','audit']]+[OUT/'fidelity-review/fidelity-status.json']
  errors=[e for p in files if p.exists() for e in read(p)['errors']]
  if not errors or not all('retries exhausted' in e for e in errors):raise RuntimeError('Generation requires inspection; accepted results retained')
  time.sleep(15)
 else:raise RuntimeError('Mechanical/capacity retries exhausted; accepted results retained')
 status('billing_and_validation')
 expected=read(OUT/'scale-report.json')['costs']['new']['account_precision_cost_usd'];before=read(OUT/'key-usage-before.json')['usage']
 for _ in range(12):
  require(RUN+['settle'],'billing.log')
  if abs(read(OUT/'key-total-after.json')['usage']-before-expected)<1e-8:break
  time.sleep(30)
 else:raise RuntimeError('Account charges not yet reconciled; publication paused')
 require(RUN+['export'],'export.log');require(RUN+['validate'],'validation.log')
 (OUT/'completion-status.json').write_text(json.dumps({'status':'complete','at_epoch':time.time(),'scope':'Generation, evaluation, provenance export and validation complete; publication tracked separately.'},indent=2))
 status('huggingface_packaging')
 require([HF,'-u',str(ROOT/'benchmarks/otd/package_paper10000_huggingface.py')],'huggingface-package.log')
 require([HF,'-u',str(ROOT/'benchmarks/otd/validate_paper10000_huggingface.py')],'huggingface-validation.log')
 status('huggingface_upload')
 require([HF,'-u',str(ROOT/'benchmarks/otd/upload_paper10000_huggingface.py')],'huggingface-upload.log')
 status('complete',papers=2000,generated_paragraphs=10000,matched_originals=10000,upload=read(OUT/'huggingface_upload.json')['url'],costs=read(OUT/'scale-report.json')['costs']['new'])
except Exception as exc:
 status('needs_inspection',error=str(exc));raise
