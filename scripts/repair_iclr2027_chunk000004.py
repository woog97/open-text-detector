"""Repair only flagged extraction pages with the existing positioned OCR stage."""
import sys,json,collections,re,subprocess,concurrent.futures,fcntl,time,os
from pathlib import Path
from process_new_positioned_papers import extract_one,verify_one,write
sys.path.insert(0,'backend');from open_text_detector_backend.result_codec import decode
ROOT=Path.cwd();root=ROOT/'research/data/openreview_iclr2027_all';chunk=root/'chunks/000004'
def repair(p):
 a=decode((ROOT/p['text_file']).read_bytes());selected=[];diagnostics=[]
 for page in a['pages']:
  n=page['page'];native=' '.join(a['text'][b['start']:b['end']] for b in a['rectangles'] if b['page']==n)
  plain=subprocess.check_output(['pdftotext','-f',str(n),'-l',str(n),'-enc','UTF-8',p['pdf'],'-']).decode()
  count=lambda t:collections.Counter(re.sub(r'\s','',t))
  difference=sum((count(plain)-count(native)).values())/max(1,sum(count(plain).values()))
  replacements=native.count('\ufffd')/max(1,len(native))
  if difference>.02 or replacements>.005:selected.append(n)
  diagnostics.append({'page':n,'plain_text_difference':difference,'replacement_fraction':replacements})
 assert selected
 p['ocr_pages']=selected;p['repair_diagnostics']=diagnostics;p['prior_extraction']={k:p[k] for k in ['text_file','text_sha256','blob_sha256','verification']}
 p.update(extract_one(p));p['verification']=verify_one(p);return p

def main():
 lock=(chunk/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 d=json.loads((chunk/'extraction.json').read_text());write(chunk/'extraction-before-repair.json',d)
 results={p['forum_id']:p for p in d['papers']};bad=[p for p in d['papers'] if p['verification']['issues']]
 write(chunk/'runner-state.json',{'pid':os.getpid(),'chunk':'000004','phase':'repairing_extractions','papers':len(bad),'updated_at':time.time()})
 errors=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
  tasks={pool.submit(repair,p):p['forum_id'] for p in bad}
  for f in concurrent.futures.as_completed(tasks):
   fid=tasks[f]
   try:r=f.result();results[fid]=r;print(json.dumps({'id':fid,'ocr_pages':r['ocr_pages'],'issues':r['verification']['issues']}),flush=True)
   except Exception as e:errors.append({'id':fid,'error':str(e)});print(json.dumps(errors[-1]),flush=True)
   write(chunk/'extraction.json',{'papers':list(results.values()),'errors':errors,'verified':False})
 issues=[{'id':p['forum_id'],'issues':p['verification']['issues']} for p in results.values() if p['verification']['issues']]
 good=not errors and not issues;write(chunk/'extraction.json',{'papers':list(results.values()),'errors':errors+issues,'verified':good})
 write(chunk/'runner-state.json',{'pid':os.getpid(),'chunk':'000004','phase':'extractions_repaired' if good else 'needs_repair','updated_at':time.time(),'errors':errors+issues})
 if good:
  fcntl.flock(lock,fcntl.LOCK_UN)
  subprocess.run([sys.executable,'scripts/run_iclr2027_pipeline.py','--chunk','000004','--overlap-download'],check=True)
if __name__=='__main__':main()
