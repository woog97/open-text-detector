"""Durable streaming extractor: freeze disjoint 1000-PDF units while download runs."""
import argparse,fcntl,hashlib,json,os,sqlite3,subprocess,sys,time
from pathlib import Path
from process_new_positioned_papers import write
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'research/data/openreview_iclr2027_all'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--parent',required=True);a=parser.parse_args();assert a.parent.isdigit()
 os.chdir(ROOT);parent=DATA/'chunks'/a.parent
 lock=(parent/'extraction-manager.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 write(parent/'microbatch-managed.json',{'parent_chunk':a.parent,'unit_size':1000,'unit_id_formula':'int(parent)*10000+index','created_at':time.time()})
 def state(phase,**kw):
  d={'pid':os.getpid(),'parent_chunk':a.parent,'phase':phase,'updated_at':time.time(),**kw};write(parent/'extraction-manager-state.json',d);print(json.dumps(d),flush=True)
 children={}
 while True:
  units=[];assigned=set()
  for p in sorted((DATA/'chunks').glob('*/manifest.json')):
   m=json.loads(p.read_text())
   if m.get('source_chunk')==a.parent and m.get('streaming_unit'):
    units.append((p.parent,m));assigned.update(x['id'] for x in m['papers'])
  def settled_exception(p,m):
   if not (p/'extraction-exceptions.json').exists():return False
   exceptions=json.loads((p/'extraction-exceptions.json').read_text())['exceptions']
   ex=json.loads((p/'extraction.json').read_text())
   good={r['forum_id'] for r in ex['papers'] if not r.get('verification',{}).get('issues') and r.get('verification')}
   bad={r['id'] for r in exceptions}
   return not good.intersection(bad) and good|bad=={r['id'] for r in m['papers']}
  exception_units=[p.name for p,m in units if settled_exception(p,m)]
  pending=[(p,m) for p,m in units if not (p/'extraction-ready.json').exists() and p.name not in exception_units]
  running=[]
  for unit,m in pending:
   stage=(unit/'extraction-stage.lock').open('a')
   try:fcntl.flock(stage,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:stage.close();running.append(unit.name);continue
   fcntl.flock(stage,fcntl.LOCK_UN);stage.close()
   previous=children.get(unit.name)
   if previous and previous.poll() is None:running.append(unit.name);continue
   if previous and previous.returncode:
    state('unit_needs_repair',unit=unit.name,returncode=previous.returncode);continue
   log=(unit/'extraction-unit.log').open('a')
   child=subprocess.Popen([sys.executable,'-u','scripts/run_iclr2027_extraction_unit.py',unit.name],stdout=log,stderr=subprocess.STDOUT)
   log.close();children[unit.name]=child;running.append(unit.name)
   write(unit/'extraction-unit-launch.json',{'pid':child.pid,'at':time.time()})
  closed=(parent/'manifest.json').exists()
  if closed: rows=json.loads((parent/'manifest.json').read_text())['papers']
  elif (parent/'timing.json').exists():
   timing=json.loads((parent/'timing.json').read_text());first=timing['first_attachment_request_at']
   db=sqlite3.connect(DATA/'download_queue/queue.sqlite3');db.row_factory=sqlite3.Row
   ids={fid for row in db.execute('SELECT ids FROM attempts WHERE started_at>=?',(first,)) for fid in json.loads(row[0])}
   rows=[dict(r) for r in db.execute("SELECT * FROM papers WHERE status='downloaded'") if r['id'] in ids];db.close()
  else:rows=[]
  available=[r for r in rows if r['id'] not in assigned]
  if len(available)>=1000 or (closed and available):
   selected=available[:1000];index=len(units)+1;unit=DATA/'chunks'/f'{int(a.parent)*10000+index:06d}'
   assert not unit.exists(),f'Unexpected unit collision: {unit}'
   for r in selected:
    p=Path(r['file']);assert p.is_file() and p.stat().st_size==r['bytes'];assert digest(p)==r['sha256']
   unit.mkdir();write(unit/'manifest.json',{'chunk_id':unit.name,'source_chunk':a.parent,'streaming_unit':True,'papers':selected,'frozen_at':time.time()});continue
  if closed and not pending:
   assert {r['id'] for r in rows}==assigned
   state('extraction_complete_with_exceptions' if exception_units else 'extraction_complete',papers=len(assigned),units=len(units),exception_units=exception_units);return
  state('extracting_units' if pending else 'waiting_for_1000',unassigned_downloads=len(available),assigned=len(assigned),running_units=running);time.sleep(15)
if __name__=='__main__':main()
