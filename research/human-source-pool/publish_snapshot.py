"""Publish a consistent read snapshot while independent collectors keep running."""
import argparse,fcntl,json,os,shutil,sqlite3,sys,traceback
from pathlib import Path
from collect_pool import atomic_json,now
from expand_pool import export
from upload_pool import upload


def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--repo-id',required=True);p.add_argument('--database',type=Path,required=True,help='Verified standalone SQLite backup on the Space; never an actively written WAL database');a=p.parse_args()
 if Path(a.name).name!=a.name:raise ValueError('Snapshot name must be a basename')
 token=sys.stdin.readline().strip()
 if not token:raise ValueError('Missing in-memory authentication')
 b=a.base;dest=b/'snapshots'/a.name;dest.mkdir(parents=True,exist_ok=False)
 with (b/'publication.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  def status(state,**extra):atomic_json(dest/'publication-status.json',{'state':state,'updated_at':now(),**extra})
  try:
   status('exporting_consistent_snapshot');shutil.copytree(b/'pipeline',dest/'pipeline',ignore=shutil.ignore_patterns('__pycache__'))
   (dest/'source-downloads').symlink_to(b/'source-downloads',target_is_directory=True)
   plan=json.loads((dest/'pipeline/sampling-plan.json').read_text());registry=json.loads((dest/'pipeline/source-registry.json').read_text())
   db=sqlite3.connect('file:'+str(a.database)+'?mode=ro&immutable=1',uri=True,timeout=120)
   try:
    if db.execute('PRAGMA quick_check').fetchone()!=('ok',):raise RuntimeError('Snapshot backup failed integrity check')
    export(db,dest,registry,plan)
   finally:db.close()
   status('validating_and_uploading')
   receipt=upload(dest,a.repo_id,token);atomic_json(b/'upload-receipt.json',receipt)
   status('verified',**receipt)
  except BaseException as e:
   status('failed',error_type=type(e).__name__,error=str(e));raise

if __name__=='__main__':main()
