"""Resume a single existing streaming source without taking the whole-run lock."""
import argparse,fcntl,json
from pathlib import Path
from expand_pool import scan_source

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--source',required=True);a=p.parse_args()
 b=a.base;config=b/'pipeline';plan=json.loads((config/'sampling-plan.json').read_text())
 source=next(s for s in plan['source_quotas'] if s['source_id']==a.source)
 specs=json.loads((config/'collection-sources.json').read_text());specs['wiki_talk']=specs['wikipedia']
 if a.source in ('cccc','ubuntu_irc'):raise ValueError('Dedicated source extraction required')
 with (b/(a.source+'.lock')).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  scan_source(b/'collection.sqlite3',b,a.source,specs[a.source],source['category'],source['planned_passages'],json.loads((config/'protected-exclusions.json').read_text()))
if __name__=='__main__':main()
