"""Space-only: extend the shared frozen windows to the official third tokenizer."""
from runtime import require_space,SPACE_CACHE
import json,gzip,hashlib,shutil
from pathlib import Path
from transformers import AutoTokenizer
from data import crop
ROOT=Path(__file__).resolve().parent

def main():
 require_space();lock=json.loads((ROOT/'models.lock.json').read_text());info=lock['qwen35'];tok=AutoTokenizer.from_pretrained(info['repo'],revision=info['revision'],cache_dir=SPACE_CACHE)
 path=ROOT/'prepared/manifest.json';m=json.loads(path.read_text());prior=hashlib.sha256(path.read_bytes()).hexdigest()
 if 'qwen35' in m['models']:print('Already added; audit_prepared.py verifies the frozen assets.');return
 archive=ROOT/'prepared-before-qwen35';archive.mkdir(exist_ok=True);shutil.copyfile(path,archive/'manifest.json');changed={}
 for name,stats in m['files'].items():
  if not (name.startswith('stage') or name.endswith('-windows')):continue
  p=ROOT/'prepared'/(name+'.jsonl.gz');old=p.read_bytes();rs=[json.loads(l) for l in gzip.decompress(old).splitlines()];n=0
  for i,r in enumerate(rs):
   enc=tok(r['text'],add_special_tokens=False,return_offsets_mapping=True)
   if len(enc['input_ids'])>510:
    stop=enc['offset_mapping'][510][0];s=crop(r,0,stop);rs[i]={**r,**s,'source_start':r['source_start'],'source_end':r['source_start']+stop};n+=1
  changed[name]=n
  if n:
   (archive/p.name).write_bytes(old);blob=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs).encode();p.write_bytes(gzip.compress(blob,mtime=0));stats['sha256']=hashlib.sha256(blob).hexdigest()
 m['models']=lock;m['previous_manifest_sha256']=prior;m['qwen35_window_adjustments']=changed;m['version']=2;path.write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(changed,indent=2))
if __name__=='__main__':main()
