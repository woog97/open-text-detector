from runtime import ROOT,require_space
from pathlib import Path
import json,gzip,hashlib,random,collections,os
from transformers import AutoTokenizer
from data import crop

def read(p):return [json.loads(l) for l in gzip.decompress(Path(p).read_bytes()).splitlines()]
def write(p,rows):
 b=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode();p.write_bytes(gzip.compress(b,mtime=0));return {'rows':len(rows),'sha256':hashlib.sha256(b).hexdigest()}
def prepare(name):
 require_space();dest=ROOT/'runs'/name;dest.mkdir(parents=True,exist_ok=True);out=dest/'prepared-v2';out.mkdir(exist_ok=True)
 if (out/'manifest.json').exists():return
 asset=ROOT/'assets'/name;tok=AutoTokenizer.from_pretrained(asset,local_files_only=True)
 src=ROOT/'source-pools';reservation=json.loads((ROOT/'reservation.json').read_text());blocked=set(reservation['exclude_from_training']['source_paper_ids'])
 excluded={};pools={};blocked_groups={'paper:'+p for p in blocked}
 for ds in ['human','mirrors','papers','gradtex','fullpapers']:
  rs=read(src/f'pool-{ds}.jsonl.gz');pools[ds]=[r for r in rs if r['paper_id'] not in blocked and r.get('group') not in blocked_groups];excluded[ds]=len(rs)-len(pools[ds]);assert pools[ds]
 def window(r,rng):
  if r.get('supervision')=='document_only':return dict(r)
  text=r['text'];off=tok(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping'];n=len(off);assert n
  start=rng.randrange(max(1,n-510+1));end=min(n,start+510);row=dict(r)
  row.setdefault('kind','paired' if r['dataset']=='papers' else 'novel_human' if r.get('label')==0 else 'generated')
  if 'regions' not in row:row.update(regions=[{'start':0,'end':len(text),'label':r['label']}],target_start=0,target_end=len(text))
  result=crop(row,off[start][0],off[end-1][1])
  while len(tok(result['text'],add_special_tokens=False)['input_ids'])>510:
   end-=1;result=crop(row,off[start][0],off[end-1][1])
  result.update(dataset=r['dataset'],group=r.get('group'));return result
 files={};counts={}
 for stage,epochs,num in [(1,1,12000),(2,3,24000)]:
  mix=['human']*5+['mirrors']*5+['papers']*5+(['gradtex']*4 if stage==2 else ['papers']*4)+['fullpapers']
  for epoch in range(epochs):
   key=f'stage{stage}-epoch{epoch}';rng=random.Random(42+stage*1000+epoch);rows=[]
   for i in range(num):
    ds=mix[i%20];r=window(rng.choice(pools[ds]),rng);r['draw_id']=key+'-'+str(i);rows.append(r)
   rng.shuffle(rows);counts[key]=dict(collections.Counter(r['dataset'] for r in rows));files[key]=write(out/(key+'.jsonl.gz'),rows)
   print(json.dumps({'phase':'prepared','model':name,'file':key,'counts':counts[key]}),flush=True)
   if stage==2:assert counts[key]=={'human':6000,'mirrors':6000,'papers':6000,'gradtex':4800,'fullpapers':1200}
 for key in ['selection-windows','calibration-windows']:
  rows=[]
  for r in read(src/(key+'.jsonl.gz')):
   e=tok(r['text'],add_special_tokens=False,return_offsets_mapping=True)
   if len(e['input_ids'])>510:r=crop(r,0,e['offset_mapping'][510][0])
   while len(tok(r['text'],add_special_tokens=False)['input_ids'])>510:r=crop(r,0,len(r['text'])-1)
   rows.append(r)
  files[key]=write(out/(key+'.jsonl.gz'),rows)
 m={'files':files,'counts':counts,'pool_counts':{k:len(v) for k,v in pools.items()},'reserved_families_excluded':excluded,'source_manifest_sha256':hashlib.sha256((src/'manifest.json').read_bytes()).hexdigest(),'reservation_sha256':hashlib.sha256((ROOT/'reservation.json').read_bytes()).hexdigest(),'policy':'Same existing binary labels and 510-token crops; whole-document GRADTEX pooled over all chunks; native tokenizer; reserved manuscript families excluded.'}
 (out/'manifest.json').write_text(json.dumps(m,indent=2))
if __name__=='__main__':
 import sys;prepare(sys.argv[1])
