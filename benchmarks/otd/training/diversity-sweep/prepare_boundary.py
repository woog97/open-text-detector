"""Contiguous short-AI boundary crops from existing training papers; no new joins."""
from prepare import *
def main():
 while not (ROOT/'prepared-complete.json').exists():time.sleep(15)
 root=ROOT/'boundary';assert not root.exists();root.mkdir();(root/'prepared').mkdir();(root/'configs').mkdir()
 for p in (ROOT/'control').glob('*.py'):shutil.copy2(p,root/p.name)
 shutil.copy2(BASE/'models.lock.json',root/'models.lock.json');shutil.copy2(BASE/'configs/encoder.json',root/'configs/encoder.json');(root/'vendor').symlink_to(BASE/'vendor')
 sys.path.insert(0,str(BASE));from transformers import AutoTokenizer
 from data import crop
 m=json.loads((ROOT/'control/prepared/manifest.json').read_text());info=m['models']['encoder'];tok=AutoTokenizer.from_pretrained(info['repo'],revision=info['revision'],local_files_only=True)
 donors=[]
 for r in read(BASE/'prepared/train.jsonl.gz'):
  if r['kind']!='paired' or not any(z['label']==1 for z in r['regions']):continue
  offsets=tok(r['text'],add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
  for z in r['regions']:
   if z['label']!=1:continue
   indexes=[i for i,(a,b) in enumerate(offsets) if a>=z['start'] and b<=z['end']]
   if indexes:donors.append((r,offsets,indexes[0],indexes[-1]+1))
 cache={};stats={}
 def candidates(n):
  if n in cache:return cache[n]
  result=[];small=max(4,min(30,n//8))
  for r,off,a,b in donors:
   for start,end in [(a+small-n,a+small),(b-small,b-small+n)]:
    if start<0 or end>len(off):continue
    row=crop(r,off[start][0],off[end-1][1]);cnt=len(tok(row['text'],add_special_tokens=False)['input_ids'])
    if abs(cnt-n)>3:continue
    chars=collections.Counter()
    for z in row['regions']:chars[z['label']]+=z['end']-z['start']
    if chars[-100] or not(20<=chars[1]<=200) or not(.02<chars[1]/len(row['text'])<.25):continue
    if cnt>n:continue
    result.append((row,cnt))
  cache[n]=result;return result
 for p in (BASE/'prepared').glob('*.jsonl.gz'):
  if not p.name.startswith('stage'):(root/'prepared'/p.name).symlink_to(p);continue
  rows=read(p);key=p.name.removesuffix('.jsonl.gz');rng=random.Random(9234+sum(map(ord,key)));baseline=actual=changed=0
  for i,row in enumerate(rows):
   n=len(tok(row['text'],add_special_tokens=False)['input_ids']);baseline+=n+2;actual+=n+2
   if i%5!=2:continue
   pool=candidates(n)
   if not pool:continue
   item,count=rng.choice(pool);item=dict(item);item['draw_id']=row.get('draw_id',str(i));item['augmentation']='contiguous_short_AI_boundary';rows[i]=item;actual+=count-n;changed+=1
  assert changed>len(rows)*.1 and abs(actual/baseline-1)<.005
  m['files'][key]=write(root/'prepared'/p.name,rows);stats[key]={'rows':len(rows),'boundary_draws':changed,'baseline_processed_tokens':baseline,'processed_tokens':actual}
 m['experiment']={'arm':'boundary','token_budget':stats,'augmentation':'contiguous original windows with20-200AI chars and2-25percentAI; no splice; replace up to20percentdraws','training_only':True};save(root/'prepared/manifest.json',m);save(root/'exposure.json',stats);save(ROOT/'boundary-prepared.json',stats);print('boundary ready',stats,flush=True)
if __name__=='__main__':main()
