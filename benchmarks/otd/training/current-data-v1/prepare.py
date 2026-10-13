"""Versioned preparation; preserve originals and existing evaluation partitions."""
from pathlib import Path
import json,gzip,hashlib,re,random,collections,time,shutil
import pyarrow.parquet as pq
from transformers import AutoTokenizer
R=Path(__file__).resolve().parent;P=R/'prepared-v2';P.mkdir(exist_ok=True)
def save(p,v):p.write_text(json.dumps(v,indent=2))
def status(**kw):save(R/'prepare-status.json',{'time':time.time(),**kw})
def norm(t):return ' '.join(re.findall(r'\w+',t.lower()))
def digest(t):return hashlib.sha256(t.encode()).hexdigest()
def shingles(t):
 w=norm(t).split()
 return {hashlib.blake2b(' '.join(w[i:i+13]).encode(),digest_size=8).digest() for i in range(0,len(w)-12,4)}
def read(p):
 with gzip.open(p,'rt') as f:return [json.loads(l) for l in f]
def write(name,rs):
 blob=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs).encode();(P/(name+'.jsonl.gz')).write_bytes(gzip.compress(blob,mtime=0));return {'rows':len(rs),'papers':len({r['paper_id'] for r in rs}),'sha256':hashlib.sha256(blob).hexdigest()}
def main():
 assert not (P/'manifest.json').exists(),'Frozen preparation already exists'
 status(state='running',phase='heldout-index')
 held=set();exact=set();held_ids=set()
 def protect(r):
  for key in ['paper_id','source_paper_id','forum_id']:
   if r.get(key):held_ids.add(str(r[key]))
  for key in ['text','original_target','human_source_text']:
   t=r.get(key)
   if isinstance(t,str):exact.add(digest(norm(t)));held.update(shingles(t))
 old=R/'sources/frozen-paper-evaluation'
 for name in ['selection','calibration']:
  for r in read(old/(name+'.jsonl.gz')):protect(r)
 suite=Path('/data/workspace/paper-v3-modernbert-20260930/eval-suite-v1')
 for p in suite.glob('*.jsonl.gz'):
  for r in read(p):protect(r)
 for split in ['validation','test']:
  for r in pq.read_table(R/'sources/papers/data/passages'/f'{split}-00000-of-00001.parquet',columns=['text','paper_id']).to_pylist():protect(r)
 for split in ['validation','test']:
  p=Path('/data/workspace/paper-diversity-v1/public-source-audit')/f'gradtex-{split}.parquet'
  for r in pq.read_table(p,columns=['text','human_source_text']).to_pylist():protect(r)
 rows=[];human_by_id={};parents={}
 def find(x):
  parents.setdefault(x,x)
  while parents[x]!=x:parents[x]=parents[parents[x]];x=parents[x]
  return x
 def union(a,b):
  a,b=find(a),find(b)
  if a!=b:parents[max(a,b)]=min(a,b)
 aliases={}
 def add(row,keys):
  g=row['group'];find(g)
  keys=['text:'+digest(norm(row['text']))]+keys
  for k in keys:
   if k in aliases:union(g,aliases[k])
   else:aliases[k]=g
  rows.append(row)
 status(state='running',phase='source-pools')
 for p in sorted((R/'sources/human/data').glob('*.parquet')):
  for r in pq.read_table(p).to_pylist():
   assert r['usage_approved']
   g='human:'+str(r.get('provisional_family_id') or r['parent_document_id']);human_by_id[r['record_id']]=g
   keys=[k+':'+str(r[k]) for k in ['parent_document_id','prompt_family_id','reviewer_family_id','product_family_id'] if r.get(k)]
   add({'id':r['record_id'],'paper_id':g,'group':g,'dataset':'human','text':r['text'],'label':0,'source':r['source_id'],'protected':str(r.get('original_split','')).lower() in ['test','validation','valid']},keys)
 assert len(human_by_id)==100000
 for r in pq.read_table('/datasets/synthetic-mirrors-luna-28120/data/train-00000-of-00001.parquet').to_pylist():
  g=human_by_id[r['source_record_id']]
  add({'id':r['record_id'],'paper_id':g,'group':g,'dataset':'mirrors','text':r['text'],'label':1,'development_exposed':bool(r.get('development_exposed')),'protected':False},[])
 for r in pq.read_table(R/'sources/papers/data/passages/train-00000-of-00001.parquet').to_pylist():
  g='paper:'+r['paper_id'];add({'id':r['id'],'paper_id':r['paper_id'],'group':g,'dataset':'papers','text':r['text'],'regions':r['regions'],'target_start':r['target_start'],'target_end':r['target_end'],'protected':bool(r.get('development_exposed')),'original_split':'train'},['paper:'+r['paper_id']])
 for i,r in enumerate(pq.read_table('/data/workspace/paper-diversity-v1/public-source-audit/gradtex-train.parquet').to_pylist()):
  source=r.get('human_source_text') or r['text'];g='gradtex:'+digest(norm(source))
  add({'id':f'gradtex-{i}','paper_id':g,'group':g,'dataset':'gradtex','text':r['text'],'document_label':1-int(r['binary_label']),'supervision':'document_only','token_labels':None,'original_split':'train','source_text':source},['text:'+digest(norm(source))])
 for r in pq.read_table(next((R/'sources/fullpapers/data').glob('*.parquet'))).to_pylist():
  text=r['paper_markdown'];abstract=r['rendered_abstract'];assert abstract in text
  start=text.index(abstract)+len(abstract);body=text[start:];assert body.strip()
  g='paper:'+r['source_paper_id'];add({'id':'fullpaper-'+str(r['paper_id']),'paper_id':r['source_paper_id'],'group':g,'dataset':'fullpapers','text':body,'label':1,'model':r['model'],'excluded_prefix_chars':start},['paper:'+r['source_paper_id']])
 status(state='running',phase='split-and-overlap-check',rows=len(rows))
 blocked=set();reasons=collections.Counter();labels=collections.defaultdict(set)
 for ri,r in enumerate(rows):
  if ri%20000==0:status(state='running',phase='split-and-overlap-check',rows=ri,total=len(rows))
  g=find(r['group']);r['group']=g;h=digest(norm(r['text']));labels[h].add(r.get('label',r.get('document_label',-100)))
  reason=None
  if r.get('protected') or r['paper_id'] in held_ids:reason='protected_family'
  elif h in exact or shingles(r['text'])&held:reason='heldout_text_overlap'
  elif r.get('source_text') and (digest(norm(r['source_text'])) in exact or shingles(r['source_text'])&held):reason='heldout_source_overlap'
  if reason:blocked.add(g);reasons[reason]+=1
 conflicts={h for h,ls in labels.items() if 0 in ls and 1 in ls}
 for r in rows:
  if digest(norm(r['text'])) in conflicts:blocked.add(r['group']);reasons['conflicting_exact_labels']+=1
 # Existing paper and GRADTEX train assignments are retained; linked new data follow them.
 train_groups={r['group'] for r in rows if r.get('original_split')=='train'}
 pools=collections.defaultdict(list);splits=collections.Counter();assignments=[];seen=set()
 for r in rows:
  g=r['group'];v=int(digest('current-data-v1/'+g)[:8],16)%100
  split='excluded' if g in blocked else 'train' if g in train_groups or v<80 else 'validation' if v<90 else 'test'
  splits[(r['dataset'],split)]+=1;assignments.append({'id':r['id'],'dataset':r['dataset'],'group':g,'split':split})
  if split!='train':continue
  key=(r['dataset'],digest(norm(r['text'])))
  if key in seen:continue
  seen.add(key);r.pop('source_text',None);pools[r['dataset']].append(r)
 save(P/'split-assignments.json',assignments)
 tok=AutoTokenizer.from_pretrained(R/'assets/modernbert',local_files_only=True)
 from data import crop
 def window(r,rng):
  if r.get('supervision')=='document_only':return dict(r)
  text=r['text'];off=tok(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping'];n=len(off)
  if not n:raise ValueError('Empty source')
  start=rng.randrange(max(1,n-510+1)) if n>510 else 0;end=min(n,start+510)
  row=dict(r);row.setdefault('kind','paired' if r['dataset']=='papers' else 'novel_human' if r.get('label')==0 else 'generated')
  if 'regions' not in row:row['regions']=[{'start':0,'end':len(text),'label':r['label']}];row['target_start']=0;row['target_end']=len(text)
  out=crop(row,off[start][0],off[end-1][1])
  while len(tok(out['text'],add_special_tokens=False)['input_ids'])>510:
   end-=1;out=crop(row,off[start][0],off[end-1][1])
  out['dataset']=r['dataset'];out['group']=r['group'];return out
 files={}
 for name,rs in pools.items():files['pool-'+name]=write('pool-'+name,rs)
 for stage,epochs,count in [(1,1,12000),(2,3,24000)]:
  mix=['human']*5+['mirrors']*5+['papers']*5+(['gradtex']*4 if stage==2 else ['papers']*4)+['fullpapers']
  for epoch in range(epochs):
   rng=random.Random(42+stage*1000+epoch);schedule=[]
   for i in range(count):
    src=mix[i%20];r=rng.choice(pools[src]);w=window(r,rng);w['draw_id']=f's{stage}-e{epoch}-{i}';schedule.append(w)
    if i%2000==0:status(state='running',phase='training-windows',stage=stage,epoch=epoch,rows=i)
   rng.shuffle(schedule);files[f'stage{stage}-epoch{epoch}']=write(f'stage{stage}-epoch{epoch}',schedule)
 for name in ['selection-windows','calibration-windows']:
  files[name]=write(name,read(old/(name+'.jsonl.gz')))
 manifest={'version':1,'models':json.loads((R/'models.lock.json').read_text()),'files':files,'pool_counts':{k:len(v) for k,v in pools.items()},'split_counts':{str(k):v for k,v in splits.items()},'excluded_families':len(blocked),'exclusion_reasons':dict(reasons),'policy':'Original paper and GRADTEX train splits preserved. New families 80/10/10. Connected source families and normalized duplicates grouped. Heldout exact and 13-word stride4 matches block family; not a semantic overlap guarantee. No audit-outcome filtering of user-approved human corpus. GRADTEX whole-document supervision only. Full manuscripts omit title/abstract prefix.','draws':84000,'stage2_mix':{'human':.25,'mirrors':.25,'papers':.25,'gradtex':.20,'fullpapers':.05}}
 save(P/'manifest.json',manifest);status(state='complete',pool_counts=manifest['pool_counts'],draws=84000)
if __name__=='__main__':
 try:main()
 except Exception as e:status(state='failed',error=repr(e));raise
