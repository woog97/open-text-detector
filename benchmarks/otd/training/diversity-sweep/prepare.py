"""Prepare fixed-budget diversity pilots; never accesses generation APIs or model weights."""
import csv,gzip,json,hashlib,re,random,collections,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=Path('/data/workspace/paper-lora-comparison-v1')
SUITE=Path('/data/workspace/paper-v3-modernbert-20260930/eval-suite-v1')
def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def words(t):return re.findall(r'\w+',t.lower())
def shingles(t,stride=1):
 w=words(t)
 return (hashlib.blake2b(' '.join(w[i:i+13]).encode(),digest_size=8).digest() for i in range(0,max(0,len(w)-12),stride))
def norm(t):return ' '.join(words(t))
def read(p):
 with gzip.open(p,'rt') as f:return [json.loads(l) for l in f]
def save(p,x):p.write_text(json.dumps(x,indent=2))
def write(p,rows):
 b=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode();p.write_bytes(gzip.compress(b,mtime=0))
 return {'rows':len(rows),'sha256':hashlib.sha256(b).hexdigest(),'papers':len({r['paper_id'] for r in rows})}
def main():
 assert (ROOT/'download-complete.json').exists()
 if (ROOT/'prepared-complete.json').exists():return
 blocked=set();exact=set();n=0
 def add(text):
  exact.add(sha(norm(text)));blocked.update(shingles(text,8))
 # All current suite profiles, including full, and original selection/calibration windows.
 paths=list(SUITE.glob('*.jsonl.gz'))+[p for p in (BASE/'prepared').glob('*.jsonl.gz') if not p.name.startswith(('stage','train'))]
 for p in paths:
  for row in read(p):
   if 'text' in row:add(row['text']);n+=1
 print('suite exclusion indexed',n,len(blocked),flush=True)
 for split in ['valid','test']:
  with (ROOT/f'mage-{split}.csv').open(encoding='utf-8-sig') as f:
   for row in csv.DictReader(f):add(row['text'])
 def overlap(t):return sha(norm(t)) in exact or any(s in blocked for s in shingles(t))
 pools={'raid':[],'mage':[]};rejected=collections.Counter();group_split={};bad=set();seen=set()
 with (ROOT/'raid.csv').open() as f:
  for r in csv.DictReader(f):
   if r['domain']!='abstracts' or r['attack']!='none':continue
   g=r['source_id'];group_split[g]='train' if int(sha(g)[:8],16)%10<8 else ('validation' if int(sha(g)[:8],16)%10==8 else 'test')
   if group_split[g]!='train':continue
   text=r['generation'];h=sha(norm(text))
   if overlap(text):bad.add(g)
   if len(words(text))<60 or h in seen:continue
   seen.add(h);pools['raid'].append({'id':'raid/'+r['id'],'paper_id':'raid/'+g,'kind':'external','text':text,'label':int(r['model']!='human'),'source':r['model'],'group':g})
 pools['raid']=[r for r in pools['raid'] if r['group'] not in bad]
 save(ROOT/'raid-group-split.json',group_split)
 # Exclude continuation setups because their prefix ownership requires additional provenance work.
 buckets=collections.defaultdict(list);rng=random.Random(6201);counts=collections.Counter()
 with (ROOT/'mage-train.csv').open(encoding='utf-8-sig') as f:
  for i,r in enumerate(csv.DictReader(f)):
   src=r['src'];label=1-int(r['label'])
   if not (src.endswith('_human') or '_machine_specified_' in src or '_machine_topical_' in src):continue
   assert (label==0)==src.endswith('_human')
   text=r['text'];h=sha(norm(text))
   if len(words(text))<60 or h in seen:continue
   if overlap(text):rejected['mage_overlap']+=1;continue
   seen.add(h);item={'id':f'mage/{i}','paper_id':'mage/'+h,'kind':'external','text':text,'label':label,'source':src,'group':h}
   counts[src]+=1;b=buckets[src]
   if len(b)<700:b.append(item)
   else:
    j=rng.randrange(counts[src])
    if j<700:b[j]=item
 domains={s.split('_machine_')[0] for s in buckets if '_machine_' in s}
 pools['mage']=[r for s,b in buckets.items() for r in b if s.removesuffix('_human').split('_machine_')[0] in domains]
 save(ROOT/'audit.json',{'excluded_indexed_rows':n,'overlap_rule':'normalized exact OR shared 13-word span (heldout stride8, candidate stride1); conservative lexical exclusion, not semantic paraphrase proof','raid_blocked_source_groups':len(bad),'rejected':dict(rejected),'eligible_counts':{k:dict(collections.Counter(r['source'] for r in v)) for k,v in pools.items()},'mage_excluded':'continuation setups and domains lacking both classes','sources':json.loads((ROOT/'download-complete.json').read_text())})
 sys.path.insert(0,str(BASE));from transformers import AutoTokenizer
 manifest=json.loads((BASE/'prepared/manifest.json').read_text());info=manifest['models']['encoder'];tok=AutoTokenizer.from_pretrained(info['repo'],revision=info['revision'],local_files_only=True)
 for key,pool in pools.items():
  for r in pool:r['offsets']=tok(r['text'],add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
  assert all(sum(r['label']==label for r in pool)>100 for label in [0,1]),key
 print('audited pools', {k:len(v) for k,v in pools.items()},flush=True)
 for name in ['control','raid','mage']:
  root=ROOT/name;assert not root.exists(),'Refuse overwriting partial trial';root.mkdir();(root/'prepared').mkdir();(root/'configs').mkdir()
  for p in BASE.glob('*.py'):shutil.copy2(p,root/p.name)
  shutil.copy2(BASE/'models.lock.json',root/'models.lock.json');shutil.copy2(BASE/'configs/encoder.json',root/'configs/encoder.json');(root/'vendor').symlink_to(BASE/'vendor')
  m=json.loads(json.dumps(manifest));stats={};used=[]
  for p in (BASE/'prepared').glob('*.jsonl.gz'):
   if not p.name.startswith('stage'):(root/'prepared'/p.name).symlink_to(p);continue
   key=p.name.removesuffix('.jsonl.gz');rows=read(p);base_tokens=sum(len(tok(r['text'],add_special_tokens=False)['input_ids'])+2 for r in rows)
   replaced_tokens=0;actual_tokens=base_tokens
   if name!='control':
    rng=random.Random(9234+sum(map(ord,key)));replacement=0
    order=[i for i in range(len(rows)) if i%5==2];remaining=[i for i in range(len(rows)) if i%5!=2];rng.shuffle(remaining);order+=remaining
    for i in order:
     if replaced_tokens>=.2*base_tokens:break
     row=rows[i]
     desired=len(tok(row['text'],add_special_tokens=False)['input_ids']);label=replacement%2
     eligible=collections.defaultdict(list)
     for r in pools[name]:
      if r['label']==label and len(r['offsets'])>=desired:eligible[r['source']].append(r)
     if not eligible:continue
     source=rng.choice(sorted(eligible));r=rng.choice(eligible[source]);text=r['text'][:r['offsets'][desired-1][1]]
     while len(tok(text,add_special_tokens=False)['input_ids'])>desired:text=text[:-1]
     nt=len(tok(text,add_special_tokens=False)['input_ids']);assert nt>=desired-3
     rows[i]={'id':r['id'],'paper_id':r['paper_id'],'kind':'external','text':text,'regions':[{'start':0,'end':len(text),'label':label}],'target_start':0,'target_end':len(text),'draw_id':row.get('draw_id',str(i)),'dataset':name,'source':source,'provenance':'human source or fully generated topical/specified response; no continuation prefixes'}
     replacement+=1;replaced_tokens+=nt+2;actual_tokens+=nt-desired;used.append({'id':r['id'],'source':source,'group':r['group'],'label':label})
   assert abs(actual_tokens/base_tokens-1)<.005
   if name!='control':assert .199<=replaced_tokens/actual_tokens<=.201
   m['files'][key]=write(root/'prepared'/p.name,rows);stats[key]={'rows':len(rows),'baseline_processed_tokens':base_tokens,'processed_tokens':actual_tokens,'external_tokens':replaced_tokens,'external_token_share':replaced_tokens/actual_tokens}
  m['experiment']={'arm':name,'token_budget':stats,'recipe':'same paper recipe; fifth-position candidates then seeded fallback until20percent token exposure, alternating classes, balanced source buckets, length matched within3 tokens','seed':9234,'audit_sha256':hashlib.sha256((ROOT/'audit.json').read_bytes()).hexdigest()};save(root/'prepared/manifest.json',m);save(root/'exposure.json',stats);save(root/'external-used.json',used)
 save(ROOT/'prepared-complete.json',{'time':time.time(),'arms':['control','raid','mage']})
if __name__=='__main__':main()
