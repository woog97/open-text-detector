from pathlib import Path
import collections,hashlib,json,re,time
O=Path('/data/workspace/paper-diversity-v1/public-source-audit/hip-audit');P=O/'pair-preparation';D=O/'pair-preparation-grouped-v2';D.mkdir(exist_ok=True)
def save(n,v):
 p=D/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def status(**x):save('status.json',dict(time=time.time(),**x))
assert not (D/'manifest.json').exists()
assert (P/'manifest.json').exists()
status(state='running',phase='strengthen-near-duplicate-groups')
rows=[json.loads(l) for s in ['train','selection','reserved_test'] for l in (P/(s+'.jsonl')).read_text().splitlines()]
parent=list(range(len(rows)))
def root(i):
 while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
 return i
def union(a,b):
 a=root(a);b=root(b)
 if a!=b:parent[max(a,b)]=min(a,b)
def shingles(text,step=1):
 w=re.findall(r'\w+',text.lower())
 for i in range(0,len(w)-12,step):yield hashlib.blake2b(' '.join(w[i:i+13]).encode(),digest_size=8).digest()
index={};exact={}
for i,r in enumerate(rows):
 for text in [r['human_text'],r['ai_text']]:
  h=hashlib.sha256(' '.join(re.findall(r'\w+',text.lower())).encode()).hexdigest()
  if h in exact:union(i,exact[h])
  else:exact[h]=i
  for h in shingles(text,8):
   if h in index:union(i,index[h])
   else:index[h]=i
for i,r in enumerate(rows):
 for text in [r['human_text'],r['ai_text']]:
  for h in shingles(text):
   if h in index:union(i,index[h])
components=collections.defaultdict(list)
for i,r in enumerate(rows):components[root(i)].append(r)
counts=collections.Counter();tokens=collections.Counter();maxpair=collections.Counter();groups=[]
for rs in components.values():
 g=hashlib.sha256('\n'.join(sorted({r['source_group'] for r in rs})).encode()).hexdigest()
 seen=any(r['existing_training_match'] for r in rs)
 u=int(hashlib.sha256(('hip-pairs-grouped-v2:20261002:'+g).encode()).hexdigest()[:16],16)/16**16
 split='train' if seen or u<.8 else 'selection' if u<.9 else 'reserved_test'
 groups.append({'group':g,'split':split,'pairs':len(rs),'existing_training_match':seen})
 for r in rs:
  r['original_source_group']=r['source_group'];r['source_group']=g;r['split']=split
  r['human_origin']={'dataset':r['dataset'],'source':r['source'],'released_row_id':r['id']}
  r['ai_transform']='released OpenAI paraphrase; per-row model identifier unavailable'
  r['verified_pair_relation']='Human-original/AI-paraphrase field pairing verified against pinned source dataset card; no independent semantic or generation-offset verification.'
  r['normalized_human_hash']=r['original_source_group']
  r['normalized_ai_hash']=hashlib.sha256(' '.join(re.findall(r'\w+',r['ai_text'].lower())).encode()).hexdigest()
  counts[split]+=1;tokens[split]+=r['total_tokens'];maxpair[split]=max(maxpair[split],r['total_tokens'])
for split in ['train','selection','reserved_test']:
 p=D/(split+'.jsonl');tmp=p.with_suffix('.tmp')
 with tmp.open('w') as f:
  for r in rows:
   if r['split']==split:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 tmp.replace(p)
assert len({r['id'] for r in rows})==len(rows)
assert all(not r['existing_training_match'] for r in rows if r['split']!='train')
# Group construction explicitly puts all normalized exact / sampled-shingle links together.
assert all(len({r['split'] for r in rs})==1 for rs in components.values())
manifest={'state':'prepared','time':time.time(),'version':'hip-pairs-grouped-v2','pair_counts':dict(counts),'one_pass_processed_tokens':dict(tokens),'max_pair_tokens':dict(maxpair),'near_duplicate_components':len(components),'largest_component_pairs':max(map(len,components.values())),'forced_train_components':sum(g['existing_training_match'] for g in groups),'groups_by_split':dict(collections.Counter(g['split'] for g in groups)),'grouping':'Transitive normalized exact and shared13word links across human and paraphrase texts, stride8 indexed/fullstride1 queried; all componentmembers sharepartition','split':'Deterministic80/10/10SHA256component key; seen existingtrainingmatches forcedtrain','source_preparation_manifest':str(P/'manifest.json'),'source_manifest_sha256':hashlib.sha256((P/'manifest.json').read_bytes()).hexdigest(),'files':{s:{'path':str(D/(s+'.jsonl')),'sha256':hashlib.sha256((D/(s+'.jsonl')).read_bytes()).hexdigest()} for s in counts},'whole_pair_max_tokens_each_including_specials':512,'supervision':'document-paironly, no token/spangold','training_ready':False,'remaining':['Matched paired-ranking protocol/control validation','Choose fixed token-exposure sampler; complete wholepair totals are recorded','Review GRADTEXresult before registration'],'limits':['Lexical grouping does not prove semantic independence','Existingtraining inventory exclusion is normalized exact; nearvarianttraining exposure cannot be ruled out','Reservedtest prospective only, no model scores accessed']}
save('groups.json',groups);save('manifest.json',manifest);status(state='complete',counts=dict(counts),training_ready=False);print(json.dumps(manifest,indent=2))
