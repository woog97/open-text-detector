from pathlib import Path
import collections,gzip,hashlib,json,re,time
import pyarrow.parquet as pq
from transformers import AutoTokenizer
R=Path('/data/workspace/paper-diversity-v1');O=R/'public-source-audit/hip-audit';P=O/'pair-preparation';P.mkdir(exist_ok=True)
def save(n,v):
 p=P/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def norm(t):return ' '.join(re.findall(r'\w+',t.lower()))
def hx(t):return hashlib.sha256(norm(t).encode()).hexdigest()
def status(**v):save('status.json',dict(time=time.time(),**v))
assert not (P/'manifest.json').exists()
assert json.loads((O/'audit.json').read_text())['state']=='audit_complete'
bad=set(json.loads((O/'excluded-source-hashes.json').read_text()))
rows=pq.read_table(O/'train.parquet').to_pylist()
tok=AutoTokenizer.from_pretrained(str(R/'control/run/tokenizer'),local_files_only=True)
status(state='running',phase='index-existing-training')
# Freeze an exact inventory of all existing stage training examples including pending GRADTEX data.
existing=set();files=[]
paths=list(R.glob('*/prepared/stage*.gz'))+[p for p in (R/'gradtex-document-preparation/prepared').glob('stage*.gz')]
resolved=set()
for p in paths:
 rp=str(p.resolve())
 if rp in resolved:continue
 resolved.add(rp);files.append(str(p))
 for line in gzip.decompress(p.read_bytes()).splitlines():existing.add(hx(json.loads(line)['text']))
status(state='running',phase='whole-pair-eligibility')
records=[];excluded=collections.Counter();seen=set()
for i,x in enumerate(rows):
 g=hx(x['text']);a=hx(x['paraphrased_text'])
 if g in bad:excluded['audited_heldout_overlap']+=1;continue
 if g==a:excluded['unchanged_pair']+=1;continue
 key=(g,a)
 if key in seen:excluded['duplicate_pair']+=1;continue
 seen.add(key)
 ht=len(tok(x['text'],add_special_tokens=True,truncation=False)['input_ids']);at=len(tok(x['paraphrased_text'],add_special_tokens=True,truncation=False)['input_ids'])
 if max(ht,at)>512:excluded['whole_pair_member_over512']+=1;continue
 if min(ht,at)<8:excluded['too_short']+=1;continue
 records.append(dict(id='hip-train-'+str(i),source_group=g,human_text=x['text'],ai_text=x['paraphrased_text'],human_token_count=ht,ai_token_count=at,total_tokens=ht+at,supervision='paired_document',human_document_label=0,ai_document_label=1,token_labels=None,dataset=x['dataset'],source=x['source'],semantic_score=x['gpt5_nano_semantic_score'],human_text_sha256=hashlib.sha256(x['text'].encode()).hexdigest(),ai_text_sha256=hashlib.sha256(x['paraphrased_text'].encode()).hexdigest(),existing_training_match=(g in existing or a in existing)))
# Assign whole normalized-human groups deterministically. Exact known-training groups only train.
groups=collections.defaultdict(list)
for r in records:groups[r['source_group']].append(r)
assignment={}
for g,rs in groups.items():
 if any(x['existing_training_match'] for x in rs):split='train'
 else:
  u=int(hashlib.sha256(('hip-pairs-v1:20261002:'+g).encode()).hexdigest()[:16],16)/16**16
  split='train' if u<.8 else 'selection' if u<.9 else 'reserved_test'
 assignment[g]=split
counts=collections.Counter();tokens=collections.Counter();sourcecounts=collections.Counter();maxpair=collections.Counter()
for split in ['train','selection','reserved_test']:
 p=P/(split+'.jsonl');t=p.with_suffix('.tmp')
 with t.open('w') as f:
  for r in records:
   if assignment[r['source_group']]!=split:continue
   r['split']=split;f.write(json.dumps(r,ensure_ascii=False)+'\n');counts[split]+=1;tokens[split]+=r['total_tokens'];sourcecounts[(split,r['dataset'],r['source'])]+=1;maxpair[split]=max(maxpair[split],r['total_tokens'])
 t.replace(p)
assert all(counts[s]>0 for s in ['train','selection','reserved_test'])
assert all(not r['existing_training_match'] for r in records if r['split']!='train')
# Each source group has exactly one partition; texts remain unmodified full documents.
assert all(len({r['split'] for r in rs})==1 for rs in groups.values())
manifest={'state':'prepared','time':time.time(),'source_revision':json.loads((O/'metadata.json').read_text())['sha'],'overlap_audit_sha256':hashlib.sha256((O/'audit.json').read_bytes()).hexdigest(),'pair_counts':dict(counts),'total_processed_tokens_one_pass':dict(tokens),'max_pair_tokens':dict(maxpair),'counts_by_split_source':{str(k):v for k,v in sourcecounts.items()},'source_groups':len(groups),'forced_train_existing_source_groups':sum(any(r['existing_training_match'] for r in rs) for rs in groups.values()),'exclusions':dict(excluded),'split_policy':'80/10/10 deterministic SHA256 of normalized-human source group; any exact human OR paraphrase match with existing stage-training inventory forced train','split_seed':'hip-pairs-v1:20261002','existing_training_files':files,'files':{s:{'path':str(P/(s+'.jsonl')),'sha256':hashlib.sha256((P/(s+'.jsonl')).read_bytes()).hexdigest()} for s in counts},'whole_documents':True,'max_tokens_each_including_specials':512,'tokenizer':str(R/'control/run/tokenizer'),'labels':'Paired human original=0, AI paraphrase=1 at document level only; no tokenlabels; not a localized editing dataset','training_ready':False,'remaining':['Cross-partition near-duplicate group strengthening before blind-use claims','Matched paired-ranking loss/control validation','Choose fixed token-exposure sampling protocol; one-pass totals recorded','Review GRADTEX experiment before new training registration'],'test_status':'Reserved prospective split; no model scores accessed. Historical broader research exposure and semantic paraphrase leakage are not ruled out.'}
save('manifest.json',manifest);status(state='complete',counts=dict(counts),training_ready=False);print(json.dumps(manifest,indent=2))
