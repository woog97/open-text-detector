"""Prepare shared coefficient0/.1 paper-pair data, without registering GPU jobs."""
from pathlib import Path
import json,gzip,collections,random,hashlib,sys
R=Path('/data/workspace/paper-diversity-v1');O=R/'objective-pair-ranking';P=O/'prepared'
assert not P.exists(),'Do not overwrite existing paired data'
P.mkdir();sys.path.insert(0,str(R/'control'))
from transformers import AutoTokenizer
from data import encode_example
from objective_pairwise import pair_indices
tok=AutoTokenizer.from_pretrained(R/'control/run/tokenizer',local_files_only=True)
def rows(p):return [json.loads(x) for x in gzip.decompress(p.read_bytes()).splitlines()]
pool=rows(O/'paper-pair-pool.jsonl.gz');index=collections.defaultdict(list)
for x in pool:index[(x['human_token_count'],x['ai_token_count'])].append(x)
m=json.loads((R/'control/prepared/manifest.json').read_text());checks={}
for p in sorted((R/'control/prepared').glob('*.gz')):
 if not p.name.startswith('stage2'):(P/p.name).symlink_to(p);continue
 base=rows(p);ls=[len(t)+2 for t in tok([r['text'] for r in base],add_special_tokens=False)['input_ids']];out=[dict(r) for r in base]
 rng=random.Random(42001+len(checks));slots=list(range(0,len(base),2));rng.shuffle(slots);sourcecounts=collections.Counter();usedtokens=0;pairedrows=set();class_tokens=collections.Counter()
 for i in slots:
  if usedtokens>=.1*sum(ls):break
  candidates=[]
  for dh in range(-3,4):
   for da in range(-3,4):candidates.extend(x for x in index[(ls[i]+dh,ls[i+1]+da)] if sourcecounts[x['source_group']]<3)
  if not candidates:continue
  x=rng.choice(candidates);pid=p.stem+'-draw-'+str(i)+'-'+x['id'];sourcecounts[x['source_group']]+=1
  for j,role in [(i,'human'),(i+1,'ai')]:
   r=dict(x[role]);r.update(ranking_pair_id=pid,ranking_role=role,verified_pair_relation=True,draw_id=f'{p.stem}-{j}')
   out[j]=r;pairedrows.add(j);class_tokens[role]+=x[role+'_token_count']
  usedtokens+=x['total_tokens']
 lengths=[len(encode_example(r,tok,'encoder',2)['ids']) for r in out]
 for i in pairedrows:assert abs(lengths[i]-ls[i])<=3
 for start in range(0,len(out),8):pair_indices(out[start:start+8])
 share=usedtokens/sum(lengths);assert .099<=share<=.101,(p.name,share)
 assert abs(sum(lengths)/sum(ls)-1)<.005
 assert max(sourcecounts.values())<=3 and len(out)==12000
 blob=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out).encode();(P/p.name).write_bytes(gzip.compress(blob,mtime=0));key=p.name.removesuffix('.jsonl.gz')
 m['files'][key]={'rows':len(out),'sha256':hashlib.sha256(blob).hexdigest(),'papers':len({r['paper_id'] for r in out})}
 checks[key]={'paired_rows':len(pairedrows),'pairs':len(pairedrows)//2,'paired_token_share':share,'processed_tokens':sum(lengths),'control_tokens':sum(ls),'paired_class_tokens':dict(class_tokens),'source_groups':len(sourcecounts),'max_pair_draws_per_source':max(sourcecounts.values()),'all_pairs_within_microbatch8':True,'length_match_within3':True,'all_adapter_rows_valid':True}
m['experiment']={'purpose':'shared paired-data control/ranking manifests','production_ready':False,'ranking_weights':[0,.1],'stage1':'unchanged','total_draws':42000,'generation':False}
(P/'manifest.json').write_text(json.dumps(m,indent=2));report={'state':'data_prepared_not_gpu_ready','checks':checks,'production_gate':'GRADTEX review, controlled question, pair-aware BF16 preflight; no registered GPU job'}
(O/'budget-manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
