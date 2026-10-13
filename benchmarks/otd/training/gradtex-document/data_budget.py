"""Build deterministic matched-budget document-supervision rows; no training registration."""
from pathlib import Path
import collections,gzip,hashlib,json,random,time
from transformers import AutoTokenizer
R=Path('/data/workspace/paper-diversity-v1');O=R/'gradtex-document-preparation';P=O/'prepared'
def save(p,v):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def read(p):return [json.loads(l) for l in gzip.decompress(p.read_bytes()).splitlines()]
assert (O/'donor-manifest.json').exists()
assert not (O/'budget-manifest.json').exists()
P.mkdir(exist_ok=True)
donors=[json.loads(l) for l in (O/'donors.jsonl').read_text().splitlines()]
by=collections.defaultdict(list)
for d in donors:by[(d['document_label'],d['token_count'])].append(d)
tok=AutoTokenizer.from_pretrained(str(R/'control/run/tokenizer'),local_files_only=True)
manifest=json.loads((R/'control/prepared/manifest.json').read_text());checks={}
for p in sorted((R/'control/prepared').glob('*.gz')):
 if not p.name.startswith('stage2'):
  if not (P/p.name).exists():(P/p.name).symlink_to(p)
  continue
 base=read(p);out=list(base)
 lens=[len(x) for x in tok([x['text'] for x in base],add_special_tokens=True,truncation=False)['input_ids']]
 total=sum(lens);rng=random.Random(72904+sum(map(ord,p.name)));order=list(range(len(base)));rng.shuffle(order)
 class_tokens=collections.Counter();class_draws=collections.Counter();sources=collections.Counter();scenarios=collections.Counter();generators=collections.Counter();delta=0
 for i in order:
  if sum(class_tokens.values())>=.1*(total+delta):break
  label=0 if class_tokens[0]<=class_tokens[1] else 1
  candidates=[d for n in range(max(1,lens[i]-3),lens[i]+4) for d in by[(label,n)] if sources[d['source_group']]<3]
  if not candidates:continue
  rng.shuffle(candidates)
  # Underrepresented scenarios then generators; random tie-break within exact lengths.
  d=min(candidates,key=lambda d:(scenarios[(label,d['scenario'])],generators[(label,d['generator_model'])]))
  row=dict(d,draw_id=base[i].get('draw_id',str(i)),paper_id='gradtex/'+d['source_group'],kind='external_document',dataset='gradtex',regions=[],target_start=0,target_end=len(d['text']))
  out[i]=row;class_tokens[label]+=d['token_count'];class_draws[label]+=1;sources[d['source_group']]+=1;scenarios[(label,d['scenario'])]+=1;generators[(label,d['generator_model'])]+=1;delta+=d['token_count']-lens[i]
 ext=sum(class_tokens.values());share=ext/(total+delta)
 assert .099<=share<=.101,(p.name,share)
 assert abs(delta)/total<=.005
 assert abs(class_tokens[0]-class_tokens[1])/ext<=.01
 assert max(sources.values())<=3
 for i,x in enumerate(out):
  if x.get('dataset')=='gradtex':
   assert x['token_labels'] is None and x['regions']==[]
   assert abs(x['token_count']-lens[i])<=3
   assert len(tok(x['text'],add_special_tokens=True,truncation=False)['input_ids'])==x['token_count']<=512
 blob=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out).encode();(P/p.name).write_bytes(gzip.compress(blob,mtime=0));key=p.name.removesuffix('.jsonl.gz')
 manifest['files'][key]={'rows':len(out),'sha256':hashlib.sha256(blob).hexdigest(),'papers':len({x['paper_id'] for x in out})}
 checks[key]={'external_token_share':share,'processed_tokens':total+delta,'control_tokens':total,'class_tokens':dict(class_tokens),'class_draws':dict(class_draws),'scenarios':{str(k):v for k,v in scenarios.items()},'generators':{str(k):v for k,v in generators.items()},'source_groups':len(sources),'max_draws_per_source':max(sources.values()),'rows':len(out)}
 print(json.dumps({'file':p.name,**checks[key]}),flush=True)
manifest['experiment']={'whole_document_supervision_only':True,'stage1':'unchanged','stage2_external_token_target':.1,'donor_manifest':str(O/'donor-manifest.json'),'no_token_gold':True,'data_checks':checks,'training_ready':False}
save(P/'manifest.json',manifest)
save(O/'budget-manifest.json',{'state':'data_prepared','time':time.time(),'checks':checks,'prepared':str(P),'training_ready':False,'remaining':'Integrate document loss and validate adapter, matched-head control, BF16 preflight before registration.'})
