"""Freeze train/development/calibration data and identical cross-backbone windows."""
from runtime import require_space, SPACE_CACHE
import collections,gzip,json,random,difflib,re
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from data import sha,normalized_hash,shared_crop
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[3]
OLD=ROOT.parent/'paper-v3-modernbert';EXPORT=PROJECT/'benchmarks/otd/exports/ai-paper-provenance-v3-10000/data/fresh_passages'
GOOD={'fully_faithful','mostly_faithful_with_minor_differences'}
def write(path,rows):
 blob=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode();path.write_bytes(gzip.compress(blob,mtime=0));return {'sha256':__import__('hashlib').sha256(blob).hexdigest(),'rows':len(rows),'papers':len({r['paper_id'] for r in rows})}
def main():
 require_space()
 dest=ROOT/'prepared'
 if (dest/'manifest.json').exists():raise RuntimeError('Frozen data already prepared; use a new version to rebuild')
 dest.mkdir(exist_ok=True);lock=json.loads((ROOT/'models.lock.json').read_text());toks=[AutoTokenizer.from_pretrained(**{'pretrained_model_name_or_path':r['repo'],'revision':r['revision'],'local_files_only':True}) for r in lock.values()]
 cols=['id','pair_id','paper_id','forum_id','text','regions','target_start','target_end','target_label','development_exposed','eligible_for_pilot_training','paired_generation_fidelity_verdict','authors']
 source={s:pq.read_table(EXPORT/(s+'-00000-of-00001.parquet'),columns=cols).to_pylist() for s in ['train','validation','test']}
 forbidden=set();testpapers={r['paper_id'] for r in source['test']}
 for r in source['test']:forbidden.add(normalized_hash(r['text']));forbidden.add(normalized_hash(r['text'][r['target_start']:r['target_end']]))
 workflow=PROJECT/'research/data/paper-eval-workflows-luna-20260930'
 for name in ['papers.jsonl','human-body.jsonl','dataset.jsonl']:
  for line in (workflow/name).open():
   r=json.loads(line);testpapers.add(r['paper_id'])
   if r.get('text'):forbidden.add(normalized_hash(r['text']))
   if r.get('original_target'):forbidden.add(normalized_hash(r['original_target']))
 # Exclude all exact public benchmark texts/targets, including previously inspected diagnostics.
 bundle=PROJECT/'benchmarks/otd/eval_suite/bundle'
 for profile in ['comparison','full','context']:
  for line in gzip.open(bundle/(profile+'.jsonl.gz'),'rt'):
   r=json.loads(line);forbidden.add(normalized_hash(r['text']))
   if 'target_start' in r and 'target_end' in r:forbidden.add(normalized_hash(r['text'][r['target_start']:r['target_end']]))
 target_splits=collections.defaultdict(set);target_labels=collections.defaultdict(set)
 for split,rs in source.items():
  for r in rs:
   h=normalized_hash(r['text'][r['target_start']:r['target_end']]);target_splits[h].add(split);target_labels[h].add(r['target_label'])
 forbidden|={h for h,s in target_splits.items() if len(s)>1 or len(target_labels[h])>1}
 pools={};rejected=collections.Counter()
 for split in ['train','validation']:
  pairs=collections.defaultdict(list)
  for r in source[split]:pairs[r['pair_id']].append(r)
  retained=[]
  for pair,rs in pairs.items():
   if len(rs)!=2 or {r['target_label'] for r in rs}!={0,1}:rejected['incomplete_pair']+=1;continue
   if not all(not r['development_exposed'] and r['eligible_for_pilot_training'] and r['paired_generation_fidelity_verdict'] in GOOD for r in rs):rejected['quality_or_development']+=1;continue
   if any(r['paper_id'] in testpapers or normalized_hash(r['text']) in forbidden or normalized_hash(r['text'][r['target_start']:r['target_end']]) in forbidden for r in rs):rejected['heldout_overlap']+=1;continue
   for r in rs:
    r['kind']='paired';r['split']=split;retained.append(r)
  path=OLD/'wide-eval-v1'/('human_remaining_'+split+'.jsonl')
  for line in path.open():
   r=json.loads(line)
   if r['development_exposed'] or not r['clean_prose'] or r['cohort']!='novel_body' or r['text_appears_in_multiple_splits'] or r['paper_id'] in testpapers or normalized_hash(r['text']) in forbidden:continue
   retained.append({'id':r['id'],'paper_id':r['paper_id'],'forum_id':r.get('forum_id'),'text':r['text'],'regions':r['regions'],'kind':'novel_human','target_start':0,'target_end':len(r['text']),'split':split})
  pools[split]=retained
 # Remove exact train/validation collisions, at whole-pair granularity.
 valhash={normalized_hash(r['text']) for r in pools['validation']};badpairs={r.get('pair_id') for r in pools['train'] if normalized_hash(r['text']) in valhash and r.get('pair_id')}
 pools['train']=[r for r in pools['train'] if normalized_hash(r['text']) not in valhash and r.get('pair_id','') not in badpairs]
 assert not ({r['paper_id'] for r in pools['train']}&{r['paper_id'] for r in pools['validation']})
 # Equal halves by paper hash; all views/body paragraphs of a paper stay together.
 papers=sorted({r['paper_id'] for r in pools['validation']},key=lambda p:sha('validation-partition-42/'+p));cal=set(papers[::2])
 splits={'train':pools['train'],'selection':[r for r in pools['validation'] if r['paper_id'] not in cal],'calibration':[r for r in pools['validation'] if r['paper_id'] in cal]}
 manifest={'version':1,'models':lock,'seed':42,'files':{},'excluded_pairs':dict(rejected),'forbidden_text_hashes':len(forbidden),'heldout_papers':len(testpapers),'policy':'75% paired / 25% novel human; choose paper, target and variant uniformly within stratum. Exact heldout text exclusion. Original paper splits retained; author overlap across old splits is not excluded. All new workflow splits held out.','source_hashes':{}}
 for p in [*(EXPORT.glob('*.parquet')),*(OLD/'wide-eval-v1').glob('human_remaining_*.jsonl')]:manifest['source_hashes'][str(p.relative_to(PROJECT))]=__import__('hashlib').sha256(p.read_bytes()).hexdigest()
 for split,rs in splits.items():manifest['files'][split]=write(dest/(split+'.jsonl.gz'),rs)
 # Shared schedule is sampled before tokenization. Both models see identical text and supervision.
 bykind={}
 for kind in ['paired','novel_human']:
  groups=collections.defaultdict(lambda:collections.defaultdict(list))
  for r in splits['train']:
   if r['kind']==kind:groups[r['paper_id']][r.get('pair_id',r['id'])].append(r)
  if not groups:raise ValueError('Empty training stratum '+kind)
  bykind[kind]=groups
 for stage,epochs,draws in [(1,1,6000),(2,3,12000)]:
  for epoch in range(epochs):
   rng=random.Random(42+stage*10000+epoch);windows=[]
   for i in range(draws):
    kind='novel_human' if i%4==0 else 'paired';groups=bykind[kind];paper=rng.choice(sorted(groups));pair=rng.choice(sorted(groups[paper]));row=rng.choice(groups[paper][pair]);mode=['target','context','random'][i%3]
    w=shared_crop(row,rng,toks,mode);w['draw_id']=f's{stage}-e{epoch}-{i}';windows.append(w)
   name=f'stage{stage}-epoch{epoch}';manifest['files'][name]=write(dest/(name+'.jsonl.gz'),windows)
 for split in ['selection','calibration']:
  windows=[];groups=collections.defaultdict(list)
  for r in splits[split]:groups[(r['paper_id'],r['kind'])].append(r)
  for key,rs in sorted(groups.items()):
   rng=random.Random(int(sha('/'.join(key))[:12],16));rs=sorted(rs,key=lambda r:sha(r['id']))
   # Keep paired targets, cap novel-human paragraphs at eight per paper.
   for r in (rs if key[1]=='paired' else rs[:8]):
    windows.append(shared_crop(r,rng,toks,'target' if key[1]=='paired' else 'context'))
  manifest['files'][split+'-windows']=write(dest/(split+'-windows.jsonl.gz'),windows)
 for split in splits:manifest[split+'_strata']=dict(collections.Counter(r['kind'] for r in splits[split]))
 (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k not in ['source_hashes','files']},indent=2));print(json.dumps(manifest['files'],indent=2))
if __name__=='__main__':main()
