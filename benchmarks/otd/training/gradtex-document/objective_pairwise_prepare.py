"""Audit existing TRAIN original/generated target pairs; no GPU job registration."""
from pathlib import Path
import sys,json,gzip,hashlib,collections
R=Path('/data/workspace/paper-diversity-v1');O=R/'objective-pair-ranking';O.mkdir(exist_ok=True)
out=O/'paper-pair-pool.jsonl.gz'
assert not out.exists(),'Preserve existing preparation; do not overwrite'
sys.path.insert(0,str(R/'control'))
from data import crop,encode_example
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(R/'control/run/tokenizer',local_files_only=True)
def sha(x):return hashlib.sha256(x.encode()).hexdigest()
rows=[json.loads(x) for x in gzip.decompress((R/'control/prepared/train.jsonl.gz').read_bytes()).splitlines()]
groups=collections.defaultdict(list)
for r in rows:
 if r.get('kind')=='paired':groups[r['pair_id']].append(r)
blocked=collections.Counter();pairs=[]
for pid,rs in sorted(groups.items()):
 h=[r for r in rs if r['id'].endswith('/human_original')];a=[r for r in rs if r['id'].endswith('/paragraph_generate')]
 if len(h)!=1 or len(a)!=1:blocked['missing_or_ambiguous_member']+=1;continue
 h,a=h[0],a[0]
 if any(r.get('split')!='train' or r.get('development_exposed') or not r.get('eligible_for_pilot_training') for r in [h,a]):blocked['not_clean_train']+=1;continue
 if h['paper_id']!=a['paper_id'] or h['text'][:h['target_start']]!=a['text'][:a['target_start']] or h['text'][h['target_end']:]!=a['text'][a['target_end']:]:blocked['context_mismatch']+=1;continue
 hh=crop(h,h['target_start'],h['target_end']);aa=crop(a,a['target_start'],a['target_end'])
 if ' '.join(hh['text'].lower().split())==' '.join(aa['text'].lower().split()):blocked['unchanged_AI']+=1;continue
 try:eh=encode_example(hh,tok,'encoder',2);ea=encode_example(aa,tok,'encoder',2)
 except ValueError:blocked['overlength_or_empty']+=1;continue
 if set(eh['source_labels'])!={0} or set(ea['source_labels'])!={1}:blocked['target_provenance_not_pure']+=1;continue
 for r,role in [(hh,'human'),(aa,'ai')]:r.update(ranking_pair_id=pid,ranking_role=role,verified_pair_relation=True)
 pairs.append({'id':pid,'source_group':h['paper_id'],'human':hh,'ai':aa,'human_token_count':len(eh['ids']),'ai_token_count':len(ea['ids']),'total_tokens':len(eh['ids'])+len(ea['ids']),'human_sha256':sha(hh['text']),'ai_sha256':sha(aa['text']),'supervision':'verified_target_token_gold','verified_pair_relation':'original TRAIN pair_id, identical preserved prefix/suffix, complete0human/1AI target labels'})
blob=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pairs).encode();out.write_bytes(gzip.compress(blob,mtime=0))
report={'state':'pair_pool_audited_not_training_ready','eligible_pairs':len(pairs),'paper_groups':len({r['source_group'] for r in pairs}),'excluded':dict(blocked),'sha256_uncompressed':hashlib.sha256(blob).hexdigest(),'total_tokens_range':[min(r['total_tokens'] for r in pairs),max(r['total_tokens'] for r in pairs)],'source':'existing control prepared TRAIN; original paper/source/heldout policy retained','new_generation':False,'model_assets_downloaded':False,'gate':'No GPU training registration; require current GRADTEX review, identical paired-data coefficient0 control, budget/order manifests and BF16 preflight.'}
(O/'paper-pair-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
