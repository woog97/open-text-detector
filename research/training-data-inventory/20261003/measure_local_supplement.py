from pathlib import Path
import json,collections,random,statistics,math,gzip,hashlib
import pyarrow.parquet as pq
from tokenizers import Tokenizer
R=Path(__file__).resolve().parents[3];O=Path(__file__).parent;out={}
for name in ['research/data/reviewbench/original','research/exports/paper-text-hf/data','research/cache/iclr']:
 items=[]
 for p in (R/name).rglob('*.parquet'):
  f=pq.ParquetFile(p);items.append({'path':str(p.relative_to(R)),'rows':f.metadata.num_rows,'columns':f.schema_arrow.names})
 out[name]=items
p=R/'research/exports/paper-text-hf/data/batch-001.parquet';f=pq.ParquetFile(p);key=next(k for k in ['text','canonical_text','paper_text'] if k in f.schema_arrow.names)
tab=pq.read_table(p,columns=[key]);ids=random.Random(42).sample(range(tab.num_rows),100);texts=tab.take(ids)[key].to_pylist();T=Tokenizer.from_file(str(R/'models/meld-v8/tokenizer.json'));T.no_truncation();T.no_padding();counts=[len(e.ids) for e in T.encode_batch(texts,add_special_tokens=False)]
out['paper_archive_estimate']={'rows':tab.num_rows,'sample_rows':100,'estimated_tokens':round(statistics.mean(counts)*tab.num_rows),'sampling_95pct_margin':round(1.96*statistics.stdev(counts)/10*tab.num_rows),'field':key,'classification':'unlabeled_source_archive'}
# Hash only the stored textual field to identify older paper versions already contained in the final corpus.
base=R/'research/data/paper-gap10000-v3-luna-20260930/dataset.jsonl';current=set()
for l in base.open():
 r=json.loads(l);current.add(r['text_sha256'])
old={}
for p in sorted((R/'research/data').glob('paper-*/dataset.jsonl')):
 if p==base or 'eval-workflow' in str(p):continue
 n=overlap=0;unique=set()
 for l in p.open():
  r=json.loads(l);h=r.get('text_sha256') or hashlib.sha256(r.get('text','').encode()).hexdigest();n+=1;overlap+=h in current;unique.add(h)
 old[str(p.parent.relative_to(R))]={'rows':n,'unique_passage_texts':len(unique),'rows_exactly_in_current_20000':overlap,'unique_passages_not_in_current':len(unique-current)}
out['historical_exact_overlap']=old
p=R/'benchmarks/otd/eval_suite/bundle/manifest.json';d=json.loads(p.read_text());out['eval_bundle_manifest']=d
p=R/'benchmarks/otd/training/backbone-comparison/prepared/train.jsonl.gz'
with gzip.open(p,'rt') as f:r=json.loads(next(f));out['backbone_train_fields']=list(r)
p=R/'benchmarks/otd/exports/arena-prose-100-49-models/data/test-00000-of-00001.parquet';rows=pq.read_table(p,columns=['model_id','prompt_id','mechanically_eligible']).to_pylist();out['arena_exact']={'rows':len(rows),'models':len({r['model_id'] for r in rows}),'prompts':len({r['prompt_id'] for r in rows}),'mechanically_eligible':sum(r['mechanically_eligible'] for r in rows)}
(O/'local-supplement.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='eval_bundle_manifest'},indent=2))
