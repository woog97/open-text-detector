"""Remote-only corpus census and token samples using an existing remote tokenizer."""
from pathlib import Path
import json,csv,tarfile,random,statistics,math,collections,datetime,hashlib
import pyarrow.parquet as pq
from tokenizers import Tokenizer
p=Path('/data/workspace/paper-v3-modernbert-20260930/baseline-models/meld-v8/tokenizer.json')
T=Tokenizer.from_file(str(p));T.no_truncation();T.no_padding()
print(json.dumps({'tokenizer':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}),flush=True)
def sample_stream(it,key='text',n=500):
 rng=random.Random(42);sample=[];count=0;labels=collections.Counter();sources=collections.Counter()
 for row in it:
  count+=1
  for k in ['binary_label','label','text_type']:
   if k in row:labels[str(row[k])]+=1;break
  if 'source_id' in row:sources[row['source_id']]+=1
  if len(sample)<n:sample.append(row)
  else:
   i=rng.randrange(count)
   if i<n:sample[i]=row
 ns=[len(e.ids) for e in T.encode_batch([r.get(key) or '' for r in sample],add_special_tokens=False)]
 return {'rows':count,'sample_rows':len(ns),'mean_tokens':statistics.mean(ns) if ns else 0,'estimated_tokens':round(statistics.mean(ns)*count) if ns else 0,'sampling_95pct_margin':round(1.96*statistics.stdev(ns)/math.sqrt(len(ns))*count*math.sqrt(max(0,(count-len(ns))/max(1,count-1)))) if len(ns)>1 else 0,'labels':dict(labels),'sources':dict(sources),'field':key}
def emit(name,data):print(json.dumps({'source':name,'data':data}),flush=True)
b=Path('/data/workspace/paper-diversity-v1')
for p in [b/'raid.csv',b/'mage-train.csv',b/'mage-valid.csv',b/'mage-test.csv']:
 try:
  with p.open(newline='') as f:
   reader=csv.DictReader(f);fields=reader.fieldnames;key=next((k for k in ['generation','text','content'] if k in fields),None)
   emit(str(p),sample_stream(reader,key) if key else {'columns':fields})
 except Exception as e:emit(str(p),{'error':type(e).__name__+':'+str(e)[:150]})
for p in list((b/'public-source-audit').rglob('*.parquet'))+list(Path('/data/workspace/open-pangram-private-copy/dataset').rglob('*.parquet')):
 f=pq.ParquetFile(p);cols=f.schema_arrow.names;wanted=[k for k in ['text','binary_label','text_type'] if k in cols]
 rows=(r for batch in f.iter_batches(columns=wanted,batch_size=1024) for r in batch.to_pylist());emit(str(p),sample_stream(rows))
 if 'paraphrased_text' in cols:
  rows=(r for batch in f.iter_batches(columns=['paraphrased_text'],batch_size=1024) for r in batch.to_pylist());emit(str(p)+'#paraphrases',sample_stream(rows,'paraphrased_text'))
for p in sorted((b/'public-source-audit/seqxgpt-original').glob('*.jsonl')):
 with p.open() as f:emit(str(p),sample_stream((json.loads(l) for l in f if l.strip())))
p=Path('/data/workspace/synthetic-mirrors-luna-dollar-v1-recovered-20261002/raw-30000-20261002/checkpoints/00012-1790984912.tar.gz')
with tarfile.open(p,'r|gz') as tar:
 for m in tar:
  if m.name=='run/raw-documents.jsonl':emit('raw_mirrors_final_export',sample_stream((json.loads(l) for l in tar.extractfile(m) if l.strip()),n=1000))
  elif m.name in ['run/raw-export-receipt.json','run/raw-production-status.json']:emit(m.name,json.load(tar.extractfile(m)))
emit('checked_at',datetime.datetime.now(datetime.timezone.utc).isoformat())
