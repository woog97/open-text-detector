"""Read local corpus metadata and estimate text tokens from deterministic samples."""
from pathlib import Path
import json,random,hashlib,collections,statistics,math
import pyarrow.parquet as pq
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).parent
T=Tokenizer.from_file(str(ROOT/'models/meld-v8/tokenizer.json'));T.no_truncation();T.no_padding()
def estimate(rows,key,n=500):
 sample=random.Random(42).sample(rows,min(n,len(rows)));ns=[len(e.ids) for e in T.encode_batch([x[key] for x in sample],add_special_tokens=False)]
 return {'rows':len(rows),'sample_rows':len(ns),'estimated_tokens':round(statistics.mean(ns)*len(rows)) if ns else 0,'mean_tokens':statistics.mean(ns) if ns else 0,'sampling_95pct_margin':round(1.96*statistics.stdev(ns)/math.sqrt(len(ns))*len(rows)) if len(ns)>1 else None,'text_field':key}
out={'tokenizer_sha256':hashlib.sha256((ROOT/'models/meld-v8/tokenizer.json').read_bytes()).hexdigest(),'seed':42,'counts':{}}
p=ROOT/'research/data/synthetic-papers-600-hf-20261003/data/train-00000-of-00001.parquet';rows=pq.read_table(p).to_pylist();groups=collections.defaultdict(list)
for r in rows:groups[r['model']].append(r)
out['counts']['synthetic_full_papers']={'rows':len(rows),'models':{k:estimate(v,'paper_markdown',40) for k,v in groups.items()},'unique_text_hashes':len({r['paper_sha256'] for r in rows}),'unique_source_papers':len({r['source_paper_id'] for r in rows}),'columns':list(rows[0]),'path':str(p.relative_to(ROOT))}
for view in ['paragraphs','passages']:
 base=ROOT/'benchmarks/otd/exports/ai-paper-provenance-v3-10000/data'/view;info={}
 for p in sorted(base.glob('*.parquet')):
  cols=pq.ParquetFile(p).schema_arrow.names
  rs=pq.read_table(p,columns=['text']+(['label'] if 'label' in cols else [])).to_pylist();info[p.name]=estimate(rs,'text')
 out['counts']['paper_10000_'+view]=info
base=ROOT/'benchmarks/otd/exports/arena-prose-100-49-models';fs=list(base.rglob('*.parquet'));out['counts']['arena']={}
for p in fs:
 f=pq.ParquetFile(p);rs=pq.read_table(p).to_pylist();keys=[k for k in ['text','response','response_text'] if k in rs[0] and isinstance(rs[0][k],str)];out['counts']['arena'][str(p.relative_to(ROOT))]=estimate(rs,keys[0]) if keys else {'rows':len(rs),'columns':list(rs[0])}
for name in ['ai_mdta_2025','human_pg19','iclr_2023','iclr_2026']:
 for p in (ROOT/'research/data'/name).glob('*.jsonl'):
  rows=[json.loads(l) for l in p.open() if l.strip()];keys=[k for k in ['text','response','abstract'] if k in rows[0] and isinstance(rows[0][k],str)]
  out['counts'][name]=estimate(rows,keys[0],100) if keys else {'rows':len(rows),'columns':list(rows[0])}
prepared=ROOT/'benchmarks/otd/data/prepared';out['prepared_benchmarks']={}
for p in sorted(prepared.glob('*.jsonl')):
 rows=[json.loads(l) for l in p.open() if l.strip()];out['prepared_benchmarks'][p.stem]={'rows':len(rows),'labels':dict(collections.Counter(str(r.get('label')) for r in rows)),'unique_texts':len({r.get('text') for r in rows})}
out['historical_paper_runs']={}
for p in sorted((ROOT/'research/data').glob('paper-*/summary.json')):
 d=json.loads(p.read_text());out['historical_paper_runs'][str(p.parent.relative_to(ROOT))]={k:v for k,v in d.items() if k in ['papers','rows','source_passages','generated_paragraphs','eligible_rows','ai_region_tokens','counts','operations','generated','examples','total','splits'] or isinstance(v,(int,float))}
(OUT/'local-measurements.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
