"""Verify every frozen window with both tokenizers and all paper partitions."""
from runtime import require_space, SPACE_CACHE
import gzip,json,hashlib
from pathlib import Path
from transformers import AutoTokenizer
from data import encode_example
ROOT=Path(__file__).resolve().parent
require_space()
m=json.loads((ROOT/'prepared/manifest.json').read_text());allrows={};unique={}
for name,info in m['files'].items():
 blob=gzip.decompress((ROOT/'prepared'/(name+'.jsonl.gz')).read_bytes());assert hashlib.sha256(blob).hexdigest()==info['sha256'];rows=[json.loads(l) for l in blob.splitlines()];assert len(rows)==info['rows'];allrows[name]=rows
 if name.startswith('stage') or name.endswith('-windows'):
  for r in rows:unique[r['text']]=r
papers={name:{r['paper_id'] for r in allrows[name]} for name in ['train','selection','calibration']}
assert not(papers['train']&papers['selection'] or papers['train']&papers['calibration'] or papers['selection']&papers['calibration'])
report={'paper_partitions_disjoint':True,'unique_windows':len(unique),'models':{}}
for kind,info in m['models'].items():
 tok=AutoTokenizer.from_pretrained(info['repo'],revision=info['revision'],local_files_only=True);n=0;maximum=0
 for r in unique.values():
  e=encode_example(r,tok,'encoder' if kind=='encoder' else 'causal',2);assert any(y>=0 for y in e['source_labels']);assert len(e['source_positions'])==len(e['source_labels']);maximum=max(maximum,len(e['source_labels']));n+=1
 report['models'][kind]={'windows_validated':n,'max_source_tokens':maximum}
 print(kind,n,maximum,flush=True)
(ROOT/'prepared/audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
