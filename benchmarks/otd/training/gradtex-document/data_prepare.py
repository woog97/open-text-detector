"""Build audited whole-document donors; never derives token labels from document labels."""
from pathlib import Path
import collections, hashlib, json, os, re, time
import pyarrow.parquet as pq
from transformers import AutoTokenizer
R=Path('/data/workspace/paper-diversity-v1')
A=R/'public-source-audit'
O=R/'gradtex-document-preparation'
O.mkdir(exist_ok=True)
def save(name, value):
 p=O/name; t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)
def norm(t):return ' '.join(re.findall(r'\w+',t.lower()))
def sha(t):return hashlib.sha256(t.encode()).hexdigest()
def group(t):return sha(norm(t))
def status(**kw):save('status.json',dict(time=time.time(),**kw))
assert not (O/'donor-manifest.json').exists(), 'Do not overwrite completed preparation'
status(state='running',phase='loading')
audit=json.loads((A/'gradtex-near-audit.json').read_text())
assert json.loads((A/'gradtex-near-status.json').read_text())['state']=='complete'
bad=set(json.loads((A/'gradtex-excluded-source-hashes.json').read_text()))
assert len(bad)==audit['blocked_source_groups']
tok=AutoTokenizer.from_pretrained(str(R/'control/run/tokenizer'),local_files_only=True)
rows=pq.read_table(A/'gradtex-train.parquet').to_pylist()
label_by_hash=collections.defaultdict(set)
for x in rows:label_by_hash[group(x['text'])].add(1-int(x['binary_label']))
conflicts={h for h,v in label_by_hash.items() if len(v)>1}
seen=set();excluded=collections.Counter();counts=collections.Counter();generators=collections.Counter();domains=collections.Counter();lengths=collections.Counter();records=[]
for i,x in enumerate(rows):
 text=x['text'];src=x.get('human_source_text') or text;g=group(src);h=group(text)
 if g in bad:excluded['heldout_overlap_source_group']+=1;continue
 assert x['binary_label'] in [0,1]
 label=1-int(x['binary_label'])
 assert (x['multiclass_label']=='HWT')==(label==0)
 if h in conflicts:excluded['conflicting_document_labels']+=1;continue
 if not norm(text):excluded['empty']+=1;continue
 if label and h==group(src):excluded['ai_involved_identical_to_human_source']+=1;continue
 if h in seen:excluded['duplicate_normalized_document']+=1;continue
 seen.add(h)
 ids=tok(text,add_special_tokens=True,truncation=False)['input_ids'];n=len(ids)
 lengths['<=128' if n<=128 else '<=256' if n<=256 else '<=512' if n<=512 else '>512']+=1
 # Whole-document retention is essential: MIX labels cannot survive arbitrary crops.
 if n>512:excluded['whole_document_over_512_tokens']+=1;continue
 records.append(dict(id='gradtex-train-'+str(i),text=text,document_label=label,supervision='document_only',token_labels=None,source_group=g,text_sha256=sha(text),normalized_text_sha256=h,scenario=x['scenario'],generator_model=x.get('generator_model') or 'human',domain=x['domain'],token_count=n,source_split='train',upstream_binary_label=int(x['binary_label']),upstream_multiclass_label=x['multiclass_label']))
 counts[(label,x['scenario'])]+=1;generators[(label,x.get('generator_model') or 'human')]+=1;domains[(label,x['domain'])]+=1
 if i%5000==0:status(state='running',phase='tokenizing-whole-documents',rows=i,total=len(rows),eligible=len(records))
assert {r['document_label'] for r in records}=={0,1}
assert len({r['normalized_text_sha256'] for r in records})==len(records)
assert not {r['source_group'] for r in records}&bad
p=O/'donors.jsonl';tmp=p.with_suffix('.jsonl.tmp')
with tmp.open('w') as f:
 for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
tmp.replace(p)
manifest=dict(state='data_prepared',time=time.time(),source_repo='elisabeth-pl-pl/GRADTEX',revision='553d859da0255d75a39c385c208f7522a2007f53',source_file='gradtex-train.parquet',source_sha256=hashlib.sha256((A/'gradtex-train.parquet').read_bytes()).hexdigest(),donor_file=str(p),donor_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rows=len(records),source_groups=len({r['source_group'] for r in records}),counts_by_class_scenario={str(k):v for k,v in counts.items()},counts_by_class_generator={str(k):v for k,v in generators.items()},counts_by_class_domain={str(k):v for k,v in domains.items()},exclusions=dict(excluded),lengths_before_length_filter=dict(lengths),blocked_source_groups=len(bad),audit_file=str(A/'gradtex-near-audit.json'),audit_sha256=hashlib.sha256((A/'gradtex-near-audit.json').read_bytes()).hexdigest(),label_semantics='document_label=1 means AI-involved; upstream binary labels inverted. No token or sentence gold is inferred, including HWT/MGT.',whole_documents=True,max_tokens_with_specials=512,tokenizer=str(R/'control/run/tokenizer'),budget_policy='Donor pool only. Integrated experiment must sample class/scenario/generator strata and match replacement token budget; no donor pool automatically marks a GPU job ready.',training_ready=False)
save('donor-manifest.json',manifest);status(state='complete',phase='donor-pool-prepared',rows=len(records),training_ready=False)
print(json.dumps(manifest,indent=2))
