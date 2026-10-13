"""Check every packaged span/label and load all four views through Hugging Face."""
import json,re,unicodedata
from collections import Counter,defaultdict
import pyarrow.parquet as pq
from datasets import load_dataset
from tokenizers import Tokenizer
from paper_forum_ids import resolve as resolve_forum_ids
from package_paper_huggingface import OUT,SOURCE,SPLITS,MAP,sha,rows,save

def parquet_rows(path):
 for batch in pq.ParquetFile(path).iter_batches(batch_size=50):yield from batch.to_pylist()
def main():
 expected_forums,identifier_report=resolve_forum_ids(list(rows('papers.jsonl')),SOURCE)
 assert json.loads((OUT/'identifier_mapping.json').read_text())==identifier_report
 readers={s:iter(parquet_rows(OUT/'data/passages'/f'{s}-00000-of-00001.parquet')) for s in SPLITS}
 tok=Tokenizer.from_file(str(OUT/'tokenizer/tokenizer.json'));ids=set();paper_splits={};author_splits={};pair_labels=defaultdict(list);parents={};ai_tokens=0
 for original in rows('dataset.jsonl'):
  r=next(readers[original['split']]);assert r['id']==original['id'] and r['text']==original['text'];assert r['forum_id']==expected_forums[r['paper_id']];assert r['text_sha256']==sha(r['text'])
  assert r['id'] not in ids;ids.add(r['id']);paper=r['paper_id'];assert paper not in paper_splits or paper_splits[paper]==r['split'];paper_splits[paper]=r['split']
  for name in r['authors']:
   key=re.sub(r'\W+','',unicodedata.normalize('NFKC',name).casefold());assert key not in author_splits or author_splits[key]==r['split'];author_splits[key]=r['split']
  ai=original['operation']=='paragraph_generate';pair_labels[r['pair_id']].append(r['target_label']);assert r['label']==(2 if ai else 0) and r['has_ai']==ai and r['variant']==int(ai)
  n=len(r['input_ids']);assert all(len(r[k])==n for k in ['attention_mask','token_labels','token_loss_mask','token_start','token_end','token_text'])
  encoded=tok.encode(r['text'],add_special_tokens=False);assert encoded.ids==r['input_ids'];assert encoded.offsets==list(zip(r['token_start'],r['token_end']))
  for j,t in enumerate(original['tokens']):
   assert r['token_text'][j]==r['text'][r['token_start'][j]:r['token_end'][j]]==t['text']
   expected=MAP[t['label']] if t['loss_mask'] and MAP[t['label']]<2 else -100
   assert r['token_labels'][j]==expected and r['token_loss_mask'][j]==(expected!=-100)
   # Independently derive supervision from character provenance, not diff text.
   labels={reg['label'] for reg in r['regions'] if max(t['start'],reg['start'])<min(t['end'],reg['end'])}
   assert expected==(next(iter(labels)) if len(labels)==1 else -100)
  ai_tokens+=r['token_labels'].count(1)
  assert r['regions'][0]['start']==0 and r['regions'][-1]['end']==len(r['text'])
  assert all(a['end']==b['start'] for a,b in zip(r['regions'],r['regions'][1:]))
  assert len(r['sentences'])==len(original['sentences'])
  for a,b in zip(r['sentences'],original['sentences']):
   assert a['text']==r['text'][a['start']:a['end']]==b['text'];assert a['label']==MAP[b['label']]
   assert a['binary_label']==(a['label'] if a['label']<2 else -100);assert a['loss_mask']==(a['label']<2)
  parents[r['id']]={k:r[k] for k in ['paper_id','forum_id','text','split','target_start','target_end','target_label','sentences','development_exposed','eligible_for_pilot_training','quality_flags']}
 for it in readers.values():assert next(it,None) is None
 assert len(ids)==5000 and len(paper_splits)==500 and len(pair_labels)==2500 and all(sorted(x)==[0,1] for x in pair_labels.values()) and ai_tokens==323204
 expected_counts=json.loads((OUT/'statistics.json').read_text())['splits'];loaded_counts={};seen_sentence_ids=set();sentence_counts=Counter();paragraph_counts=Counter()
 for config in ['passages','fresh_passages','paragraphs','sentences']:
  ds=load_dataset(str(OUT),config,cache_dir='/tmp/pangram-paper-hf-loadcheck')
  assert set(ds)==set(SPLITS);assert all('paper_id' in ds[s].features and 'forum_id' in ds[s].features for s in SPLITS);loaded_counts[config]={s:len(ds[s]) for s in SPLITS};assert loaded_counts[config]==expected_counts[config]
  if config in ['passages','fresh_passages']:
   assert ds['train'].features['label'].names==['human','ai','mixed']
   if config=='fresh_passages':
    for split in SPLITS:
     assert all(not x for x in ds[split]['development_exposed'])
     assert set(ds[split]['id'])=={id for id,p in parents.items() if p['split']==split and not p['development_exposed']}
   continue
  for split in SPLITS:
   for r in ds[split]:
    parent=parents[r['parent_id']];assert r['paper_id']==parent['paper_id'] and r['forum_id']==parent['forum_id'];assert r['split']==parent['split'] and r['development_exposed']==parent['development_exposed'];assert r['quality_flags']==parent['quality_flags']
    assert r['text']==parent['text'][r['start_in_passage']:r['end_in_passage']] and r['text_sha256']==sha(r['text'])
    if config=='paragraphs':assert r['label']==parent['target_label'];paragraph_counts[r['label']]+=1
    else:
     assert r['id'] not in seen_sentence_ids;seen_sentence_ids.add(r['id']);s=parent['sentences'][r['sentence_index']]
     assert all(r[k]==s[k] for k in ['text','label','binary_label','loss_mask','ai_character_fraction'])
     assert sum(x['end']-x['start'] for x in r['regions'])==len(r['text']);sentence_counts[r['label']]+=1
 assert paragraph_counts=={0:2500,1:2500} and sentence_counts=={0:57270,1:11006,2:18}
 # Scan materialized, uncompressed data and text artifacts for credential patterns and local paths.
 secret=re.compile(r'sk-or-v1-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{25,}|/Users/alicerigg|Authorization: Bearer')
 for path in OUT.rglob('*'):
  if not path.is_file():continue
  if path.suffix=='.parquet':
   for batch in pq.ParquetFile(path).iter_batches(batch_size=50):assert not secret.search(json.dumps(batch.to_pylist(),ensure_ascii=False)),path
  else:assert not secret.search(path.read_text()),path
 result={'passed':True,'configs':loaded_counts,'checks':['All 5,000 source rows preserved with exact text and SHA-256','All 2,052,212 token IDs and offsets reproduced with bundled tokenizer','Every token label independently agrees with character spans','All 68,294 sentence labels and offsets agree across views','18 mixed sentences excluded from binary loss','2,500 human/AI target pairs retained','Both paper_id and nullable scraped OpenReview forum_id verified across all views','Paper and normalized-author split isolation preserved','Fresh passage view excludes all development-exposed pilot targets','Hugging Face ClassLabel metadata survives loading','No credential patterns or local user paths in exports'],'source_dataset_sha256':sha((SOURCE/'dataset.jsonl').read_bytes())}
 save('validation.json',result)
 save('file_checksums.json',{p.relative_to(OUT).as_posix():sha(p.read_bytes()) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='file_checksums.json'})
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
