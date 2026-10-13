"""Package frozen v3 paper reconstructions with exact provenance supervision."""
import hashlib,json,re,shutil
from collections import Counter,defaultdict
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from paper_forum_ids import resolve as resolve_forum_ids
from datasets import Features,Value,ClassLabel,List,load_dataset

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'research/data/paper-gap2500-v3-luna-20260930'
OUT=ROOT/'benchmarks/otd/exports/ai-paper-provenance-v3'
REPO='woog/ai-paper-provenance-v3'
SPLITS=['train','validation','test']
CLASSES=['human','ai','mixed']
MAP={'human_preserved':0,'human':0,'ai_rewritten':1,'mixed':2,'mixed_boundary':2}
V=lambda s:Value(s)
J=lambda x:json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def sha(x):return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()
def rows(name):
 with (SOURCE/name).open() as f:
  for line in f:yield json.loads(line)
def read(name):return json.loads((SOURCE/name).read_text())
def save(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def common_features():
 return {'id':V('string'),'pair_id':V('string'),'paper_id':V('string'),'forum_id':V('string'),'split':V('string'),'variant':ClassLabel(names=['human_original','paragraph_generated']),
  'title':V('string'),'authors':List(V('string')),'conference':V('string'),'year':V('int32'),'source_page':V('int32'),'pdf_url':V('string'),'abstract_url':V('string'),'pdf_sha256':V('string'),
  'development_exposed':V('bool'),'eligible_for_pilot_training':V('bool'),'quality_status':V('string'),'quality_flags':List(V('string')),
  'generation_id':V('string'),'generator_model':V('string'),'generator_canonical_model':V('string'),'generation_service_tier':V('string'),'fidelity_verdict':V('string'),
  'paired_generation_fidelity_verdict':V('string'),'paired_quality_audit_json':V('string'),'paired_fidelity_review_json':V('string'),'sample_weight':V('float64')}

def main():
 assert read('validation.json')['passed'] and read('completion-status.json')['status']=='complete'
 OUT.mkdir(parents=True,exist_ok=True)
 papers={p['paper_id']:p for p in rows('papers.jsonl')};passages={p['passage_id']:p for p in rows('passages.jsonl')}
 forum_ids,identifier_report=resolve_forum_ids(list(papers.values()),SOURCE)
 save('identifier_mapping.json',identifier_report)
 fidelity={p['passage_id']:p['output'] for p in rows('fidelity-review/responses.jsonl')};audits={p['passage_id']:p['output'] for p in rows('audit-responses.jsonl')}
 final_ids={r['generation_id'] for r in rows('writer-responses.jsonl')};writer_requests={}
 for a in rows('attempts.jsonl'):
  if a['stage']=='writer' and a['response'].get('id') in final_ids:writer_requests[a['response']['id']]=a
 manifest=read('manifest.json');tokenizer_sha=sha((ROOT/'models/meld-v5/tokenizer.json').read_bytes())
 # Original boundary labels stay available; binary loss uses -100 for mixed units.
 sentence_struct={'start':V('int32'),'end':V('int32'),'text':V('string'),'label':ClassLabel(names=CLASSES),'binary_label':V('int64'),'loss_mask':V('bool'),'ai_character_fraction':V('float64'),'human_characters':V('int32'),'ai_characters':V('int32')}
 region_struct={'start':V('int32'),'end':V('int32'),'label':ClassLabel(names=CLASSES),'source_label':V('string'),'source_start':V('int32'),'source_end':V('int32')}
 passage_features=Features({**common_features(),'text':V('string'),'text_sha256':V('string'),'source_text_sha256':V('string'),'label':ClassLabel(names=CLASSES),'has_ai':V('bool'),
  'target_start':V('int32'),'target_end':V('int32'),'target_label':ClassLabel(names=['human','ai']),'offset_unit':V('string'),'offset_convention':V('string'),
  'regions':List(region_struct),'input_ids':List(V('int32')),'attention_mask':List(V('int8')),'token_labels':List(V('int64')),'token_loss_mask':List(V('bool')),
  'token_start':List(V('int32')),'token_end':List(V('int32')),'token_text':List(V('string')),'tokenizer_sha256':V('string'),
  'words':List(V('string')),'word_labels':List(ClassLabel(names=CLASSES)),'word_start':List(V('int32')),'word_end':List(V('int32')),'sentences':List(sentence_struct)})
 paragraph_features=Features({**common_features(),'parent_id':V('string'),'text':V('string'),'text_sha256':V('string'),'label':ClassLabel(names=['human','ai']),
  'start_in_passage':V('int32'),'end_in_passage':V('int32'),'text_has_multiple_labels':V('bool'),'text_appears_in_multiple_splits':V('bool')})
 sentence_features=Features({**common_features(),'parent_id':V('string'),'sentence_index':V('int32'),'text':V('string'),'text_sha256':V('string'),
  'start_in_passage':V('int32'),'end_in_passage':V('int32'),'label':ClassLabel(names=CLASSES),'binary_label':V('int64'),'loss_mask':V('bool'),'ai_character_fraction':V('float64'),
  'regions':List(region_struct),'overlaps_target':V('bool'),'within_target':V('bool'),'text_has_multiple_labels':V('bool'),'text_appears_in_multiple_splits':V('bool'),'inverse_duplicate_weight':V('float64')})
 features={'passages':passage_features,'fresh_passages':passage_features,'paragraphs':paragraph_features,'sentences':sentence_features}
 buffers={c:{s:[] for s in SPLITS} for c in features};writers={};counts={c:Counter() for c in features};ai_tokens=0;human_tokens=0;ignored_tokens=0;token_total=0;source_ids=set();sentence_labels=Counter()
 def flush(config,split):
  batch=buffers[config][split]
  if not batch:return
  key=(config,split)
  if key not in writers:
   folder=OUT/'data'/config;folder.mkdir(parents=True,exist_ok=True)
   writers[key]=pq.ParquetWriter(folder/(split+'-00000-of-00001.parquet'),features[config].arrow_schema,compression='zstd')
  table=pa.Table.from_pylist(batch,schema=features[config].arrow_schema);writers[key].write_table(table,row_group_size=100);batch.clear()
 def add(config,split,r):
  buffers[config][split].append(r);counts[config][split]+=1
  if config in ['passages','fresh_passages'] and len(buffers[config][split])>=100:flush(config,split)
 for r in rows('dataset.jsonl'):
  assert r['id'] not in source_ids;source_ids.add(r['id']);assert r['tokenizer_sha256']==tokenizer_sha
  p=passages[r['passage_id']];paper=papers[r['paper_id']];ai=r['operation']=='paragraph_generate';split=r['split'];f=fidelity[r['passage_id']]
  req=writer_requests[r['generation_id']] if ai else None
  common={k:r[k] for k in ['id','paper_id','split','development_exposed','eligible_for_pilot_training','quality_status','quality_flags','generation_id','sample_weight']}
  common.update(forum_id=forum_ids[r['paper_id']],pair_id=r['passage_id'],variant=int(ai),title=paper['title'],authors=paper['authors'],conference=paper['conference'],year=paper['year'],source_page=r['source_page'],pdf_url=paper['pdf_url'],abstract_url=paper['abstract_url'],pdf_sha256=paper['pdf_sha256'],generator_model=r['model'],generator_canonical_model=manifest['canonical_slug'] if ai else None,generation_service_tier=req['response'].get('service_tier','not_reported') if ai else None,fidelity_verdict=f['verdict'] if ai else 'not_applicable',paired_generation_fidelity_verdict=f['verdict'],paired_quality_audit_json=J(audits[r['passage_id']]),paired_fidelity_review_json=J(f))
  start=r['edits'][0]['target_start'] if ai else p['held_out_start'];end=r['edits'][0]['target_end'] if ai else p['held_out_end']
  regions=[{**q,'source_label':q['label'],'label':MAP[q['label']]} for q in r['regions']]
  assert r['text'][start:end]==(r['edits'][0]['replacement'] if ai else p['held_out'])
  assert all(t['text']==r['text'][t['start']:t['end']] for t in r['tokens'])
  token_labels=[MAP[t['label']] if t['loss_mask'] and MAP[t['label']]<2 else -100 for t in r['tokens']]
  ai_tokens+=token_labels.count(1);human_tokens+=token_labels.count(0);ignored_tokens+=token_labels.count(-100);token_total+=len(token_labels)
  sentences=[]
  for i,s in enumerate(r['sentences']):
   label=MAP[s['label']];sentence_labels[CLASSES[label]]+=1
   item={'start':s['start'],'end':s['end'],'text':s['text'],'label':label,'binary_label':label if label<2 else -100,'loss_mask':label<2,'ai_character_fraction':s['ai_rewritten_character_fraction'],'human_characters':s['character_counts'].get('human_preserved',0),'ai_characters':s['character_counts'].get('ai_rewritten',0)}
   sentences.append(item)
   local_regions=[]
   for q in regions:
    a,z=max(q['start'],s['start']),min(q['end'],s['end'])
    if a<z:local_regions.append({'start':a-s['start'],'end':z-s['start'],'label':q['label'],'source_label':q['source_label'],'source_start':None,'source_end':None})
   sr={**common,'id':r['id']+f'/sentence-{i:04d}','parent_id':r['id'],'sentence_index':i,'text':s['text'],'text_sha256':sha(s['text']),'start_in_passage':s['start'],'end_in_passage':s['end'],'label':label,'binary_label':item['binary_label'],'loss_mask':item['loss_mask'],'ai_character_fraction':item['ai_character_fraction'],'regions':local_regions,'overlaps_target':s['start']<end and s['end']>start,'within_target':start<=s['start'] and s['end']<=end}
   add('sentences',split,sr)
  out={**common,'text':r['text'],'text_sha256':r['text_sha256'],'source_text_sha256':r['source_text_sha256'],'label':2 if ai else 0,'has_ai':ai,'target_start':start,'target_end':end,'target_label':int(ai),'offset_unit':'unicode_code_points','offset_convention':'half_open','regions':regions,
   'input_ids':[t['token_id'] for t in r['tokens']],'attention_mask':[1]*len(r['tokens']),'token_labels':token_labels,'token_loss_mask':[x!=-100 for x in token_labels],
   'token_start':[t['start'] for t in r['tokens']],'token_end':[t['end'] for t in r['tokens']],'token_text':[t['text'] for t in r['tokens']],'tokenizer_sha256':tokenizer_sha,
   'words':[t['text'] for t in r['word_units']],'word_labels':[MAP[t['label']] for t in r['word_units']],'word_start':[t['start'] for t in r['word_units']],'word_end':[t['end'] for t in r['word_units']],'sentences':sentences}
  add('passages',split,out)
  if not r['development_exposed']:add('fresh_passages',split,out)
  text=r['text'][start:end]
  add('paragraphs',split,{**common,'id':r['id']+'/target','parent_id':r['id'],'text':text,'text_sha256':sha(text),'label':int(ai),'start_in_passage':start,'end_in_passage':end})
 # Preserve duplicates but expose conflicts and repetition so evaluation can control them.
 duplicate_summary={}
 for config in ['paragraphs','sentences']:
  allrows=[r for split in SPLITS for r in buffers[config][split]];labels=defaultdict(set);splits=defaultdict(set);freq=Counter(r['text_sha256'] for r in allrows)
  for r in allrows:labels[r['text_sha256']].add(r['label']);splits[r['text_sha256']].add(r['split'])
  for r in allrows:
   r['text_has_multiple_labels']=len(labels[r['text_sha256']])>1;r['text_appears_in_multiple_splits']=len(splits[r['text_sha256']])>1
   if config=='sentences':r['inverse_duplicate_weight']=1/freq[r['text_sha256']]
  duplicate_summary[config]={'unique_texts':len(freq),'conflicting_label_texts':sum(len(x)>1 for x in labels.values()),'cross_split_duplicate_texts':sum(len(x)>1 for x in splits.values())}
 for config in features:
  for split in SPLITS:flush(config,split)
 for w in writers.values():w.close()
 assert dict(counts['passages'])=={'train':3000,'validation':1000,'test':1000}
 assert dict(counts['fresh_passages'])=={'train':2700,'validation':900,'test':900}
 assert ai_tokens==323204 and token_total==2052212 and sum(sentence_labels.values())==68294
 (OUT/'tokenizer').mkdir(exist_ok=True);shutil.copyfile(ROOT/'models/meld-v5/tokenizer.json',OUT/'tokenizer/tokenizer.json')
 save('label_schema.json',{'binary_labels':{'0':'human','1':'ai','-100':'ignore for loss'},'provenance_labels':{'0':'human','1':'ai','2':'mixed'},'variant_labels':{'0':'human_original','1':'paragraph_generated'},'passage_label':'0 for all-human control; 2 for human context with AI middle paragraph. Use has_ai for binary document detection.','token_labels':'Aligned to input_ids, no special tokens, padding or truncation. Full replacement provenance, not word-diff labels.','mixed_sentences':'Retain class 2; binary_label=-100 and loss_mask=false.','offsets':'Half-open Unicode code-point offsets [start,end), not UTF-8 byte or JavaScript UTF-16 offsets.','tokenizer_sha256':tokenizer_sha})
 save('features.json',{c:f.to_dict() for c,f in features.items()})
 save('generation_protocol.json',{'model':manifest['model'],'canonical_model':manifest['canonical_slug'],'version':3,'system':manifest['system'],'instructions':manifest['instructions'],'settings':manifest['settings'],'writer_input_fields':manifest['writer_input_fields'],'original_hidden_from_writer':True,'context':'Abstract, immediate previous paragraph, immediate next paragraph, structured content notes; broad length range.','labels_are_controlled_provenance':True})
 save('costs.json',read('scale-report.json')['costs'])
 save('statistics.json',{'papers':500,'pairs':2500,'rows':5000,'splits':{k:dict(v) for k,v in counts.items()},'token_labels':{'human':human_tokens,'ai':ai_tokens,'ignored':ignored_tokens},'sentence_labels':dict(sentence_labels),'ready_passages':read('summary.json')['eligible_rows'],'ready_generated_paragraphs':read('summary.json')['eligible_generated_paragraphs'],'duplicate_summary':duplicate_summary,'new_generation_fidelity':read('scale-report.json')['new_verdicts']})
 save('provenance.json',{'source_collection':'paper-gap2500-v3-luna-20260930','source_dataset_sha256':sha((SOURCE/'dataset.jsonl').read_bytes()),'source_passages_sha256':sha((SOURCE/'passages.jsonl').read_bytes()),'source_papers_sha256':sha((SOURCE/'papers.jsonl').read_bytes()),'tokenizer_sha256':tokenizer_sha,'sampling':manifest['sampling_plan'],'source_corrections':manifest['source_corrections'],'originals':'Historical proceedings and pre-2022 PDF metadata support human provenance, without direct authorship logs.','private_repository':True})
 configs=[{'config_name':c,**({'default':True} if c=='passages' else {}),'data_files':[{'split':s,'path':f'data/{c}/{s}-*.parquet'} for s in SPLITS]} for c in features]
 metadata={'pretty_name':'AI Research Paper Provenance — V3, 500 Papers','language':['en'],'license':'other','license_name':'mixed-paper-and-model-output-terms','license_link':'LICENSE.md','task_categories':['token-classification','text-classification'],'tags':['ai-text-detection','synthetic','sentence-classification','research-papers'],'size_categories':['1K<n<10K'],'configs':configs}
 card='---\n'+yaml.safe_dump(metadata,sort_keys=False)+'---\n\n'+card_body(counts,duplicate_summary)
 (OUT/'README.md').write_text(card)
 (OUT/'LICENSE.md').write_text('# Component rights\n\nSource excerpts come from NeurIPS, ICML/PMLR and ACL proceedings. Paper authors, official URLs and PDF hashes are retained per row. Rights and redistribution terms can differ by paper; this package does not grant new rights to those texts. Generated text is retained from OpenRouter/OpenAI Luna responses; applicable provider terms remain relevant. The bundled tokenizer reproduces the local MELD v5/Ettin annotations; its upstream terms also remain applicable.\n\nThe `license: other` metadata describes these component-specific terms, not a blanket permissive license. The repository is created private.\n')
 save('file_checksums.json',{p.relative_to(OUT).as_posix():sha(p.read_bytes()) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='file_checksums.json'})
 print(json.dumps({'package':str(OUT),'counts':{c:dict(v) for c,v in counts.items()},'duplicates':duplicate_summary,'ai_tokens':ai_tokens},indent=2))

def card_body(counts,duplicates):
 return f'''# AI Research Paper Provenance: v3

**2,500 generated paragraphs and 2,500 matched original controls from 500 historical AI research papers.** Each full passage has three paragraphs: an unchanged human-source paragraph, the original or generated target, and an unchanged human-source paragraph. All final v3 examples are included, even when flagged. Retired source-selection attempts and earlier experimental prompt versions are not training rows.

## Load

```python
from datasets import load_dataset

repo = "{REPO}"
passages = load_dataset(repo, "passages", token=True)
paragraphs = load_dataset(repo, "paragraphs", token=True)
sentences = load_dataset(repo, "sentences", token=True)
# Recommended for evaluations that exclude development-exposed pilot sources:
fresh = load_dataset(repo, "fresh_passages", token=True)
```

The repo is private; authenticate with your own Hugging Face token. No custom dataset loading script is needed. See [Hugging Face loading documentation](https://huggingface.co/docs/datasets/loading).

| View | Train | Validation | Test | Unit |
|---|---:|---:|---:|---|
'''+'\n'.join(f'| `{c}` | {counts[c]["train"]:,} | {counts[c]["validation"]:,} | {counts[c]["test"]:,} | '+({'passages':'Full passage','fresh_passages':'Full passage, pilot excluded','paragraphs':'Target paragraph only','sentences':'Individual sentence'}[c])+' |' for c in counts)+f'''

## Labels and training

- Binary supervision: **0 = human, 1 = AI, -100 = ignore for loss**.
- Three-way provenance: **0 = human, 1 = AI, 2 = mixed** (stored as Hugging Face `ClassLabel`).
- In `passages`, `label=0` for original controls and `label=2` for mixed human-context/AI-target examples. `has_ai` is the binary document target; `target_label` is the middle-paragraph target.
- `input_ids`, `token_labels`, `token_loss_mask`, `token_start`, `token_end`, and `token_text` are aligned. Tokens use the bundled tokenizer with **no special tokens, padding or truncation**. There are 1,729,008 human-labeled tokens and 323,204 AI-labeled tokens in the full view.
- `words` and `word_labels` preserve the source whitespace-unit annotations. Character `regions` support alignment with other model tokenizers. Do not reuse these token IDs with a different tokenizer.
- Sentence labels include **18 mixed sentences**, masked out of binary loss. Segmentation is heuristic and can cross a paragraph boundary; the existing annotations are preserved. Nested sentences in `passages` and rows in `sentences` agree exactly.
- The **entire returned middle paragraph is AI-labeled**, including wording identical to the source. Visual word diffs are not authorship labels. Human surrounding paragraphs stay human.
- Offsets use half-open Python Unicode code points, `[start,end)`. They are not byte offsets or UTF-16 indexes. Paragraph and sentence views expose parent-passage offsets; sentence `regions` use sentence-local offsets.

```python
# Ready-to-align token-classification inputs; add special tokens/windows as needed.
row = fresh["train"][0]
inputs = {{"input_ids": row["input_ids"], "attention_mask": row["attention_mask"], "labels": row["token_labels"]}}
# Your model/collator should use token_labels as labels, with special/pad labels -100.

# Binary sentence data: preserve mixed rows in the archive, exclude them from loss.
binary_sentences = sentences["train"].filter(lambda x: x["loss_mask"])
# Train/evaluate only on text and labels, not metadata, pair originals or judge output.
```

## Paper identifiers

Every view includes `paper_id` (the stable local corpus identifier) and nullable `forum_id` (the corresponding ID from the scraped OpenReview catalogue). All controls, generated variants, paragraphs and sentences from a paper share both identifiers. Matches require normalized title, conference and year agreement; the official proceedings link disambiguates multiple matching forums and must agree when present. No fuzzy or cross-year match is inferred. A null `forum_id` means no unambiguous scraped match was found, not proof that no forum exists. `identifier_mapping.json` records coverage and the matching rule.

## Splits, pairing and repeated text

The 500 papers are split 300/100/100. All five targets per paper, matched controls, derived sentences and variants stay together. Normalized author identities do not cross splits. The 450 new papers comprise 30 papers per venue/year cell across NeurIPS, ICML and ACL, 2017–2021; 50 earlier NeurIPS 2020 papers are retained. ACL 2018 includes two short papers after eligible long-paper candidates were exhausted.

All 250 pilot targets (500 passage rows) have `development_exposed=true`, including pilot rows in the historical validation/test splits. `fresh_passages` excludes them from every split and contains 450 papers. Filter `development_exposed` similarly in paragraph/sentence views. These flags address development exposure, not possible foundation-model pretraining contamination.

`pair_id` links each control to its generated variant. Repeated human context is deliberately retained. Paragraph/sentence views expose `text_sha256`, `text_has_multiple_labels`, and `text_appears_in_multiple_splits`; sentences additionally provide `inverse_duplicate_weight` (inverse exact-text frequency across this packaged view). Exact strings can have different provenance labels when generated wording matches human text; these conflicts are retained, not relabeled. Do not let duplicated context inflate evaluation: restrict to target sentences or group/deduplicate by exact text where appropriate. There are {duplicates['sentences']['cross_split_duplicate_texts']} exact sentence texts occurring across paper splits; use the flag to exclude them for strict text-disjoint evaluation.

## Quality and fidelity

Quality and provenance are separate. All 5,000 passage rows remain in the full dataset. `eligible_for_pilot_training` reproduces the existing conservative ready-data filter: **1,724 passage rows, including 740 generated paragraphs**, pass. Most excluded generated rows have source-extraction warnings, often in surrounding context. This filter does not incorporate the later strict-fidelity verdict; select any additional fidelity threshold explicitly.

`fidelity_verdict` applies to generated rows; human controls use `not_applicable`. `paired_generation_fidelity_verdict`, `paired_quality_audit_json` and `paired_fidelity_review_json` retain the generated counterpart's review for either variant. On the 2,250 new generations, Luna rated 1,600 fully faithful, 595 with minor differences, 45 with material differences and 10 uncertain. These are same-family model judgments, not independent expert gold labels. Flags propagate from the parent passage to derived rows and are not sentence-specific defect claims. Generator metadata also describes the parent intervention; a human-context sentence in a generated passage remains human-labeled even when its parent has a generator model.

## Generation and provenance

The frozen v3 writer sees the abstract, the immediate paragraph before and after, structured content notes and a broad length range. It never receives the held-out original directly. A separate notes call sees the original; generation, audit and strict review are isolated calls. First mechanically valid outputs are retained; there are no quality-based rerolls. `generation_protocol.json` preserves the instructions and settings.

The 2,250-example expansion used `openai/gpt-6-luna` on verified OpenRouter Flex. The prior 250 outputs are reused unchanged; their service tier is recorded as reported. The expansion cost **$2.17329542** in response-reported charges, including retries and retired source work: **15,444,938 input and 4,876,162 output tokens**. Account reconciliation accounts for per-call nine-decimal truncation. These API output counts include notes, reasoning and reviews; they are not the 323,204 final AI-labeled text tokens. `costs.json` keeps new and historical spending separate.

Three source-context leakage cases were corrected: two targets were replaced within their papers and one paper lacking five non-leaking targets was replaced in the same venue/year/split. All final generation counts and sampling quotas are unchanged. Costs of retired work remain in the accounting; retired examples are excluded from this dataset.

Human labels rely on historical proceedings and pre-2022 PDF metadata, not observed human writing logs. Source extraction can contain OCR/layout errors. The generated data represents one model family and one reconstruction protocol; performance here does not establish generalization to other generators, subtle mixed edits, or current research papers. The source selection requires five eligible prose targets and author non-overlap, so it is not a uniform sample of all proceedings. The fixed middle-paragraph intervention can create positional shortcuts; use span-level metrics and additional intervention placements in future data.

Paper authors, official source links and hashes are included per row. See `LICENSE.md` for component-rights notes and `provenance.json` for sampling and source hashes.
'''

if __name__=='__main__':main()
