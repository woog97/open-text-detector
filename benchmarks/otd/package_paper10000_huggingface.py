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
SOURCE=ROOT/'research/data/paper-gap10000-v3-luna-20260930'
OUT=ROOT/'benchmarks/otd/exports/ai-paper-provenance-v3-10000'
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
def read_package(name):return json.loads((OUT/name).read_text())
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
 assert dict(counts['passages'])=={'train':12000,'validation':4000,'test':4000}
 assert dict(counts['fresh_passages'])=={'train':11700,'validation':3900,'test':3900}
 assert ai_tokens==read('summary.json')['ai_region_tokens'] and token_total>ai_tokens and sum(sentence_labels.values())>0
 (OUT/'tokenizer').mkdir(exist_ok=True);shutil.copyfile(ROOT/'models/meld-v5/tokenizer.json',OUT/'tokenizer/tokenizer.json')
 save('label_schema.json',{'binary_labels':{'0':'human','1':'ai','-100':'ignore for loss'},'provenance_labels':{'0':'human','1':'ai','2':'mixed'},'variant_labels':{'0':'human_original','1':'paragraph_generated'},'passage_label':'0 for all-human control; 2 for human context with AI middle paragraph. Use has_ai for binary document detection.','token_labels':'Aligned to input_ids, no special tokens, padding or truncation. Full replacement provenance, not word-diff labels.','mixed_sentences':'Retain class 2; binary_label=-100 and loss_mask=false.','offsets':'Half-open Unicode code-point offsets [start,end), not UTF-8 byte or JavaScript UTF-16 offsets.','tokenizer_sha256':tokenizer_sha})
 save('features.json',{c:f.to_dict() for c,f in features.items()})
 save('generation_protocol.json',{'model':manifest['model'],'canonical_model':manifest['canonical_slug'],'version':3,'system':manifest['system'],'instructions':manifest['instructions'],'settings':manifest['settings'],'writer_input_fields':manifest['writer_input_fields'],'original_hidden_from_writer':True,'context':'Abstract, immediate previous paragraph, immediate next paragraph, structured content notes; broad length range.','labels_are_controlled_provenance':True})
 save('costs.json',read('scale-report.json')['costs'])
 save('statistics.json',{'papers':len(papers),'pairs':len(passages),'rows':len(source_ids),'splits':{k:dict(v) for k,v in counts.items()},'token_labels':{'human':human_tokens,'ai':ai_tokens,'ignored':ignored_tokens},'sentence_labels':dict(sentence_labels),'ready_passages':read('summary.json')['eligible_rows'],'ready_generated_paragraphs':read('summary.json')['eligible_generated_paragraphs'],'duplicate_summary':duplicate_summary,'new_generation_fidelity':read('scale-report.json')['new_verdicts']})
 save('provenance.json',{'source_collection':'paper-gap10000-v3-luna-20260930','source_dataset_sha256':sha((SOURCE/'dataset.jsonl').read_bytes()),'source_passages_sha256':sha((SOURCE/'passages.jsonl').read_bytes()),'source_papers_sha256':sha((SOURCE/'papers.jsonl').read_bytes()),'tokenizer_sha256':tokenizer_sha,'sampling':manifest['sampling_plan'],'source_corrections':manifest.get('source_corrections'),'originals':'Historical proceedings and pre-2022 PDF metadata support human provenance, without direct authorship logs.','private_repository':False})
 configs=[{'config_name':c,**({'default':True} if c=='passages' else {}),'data_files':[{'split':s,'path':f'data/{c}/{s}-*.parquet'} for s in SPLITS]} for c in features]
 metadata={'pretty_name':'AI Research Paper Provenance — V3, 2,000 Papers','language':['en'],'license':'other','license_name':'mixed-paper-and-model-output-terms','license_link':'LICENSE.md','task_categories':['token-classification','text-classification'],'tags':['ai-text-detection','synthetic','sentence-classification','research-papers'],'size_categories':['10K<n<100K'],'configs':configs}
 card='---\n'+yaml.safe_dump(metadata,sort_keys=False)+'---\n\n'+card_body(counts,duplicate_summary)
 (OUT/'README.md').write_text(card)
 (OUT/'LICENSE.md').write_text('# Component rights\n\nSource excerpts come from NeurIPS, ICML/PMLR and ACL proceedings. Paper authors, official URLs and PDF hashes are retained per row. Rights and redistribution terms can differ by paper; this package does not grant new rights to those texts. Generated text is retained from OpenRouter/OpenAI Luna responses; applicable provider terms remain relevant. The bundled tokenizer reproduces the local MELD v5/Ettin annotations; its upstream terms also remain applicable.\n\nThe `license: other` metadata describes these component-specific terms, not a blanket permissive license. The repository is public.\n')
 save('file_checksums.json',{p.relative_to(OUT).as_posix():sha(p.read_bytes()) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='file_checksums.json'})
 print(json.dumps({'package':str(OUT),'counts':{c:dict(v) for c,v in counts.items()},'duplicates':duplicate_summary,'ai_tokens':ai_tokens},indent=2))

def card_body(counts,duplicates):
 stats=read_package('statistics.json');report=read('scale-report.json');new=report['costs']['new']
 table='\n'.join(f'| `{c}` | {counts[c]["train"]:,} | {counts[c]["validation"]:,} | {counts[c]["test"]:,} |' for c in counts)
 return f'''# AI Research Paper Provenance: v3

**2,000 papers; 10,000 generated paragraphs + 10,000 matched human originals**, for contextual token/sentence detection of LLM-produced wording. Each passage contains the target and its unchanged human neighbors. Flagged examples are retained.

## Load and choose a view

```python
from datasets import load_dataset
repo = "{REPO}"
data = load_dataset(repo, "fresh_passages")
```

The repository is public; no authentication is required.

| View | Train | Validation | Test |
|---|---:|---:|---:|
{table}

`fresh_passages` is the recommended context-preserving view: it excludes development-exposed pilots. `paragraphs` and `sentences` are target-only and sentence views; filter `development_exposed == false` yourself. These are **alternate views of the same examples**, not independent datasets.

## Labels and identifiers

- **Token loss:** `token_labels` uses human `0`, AI `1`, ignored `-100`; `token_loss_mask` identifies supervised tokens. Special/padding tokens must also receive `-100`.
- **Three-way provenance:** human `0`, AI `1`, mixed `2`. Passage `label=2` means human context plus an AI target; it is not a token target. Use `target_label` for the target paragraph and `has_ai` for binary passage detection.
- **Mixed boundaries:** {stats['sentence_labels'].get('mixed',0):,} sentences are mixed; their binary label is `-100`. There are {stats['token_labels'].get('ignored',0):,} ignored tokens. Preserve these masks.
- **Provenance rule:** the entire generated target is AI-labeled, including wording copied from its source. Unchanged surrounding paragraphs are human. Production history is not always inferable from text.
- **Alignment:** `input_ids`, token labels/masks, and token start/end offsets align with the bundled tokenizer, without special tokens, padding, or truncation. For another tokenizer, rebuild labels from character `regions`; do not reuse these IDs. Offsets are half-open Unicode code points `[start,end)`, not bytes or UTF-16 positions.
- **IDs:** `paper_id` is the stable corpus ID; `pair_id` joins matched variants. Nullable `forum_id` is the verified scraped OpenReview ID; null means no verified match. See `identifier_mapping.json`.

## Recommended training and evaluation recipe

This is a starting protocol, not a claim that hyperparameters or mixture weights have been optimized.

1. **Filter whole pairs.** Start with pairs where both variants pass `eligible_for_pilot_training` and `paired_generation_fidelity_verdict` is `fully_faithful` or `mostly_faithful_with_minor_differences`. Use that paired field for human rows too. Report retained counts, a fully-faithful-only sensitivity check, and broader flagged-data results separately.
2. **Use text-only inputs and masked token cross-entropy.** Feed one variant at a time. Originals, notes, judgments and metadata are not classifier inputs. Pairing controls topic; it does not prescribe contrastive loss. Sentence-only classification is a separate baseline.
3. **Sample by paper/target; balance variants.** Mix target-only examples with varied-offset context windows, covering boundaries and human-only text. Preserve offsets/masks and avoid truncating away targets. Token classes remain imbalanced; tune weighting on validation data. This reduces the fixed-middle-position shortcut.
4. **Control duplicates.** For paragraph/sentence baselines, exclude `text_has_multiple_labels` from the primary subset. Deduplicate or weight repeated context using training-only frequencies. For strict text-disjoint evaluation, exclude `text_appears_in_multiple_splits`; exported duplicate weights describe the whole view.
5. **Calibrate on validation, then freeze.** Choose thresholds and sentence aggregation/smoothing at a stated human false-positive target. Report token/sentence precision, recall, false positives and boundary performance, with uncertainty clustered by paper. Preserve mixed-boundary masks.
6. **Keep transfer tests and mining separate.** Train/tune on Luna; test expensive generators on unseen held-out papers. Mine human false positives only from a separate training/development pool, add synthetic mirrors, and retrain. Never mine the final test set.

## How this differs from Pangram 4

Our labels describe **binary surface-production provenance**. Pangram distinguishes human, AI-assisted and AI-generated text. **Our `mixed=2` must not map to AI-assisted:** it describes composition/boundaries. Our reconstructions preserve human-specified ideas and could qualify as assisted under Pangram's definition. Reproducing its taxonomy requires additional labeling and data.

| Aspect | This dataset | Pangram 4 reference |
|---|---|---|
| Generation | Luna reconstructs a hidden paragraph from detailed notes, abstract and neighbors | Broader generator/domain mixture, synthetic mirrors and edited text |
| Labels | Known replacement spans; copied output remains AI-labeled | Clause-level lexical/semantic matching distinguishes human, assisted and generated text |
| Training | Supplies supervision; no required backbone or tuned schedule | Segment-first training, then tokenwise and auxiliary objectives |
| Causal context | Choose a suitable contextual classifier | Repeat2 duplicates each sampled window and masks loss on the first copy |

Pangram uses 512-token source windows and inference stride 256. Repeat2 supervises only the second copy; a bidirectional encoder does not require it. This corpus alone cannot reproduce its humanizer or full three-class objectives. See [Pangram 4 §§2–4](https://arxiv.org/html/2607.27183v1#S2).

## Splits, quality and limitations

Paper splits are **1,200 train / 400 validation / 400 test**. All five targets, variants and derived rows stay with their paper; normalized author identities do not cross splits. The newest 1,500 papers comprise 500 each from ACL, ICML and NeurIPS, 2013–2021. The original 250 pilot targets are marked `development_exposed=true`; `fresh_passages` excludes their 500 rows. See `provenance.json` for sampling and hashes.

The existing eligibility flag passes {stats['ready_passages']:,} passage rows, including {stats['ready_generated_paragraphs']:,} generated paragraphs, **before** the paired fidelity filter above. Flags apply to parent passages, not individual sentence defects. Luna reviews its own model family; these are not expert gold labels. Historical dates support human provenance but do not prove it. PDF extraction can introduce artifacts.

The writer sees the abstract, immediate neighbors, structured notes, and a broad length range, but never the held-out paragraph directly. A separate call extracts notes from that paragraph. First mechanically valid outputs are retained; there are no quality-based rerolls. Full prompts are in `generation_protocol.json`.

The full view contains {stats['token_labels']['ai']:,} AI-labeled and {stats['token_labels']['human']:,} human-labeled tokens. The latest expansion cost **${new['cost_usd']:.4f}**, using **{new['input_tokens']:,} input / {new['output_tokens']:,} output API tokens**, including notes, retries and reviews. All accepted new outputs were verified as Flex. See `costs.json` and `statistics.json` for details. Source/model/tokenizer rights are described in `LICENSE.md`; this corpus does not establish generalization to other generators or writing workflows.
'''

if __name__=='__main__':main()
