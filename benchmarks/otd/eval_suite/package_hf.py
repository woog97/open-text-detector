"""Publish an explicit allowlist: workflow excerpts + text-free suite index."""
import json,gzip,hashlib,shutil
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from suite import HERE,NEW,PROJECT,digest,read
OUT=HERE/'hf-package'
def main():
 OUT.mkdir(exist_ok=True); manifest=json.loads((HERE/'bundle/manifest.json').read_text())
 configs={};papers={r['paper_id']:r for r in map(json.loads,(NEW.parent/'papers.jsonl').read_text().splitlines())}
 for profile in ['workflow','workflow_validation','assistance']:
  for r in map(json.loads,read(HERE/'bundle'/(profile+'.jsonl.gz')).splitlines()):
   if r['dataset']=='ellipse':continue
   config='assistance' if profile=='assistance' else 'reconstruction' if r['dataset']=='paper_workflow_reconstruction' else 'human_controls'
   split='validation' if r['split']=='calibration' else r['split']
   p=papers[r['paper_id']]
   record={k:r.get(k) for k in ['id','paper_id','forum_id','family_id','author_component_id','dataset','condition','view','text','regions','target_start','target_end','original_target','generated_target','label','target_label','text_sha256','conference','year','pdf_url','pdf_sha256','clean_prose','eligible_clean_novel_control','label_policy','development_exposed','requested_sentence_count','generated_sentence_count','quality_stratum','generator']}
   record.update(split=split,authors=p.get('authors'),title=p.get('title'),metadata_json=json.dumps({k:v for k,v in r.items() if k not in record},ensure_ascii=False))
   configs.setdefault(config,{}).setdefault(split,[]).append(record)
 # Fixed per-config schema across splits, including all-null fields.
 counts={}
 for config,splits in configs.items():
  allrows=[r for rs in splits.values() for r in rs];schema=pa.Table.from_pylist(allrows).schema
  for split,rows in splits.items():
   dest=OUT/'data'/config;dest.mkdir(parents=True,exist_ok=True)
   pq.write_table(pa.Table.from_pylist(rows,schema=schema),dest/(split+'.parquet'));counts[config+'/'+split]=len(rows)
 index=[]
 for profile in manifest['profiles']:
  for r in map(json.loads,read(HERE/'bundle'/(profile+'.jsonl.gz')).splitlines()):
   index.append({'profile':profile,'id':r['id'],'dataset':r['dataset'],'text_sha256':r['text_sha256'],'group_id':str(r.get('group_id','')),'label':r.get('label'),'source_metadata_json':json.dumps({k:r.get(k) for k in ['source','source_file','paper_id','forum_id','cohort','generator','split']})})
 (OUT/'data/index').mkdir(parents=True,exist_ok=True);pq.write_table(pa.Table.from_pylist(index),OUT/'data/index/test.parquet');counts['index/test']=len(index)
 for source,name in [(HERE/'bundle/manifest.json','suite-manifest.json'),(NEW.parent/'protocol.json','generation-protocol.json'),(NEW.parent/'PROTOCOL.md','GENERATION.md'),(NEW.parent/'RESULTS.md','GENERATION_RESULTS.md'),(PROJECT/'benchmarks/otd/COVERAGE.md','SOURCES.md')]:shutil.copyfile(source,OUT/name)
 for name in ['suite.py','common.py','compare_models.py','score_wide_eval.py','meld_model.py','README.md']:
  (OUT/'runner').mkdir(exist_ok=True);shutil.copyfile(HERE/'bundle'/name,OUT/'runner'/name)
 shutil.copyfile(HERE/'requirements-a100.txt',OUT/'requirements-a100.txt')
 header='---\npretty_name: AI Paper Workflow Evaluation\nlanguage: [en]\nlicense: other\nlicense_name: mixed-paper-and-model-output-terms\nlicense_link: LICENSE.md\ntask_categories: [token-classification, text-classification]\nconfigs:\n'
 for config in [*configs,'index']:
  header+=f'- config_name: {config}\n'
  if config=='reconstruction':header+='  default: true\n'
  header+='  data_files:\n'
  for split in (configs[config] if config in configs else ['test']):header+=f'  - split: {split}\n    path: data/{config}/{split}.parquet\n'
 card='''---
# AI Paper Workflow Evaluation

Frozen GPT-6 Luna Flex outputs and historical research-paper controls. **Evaluation only: keep test out of training, prompt development and threshold selection.** 162 main papers: 54 validation and 108 test, stratified across NeurIPS, ICML and ACL, 2013–2021. The 27 development pilot papers are excluded. Author-name connected components do not cross splits; this is not perfect author identity resolution.

```python
from datasets import load_dataset
# Pin revision to the upload commit recorded by your experiment.
ds = load_dataset("woog/ai-paper-workflow-eval", "reconstruction", revision=REVISION)
```

| Config | Validation / test rows | Purpose |
|---|---|---|
| reconstruction | 432 / 864 | One/two sentences, v3 paragraph, concise paragraph; target hidden from writer |
| human_controls | 554 / 5,117 | Matched originals and other human body paragraphs; use clean-novel flags for primary FPR |
| assistance | 324 / 648 | Proofread, light polish, substantial rewrite; diagnostic, no binary target gold |
| index | See suite manifest | Text-free frozen row selections for every local profile; **not a test split to score or train on** |

Paragraph and contextual views are correlated, not additional generations. Main reconstruction counts are 216 validation + 432 test generated targets; assistance counts are 162 + 324. Match originals via `family_id`, cluster by paper (author-component sensitivity), and report condition/view separately. Human controls deliberately retain extraction diagnostics; validation remaining controls are restricted to clean-novel prose.

`regions` are character `[start,end)` spans: 0 human, 1 generated replacement, -100 unknown/assisted. These are provenance labels, not measures of correctness. `paper_id` is stable; `forum_id` is null when not sourced from OpenReview. Quality reviews are by the same Luna model, not human gold; original verdicts and evidence-format corrections are retained in `metadata_json`. All first valid writer outputs remain, including quality failures and sentence-count mismatches. No detector scores selected this test set.

The `index` config inventories the older paper-v3 comparison, Arena, PELIC, Liang, VUB, Perkins, MELD-eval, DetectRL, Epoch, OpAI, Sem-Detect, GEDE, Saha, ELLIPSE and local diagnostic proxies. Those third-party texts are **not mirrored here**. ELLIPSE is upstream CC-BY-NC-SA-4.0. See SOURCES.md for acquisition choices. Existing public generated corpora: [v3 paper pairs](https://huggingface.co/datasets/woog/ai-paper-provenance-v3), [Arena](https://huggingface.co/datasets/woog/arena-prose-100-49-models).

Local frozen bundle + checkpoint artifacts run offline through `runner/suite.py` (see runner/README.md). This HF dataset alone does not hydrate every third-party profile or download our checkpoint. Code/data/model hashes, fixed comparison-v1 thresholds and BF16 inference are recorded; changed inputs invalidate score caches. Same-environment rescoring is reproducible subject to GPU numerics; new Luna generation is stochastic. These are our benchmarks, not Pangram 4's exact private cohorts. See GENERATION.md and GENERATION_RESULTS.md for generation design and quality results.
'''
 (OUT/'README.md').write_text(header+card)
 (OUT/'LICENSE.md').write_text('''# Component terms

Paper excerpts retain their authors’ rights and the applicable NeurIPS, ICML/PMLR or ACL publication terms; per-row authors, paper titles, URLs and PDF hashes identify sources. No new blanket license is granted to source text. Generated outputs remain subject to applicable provider terms. This public dataset uses `license: other` for these mixed terms. Raw PDFs are excluded. Third-party benchmark texts and ELLIPSE essays are not redistributed here; the index contains IDs and hashes only.
''')
 (OUT/'counts.json').write_text(json.dumps(counts,indent=2))
 hashes={str(p.relative_to(OUT)):digest(p.read_bytes()) for p in OUT.rglob('*') if p.is_file() and p.name!='checksums.json'}
 (OUT/'checksums.json').write_text(json.dumps(hashes,indent=2));print(json.dumps(counts,indent=2))
if __name__=='__main__':main()
