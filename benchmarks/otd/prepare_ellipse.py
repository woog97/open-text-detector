"""Prepare the publisher's ELLIPSE test split without using detector scores."""
import csv,hashlib,json,re
from collections import Counter
from pathlib import Path
B=Path(__file__).resolve().parent
RAW=B/'data/raw/ellipse';DEST=B/'data/prepared/ellipse.jsonl'
def sha(s):return hashlib.sha256(s.encode() if isinstance(s,str) else s).hexdigest()
def normalized(s):return re.sub(r'\s+',' ',s).strip().casefold()
def main():
 index=json.loads((RAW/'source-index.json').read_text());existing={}
 for path in sorted((B/'data/prepared').glob('*.jsonl')):
  if path==DEST:continue
  for line in path.open():
   r=json.loads(line)
   if isinstance(r.get('text'),str):existing.setdefault(sha(normalized(r['text'])),[]).append(r.get('id',path.stem))
 rows=[];seen=set();excluded=[]
 with (RAW/'official-test.csv').open(encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):
   text=r['full_text'].strip();tid=r['text_id_kaggle'];key=sha(normalized(text))
   if not text or key in seen:excluded.append({'id':tid,'reason':'empty_or_duplicate_text'});continue
   seen.add(key);n=len(text.split());score={}
   for col in ['Overall','Cohesion','Syntax','Vocabulary','Phraseology','Grammar','Conventions']:
    try:score[col.lower()]=float(r[col])
    except (ValueError,TypeError):score[col.lower()]=None
   rows.append({'id':'ellipse:'+tid,'dataset':'ellipse','label':'human','language':'en','text':text,'text_sha256':sha(text),'group_id':'ellipse:'+tid,'author_id':None,'cohort':'publisher_official_test','domain':'learner_english','task':'human','words':n,'length_bucket':'under_50' if n<50 else '50-99' if n<100 else '100-249' if n<250 else '250-499' if n<500 else '500plus','source_split':r['set'],'publisher_proficiency_scores':score,'prompt_id':r['prompt'],'exact_overlap_existing_ids':existing.get(key,[]),'provenance':'Publisher-designated human English-language-learner essays; official ELLIPSE test CSV at pinned repository revision; no model filtering','source_revision':index['revision'],'license':'CC-BY-NC-SA-4.0'})
 assert len({r['id'] for r in rows})==len(rows)
 DEST.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 summary={'source':'https://github.com/scrosseye/ELLIPSE-Corpus','revision':index['revision'],'license':'CC-BY-NC-SA-4.0','source_file_sha256':sha((RAW/'official-test.csv').read_bytes()),'prepared_sha256':sha(DEST.read_bytes()),'rows':len(rows),'excluded':excluded,'exact_overlap_rows':sum(bool(r['exact_overlap_existing_ids']) for r in rows),'length_buckets':dict(Counter(r['length_bucket'] for r in rows)),'evaluation_rule':'Use all nonempty, deduplicated official-test essays, with no detector-dependent selection; report FPR and lengths/proficiency slices separately from papers. Also report a no-existing-overlap sensitivity view. Do not tune thresholds on this set.','author_grouping':'No author identifier is supplied in the final CSV; group by essay and do not claim author-level independence.','limitations':['Not the unreleased Pangram evaluation subset','Publisher test split is for proficiency scoring; it does not establish exclusion from third-party detector training','Student essays differ from research papers','Noncommercial share-alike license retained; data not uploaded or republished'],'sensitive_metadata_policy':'Race/ethnicity, gender, grade and economic status excluded from the prepared evaluation; no sensitive traits inferred.'}
 (RAW/'preparation-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
