"""Paired context sensitivity check for the same remaining human paragraphs."""
from build_wide_eval import *
import gzip,shutil

def main():
 parent=Path(__file__).parent/'wide-eval-v1';out=Path(__file__).parent/'wide-eval-context-v1';out.mkdir(exist_ok=True)
 assert not (out/'manifest.json').exists()
 chosen={r['id']:r for r in readl(parent/'suite.jsonl') if r['dataset']=='human_paper_remaining'}
 papers=[p for p in readl(SOURCE/'papers.jsonl') if p['paper_id'] in {r['paper_id'] for r in chosen.values()}]
 result=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
  for pid,paras,error in pool.map(extract,papers):
   assert error is None,error
   for j,q in enumerate(paras):
    old_id=f'human_remaining:{pid}:{j:04d}'
    if old_id not in chosen:continue
    r=chosen[old_id];assert r['text']==q['text'];before=paras[j-1]['text']+'\n\n' if j else '';after='\n\n'+paras[j+1]['text'] if j+1<len(paras) else '';text=before+r['text']+after;start=len(before);end=start+len(r['text']);reg=[]
    if start:reg.append({'start':0,'end':start,'label':-100})
    reg.append({'start':start,'end':end,'label':0})
    if end<len(text):reg.append({'start':end,'end':len(text),'label':-100})
    result.append({**r,'id':'context:'+old_id,'paired_isolated_id':old_id,'text':text,'text_sha256':sha(text),'regions':reg,'target_start':start,'target_end':end,'granularity':'historical_human_paragraph_with_context','provenance':'Unmodified original adjacent paragraphs supplied as context; only the non-targeted focal paragraph is scored. Context labels -100 are metric masks, not unknown authorship.'})
 assert len(result)==len(chosen)
 writel(out/'suite.jsonl',result)
 with (out/'suite.jsonl').open('rb') as a,gzip.open(out/'suite.jsonl.gz','wb',compresslevel=6) as b:shutil.copyfileobj(a,b)
 parent_manifest=json.loads((parent/'manifest.json').read_text());m={'version':'wide-eval-context-v1','created_at':time.time(),'evaluation_rows':len(result),'unique_evaluation_texts':len({r['text_sha256'] for r in result}),'by_dataset':{'human_paper_remaining':len(result)},'checkpoint_sha256':parent_manifest['checkpoint_sha256'],'thresholds':parent_manifest['thresholds'],'new_generation_calls':0,'metric_policy':'Sensitivity analysis added after observing paragraph-only false positives. Same frozen checkpoint/thresholds, same selected human focal paragraphs, original adjacent paragraphs supplied as context. Primary span metrics mask neighbors. Not an independent test cohort. No training or calibration.','files':{'suite.jsonl':hashlib.sha256((out/'suite.jsonl').read_bytes()).hexdigest()}}
 (out/'manifest.json').write_text(json.dumps(m,indent=2));print(json.dumps({'context_rows':len(result),'gzip_bytes':(out/'suite.jsonl.gz').stat().st_size}))
if __name__=='__main__':main()
