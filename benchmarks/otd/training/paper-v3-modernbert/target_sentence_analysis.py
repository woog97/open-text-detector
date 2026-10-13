"""Supplementary denominator check using stored predictions and frozen thresholds.

Added after observing the gap between full-context and target-token false positives.
Does not retrain, select models, change thresholds, or rerun model inference.
"""
import collections,json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
import train as run
from common import digest

def main(root):
    root=Path(root);results=json.loads((root/'results.json').read_text());cfg=results['config'];thresholds=results['thresholds']
    split_hashes=collections.defaultdict(set)
    for split in ['train','validation','test']:
        rows=run.load_rows(split)
        for r in rows:
            for s in r['sentences']:split_hashes[digest(s['text'])].add(split)
    run.CROSS_SENTENCES={h for h,ss in split_hashes.items() if len(ss)>1}
    tok=AutoTokenizer.from_pretrained(root/'best_model')
    test=run.encode(rows,tok);ids=json.loads((root/'test_prediction_ids.json').read_text());assert ids==[r['id'] for r in test]
    stored=np.load(root/'test_predictions.npz');pred=[stored[str(i)] for i in range(len(test))]
    kept=set(json.loads((root/'test_subset_ids.json').read_text()))
    for r in test:r['sentences']=[s for s in r['sentences'] if s['start']>=r['target_start'] and s['end']<=r['target_end']]
    for name,mask in [('quality_filtered',[r['id'] in kept for r in test]),('fully_faithful',[r['id'] in kept and r['paired_generation_fidelity_verdict']=='fully_faithful' for r in test]),('all_fresh',[True]*len(test))]:
        rr=[r for r,m in zip(test,mask) if m];pp=[p for p,m in zip(pred,mask) if m]
        results['test'][name]['target_sentences']=run.summarize(run.units(rr,pp,'sentences'),thresholds['sentences'],500)
    results['supplementary_analysis']='Target-sentence denominator check added after observing higher target-token than full-context false positives. Uses stored predictions and unchanged frozen thresholds; no model or threshold selection.'
    run.save(root/'results.json',results)
    print(json.dumps({k:v['target_sentences'] for k,v in results['test'].items() if 'target_sentences' in v}))
if __name__=='__main__':
    import sys
    main(sys.argv[1])
