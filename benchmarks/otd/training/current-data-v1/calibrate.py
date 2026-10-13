"""Freeze operating points using calibration papers only, with human-target FPR guard."""
from runtime import require_space, SPACE_CACHE
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from train import ROOT,require_gpu,read_data,save
from inference import load_checkpoint,predict

def main(a):
 require_gpu();run=Path(a.run)
 if (run/'thresholds.json').exists():raise RuntimeError('Calibration already frozen')
 manifest=json.loads((ROOT/'prepared-v2/manifest.json').read_text());rows=read_data('calibration-windows',manifest)
 sys.path.insert(0,str(Path(a.suite)));import compare_models as c
 ref=AutoTokenizer.from_pretrained(a.reference,local_files_only=True);model,tok,contract=load_checkpoint(run)
 if contract['data_manifest_sha256']!=__import__('data').sha((ROOT/'prepared-v2/manifest.json').read_text()):raise ValueError('Training/calibration dataset mismatch')
 values={kind:{unit:([],[]) for unit in ['tokens','sentences','document']} for kind in ['paired','novel_human']}
 for start in range(0,len(rows),32):
  chunk=rows[start:start+32];pred=predict(chunk,model,tok);offs=ref([r['text'] for r in chunk],add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
  for r,off,(native,prob) in zip(chunk,offs,pred):
   p=c.project(r['text'],off,native,prob);ts,ty,ss,sy=c.units(r,off,p)
   bucket=values[r['kind']]
   for name,scores,labels in [('tokens',ts,ty),('sentences',ss,sy)]:bucket[name][0].extend(map(float,scores));bucket[name][1].extend(map(int,labels))
   labs={x['label'] for x in r['regions']}
   if labs=={0}:bucket['document'][0].append(float(p[[bool(r['text'][a:b].strip()) for a,b in off]].mean()));bucket['document'][1].append(0)
 thresholds={};diagnostics={}
 for unit in ['tokens','sentences','document']:
  # The hardest human stratum determines the operating point, instead of easy context dilution.
  thresholds[unit]=max(c.choose_threshold(*bucket[unit],fpr=.01) for bucket in values.values())
  diagnostics[unit]={kind:c.metrics(c.counts(*bucket[unit],thresholds[unit])) for kind,bucket in values.items()}
 reference_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(a.reference).iterdir() if p.is_file() and p.suffix in ['.json','.txt']}
 save(run/'thresholds.json',{'thresholds':thresholds,'empirical_calibration':diagnostics,'calibration_sha256':manifest['files']['calibration-windows']['sha256'],'checkpoint_sha256':hashlib.sha256((run/'stage2-best.safetensors').read_bytes()).hexdigest(),'reference_tokenizer_files':reference_hashes,'suite_code':json.loads((Path(a.suite)/'manifest.json').read_text())['code'],'policy':'Maximum 1% empirical human FPR separately on paired original targets and clean novel humans; no test fitting, not a population guarantee.'})
 print(json.dumps(thresholds,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--suite',required=True);p.add_argument('--reference',required=True);main(p.parse_args())
