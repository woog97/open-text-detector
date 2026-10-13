"""Evaluate either trained backbone against a frozen suite using the common reference grid."""
from runtime import require_space, SPACE_CACHE
import argparse,gzip,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from train import require_gpu,save
from inference import load_checkpoint,predict

def main(a):
 require_gpu();bundle=Path(a.suite);sys.path.insert(0,str(bundle));import suite,compare_models as c
 manifest=suite.validate(bundle);run=Path(a.run);cal=json.loads((run/'thresholds.json').read_text());th=cal['thresholds'];checkpoint=hashlib.sha256((run/'stage2-best.safetensors').read_bytes()).hexdigest()
 if checkpoint!=cal['checkpoint_sha256']:raise ValueError('Checkpoint changed after calibration')
 reference={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(a.reference).iterdir() if p.is_file() and p.suffix in ['.json','.txt']}
 if reference!=cal['reference_tokenizer_files'] or manifest['code']!=cal['suite_code']:raise ValueError('Reference grid or metric code differs from calibration')
 out=Path(a.output)
 if out.exists() and any(out.iterdir()):raise RuntimeError('Use a new output directory')
 out.mkdir(parents=True);rows=[json.loads(l) for l in gzip.decompress((bundle/(a.profile+'.jsonl.gz')).read_bytes()).splitlines()]
 model,tok,contract=load_checkpoint(run);ref=AutoTokenizer.from_pretrained(a.reference,local_files_only=True);records=[];starttime=time.time()
 for start in range(0,len(rows),32):
  chunk=rows[start:start+32];pred=predict(chunk,model,tok);offs=ref([r['text'] for r in chunk],add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
  for r,off,(native,p) in zip(chunk,offs,pred):
   r.setdefault('granularity','character_provenance' if 'regions' in r else 'native_document_label');p=c.project(r['text'],off,native,p);rec=c.summarize_row(r,off,p,th);rec['document_flag']=rec['mean_ai_probability']>=th['document'];rec['span_annotation_conflict']=r.get('span_annotation_conflict',False);rec['condition_view_split']=' / '.join(str(r.get(k,'unspecified')) for k in ['condition','view','split']);records.append(rec)
  save(out/'status.json',{'rows':min(start+32,len(rows)),'total':len(rows)})
 report=c.report(records)
 for ds,result in report.items():
  rs=[r for r in records if r['dataset']==ds];result['breakdowns']['condition_view_split']={v:c.aggregate([r for r in rs if r['condition_view_split']==v],False) for v in sorted({r['condition_view_split'] for r in rs})}
 with gzip.open(out/'predictions.jsonl.gz','wt') as f:
  for r in records:f.write(json.dumps(r)+'\n')
 save(out/'results.json',{'training':contract,'calibration':cal,'profile':a.profile,'profile_sha256':manifest['profiles'][a.profile]['sha256'],'elapsed_seconds':time.time()-starttime,'datasets':report})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--suite',required=True);p.add_argument('--reference',required=True);p.add_argument('--profile',choices=['workflow','comparison','full','context'],default='workflow');p.add_argument('--output',required=True);main(p.parse_args())
