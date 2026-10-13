"""Frozen, offline-capable evaluation entry point. No generation or threshold fitting."""
import argparse, collections, gzip, hashlib, json, shutil, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
OLD=PROJECT/'benchmarks/pangram4/training/paper-v3-modernbert'
NEW=PROJECT/'research/data/paper-eval-workflows-luna-20260930/scoring'
def digest(b): return hashlib.sha256(b).hexdigest()
def read(path):
 b=path.read_bytes(); return gzip.decompress(b) if path.suffix=='.gz' else b

def build(dest):
 dest.mkdir(parents=True,exist_ok=True)
 sources={'comparison':OLD/'comparison-v1/suite.jsonl.gz','workflow':NEW/'suite.jsonl.gz','assistance':NEW/'assistance.jsonl.gz','full':OLD/'wide-eval-v1/suite.jsonl','context':OLD/'wide-eval-context-v1/suite.jsonl','comparison_validation':OLD/'comparison-v1/validation.jsonl.gz','workflow_validation':NEW/'validation.jsonl.gz'}
 manifest={'version':1,'profiles':{},'code':{},'precision':'bfloat16','thresholds':json.loads((NEW/'manifest.json').read_text())['primary_thresholds'],'sentence_policy':'Frozen comparison-v1 regex segmentation, uniformly applied to all models; not linguistic gold.','profile_policy':'Run profiles separately. Full overlaps comparison; views and conditions are correlated. Validation profiles are not test benchmarks.'}
 for name,path in sources.items():
  if not path.exists() and path.with_suffix(path.suffix+'.gz').exists():path=path.with_suffix(path.suffix+'.gz')
  blob=read(path);rows=[json.loads(l) for l in blob.splitlines()];(dest/(name+'.jsonl.gz')).write_bytes(gzip.compress(blob,mtime=0))
  manifest['profiles'][name]={'sha256':digest(blob),'rows':len(rows),'datasets':dict(collections.Counter(r['dataset'] for r in rows)),'source':str(path.relative_to(PROJECT))}
 for name in ['common.py','compare_models.py','score_wide_eval.py','meld_model.py']:
  blob=(OLD/name).read_bytes();(dest/name).write_bytes(blob);manifest['code'][name]=digest(blob)
 for name in ['suite.py','README.md']:
  blob=(HERE/name).read_bytes();(dest/name).write_bytes(blob);manifest['code'][name]=digest(blob)
 (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 validate(dest);return manifest

def validate(bundle):
 m=json.loads((bundle/'manifest.json').read_text())
 for name,h in m['code'].items():
  if digest((bundle/name).read_bytes())!=h:raise ValueError('Code drift: '+name)
 for name,s in m['profiles'].items():
  blob=read(bundle/(name+'.jsonl.gz'))
  if digest(blob)!=s['sha256']:raise ValueError('Data drift: '+name)
  rows=[json.loads(l) for l in blob.splitlines()]
  if len(rows)!=s['rows'] or len({r['id'] for r in rows})!=len(rows):raise ValueError('Counts/IDs: '+name)
  for r in rows:
   if digest(r['text'].encode())!=r['text_sha256']:raise ValueError('Text hash: '+r['id'])
   for region in r.get('regions',[]):
    if region['label'] not in [0,1,-100] or not 0<=region['start']<region['end']<=len(r['text']):raise ValueError('Region: '+r['id'])
 print('Validated all code, data hashes, IDs, and label intervals.');return m

def run(args):
 m=validate(args.bundle)
 if args.profile.endswith('validation'):raise ValueError('Calibration data are not a test profile')
 import torch, numpy as np, importlib.metadata as meta, platform, time
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('A BF16-capable CUDA GPU is required; no FP32 fallback.')
 sys.path.insert(0,str(args.bundle));import compare_models as c
 from transformers import AutoTokenizer,AutoModelForTokenClassification
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 rows=[json.loads(l) for l in read(args.bundle/(args.profile+'.jsonl.gz')).splitlines()]
 if args.limit:rows=rows[:args.limit]
 for r in rows:r.setdefault('granularity','character_provenance' if 'regions' in r else 'native_document_label')
 refdir=args.models_root/'run-01/best_model'
 ref=AutoTokenizer.from_pretrained(refdir,local_files_only=True);ref.model_max_length=10**9
 for name in args.models.split(','):
  directory=refdir if name=='ours' else args.models_root/'baseline-models'/name
  fingerprint={str(p.relative_to(directory)):digest(p.read_bytes()) for p in sorted(directory.rglob('*')) if p.is_file()}
  reference={p.name:digest(p.read_bytes()) for p in refdir.iterdir() if p.is_file() and p.suffix in ['.json','.txt']}
  contract={'profile':args.profile,'data':m['profiles'][args.profile]['sha256'],'limit':args.limit,'model':name,'model_files':fingerprint,'reference_tokenizer':reference,'thresholds':m['thresholds'][name],'code':m['code'],'precision':'bfloat16','environment':{p:meta.version(p) for p in ['torch','transformers','numpy','safetensors','tokenizers']},'python':platform.python_version(),'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda}
  dest=args.output/args.profile/name;dest.mkdir(parents=True,exist_ok=True);lock=dest/'run-lock.json'
  if lock.exists() and json.loads(lock.read_text())!=contract:raise ValueError('Resume mismatch; use a new output directory.')
  lock.write_text(json.dumps(contract,indent=2));c.OUT=dest
  if (dest/'results.json').exists():print('Already complete:',dest);continue
  started=time.time()
  if name=='ours':model=AutoModelForTokenClassification.from_pretrained(directory,local_files_only=True,attn_implementation='sdpa').cuda().eval();tok=ref
  else:model=c.MeldModel(directory).cuda().eval();tok=AutoTokenizer.from_pretrained(directory,local_files_only=True);tok.model_max_length=10**9
  scored=c.compute(rows,model,tok,ref,name,dest/'scores');pred=[]
  for r,o,p in scored:
   z=c.summarize_row(r,o,p,m['thresholds'][name]);z['document_flag']=z['mean_ai_probability']>=m['thresholds'][name]['document'];z['span_annotation_conflict']=r.get('span_annotation_conflict',False)
   for k in ['condition','view','split','family_id','author_component_id']:z[k]=r.get(k)
   z['condition_view_split']=' / '.join(str(r.get(k,'unspecified')) for k in ['condition','view','split'])
   if args.profile=='assistance':
    unknown=np.asarray(c.labels_from_regions(r['text'],o,r['regions']))==-100
    z['assisted_target_flagged_token_fraction']=float((p[unknown]>=m['thresholds'][name]['tokens']).mean()) if unknown.any() else None
   pred.append(z)
  with gzip.open(dest/'predictions.jsonl.gz','wt') as f:
   for z in pred:f.write(json.dumps(z)+'\n')
  report=c.report(pred)
  for ds,out in report.items():
   rs=[r for r in pred if r['dataset']==ds]
   for field in ['condition','view','split','condition_view_split']:
    out['breakdowns'][field]={str(v):c.aggregate([r for r in rs if r.get(field)==v],False) for v in sorted({r[field] for r in rs if r.get(field) is not None})}
  if args.profile=='assistance':
   report['assisted_target_diagnostics']={key:{'rows':len(rs),'mean_flagged_token_fraction':float(np.mean([r['assisted_target_flagged_token_fraction'] for r in rs if r['assisted_target_flagged_token_fraction'] is not None]))} for key in sorted({r['condition_view_split'] for r in pred}) for rs in [[r for r in pred if r['condition_view_split']==key]]}
  c.write(dest/'results.json',{'run':contract,'elapsed_seconds':time.time()-started,'datasets':report,'diagnostic_only':args.profile=='assistance'})
  del model,scored;torch.cuda.empty_cache()
  print('Completed:',dest)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['build','validate','list','run']);p.add_argument('--bundle',type=Path,default=HERE/'bundle');p.add_argument('--profile',default='workflow',choices=['comparison','workflow','assistance','full','context']);p.add_argument('--models',default='ours,meld-v5,meld-v8');p.add_argument('--models-root',type=Path,default=OLD);p.add_argument('--output',type=Path,default=HERE/'results');p.add_argument('--limit',type=int,default=0,help='Smoke test only; included in resume lock');a=p.parse_args()
 if a.command=='build':build(a.bundle)
 elif a.command=='validate':validate(a.bundle)
 elif a.command=='list':print(json.dumps(validate(a.bundle)['profiles'],indent=2))
 else:run(a)
if __name__=='__main__':main()
