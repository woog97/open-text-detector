"""Local dataset-contract tests with a deterministic length-only fake tokenizer.

No model inference, no tokenizer/model download, and no Space access.
"""
from pathlib import Path
import tempfile,json,gzip,hashlib,copy
from objective_pairwise_validate import validate

def put(path,rows):
 blob=''.join(json.dumps(x)+'\n' for x in rows).encode();path.write_bytes(gzip.compress(blob))
 return {'rows':len(rows),'sha256':hashlib.sha256(blob).hexdigest()}

def tok(texts,**kwargs):return {'input_ids':[[1]*len(t) for t in texts]}

def fixture():
 tmp=Path(tempfile.mkdtemp(prefix='ranking-local-contract-'));root=tmp/'ranking';control=tmp/'control'
 for p in [root/'prepared',control/'prepared']:p.mkdir(parents=True)
 base=[{'id':'original-'+str(i),'paper_id':'old-'+str(i),'text':'control!'} for i in range(12000)]
 rows=copy.deepcopy(base);pool=[]
 for n in range(600):
  pid='pair'+str(n);pair={}
  for role,offset,text in [('human',0,'human000'),('ai',1,'ai000000')]:
   r={'id':pid+'/'+role,'paper_id':'paper'+str(n),'text':text,'ranking_role':role,'ranking_pair_id':pid,'verified_pair_relation':True};pair[role]=r
   rows[2*n+offset]=dict(r,ranking_pair_id='draw-'+pid)
  pool.append(pair)
 poolinfo=put(root/'paper-pair-pool.jsonl.gz',pool)
 (root/'paper-pair-audit.json').write_text(json.dumps({'eligible_pairs':len(pool),'sha256_uncompressed':poolinfo['sha256']}))
 controlinfo=put(control/'prepared/stage2-epoch0.jsonl.gz',base)
 info=put(root/'prepared/stage2-epoch0.jsonl.gz',rows)
 (control/'prepared/manifest.json').write_text(json.dumps({'files':{'stage2-epoch0':controlinfo}}))
 (root/'prepared/manifest.json').write_text(json.dumps({'files':{'stage2-epoch0':info}}))
 return root,control,rows

def update(root,rows):
 info=put(root/'prepared/stage2-epoch0.jsonl.gz',rows);(root/'prepared/manifest.json').write_text(json.dumps({'files':{'stage2-epoch0':info}}))

passed=[]
root,control,rows=fixture();result=validate(root,control,tok);assert result['stage2-epoch0']['pairs']==600;passed.append('valid10percentpaired_fixture')
for name,mutate in [
 ('cross_microbatch_pair',lambda rs:rs.__setitem__(slice(1,9,7),[rs[8],rs[1]])),
 ('unpaired_row_drift',lambda rs:rs[1500].update(text='changed!')),
 ('source_provenance_drift',lambda rs:rs[0].update(paper_id='different')),
 ('role_drift',lambda rs:rs[1].update(ranking_role='human')),
 ('pair_id_reuse',lambda rs:[rs[i].update(ranking_pair_id=rs[0]['ranking_pair_id']) for i in [8,9]])
]:
 root,control,rows=fixture();mutate(rows);update(root,rows)
 try:validate(root,control,tok)
 except AssertionError:passed.append('reject_'+name)
 else:raise AssertionError('Invalid fixture accepted: '+name)
print(json.dumps({'passed':True,'checks':passed,'model_inference':False,'Space_access':False}))
