"""Score clean paper text with fast10 backbone-comparison classifiers (BF16).

Run on the training Space. Uses each model's frozen evaluation bundle (checkpoint,
code and inference adapter from /tmp/pangram-eval-20261003/remaining/<model>) with
510-token windows (--stride 510 = no overlap; the eval used 256). LoRA is merged
into the BF16 base weights for speed (small rounding shift vs unmerged). Per paper
it stores token AI probabilities with character offsets, plus the document head's
token-weighted mean over windows.
The backbone is torch.compiled with fixed shapes. Uncalibrated: no thresholds are applied.

Several workers (one per GPU) share the job through claim files in --claims: each
claims the next unfinished shard, so stopping a worker (touch CLAIMS/stop-<worker>
to finish the current shard first, or kill it) hands its work to the others; a
dead worker's claim is taken over automatically.
Usage: classify_clean_papers.py --source DATASET [--source ...] --output DIR --model NAME --claims DIR --worker NAME
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

EVAL=Path('/tmp/pangram-eval-20261003/remaining')
SEQ=1024   # Repeat2 doubles each 510-token window (1020), rounded up


def atomic(path,raw):
    # The /data bucket mount occasionally returns EIO; retry before failing.
    for attempt in range(5):
        try:
            path.parent.mkdir(parents=True,exist_ok=True)
            temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
            with temp.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            temp.replace(path);return
        except OSError:
            if attempt==4:raise
            time.sleep(10*(attempt+1))


def digest(raw):return hashlib.sha256(raw).hexdigest()


def load_rows(sources):
    import pyarrow.parquet as pq
    rows=[]
    for source in sources:
        for shard in sorted((source/'data').glob('train-*.parquet')):
            for r in pq.read_table(shard,columns=['id','text','text_sha256','pdf_sha256']).to_pylist():
                rows.append({**r,'source_dataset':source.name})
    ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate paper IDs across sources')
    return sorted(rows,key=lambda r:r['id'])


def windows(n,stride,width=510):return [0] if n<=width else sorted(set(list(range(0,n-width+1,stride))+[n-width]))


def score(rows,model,tok,batch_size,stride):
    import numpy as np,torch
    from data import layout
    from modeling import collate
    enc=tok([r['text'] for r in rows],add_special_tokens=False,return_offsets_mapping=True,truncation=False)
    jobs=[];sums=[np.zeros(len(x),np.float64) for x in enc['input_ids']];den=[np.zeros(len(x),np.int32) for x in enc['input_ids']]
    doc=[[] for _ in rows]
    for i,ids in enumerate(enc['input_ids']):
        for start in windows(len(ids),stride) if ids else []:jobs.append((i,start,min(start+510,len(ids))))
    with torch.inference_mode():
        for pos in range(0,len(jobs),batch_size):
            part=jobs[pos:pos+batch_size];real=len(part)
            # Fixed shapes for the compiled backbone: full batches, sequences padded to 1024.
            part=part+[part[-1]]*(batch_size-real);examples=[]
            for i,a,b in part:
                e=layout(enc['input_ids'][i][a:b],[-100]*(b-a),model.kind,2,tok.cls_token_id,tok.sep_token_id)
                mask=[bool(rows[i]['text'][c:d].strip()) for c,d in enc['offset_mapping'][i][a:b]]
                e.update(source_labels=[-100]*(b-a),target=[False]*(b-a),segment_label=-100,mixed_label=-100,sentence_groups=[],
                         document_label=-100,document_only=False,document_mask=mask);examples.append(e)
            batch=collate(examples,tok.pad_token_id);extra=SEQ-batch['input_ids'].shape[1]
            if extra>0:
                batch['input_ids']=torch.nn.functional.pad(batch['input_ids'],(0,extra),value=tok.pad_token_id)
                batch['attention_mask']=torch.nn.functional.pad(batch['attention_mask'],(0,extra),value=0)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(batch)
            probs=out['tokens'].float().softmax(-1)[...,1].cpu().numpy()
            dlogits=out['document'].float().cpu().numpy();weights=batch['document_mask'].sum(1).cpu().numpy()
            for (i,a,b),p,dl,w in zip(part[:real],probs,dlogits,weights):
                sums[i][a:b]+=p[:b-a];den[i][a:b]+=1;doc[i].append((dl,float(w)))
    results=[]
    for i,r in enumerate(rows):
        n=len(enc['input_ids'][i])
        if n and not (den[i]>0).all():raise AssertionError('Uncovered source token')
        p=sums[i]/np.maximum(den[i],1);off=np.asarray(enc['offset_mapping'][i],dtype=np.int64).reshape(-1,2)
        w=np.array([x[1] for x in doc[i]]);dl=np.array([x[0] for x in doc[i]]).reshape(-1,2)
        mean=(dl*w[:,None]).sum(0)/max(w.sum(),1) if len(w) else None
        doc_prob=float(softmax1(mean)) if mean is not None else None
        win=[float(softmax1(x)) for x in dl]
        results.append((r,off,p,doc_prob,win))
    return results


def softmax1(logits):
    import numpy as np
    z=np.exp(logits-np.max(logits));return z[1]/z.sum()


def alive(claim):
    """A claim is live while its process exists, is not a zombie and has the same start time."""
    try:
        pid,ticks=claim.read_text().split()
        stat=Path(f'/proc/{pid}/stat').read_text()
    except (OSError,ValueError):return False
    fields=stat.rsplit(')',1)[1].split()
    return fields[0]!='Z' and fields[19]==ticks


def claim_shard(claims,k):
    """Atomically claim shard k (O_EXCL); a claim left by a dead worker is taken over."""
    path=claims/f'{k:05}.claim'
    me=f"{os.getpid()} {Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]}"
    for _ in range(3):
        try:
            fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
            os.write(fd,me.encode());os.close(fd);return path
        except FileExistsError:
            if alive(path):return None
            try:path.rename(path.with_suffix(f'.stale.{os.getpid()}'))   # only one taker wins the rename
            except FileNotFoundError:pass
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',action='append',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--model',required=True)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--microbatch',type=int,default=16)
    p.add_argument('--stride',type=int,default=510)
    p.add_argument('--rows-per-shard',type=int,default=100)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--claims',type=Path,required=True,help='Local directory shared by all workers of this job')
    p.add_argument('--worker',required=True,help='Worker name; touch CLAIMS/stop-<worker> to stop after the current shard')
    args=p.parse_args()
    import numpy as np,torch
    import pyarrow as pa,pyarrow.parquet as pq
    rows=load_rows(args.source)
    if args.limit:rows=rows[:args.limit]
    manifest=digest('\n'.join(f"{r['id']}\t{r['text_sha256']}" for r in rows).encode())
    name=args.model
    bundle=EVAL/name;code=bundle/'code';run=bundle/'checkpoint'
    sys.path.insert(0,str(code))
    from inference import load_checkpoint
    out=args.output/f'{name}-fast10'
    sel=json.loads((run/'stage2-selection.json').read_text())
    script=digest(Path(__file__).read_bytes())
    provenance={'model':name,'checkpoint':sel['checkpoint'],'checkpoint_sha256':digest((run/sel['checkpoint']).read_bytes()),
                'run_json_sha256':digest((run/'run.json').read_bytes()),
                'code_sha256':{f.name:digest(f.read_bytes()) for f in sorted(code.glob('*.py'))},
                'script_sha256':script,
                'sources':[s.name for s in args.source],'rows':len(rows),'paper_manifest_sha256':manifest,
                'windows':{'width':510,'stride':args.stride},'lora_merged':True,'torch_compile':True,'padded_sequence':SEQ,'batch_size':args.batch_size,'precision':'BF16','calibrated':False,
                'outputs':'token AI probability (class 1) averaged over overlapping windows; document head token-weighted mean over windows'}
    out.mkdir(parents=True,exist_ok=True)
    if (out/'provenance.json').exists():
        # Later script revisions change only shard scheduling, never scoring; they are logged separately.
        old=json.loads((out/'provenance.json').read_text())
        if {**old,'script_sha256':None}!={**provenance,'script_sha256':None}:raise ValueError('Output belongs to another model/input set')
        if old['script_sha256']!=script:
            log=out/'script-revisions.json';revs=json.loads(log.read_text()) if log.exists() else []
            if script not in [r['sha256'] for r in revs]:revs.append({'sha256':script,'first_used':time.time()});atomic(log,json.dumps(revs,indent=2).encode())
    else:atomic(out/'provenance.json',json.dumps(provenance,indent=2).encode())
    shards=[(k,rows[o:o+args.rows_per_shard]) for k,o in enumerate(range(0,len(rows),args.rows_per_shard))]
    claims=args.claims;claims.mkdir(parents=True,exist_ok=True)
    stop=claims/f'stop-{args.worker}';stop.unlink(missing_ok=True)
    def finished():
        names=set(os.listdir(out/'data')) if (out/'data').exists() else set()
        arrays=set(os.listdir(out/'tokens')) if (out/'tokens').exists() else set()
        return {k for k,_ in shards if f'summary-{k:05}.parquet' in names and f'tokens-{k:05}.npz' in arrays}
    def report(state,**extra):
        status={'state':state,'worker':args.worker,'pid':os.getpid(),'updated_at':time.time(),**extra}
        atomic(claims/'workers'/f'{args.worker}.json',json.dumps(status).encode());print(json.dumps(status),flush=True)
    report('loading')
    model,tok,_=load_checkpoint(run);model.physical_microbatch=args.microbatch
    merged=0
    for m in model.backbone.modules():
        if hasattr(m,'merge') and hasattr(m,'lora_A'):m.merge();merged+=1
    if not merged:raise AssertionError('No LoRA modules merged')
    model.backbone=torch.compile(model.backbone,dynamic=False)
    started=time.time();mine=0
    while True:
        if stop.exists():report('stopped',shards_done=mine);return
        done=finished()
        rows_done=sum(len(b) for k,b in shards if k in done)
        atomic(out/'status.json',json.dumps({'state':'complete' if len(done)==len(shards) else 'running','model':name,
               'rows':rows_done,'total':len(rows),'shards':len(done),'of':len(shards),'updated_at':time.time()}).encode())
        if len(done)==len(shards):report('complete',shards_done=mine);return
        held=None
        for k,batch in shards:
            if k in done:continue
            held=claim_shard(claims,k)
            if held:break
        if held is None:report('waiting',shards_done=mine);time.sleep(30);continue   # take over shards of workers that die
        if k in finished():held.unlink();continue
        report('scoring',shard=k,shards_done=mine,elapsed_seconds=time.time()-started)
        results=score(batch,model,tok,args.batch_size,args.stride);table=[];packed={}
        for r,off,pr,dp,win in results:
            packed[r['id']+'/offsets']=off.astype(np.int32);packed[r['id']+'/probs']=pr.astype(np.float16)
            table.append({'id':r['id'],'source_dataset':r['source_dataset'],'pdf_sha256':r['pdf_sha256'],'text_sha256':r['text_sha256'],
                          'tokens':len(pr),'windows':len(win),'document_prob':dp,
                          'window_document_prob_max':max(win) if win else None,
                          'token_prob_mean':float(pr.mean()) if len(pr) else None,
                          'token_prob_max':float(pr.max()) if len(pr) else None,
                          'token_frac_over_0_5':float((pr>0.5).mean()) if len(pr) else None})
        buf=io.BytesIO();np.savez_compressed(buf,**packed);atomic(out/'tokens'/f'tokens-{k:05}.npz',buf.getvalue())
        buf=io.BytesIO();pq.write_table(pa.Table.from_pylist(table),buf,compression='zstd');atomic(out/'data'/f'summary-{k:05}.parquet',buf.getvalue())
        held.unlink(missing_ok=True);mine+=1


if __name__=='__main__':main()
