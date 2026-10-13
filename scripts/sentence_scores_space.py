"""Sentence scores from classify_clean_papers.py token outputs (run on the Space).

Sentence rule matches research/evaluation/backbone-results-20261003/sentence_roc.py:
regex sentences; score = mean AI probability of non-whitespace tokens overlapping
the sentence. Eval-suite rows also get the suite's per-sentence gold (kept only when
all those tokens share one label).

  sentence_scores_space.py suite  SUITE_DIR TEXT_PARQUET RUN_DIR OUT.json.gz
  sentence_scores_space.py papers TEXT_DATASET... --run RUN_DIR --out DIR [--workers N]
The papers mode writes per-shard npz (id/starts, id/ends, id/scores) next to the
token shards, under DIR/sentences-xxxxx.npz, plus DIR/paper-sentences.parquet with
per-paper sentence counts and score quantiles.
"""
from concurrent.futures import ProcessPoolExecutor
import gzip
import io
import json
from pathlib import Path
import re
import sys
import zipfile

import numpy as np

SENT=re.compile(r'\S.*?(?:[.!?](?=\s|$)|$)',re.S)


def sentence_spans(text):
    return [(m.start(),m.end()) for m in SENT.finditer(text) if m.group().strip()]


def sentence_scores(text,off,prob):
    """Mean probability of valid tokens overlapping each sentence, via prefix sums."""
    off=np.asarray(off).reshape(-1,2);prob=np.asarray(prob,dtype=np.float64)
    valid=np.array([bool(text[a:b].strip()) for a,b in off],dtype=np.float64)
    cs=np.r_[0,np.cumsum(prob*valid)];cv=np.r_[0,np.cumsum(valid)]
    starts=[];ends=[];scores=[]
    for a,b in sentence_spans(text):
        lo=np.searchsorted(off[:,1],a,side='right');hi=np.searchsorted(off[:,0],b,side='left')
        n=cv[hi]-cv[lo]
        if hi<=lo or n<=0:continue
        starts.append(a);ends.append(b);scores.append((cs[hi]-cs[lo])/n)
    return np.array(starts,np.int32),np.array(ends,np.int32),np.array(scores,np.float32)


def load_npz(path):
    z=np.load(path);out={}
    for k in z.files:
        pid,kind=k.rsplit('/',1);out.setdefault(pid,{})[kind]=z[k]
    return out


def suite(suite_dir,text_parquet,run_dir,out):
    import pyarrow.parquet as pq
    sys.path.insert(0,str(suite_dir));from common import labels_from_regions
    meta={}
    for prof in ['workflow','comparison','manuscripts']:
        with gzip.open(Path(suite_dir)/f'{prof}.jsonl.gz','rt') as g:
            for line in g:
                r=json.loads(line)
                meta[r['id']]={'text':r['text'],'regions':r.get('regions'),'profile':prof,
                               **{k:r.get(k) for k in ['dataset','label','view','condition','paper_id']}}
    rows=[]
    for f in sorted((Path(run_dir)/'tokens').glob('tokens-*.npz')):
        for pid,d in load_npz(f).items():
            m=meta[pid];text=m['text'];off=d['offsets'];prob=d['probs'].astype(np.float64)
            item={k:m[k] for k in ['profile','dataset','label','view','condition','paper_id']};item['id']=pid
            if m['regions'] is not None:
                ys=np.asarray(labels_from_regions(text,off,m['regions']))
                valid=np.array([bool(text[a:b].strip()) for a,b in off])
                sent=[]
                for a,b in sentence_spans(text):
                    use=valid&(off[:,1]>a)&(off[:,0]<b)
                    if not use.any():continue
                    labs=set(ys[use].tolist())
                    if len(labs)==1 and next(iter(labs)) in (0,1):sent.append([round(float(prob[use].mean()),5),int(next(iter(labs)))])
                item['sentences']=sent
            rows.append(item)
    with gzip.open(out,'wt') as g:json.dump(rows,g)
    print('suite rows',len(rows),'sentences',sum(len(r.get('sentences',[])) for r in rows))


def shard_job(job):
    texts,tokens_path,out_path=job
    arrays=load_npz(tokens_path);packed={};summary=[]
    for pid,d in arrays.items():
        s,e,sc=sentence_scores(texts[pid],d['offsets'],d['probs'])
        packed[pid+'/starts']=s;packed[pid+'/ends']=e;packed[pid+'/scores']=sc.astype(np.float16)
        q=np.quantile(sc,[.5,.9,.99]).tolist() if len(sc) else [None]*3
        summary.append({'id':pid,'sentences':len(sc),'sentence_mean':float(sc.mean()) if len(sc) else None,
                        'sentence_p50':q[0],'sentence_p90':q[1],'sentence_p99':q[2]})
    buf=io.BytesIO();np.savez_compressed(buf,**packed);tmp=Path(out_path+'.tmp');tmp.write_bytes(buf.getvalue());tmp.replace(out_path)
    return summary


def papers(datasets,run_dir,out,workers):
    import pyarrow as pa,pyarrow.parquet as pq
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    texts={}
    for d in datasets:
        for p in sorted((Path(d)/'data').glob('train-*.parquet')):
            for r in pq.read_table(p,columns=['id','text']).to_pylist():texts[r['id']]=r['text']
    def jobs():
        for f in sorted((Path(run_dir)/'tokens').glob('tokens-*.npz')):
            dest=out/f.name.replace('tokens-','sentences-')
            if dest.exists():continue
            ids={n.rsplit('/',1)[0] for n in zipfile.ZipFile(f).namelist()}
            yield ({i:texts[i] for i in ids},str(f),str(dest))
    rows=[]
    with ProcessPoolExecutor(workers) as pool:
        for i,s in enumerate(pool.map(shard_job,jobs())):
            rows+=s
            if i%25==0:print('shards',i+1,flush=True)
    if rows:
        old=pq.read_table(out/'paper-sentences.parquet').to_pylist() if (out/'paper-sentences.parquet').exists() else []
        pq.write_table(pa.Table.from_pylist(old+rows),out/'paper-sentences.parquet',compression='zstd')
    print('papers',len(rows))


if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='suite':suite(*sys.argv[2:6])
    else:
        a=sys.argv[2:];run=a[a.index('--run')+1];out=a[a.index('--out')+1]
        workers=int(a[a.index('--workers')+1]) if '--workers' in a else 32
        datasets=[x for x in a[:a.index('--run')]]
        papers(datasets,run,out,workers)
