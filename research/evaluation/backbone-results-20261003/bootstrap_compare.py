"""Paired paper/group-clustered bootstrap of key metrics, each model vs Qwen3.6 35B-A3B (run from this folder)."""
import json,gzip,numpy as np,sys
sys.path.insert(0,'.')
from analyze import load, clean, PAPER_HUMAN, MODELS
def rankdata(s):
    o=np.argsort(s,kind="mergesort");r=np.empty(len(s));r[o]=np.arange(1,len(s)+1);ss=s[o];b=np.r_[True,ss[1:]!=ss[:-1]];gid=np.cumsum(b)-1;sums=np.bincount(gid,r[o]);cnt=np.bincount(gid);return (sums/cnt)[gid[np.argsort(o)]]
keys=[k for k,_,_ in MODELS]
D={k:load(k) for k in keys}
def idx(rows): return {r['id']:i for i,r in enumerate(rows)}
# Build aligned arrays per metric. Use the first model's row order; verify ids match.
def aligned(prof, filt):
    base=[r for r in D[keys[0]][prof] if filt(r)]; ids=[r['id'] for r in base]
    out={}
    for k in keys:
        m={r['id']:r for r in D[k][prof]}; out[k]=[m[i] for i in ids]
    return base,out
def auc(s,y):
    r=rankdata(s); p=y.sum(); n=len(y)-p
    return (r[y==1].sum()-p*(p+1)/2)/(p*n)
def tpr_at(s,y,f=0.01):
    neg=np.sort(s[y==0])[::-1]; t=neg[int(f*len(neg))]; return (s[y==1]>t).mean()
rng=np.random.default_rng(0); B=400
def boot(rows_by_model, base, stat):
    g=np.array([r['group_id'] for r in base]); ug,inv=np.unique(g,return_inverse=True)
    members=[np.nonzero(inv==j)[0] for j in range(len(ug))]
    point={k:stat(rows_by_model[k],np.arange(len(base))) for k in keys}
    reps={k:[] for k in keys}
    for b in range(B):
        pick=rng.integers(0,len(ug),len(ug)); ix=np.concatenate([members[j] for j in pick])
        for k in keys: reps[k].append(stat(rows_by_model[k],ix))
    return point,{k:np.array(v) for k,v in reps.items()}
def report(name,point,reps,ref='qwen36-35b-a3b'):
    print(f'\n## {name}')
    for k in keys:
        lo,hi=np.percentile(reps[k],[2.5,97.5]); d=reps[k]-reps[ref]; dlo,dhi=np.percentile(d,[2.5,97.5])
        print(f'  {k:15s} {point[k]:.3f} [{lo:.3f},{hi:.3f}]   vs MoE: {point[k]-point[ref]:+.3f} [{dlo:+.3f},{dhi:+.3f}]')
# Papers ROC: human paper paragraphs (workflow+comparison) vs rewritten (wf ai) + manuscripts
def paper_rows(k):
    wf=D[k]['workflow'];co=D[k]['comparison'];ms=D[k]['manuscripts']
    h=[r for r in clean(co,'human')+clean(wf,'human') if r['dataset'] in PAPER_HUMAN]
    a=clean(wf,'ai')+ms
    return sorted(h+a,key=lambda r:r['id'])
base=paper_rows(keys[0]); P={k:paper_rows(k) for k in keys}
assert all([r['id'] for r in P[k]]==[r['id'] for r in base] for k in keys)
arr={k:(np.array([r['mean_ai_probability'] for r in P[k]]),np.array([r['native_label']=='ai' for r in P[k]])) for k in keys}
pt,rp=boot(arr,base,lambda a,ix:auc(a[0][ix],a[1][ix])); report('Papers AUROC',pt,rp)
pt,rp=boot(arr,base,lambda a,ix:tpr_at(a[0][ix],a[1][ix])); report('Papers recall @1% FPR (oracle)',pt,rp)
# Rewritten paragraphs only (exclude manuscripts) at 1% FPR
base2=[r for r in base if r['dataset']!='full_manuscripts']; keep=[i for i,r in enumerate(base) if r['dataset']!='full_manuscripts']
arr2={k:(arr[k][0][keep],arr[k][1][keep]) for k in keys}
pt,rp=boot(arr2,base2,lambda a,ix:tpr_at(a[0][ix],a[1][ix])); report('Rewritten paper paragraphs recall @1% paper-human FPR (oracle)',pt,rp)
# Public pooled
def pub(k):
    co=D[k]['comparison'];return sorted(clean(co,'human')+clean(co,'ai'),key=lambda r:r['id'])
base=pub(keys[0]); P={k:pub(k) for k in keys}
arr={k:(np.array([r['mean_ai_probability'] for r in P[k]]),np.array([r['native_label']=='ai' for r in P[k]])) for k in keys}
pt,rp=boot(arr,base,lambda a,ix:auc(a[0][ix],a[1][ix])); report('Public pooled AUROC',pt,rp)
# Edits: paragraph conditions caught, two-sentence caught, untouched false alarm
for cond,label,fn in [(('paragraph_concise','paragraph_v3'),'Edited paragraphs caught',lambda r:r['sentence_counts'][0]>0),(('two_sentence',),'Two-sentence edits caught',lambda r:r['sentence_counts'][0]>0),(('sentence',),'One-sentence edits caught',lambda r:r['sentence_counts'][0]>0),(('untouched',),'Untouched passages with any flagged sentence',lambda r:r['sentence_counts'][1]>0)]:
    base,al=aligned('workflow',lambda r:r['condition'] in cond and r['native_label']!='ai')
    arr={k:np.array([fn(r) for r in al[k]],float) for k in keys}
    pt,rp=boot(arr,base,lambda a,ix:a[ix].mean()); report(label,pt,rp)
# Workflow token F1
base,al=aligned('workflow',lambda r:r['native_label']=='mixed')
arr={k:np.array([r['token_counts'] for r in al[k]],float) for k in keys}
def f1(a,ix): tp,fp,fn,tn=a[ix].sum(0); return 2*tp/(2*tp+fp+fn)
pt,rp=boot(arr,base,f1); report('Workflow reconstruction token F1',pt,rp)
