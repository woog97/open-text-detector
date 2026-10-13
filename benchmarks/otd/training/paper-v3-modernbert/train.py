"""One reproducible, text-only ModernBERT token-classifier training/evaluation run."""
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('HF_HOME','/home/user/.cache/huggingface')
import argparse,collections,copy,json,math,random,time,traceback,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from transformers import AutoTokenizer,AutoModelForTokenClassification,get_cosine_schedule_with_warmup
from huggingface_hub import HfApi,hf_hub_download
import pyarrow.parquet as pq
from common import GOOD,digest,labels_from_regions,window_starts,choose_threshold,counts,metrics

DATASET='woog/ai-paper-provenance-v3'
REVISION='98ca42d9ff340d8a8fddba50e880a8528aafecf2'
MODEL='answerdotai/ModernBERT-base'
SEED=42
COLS=['id','pair_id','paper_id','variant','text','regions','sentences','target_start','target_end','target_label','development_exposed','eligible_for_pilot_training','paired_generation_fidelity_verdict']

def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

def emit(event,**kw):
    record={'event':event,'time':time.time(),**kw}
    print(json.dumps(record,allow_nan=False),flush=True)
    if 'OUT' in globals():save(OUT/'status.json',record)

def load_rows(split):
    p=hf_hub_download(DATASET,repo_type='dataset',revision=REVISION,filename=f'data/fresh_passages/{split}-00000-of-00001.parquet')
    rows=pq.read_table(p,columns=COLS).to_pylist()
    assert all(not r['development_exposed'] for r in rows)
    return rows

def good_pairs(rows):
    groups=collections.defaultdict(list)
    for r in rows:groups[r['pair_id']].append(r)
    return {p for p,rs in groups.items() if len(rs)==2 and {r['target_label'] for r in rs}=={0,1} and all(r['eligible_for_pilot_training'] and r['paired_generation_fidelity_verdict'] in GOOD for r in rs)}

def encode(rows,tok):
    result=[]
    for r in rows:
        e=tok(r['text'],add_special_tokens=False,return_offsets_mapping=True,truncation=False)
        labels=labels_from_regions(r['text'],e['offset_mapping'],r['regions'])
        assert len(labels)==len(e['input_ids'])
        if not any(y>=0 for y in labels):continue
        result.append({**r,'ids':e['input_ids'],'offsets':e['offset_mapping'],'labels':labels})
    return result

class TrainData(Dataset):
    def __init__(self,rows,tok):self.rows=rows;self.tok=tok;self.epoch=0
    def __len__(self):return len(self.rows)*3
    def __getitem__(self,i):
        r=self.rows[i//3];mode=i%3;rng=random.Random(SEED+self.epoch*1000003+i)
        if mode==0:
            text=r['text'][r['target_start']:r['target_end']]
            ids=self.tok(text,add_special_tokens=False)['input_ids'];ys=[r['target_label']]*len(ids)
            start=rng.randrange(max(1,len(ids)-510+1));ids=ids[start:start+510];ys=ys[start:start+510]
        else:
            n=len(r['ids']);target=[j for j,(a,b) in enumerate(r['offsets']) if b>r['target_start'] and a<r['target_end']]
            anchor=rng.choice(target) if mode==1 else rng.randrange(n)
            start=max(0,min(n-510,anchor-rng.randrange(510)))
            ids=r['ids'][start:start+510];ys=r['labels'][start:start+510]
        return {'ids':ids,'labels':ys}

def collate(examples,tok):
    n=max(len(x['ids'])+2 for x in examples);n=math.ceil(n/8)*8
    ids=[];mask=[];labels=[]
    for e in examples:
        seq=[tok.cls_token_id]+e['ids']+[tok.sep_token_id];ys=[-100]+e['labels']+[-100];p=n-len(seq)
        ids.append(seq+[tok.pad_token_id]*p);mask.append([1]*len(seq)+[0]*p);labels.append(ys+[-100]*p)
    return {k:torch.tensor(v,device='cuda') for k,v in [('input_ids',ids),('attention_mask',mask),('labels',labels)]}

@torch.inference_mode()
def predict(model,tok,rows,batch_size=32,target_only=False):
    model.eval();sums=[np.zeros(len(r['ids']),np.float64) for r in rows];ns=[np.zeros(len(r['ids']),np.int32) for r in rows]
    jobs=[]
    for ri,r in enumerate(rows):
        if target_only:
            indices=[j for j,(a,b) in enumerate(r['offsets']) if a>=r['target_start'] and b<=r['target_end']]
            if not indices:continue
            left,right=indices[0],indices[-1]+1
        else:left,right=0,len(r['ids'])
        for relative in window_starts(right-left):
            start=left+relative;end=min(start+510,right);jobs.append((ri,start,end))
    for b in range(0,len(jobs),batch_size):
        part=jobs[b:b+batch_size];examples=[{'ids':rows[i]['ids'][s:e],'labels':[-100]*(e-s)} for i,s,e in part]
        batch=collate(examples,tok);batch.pop('labels')
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(**batch).logits
        probabilities=logits.float().softmax(-1)[...,1].cpu().numpy()
        for p,(i,s,e) in zip(probabilities,part):sums[i][s:e]+=p[1:1+e-s];ns[i][s:e]+=1
    result=[(s/np.maximum(n,1)).astype(np.float32) for s,n in zip(sums,ns)]
    if not target_only:assert all(np.all(n>0) for n in ns)
    return result

def cross_entropy(rows,probs):
    total=0.;n=0
    for r,p in zip(rows,probs):
        y=np.asarray(r['labels']);mask=y>=0;p=np.clip(p[mask],1e-7,1-1e-7);y=y[mask]
        total+=float(-(y*np.log(p)+(1-y)*np.log1p(-p)).sum());n+=len(y)
    return total/n

def units(rows,probs,kind):
    result=[]
    for r,p in zip(rows,probs):
        off=np.asarray(r['offsets']);ys=np.asarray(r['labels']);valid=ys>=0
        if kind in ('tokens','target_tokens','boundary_tokens'):
            if kind=='target_tokens':valid &= (off[:,0]>=r['target_start']) & (off[:,1]<=r['target_end'])
            if kind=='boundary_tokens':
                if r['target_label']!=1:continue
                valid &= (np.minimum(abs(off[:,0]-r['target_start']),abs(off[:,0]-r['target_end']))<=32)
            result.append((r['paper_id'],p[valid],ys[valid]))
        elif kind=='sentences':
            for s in r['sentences']:
                if not s['loss_mask'] or s['binary_label'] not in (0,1):continue
                selected=valid & (off[:,1]>s['start']) & (off[:,0]<s['end'])
                # Sentence label requires every scored token to agree with the sentence label.
                if not selected.any() or np.any(ys[selected]!=s['binary_label']):continue
                result.append((r['paper_id'],np.asarray([p[selected].mean()]),np.asarray([s['binary_label']]),digest(s['text'])))
    if kind=='sentences':
        labels=collections.defaultdict(set)
        for paper,p,y,h in result:labels[h].add(int(y[0]))
        seen=set();clean=[]
        for paper,p,y,h in result:
            if h in seen or h in CROSS_SENTENCES or len(labels[h])!=1:continue
            seen.add(h);clean.append((paper,p,y))
        result=clean
    return result

def flat(u):return np.concatenate([x[1] for x in u]),np.concatenate([x[2] for x in u])

def summarize(u,threshold,bootstrap=0):
    bypaper=collections.defaultdict(lambda:np.zeros(4,np.int64))
    for paper,p,y in u:bypaper[paper]+=counts(p,y,threshold)
    if not bypaper:return {'n':0,'papers':0}
    arr=np.asarray(list(bypaper.values()));out={**metrics(arr.sum(0)),'n':int(arr.sum()),'papers':len(arr),'threshold':threshold}
    if bootstrap:
        rng=np.random.default_rng(SEED);samples={k:[] for k in ['precision','recall','human_fpr']}
        for _ in range(bootstrap):
            m=metrics(arr[rng.integers(0,len(arr),len(arr))].sum(0))
            for k in samples:
                if m[k] is not None:samples[k].append(m[k])
        out['paper_bootstrap_95ci']={k:np.quantile(v,[.025,.975]).tolist() for k,v in samples.items() if v}
    return out

def evaluate(rows,probs,thresholds,bootstrap=0):
    return {kind:summarize(units(rows,probs,kind),thresholds['sentences' if kind=='sentences' else 'tokens'],bootstrap) for kind in ['tokens','target_tokens','sentences','boundary_tokens']}

def main(args):
    global OUT,CROSS_SENTENCES
    OUT=Path(args.output);OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'results.json').exists():raise RuntimeError('This run is already complete; do not overwrite test results.')
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    model_revision=HfApi().model_info(MODEL).sha
    config={'dataset':DATASET,'dataset_revision':REVISION,'base_model':MODEL,'base_model_revision':model_revision,'seed':SEED,'epochs':args.epochs,'learning_rate':2e-5,'weight_decay':.01,'batch_size':16,'gradient_accumulation':2,'max_length':512,'inference_stride':256,'precision':'bfloat16','sampling':'inverse retained targets per paper; paired variants balanced; 1/3 target-only, 1/3 target-anchored crop, 1/3 random-context crop','checkpoint_selection':'lowest validation unweighted token cross-entropy','threshold_selection':'separate token and mean-token sentence thresholds at <=1% validation human FPR','test_policy':'one final pass after checkpoint/threshold freeze','gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'started_at':time.time()}
    save(OUT/'config.json',config);emit('loading_data')
    train_all=load_rows('train');val_all=load_rows('validation');gp=good_pairs(train_all);vp=good_pairs(val_all)
    # Only identity/label metadata from test is used for overlap exclusions, never test predictions.
    split_hashes=collections.defaultdict(set);target_labels=collections.defaultdict(set);target_splits=collections.defaultdict(set)
    for split in ['train','validation','test']:
        rs=train_all if split=='train' else val_all if split=='validation' else load_rows('test')
        for r in rs:
            h=digest(r['text'][r['target_start']:r['target_end']]);target_labels[h].add(r['target_label']);target_splits[h].add(split)
            for s in r['sentences']:split_hashes[digest(s['text'])].add(split)
    CROSS_SENTENCES={h for h,ss in split_hashes.items() if len(ss)>1}
    forbidden={h for h,ls in target_labels.items() if len(ls)>1 or len(target_splits[h])>1}
    def remove_pairs(rows,good):
        bad={r['pair_id'] for r in rows if digest(r['text'][r['target_start']:r['target_end']]) in forbidden}
        return good-bad
    gp=remove_pairs(train_all,gp);vp=remove_pairs(val_all,vp)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=model_revision)
    assert tok.is_fast and tok.cls_token_id is not None and tok.sep_token_id is not None
    train=encode([r for r in train_all if r['pair_id'] in gp],tok);val=encode([r for r in val_all if r['pair_id'] in vp],tok)
    assert not ({r['paper_id'] for r in train}&{r['paper_id'] for r in val})
    manifest={'train_rows':len(train),'train_pairs':len(gp),'train_papers':len({r['paper_id'] for r in train}),'validation_rows':len(val),'validation_pairs':len(vp),'validation_papers':len({r['paper_id'] for r in val}),'train_ids':[r['id'] for r in train],'validation_ids':[r['id'] for r in val],'cross_split_sentence_hashes':len(CROSS_SENTENCES),'forbidden_target_hashes':len(forbidden)}
    save(OUT/'data_manifest.json',manifest);emit('data_ready',**{k:v for k,v in manifest.items() if not k.endswith('_ids')})
    model=AutoModelForTokenClassification.from_pretrained(MODEL,revision=model_revision,num_labels=2,id2label={0:'human',1:'ai'},label2id={'human':0,'ai':1},attn_implementation='sdpa').cuda()
    dataset=TrainData(train,tok);paper_n=collections.Counter(r['paper_id'] for r in train)
    weights=torch.tensor([1/paper_n[r['paper_id']] for r in train for _ in range(3)],dtype=torch.double)
    sample_n=len(dataset);sampler=WeightedRandomSampler(weights,sample_n,replacement=True,generator=torch.Generator().manual_seed(SEED))
    loader=DataLoader(dataset,batch_size=16,sampler=sampler,collate_fn=lambda es:collate(es,tok),num_workers=0)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    total_steps=math.ceil(len(loader)/2)*args.epochs;sched=get_cosine_schedule_with_warmup(opt,int(.06*total_steps),total_steps)
    best=float('inf');history=[];global_step=0;started=time.time()
    for epoch in range(args.epochs):
        dataset.epoch=epoch;model.train();opt.zero_grad(set_to_none=True);loss_sum=0;nb=0
        for bi,batch in enumerate(loader):
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch).loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            # Last partial accumulation has the correct denominator.
            divisor=2 if bi < (len(loader)//2)*2 else 1
            (loss/divisor).backward();loss_sum+=loss.item();nb+=1
            if (bi+1)%2==0 or bi+1==len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();sched.step();opt.zero_grad(set_to_none=True);global_step+=1
                if global_step%25==0:emit('training',epoch=epoch+1,step=global_step,total_steps=total_steps,mean_loss=loss_sum/nb,elapsed_seconds=time.time()-started)
        pred=predict(model,tok,val);vl=cross_entropy(val,pred)
        p,y=flat(units(val,pred,'tokens'));threshold=choose_threshold(p,y);vm=metrics(counts(p,y,threshold))
        rec={'epoch':epoch+1,'train_loss':loss_sum/nb,'validation_loss':vl,'validation_token_operating_point':vm,'elapsed_seconds':time.time()-started};history.append(rec)
        if vl<best:
            best=vl;model.save_pretrained(OUT/'best_model');tok.save_pretrained(OUT/'best_model');save(OUT/'best_checkpoint.json',rec)
        # Optimizer state and current weights enable an explicit recovery without losing work.
        torch.save({'epoch':epoch+1,'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'model':model.state_dict(),'random_state':random.getstate(),'numpy_state':np.random.get_state(),'torch_state':torch.get_rng_state(),'cuda_state':torch.cuda.get_rng_state_all(),'sampler_state':sampler.generator.get_state()},'/tmp/paper-v3-last-state.pt')
        shutil.copyfile('/tmp/paper-v3-last-state.pt',OUT/'last-training-state.pt')
        save(OUT/'history.json',history);emit('epoch_complete',**rec)
    del model,opt,sched;torch.cuda.empty_cache()
    model=AutoModelForTokenClassification.from_pretrained(OUT/'best_model',attn_implementation='sdpa').cuda()
    vpred=predict(model,tok,val);thresholds={}
    for kind in ['tokens','sentences']:
        p,y=flat(units(val,vpred,kind));thresholds[kind]=choose_threshold(p,y)
    save(OUT/'thresholds.json',{'frozen_at':time.time(),'target_human_fpr':.01,**thresholds})
    validation=evaluate(val,vpred,thresholds)
    emit('test_evaluation_started',thresholds=thresholds)
    test_all=load_rows('test');tp=remove_pairs(test_all,good_pairs(test_all));test=encode(test_all,tok)
    assert not ({r['paper_id'] for r in test}&({r['paper_id'] for r in train}|{r['paper_id'] for r in val}))
    testpred=predict(model,tok,test)
    results={'config':config,'best_checkpoint':json.loads((OUT/'best_checkpoint.json').read_text()),'thresholds':thresholds,'validation':validation,'test':{},'limitations':['Single seed; Luna-generated research prose only.','Automated quality judgments, not expert gold.','Token metrics use ModernBERT tokenization; full-context tokens include repeated human context.','Sentence results deduplicate identical text and exclude cross-split/conflicting sentence text.','Target-only metrics are the primary control against inflated repeated-context performance.','Bootstrap samples papers and conditions on this checkpoint and validation-selected thresholds.']}
    for name,mask in [('quality_filtered',[r['pair_id'] in tp for r in test]),('fully_faithful',[r['pair_id'] in tp and r['paired_generation_fidelity_verdict']=='fully_faithful' for r in test]),('all_fresh',[True]*len(test))]:
        rr=[r for r,m in zip(test,mask) if m];pp=[p for p,m in zip(testpred,mask) if m]
        results['test'][name]={'rows':len(rr),'pairs':len({r['pair_id'] for r in rr}),**evaluate(rr,pp,thresholds,500)}
    # Positional-shortcut sensitivity: score target tokens with only their paragraph as input.
    filtered=[r for r in test if r['pair_id'] in tp];isolated=predict(model,tok,filtered,target_only=True)
    results['test']['isolated_paragraph_sensitivity']=summarize(units(filtered,isolated,'target_tokens'),thresholds['tokens'],500)
    results['elapsed_seconds']=time.time()-config['started_at'];results['completed_at']=time.time()
    save(OUT/'results.json',results)
    np.savez_compressed(OUT/'test_predictions.npz',**{str(i):p for i,p in enumerate(testpred)})
    save(OUT/'test_prediction_ids.json',[r['id'] for r in test]);save(OUT/'test_subset_ids.json',[r['id'] for r in filtered])
    emit('complete',output=str(OUT),elapsed_seconds=results['elapsed_seconds'],test=results['test']['quality_filtered'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=4);args=p.parse_args()
    try:main(args)
    except Exception as e:
        emit('failed',error=type(e).__name__,message=str(e));traceback.print_exc();raise
