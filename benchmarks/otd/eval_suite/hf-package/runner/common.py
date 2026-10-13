"""Offset-safe supervision and validation-only operating-point calibration."""
import hashlib
import numpy as np

GOOD = {'fully_faithful', 'mostly_faithful_with_minor_differences'}

def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()

def labels_from_regions(text, offsets, regions):
    labels=[]
    for start,end in offsets:
        # Ignore whitespace and tokens spanning two production labels.
        active={r['label'] for r in regions if min(end,r['end'])>max(start,r['start']) and text[max(start,r['start']):min(end,r['end'])].strip()}
        labels.append(next(iter(active)) if len(active)==1 and next(iter(active)) in (0,1) else -100)
    return labels

def window_starts(n, width=510, stride=256):
    if n<=width:return [0]
    return sorted(set(list(range(0,n-width+1,stride))+[n-width]))

def choose_threshold(scores, labels, fpr=.01):
    human=np.asarray(scores,dtype=np.float64)[np.asarray(labels)==0]
    if not len(human):raise ValueError('No human validation samples')
    # Strictly above the empirical upper-tail cut prevents tie-driven FPR overshoot.
    order=np.sort(human);allowed=int(np.floor(fpr*len(order)))
    cutoff=order[len(order)-allowed-1]
    return float(np.nextafter(cutoff,np.inf))

def counts(scores, labels, threshold):
    p=np.asarray(scores,dtype=np.float64)>=threshold;y=np.asarray(labels)==1
    return np.array([np.sum(p&y),np.sum(p&~y),np.sum(~p&y),np.sum(~p&~y)],dtype=np.int64)

def metrics(c):
    tp,fp,fn,tn=map(int,c)
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'human_fpr':fp/(fp+tn) if fp+tn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}
