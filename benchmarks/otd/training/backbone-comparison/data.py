"""Shared, offset-safe examples and Repeat2 layout; no model-specific token labels on disk."""
import hashlib,json,re,random
from collections import defaultdict
IGNORE=-100

def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def normalized_hash(text):return sha(' '.join(text.lower().split()))
def sentence_spans(text):return [(m.start(),m.end()) for m in re.finditer(r'\S.*?(?:[.!?](?=\s|$)|$)',text,re.S)]
def token_labels(text,offsets,regions):
 out=[]
 for a,b in offsets:
  labs={r['label'] for r in regions if min(b,r['end'])>max(a,r['start']) and text[max(a,r['start']):min(b,r['end'])].strip()}
  out.append(next(iter(labs)) if len(labs)==1 and next(iter(labs)) in [0,1] else IGNORE)
 return out

def crop(row,a,b):
 text=row['text'][a:b]
 return {'id':row['id'],'paper_id':row['paper_id'],'kind':row['kind'],'text':text,'source_start':a,'source_end':b,'regions':[{'start':max(r['start'],a)-a,'end':min(r['end'],b)-a,'label':r['label']} for r in row['regions'] if min(r['end'],b)>max(r['start'],a)],'target_start':max(0,min(b-a,row.get('target_start',0)-a)),'target_end':max(0,min(b-a,row.get('target_end',len(row['text']))-a))}

def shared_crop(row,rng,tokenizers,mode='context'):
 text=row['text'];start,end=0,len(text)
 if mode=='target':start,end=row['target_start'],row['target_end']
 elif len(text)>1600:
  # Character anchors make the raw context identical for both tokenizers.
  anchor=rng.randrange(len(text)) if mode=='random' else rng.randrange(row.get('target_start',0),max(row.get('target_start',0)+1,row.get('target_end',len(text))))
  start=max(0,min(len(text)-1600,anchor-rng.randrange(1600)))
  while start and start<len(text) and not text[start-1].isspace():start+=1
  end=min(len(text),start+1600)
 # Bound source tokens for BOTH models, without model-specific truncation.
 fragment=text[start:end]
 for tok in tokenizers:
  offsets=tok(fragment,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
  if len(offsets)>510:fragment=fragment[:offsets[510][0]]
 while fragment and end<len(text) and fragment[-1:].isspace():fragment=fragment[:-1]
 if not fragment.strip():raise ValueError('Empty crop')
 return crop(row,start,start+len(fragment))

def layout(ids,labels,kind,stage,bos=None,eos=None):
 """Repeat real tokens before right padding; first-copy labels are always ignored."""
 if kind=='causal':
  repeat=stage==2
  sequence=ids+ids if repeat else ids[:]
  ys=[IGNORE]*len(ids)+labels if repeat else labels[:]
  source_positions=list(range(len(ids),2*len(ids))) if repeat else list(range(len(ids)))
 else:
  if bos is None or eos is None:raise ValueError('Encoder requires boundary tokens')
  sequence=[bos]+ids+[eos];ys=[IGNORE]+labels+[IGNORE];source_positions=list(range(1,len(ids)+1))
 return {'ids':sequence,'labels':ys,'source_positions':source_positions,'last_position':source_positions[-1]}

def encode_example(row,tok,kind,stage):
 e=tok(row['text'],add_special_tokens=False,return_offsets_mapping=True);ids=e['input_ids'];off=e['offset_mapping']
 if not ids or len(ids)>510:raise ValueError('Shared window token budget violated')
 ys=token_labels(row['text'],off,row['regions']);out=layout(ids,ys,kind,stage,tok.cls_token_id,tok.sep_token_id)
 out.update(offsets=off,source_labels=ys,text=row['text'],target=[a>=row['target_start'] and b<=row['target_end'] for a,b in off])
 chars={label:sum(r['end']-r['start'] for r in row['regions'] if r['label']==label) for label in [0,1,IGNORE]}
 # No document-level assertion when any part of the window has unknown provenance.
 out['segment_label']=min(14,int(15*chars[1]/max(1,chars[0]+chars[1]))) if chars[IGNORE]==0 else IGNORE
 valid=[y for y in ys if y>=0]
 out['mixed_label']=int(min(valid.count(0),valid.count(1))/len(valid)>.15) if valid and chars[IGNORE]==0 else IGNORE
 out['sentence_groups']=[]
 for a,b in sentence_spans(row['text']):
  selected=[i for i,(c,d) in enumerate(off) if d>a and c<b and row['text'][c:d].strip()];labs={ys[i] for i in selected}
  if selected and len(labs)==1 and next(iter(labs)) in [0,1]:out['sentence_groups'].append((selected,next(iter(labs))))
 return out
