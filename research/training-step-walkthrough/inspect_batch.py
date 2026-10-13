"""Reconstruct labels for a saved training batch, without model inference."""
import collections,gzip,hashlib,json,sys
from pathlib import Path
from tokenizers import Tokenizer
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'benchmarks/otd/training/lora-comparison'))
from data import token_labels,sentence_spans
p=root/'benchmarks/otd/training/backbone-comparison/prepared/stage2-epoch0.jsonl.gz'
tp=root/'benchmarks/otd/exports/ai-paper-provenance-v3-10000/tokenizer/tokenizer.json'
t=Tokenizer.from_file(str(tp));blob=gzip.decompress(p.read_bytes());rows=[json.loads(l) for l in blob.splitlines()][:32];out=[]
for i,r in enumerate(rows):
 e=t.encode(r['text'],add_special_tokens=False);ys=token_labels(r['text'],e.offsets,r['regions']);chars=collections.Counter()
 for reg in r['regions']:chars[reg['label']]+=reg['end']-reg['start']
 valid=[y for y in ys if y>=0];target=[a>=r['target_start'] and b<=r['target_end'] for a,b in e.offsets]
 groups=[]
 for a,b in sentence_spans(r['text']):
  selected=[j for j,(c,d) in enumerate(e.offsets) if d>a and c<b and r['text'][c:d].strip()];labs={ys[j] for j in selected}
  if selected and len(labs)==1 and next(iter(labs)) in [0,1]:groups.append((selected,next(iter(labs))))
 out.append(dict(row=i+1,id=r['id'],kind=r['kind'],source_tokens=len(ys),token_labels=dict(collections.Counter(ys)),char_labels=dict(chars),segment_label=min(14,int(15*chars[1]/max(1,chars[0]+chars[1]))),mixed_label=int(min(valid.count(0),valid.count(1))/len(valid)>.15),document_label=int(chars[1]>0),sentence_groups=len(groups),target_scored=sum(z and y>=0 for z,y in zip(target,ys)),context_scored=sum(not z and y>=0 for z,y in zip(target,ys))))
micro=[]
for i in range(4):
 batch=out[i*8:(i+1)*8];pad=((max(r['source_tokens'] for r in batch)+2+7)//8)*8
 micro.append(dict(microbatch=i+1,input_shape=[8,pad],token_output_shape=[8,max(r['source_tokens'] for r in batch),2],real_input_tokens=sum(r['source_tokens']+2 for r in batch),rows=batch))
record=dict(note='Reconstructed saved control schedule, not a replay of historical model outputs.',data_sha256=hashlib.sha256(blob).hexdigest(),tokenizer_sha256=hashlib.sha256(tp.read_bytes()).hexdigest(),microbatches=micro)
(root/'research/training-step-walkthrough/batch.json').write_text(json.dumps(record,indent=2))
for m in micro:
 print('MICROBATCH',m['microbatch'],m['input_shape'],m['token_output_shape'],'real tokens',m['real_input_tokens'])
 for r in m['rows']:print(r['row'],r['source_tokens'],r['token_labels'],r['segment_label'],r['mixed_label'],r['document_label'])
print('EXAMPLE',json.dumps(out[2]));print('DATA HASH',record['data_sha256'])
