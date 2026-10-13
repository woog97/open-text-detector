"""Apply a saved run with its frozen token threshold to UTF-8 research text."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForTokenClassification
from common import window_starts

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--text-file',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    run=Path(args.run);text=Path(args.text_file).read_text();device='cuda' if torch.cuda.is_available() else 'cpu'
    tok=AutoTokenizer.from_pretrained(run/'best_model');model=AutoModelForTokenClassification.from_pretrained(run/'best_model',attn_implementation='sdpa').to(device).eval()
    threshold=json.loads((run/'thresholds.json').read_text())['tokens']
    encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True);ids=encoded['input_ids'];offsets=encoded['offset_mapping'];sums=np.zeros(len(ids));n=np.zeros(len(ids))
    for start in window_starts(len(ids)):
        seq=[tok.cls_token_id]+ids[start:start+510]+[tok.sep_token_id];x=torch.tensor([seq],device=device)
        pred=model(input_ids=x,attention_mask=torch.ones_like(x)).logits.float().softmax(-1)[0,1:-1,1].cpu().numpy()
        sums[start:start+len(pred)]+=pred;n[start:start+len(pred)]+=1
    probs=sums/np.maximum(n,1);tokens=[]
    for (start,end),score in zip(offsets,probs):
        if not text[start:end].strip():continue
        tokens.append({'start':start,'end':end,'text':text[start:end],'ai_probability':float(score),'label':int(score>=threshold)})
    Path(args.output).write_text(json.dumps({'token_threshold':threshold,'offset_unit':'unicode_code_points','offset_convention':'half_open','tokens':tokens},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
