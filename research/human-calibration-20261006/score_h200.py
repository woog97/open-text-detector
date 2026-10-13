"""Score the calibration papers on the H200 with the Atlas scoring rules (one model per GPU).

Same inference as /data/workspace/classifications/classify_clean_papers.py (copied here): 510-token windows,
stride 510, LoRA merged into BF16 weights, torch.compile, Repeat2 layout padded to 1024. Sentence scores follow
sentence_scores_space.py (mean AI probability of non-whitespace tokens in each regex sentence).
Usage: score_h200.py MODEL_DIR BASE_DIR CLEAN_PARQUET_DIR OUT_DIR
  MODEL_DIR holds run.json (its config) and adapters.safetensors (the checkpoint to score).
Writes OUT_DIR/sentences.npz (id/starts|ends|scores), OUT_DIR/documents.parquet, OUT_DIR/status.json.
"""
import json, sys, time
from pathlib import Path
CODE = Path('/workspace/woog/pangram/backbones-20261003')
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, torch, pyarrow as pa, pyarrow.parquet as pq
from safetensors.torch import load_file
from transformers import AutoTokenizer
from modeling import Detector
from adapters_short import attach_lora
from classify_clean_papers import score
from sentence_scores_space import sentence_scores


def main(model_dir, base, clean, out):
    model_dir, out = Path(model_dir), Path(out); out.mkdir(parents=True, exist_ok=True)
    say = lambda **k: (out / 'status.json').write_text(json.dumps({'t': time.time(), **k}))
    cfg = json.loads((model_dir / 'run.json').read_text())['config']
    model = Detector.load_base({'repo': base, 'revision': None}, cfg['kind']); attach_lora(model, cfg)
    w = load_file(str(model_dir / 'adapters.safetensors'))
    assert set(w) == {n for n, p in model.named_parameters() if p.requires_grad}, 'adapter keys differ from model'
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in w: p.copy_(w[n])
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = model.to(dtype=torch.bfloat16, device='cuda').eval(); model.physical_microbatch = 16
    merged = 0
    for m in model.backbone.modules():
        if hasattr(m, 'merge') and hasattr(m, 'lora_A'): m.merge(); merged += 1
    assert merged, 'no LoRA modules merged'
    model.backbone = torch.compile(model.backbone, dynamic=False)
    rows = sorted((r for f in sorted(Path(clean).glob('train-*.parquet'))
                   for r in pq.read_table(f, columns=['id', 'text', 'text_sha256', 'pdf_sha256']).to_pylist()), key=lambda r: r['id'])
    packed, docs, t0 = {}, [], time.time()
    for i in range(0, len(rows), 50):
        for r, off, pr, dp, win in score(rows[i:i + 50], model, tok, 32, 510):
            s, e, sc = sentence_scores(r['text'], off, pr)
            packed[r['id'] + '/starts'] = s; packed[r['id'] + '/ends'] = e; packed[r['id'] + '/scores'] = sc
            docs.append({'id': r['id'], 'tokens': len(pr), 'windows': len(win), 'document_prob': dp,
                         'token_prob_mean': float(pr.mean()) if len(pr) else None, 'sentences': len(sc)})
        say(state='scoring', done=len(docs), total=len(rows), seconds=time.time() - t0, merged=merged)
    np.savez_compressed(out / 'sentences.npz', **packed)
    pq.write_table(pa.Table.from_pylist(docs), out / 'documents.parquet')
    say(state='done', done=len(docs), total=len(rows), seconds=time.time() - t0, merged=merged)


if __name__ == '__main__':
    main(*sys.argv[1:5])
