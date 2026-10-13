"""Score calibration windows with chosen sweep checkpoints, for honest (held-out) operating points.

Usage: calib_score.py TAG CHECKPOINT [CHECKPOINT ...]   e.g. calib_score.py q4b-A-s1 stage2-epoch2
Writes sweeps/<tag>/eval/<checkpoint>-calibration.npz with sentence scores and labels.
Papers sharing any sentence with the evaluation suite are excluded (EXCLUDE_PAPERS).
"""
import gzip, json, sys, time
import numpy as np
import torch
from safetensors.torch import load_file
import sweep_eval_run as E

EXCLUDE_PAPERS = set(json.loads(sys.argv[-1])) if sys.argv[-1].startswith('[') else set()


def main(tag, checkpoints):
    d = E.S / tag; run = json.loads((d / 'run.json').read_text())
    model_name = run['model']['name']
    rows = [json.loads(l) for l in gzip.open(E.ROOT / 'runs' / model_name / 'prepared-v2' / 'calibration-windows.jsonl.gz', 'rt')]
    rows = [r for r in rows if r['paper_id'] not in EXCLUDE_PAPERS]
    model, tok = E.load_model(run)
    for ck in checkpoints:
        out = d / 'eval' / f'{ck}-calibration.npz'
        if out.exists():
            continue
        w = load_file(str(d / f'{ck}-adapters.safetensors'))
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in w:
                    p.copy_(w[n].to(p.dtype))
        t0 = time.time(); preds = E.predict(rows, model, tok)
        scores, labels, kinds = [], [], []
        for r, (off, p) in zip(rows, preds):
            for s, l in E.sentence_rows(r, off, p):
                if l in (0, 1):
                    scores.append(s); labels.append(l); kinds.append(r['kind'])
        np.savez_compressed(out, scores=np.asarray(scores, np.float32), labels=np.asarray(labels, np.int8), kinds=np.asarray(kinds))
        print(json.dumps({'tag': tag, 'checkpoint': ck, 'rows': len(rows), 'sentences': len(scores), 'human': int(np.sum(np.asarray(labels) == 0)), 'seconds': round(time.time() - t0)}), flush=True)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('[')]
    main(args[0], args[1:])
