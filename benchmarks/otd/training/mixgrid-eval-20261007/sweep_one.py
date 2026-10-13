"""Score one checkpoint of sweeps/<tag> on the paper-edit evaluation set (sweep_eval's scoring, without its prune step,
which deletes checkpoints). Usage: sweep_one.py TAG CHECKPOINT_NAME   (e.g. moe-mixB stage2-epoch2)"""
import gzip, json, sys, time
import numpy as np
import sweep_eval as se
tag, name = sys.argv[1:3]
d = se.S / tag; run = json.loads((d / 'run.json').read_text()); ev = d / 'eval'; ev.mkdir(exist_ok=True)
rows = [json.loads(l) for l in gzip.open(se.S / 'sweep-eval-rows.jsonl.gz', 'rt')]
model, tok = se.load_model(run); t0 = time.time()
scored = se.score_checkpoint(model, tok, d / f'{name}-adapters.safetensors', rows)
res = {'checkpoint': f'{name}-adapters.safetensors', 'tag': tag, 'seconds': time.time() - t0, **{s: se.metrics(scored, s) for s in ('dev', 'test', 'all')}}
np.savez_compressed(ev / f'{name}-sentences.npz', **{f'{i}': np.asarray(x['sents'], dtype=np.float32).reshape(-1, 2) for i, x in enumerate(scored)},
                    doc=np.asarray([x['doc'] for x in scored], np.float32))
(ev / f'{name}.json').write_text(json.dumps(res, indent=1))  # written last: its presence means the npz is complete
print(json.dumps({'checkpoint': name, 'seconds': round(res['seconds']), 'test_small_r1': res['test']['edits_small']['recall_at_1pct']}), flush=True)
