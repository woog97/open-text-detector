"""Paired comparison of the compiled activation and the workspace's custom Metal GEGLU."""
import gc
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from open_text_detector_backend.providers.laya import Laya
from open_text_detector_backend.providers.laya_mlx import LayaMLX

p = Laya(model_dir=ROOT / 'models', runtime='mlx', precision='float16', batch_size=8)
p._load()
rows = p.prepare((ROOT / 'research/extractions/hyperdas/original-reviewbench-ocr.txt').read_text())[:48]
reference = np.array(p.score_rows(rows))
variants = {}
for activation in ['compiled', 'metal']:
    del p.model
    gc.collect()
    mx.clear_cache()
    p.model = LayaMLX(ROOT / 'models/laya', 'float16', activation=activation)
    scores = np.array(p.score_rows(rows))
    error = float(np.max(np.abs(scores - reference)))
    if error > 1e-4:
        variants[activation] = {'rejected': True, 'max_score_error': error,
                                'reason': 'Exceeds 0.0001 activation-only equivalence tolerance'}
        print(json.dumps({activation: variants[activation]}), flush=True)
        continue
    times = []
    for _ in range(3):
        start = time.perf_counter()
        p.score_rows(rows)
        times.append(time.perf_counter() - start)
    variants[activation] = {'seconds': times, 'max_score_error': error,
                            'phrases_per_second': len(rows) / float(np.median(times))}
    print(json.dumps({activation: variants[activation]}), flush=True)
(ROOT / 'research/benchmarks/laya/kernels.json').write_text(json.dumps(variants, indent=2))
