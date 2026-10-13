"""Measure complete-paper Laya throughput and repeated encoder work."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from open_text_detector_backend.providers.laya import Laya

text = (ROOT / 'research/extractions/hyperdas/original-reviewbench-ocr.txt').read_text()
p = Laya(model_dir=ROOT / 'models', runtime='mlx', precision='float16', batch_size=4)
t0 = time.perf_counter()
p._load()
load_seconds = time.perf_counter() - t0
t0 = time.perf_counter()
rows = p.prepare(text)
prepare_seconds = time.perf_counter() - t0
source_tokens = len(p.tokenizer(text, add_special_tokens=False)['input_ids'])
input_tokens = sum(len(r['ids']) for r in rows)
print(json.dumps({'words': len(text.split()), 'source_tokens': source_tokens, 'phrases': len(rows),
                  'input_tokens': input_tokens, 'repetition_factor': input_tokens/source_tokens,
                  'prepare_seconds': prepare_seconds}), flush=True)
p.score_rows(rows[:4])
t0 = time.perf_counter()
p.score_rows(rows)
inference_seconds = time.perf_counter() - t0
# Dense projections alone; excludes attention products, activations, normalization and head.
dense_encoder_flops = input_tokens * 28 * 2 * (4 * 1024**2 + 3 * 1024 * 2624)
report = {'paper': 'HyperDAS full extraction, including references and appendix',
          'words': len(text.split()), 'source_tokens': source_tokens, 'phrases': len(rows),
          'model_input_tokens': input_tokens, 'repetition_factor': input_tokens/source_tokens,
          'question_and_delimiter_tokens_per_phrase': len(p.prefix)+sum(map(len,p.labels))+1,
          'context_tokens': sum(r['context_tokens'] for r in rows),
          'load_seconds': load_seconds, 'prepare_seconds': prepare_seconds,
          'inference_seconds': inference_seconds,
          'warm_papers_per_minute': 60/(prepare_seconds+inference_seconds),
          'required_speedup_for_ten': (prepare_seconds+inference_seconds)/6,
          'dense_encoder_teraflops_per_paper': dense_encoder_flops/1e12,
          'dense_encoder_effective_tflops_per_second': dense_encoder_flops/inference_seconds/1e12,
          'batch_size': 4, 'runtime': 'mlx', 'precision': 'float16_encoder_float32_head'}
(ROOT/'research/benchmarks/laya/full-paper-diagnostic.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
