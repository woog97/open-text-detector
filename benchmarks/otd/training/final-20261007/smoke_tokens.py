"""Count rows of a prepared-v2 mix that exceed the 510-token window under each final backbone's tokenizer.
  python smoke_tokens.py PREPARED_DIR [ASSETS_DIR]   (prints JSON counts per key and tokenizer)"""
import gzip, json, sys
from pathlib import Path
from transformers import AutoTokenizer
P = Path(sys.argv[1]); A = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parent / 'assets')
out = {}
for name in ['qwen35-4b', 'qwen36-35b-a3b', 'embeddinggemma-2']:
    tok = AutoTokenizer.from_pretrained(A / name, local_files_only=True); out[name] = {}
    for f in sorted(P.glob('*.jsonl.gz')):
        rows = [json.loads(l) for l in gzip.open(f, 'rt')]
        n = [len(x) for x in tok([r['text'] for r in rows], add_special_tokens=False)['input_ids']]
        over = [(r.get('supervision') == 'document_only', k) for r, k in zip(rows, n) if k > 510]
        out[name][f.name] = {'rows': len(rows), 'over_510': sum(1 for d, _ in over if not d), 'doc_only_over_510': sum(1 for d, _ in over if d), 'max': max(n) if n else 0}
print(json.dumps(out, indent=1))
