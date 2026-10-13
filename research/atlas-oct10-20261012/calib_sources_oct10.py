"""Score sources for the Oct 10 Calibration page that were only ever scored with the Oct 7 model (Oct 12 2026).

Writes, under OUT, score_moe.load_rows source folders (data/train-00000.parquet with id, text, text_sha256, pdf_sha256):
- suite-manuscripts: the evaluation suite's 120 AI manuscripts (package/manuscripts.jsonl.gz), the positives of the
  page's document curves. They have no PDF, so pdf_sha256 = text_sha256.
The year-diagnostic texts (2023-2026 archive and OpenReview papers, 7,102 rows) are copied unchanged from the Oct 7 work
folder (yd-text/data/train-00000.parquet) by the caller. Both are then scored on the H200 with score_atlas_fast.py
(abl-noseg-nomix, stage2-epoch0), the scorer of the Oct 10 Atlas.
Usage: calib_sources_oct10.py SUITE_PACKAGE_DIR OUT
"""
import gzip, json, sys
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq

pkg, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = [json.loads(l) for l in gzip.open(pkg / 'manuscripts.jsonl.gz', 'rt')]
d = out / 'suite-manuscripts' / 'data'; d.mkdir(parents=True, exist_ok=True)
pq.write_table(pa.Table.from_pylist([{'id': r['id'], 'text': r['text'], 'text_sha256': r['text_sha256'], 'pdf_sha256': r['text_sha256']} for r in rows]), d / 'train-00000.parquet')
print(len(rows), 'manuscripts')
