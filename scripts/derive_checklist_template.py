"""Derive hashed NeurIPS checklist template fingerprints for text_cleanup.

Template lines are lines that recur in many checklist papers. Only truncated
hashes are written, so no checklist or paper text enters the repository.
Usage: derive_checklist_template.py 'path/to/paper-text/data/*.parquet'
"""
from collections import Counter
import glob
import json
from pathlib import Path
import re
import sys

import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from open_text_detector_backend.text_cleanup import CHECKLIST_FILE, _norm_line, _shingles, _hash

HEADING = re.compile(r'(?m)^\s*NeurIPS Paper Checklist\s*$')
KEEP = re.compile(r'(Answer|Justification)\b')   # author-written responses stay


def main():
    texts = []; others = Counter(); n_others = 0
    for f in sorted(glob.glob(sys.argv[1])):
        t = pq.read_table(f, columns=['conference', 'text'])
        for c, x in zip(t['conference'].to_pylist(), t['text'].to_pylist()):
            if c == 'neurips': texts.append(x)
            else: others.update({_norm_line(l) for l in x.split('\n')}); n_others += 1
    # Pass 1: long template lines, from the tail of papers with an explicit heading.
    df = Counter()
    for text in texts:
        m = HEADING.search(text)
        if m: df.update({_norm_line(l) for l in text[m.start():].split('\n')})
    # Lines that also occur in other venues' papers (e.g. "This completes the
    # proof.") are ordinary academic prose, not checklist template text.
    long_lines = [l for l, n in df.items() if n >= 5 and len(_shingles(l)) >= 2 and others[l] < 2 and not KEEP.match(l)]
    shingles = set()
    for l in long_lines: shingles.update(_shingles(l))
    # Pass 2: short template lines (item titles, "Guidelines:") that recur in
    # checklist papers but almost never in papers without a checklist.
    def gated(lines):
        return sum(_template_long(l, shingles) for l in lines) >= 20
    inside, outside = Counter(), Counter(); n_in = n_out = 0
    for text in texts:
        lines = {_norm_line(l) for l in text.split('\n')}
        if gated(lines): inside.update(lines); n_in += 1
        else: outside.update(lines); n_out += 1
    # Short lines are only removed inside gated papers, so they need to be
    # characteristic of checklist papers (>=20x more common) rather than unique.
    elsewhere = lambda l: (outside[l] + others[l]) / max(1, n_out + n_others)
    short = sorted(_hash(l) for l, n in inside.items()
                   if n >= 30 and n / n_in >= 20 * elsewhere(l) and not KEEP.match(l)
                   and len(re.findall(r'[A-Za-z]{2,}', l)) >= 1 and l not in long_lines)
    Path(CHECKLIST_FILE).write_text(json.dumps({
        'description': 'Truncated SHA-256 fingerprints of NeurIPS paper-checklist template text (word 4-gram shingles and short exact lines). Derived by scripts/derive_checklist_template.py.',
        'shingles': sorted(shingles), 'short_lines': short}, indent=0) + '\n')
    print(f'{len(texts)} NeurIPS papers, {n_in} gated, {len(long_lines)} template lines, '
          f'{len(shingles)} shingles, {len(short)} short lines')


def _template_long(line, shingles):
    s = _shingles(line)
    return bool(s) and sum(h in shingles for h in s) >= .8 * len(s)


if __name__ == '__main__': main()
