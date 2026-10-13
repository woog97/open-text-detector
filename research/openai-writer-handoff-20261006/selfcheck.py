"""Self-check one output file against its batch file before handing it in.

Usage (from the repository root):
  uv run -q --with requests python research/openai-writer-handoff-20261006/selfcheck.py d gpt-6-sol-007
  uv run -q --with requests python research/openai-writer-handoff-20261006/selfcheck.py e gpt-6.1-sol-003
  uv run -q --with requests python research/openai-writer-handoff-20261006/selfcheck.py d gpt-6-sol-r2-004   (round 2)

It calls the exact validation functions ingest uses (build_hosted.check_d / check_e: named-sentence placement,
byte-identical untouched text, sentence counts, overlap limits, never-train exclusion), and additionally requires
exactly one output line per input id, in input order. Passing selfcheck therefore implies passing ingest.
Checks <name>.jsonl.tmp when present (the in-progress write), otherwise <name>.jsonl.
Exit code 0 only if there are no problems.
"""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'llm-sentence-edits-20261005'))
sys.path.insert(0, str(HERE.parents[0] / 'claude-hosted-edits-20261006'))
import build_hosted as bh  # noqa: E402

job, name = sys.argv[1], sys.argv[2].removesuffix('.jsonl')
model = re.sub(r'(-r\d+)?-\d+$', '', name)  # gpt-6-sol-007 and gpt-6-sol-r2-007 -> gpt-6-sol
key = 'edited_paragraph' if job == 'd' else 'paragraph'
inp = [json.loads(l) for l in open(HERE / 'batches' / job / f'{name}.jsonl')]
items = bh.load_items(job)
outp = HERE / 'outputs' / job / model / f'{name}.jsonl'
if (outp.parent / f'{name}.jsonl.tmp').exists():  # writers check the .tmp file before renaming it into place
    outp = outp.parent / f'{name}.jsonl.tmp'
if not outp.exists():
    sys.exit(f'missing output file {outp}')
lines, problems = [], []
for k, l in enumerate(open(outp), 1):
    if not l.strip():
        continue
    try:
        x = json.loads(l)
    except ValueError:
        problems.append(f'line {k}: not valid JSON'); continue
    if set(x) != {'id', key}:
        problems.append(f'line {k}: keys must be exactly id and {key}'); continue
    lines.append(x)
ids = [x['id'] for x in lines]
if ids != [r['id'] for r in inp]:
    problems.append('output ids must match the batch ids one-to-one, in input order')
got = {}
for x in lines:
    got.setdefault(x['id'], x[key])
for r in inp:
    if r['id'] not in got:
        problems.append(f"{r['id']}: no output line"); continue
    h = items[r['id']]
    if h['writer'] != model:
        problems.append(f"{r['id']}: item belongs to writer {h['writer']}, not {model}"); continue
    _, reason = (bh.check_d if job == 'd' else bh.check_e)(h, got[r['id']])
    if reason:
        problems.append(f"{r['id']} ({h.get('edit_type') or h.get('kind')}): {reason}")
for p in problems:
    print(p)
print(f'{name}: {len(inp)} inputs, {len(lines)} output lines, {len(problems)} problems')
sys.exit(1 if problems else 0)
