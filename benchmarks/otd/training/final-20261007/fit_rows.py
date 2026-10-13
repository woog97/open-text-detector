"""Make a prepared-v2 mix fit the 510-token window under every final backbone's tokenizer (Qwen3.5 4B, Qwen3.6 MoE, EmbeddingGemma 2).
Gemma's tokenizer gives ~4% of T2 stage-2 rows and ~11% of stage-1 rows more than 510 tokens although they fit Qwen's.
Rows over the limit for any tokenizer are cropped the way build_t21.crop510 does it: a <=500-token window (by the longest
tokenization) around the AI span (or the target), snapped to whitespace, regions and target offsets shifted. Document-only rows
are left alone (the trainer chunks them). Writes OUT with the same file names, an updated manifest (sha256), and fit-report.json.
  python fit_rows.py PREPARED_DIR OUT_DIR [ASSETS_DIR]"""
import gzip, hashlib, json, random, shutil, sys
from pathlib import Path
from transformers import AutoTokenizer
P, OUT = Path(sys.argv[1]), Path(sys.argv[2]); A = Path(sys.argv[3] if len(sys.argv) > 3 else Path(__file__).resolve().parent / 'assets')
toks = [AutoTokenizer.from_pretrained(A / n, local_files_only=True) for n in ['qwen35-4b', 'qwen36-35b-a3b', 'embeddinggemma-2']]
rng = random.Random(7)

def offsets(t):
    return max((tk(t, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping'] for tk in toks), key=len)

def crop(r, limit):
    t = r['text']; off = offsets(t)
    if len(off) <= 510: return r
    ai = [g for g in r['regions'] if g['label'] == 1]
    a0 = min(g['start'] for g in ai) if ai else int(r.get('target_start', 0)); a1 = max(g['end'] for g in ai) if ai else int(r.get('target_end', len(t)))
    if not ai and a1 - a0 >= len(t) * .9: a0 = a1 = rng.randrange(len(t))
    i0 = next((i for i, (a, b) in enumerate(off) if b > a0), 0); i1 = next((i for i, (a, b) in enumerate(off) if b >= a1), len(off) - 1)
    span = i1 - i0 + 1
    st = i0 if span >= limit else max(0, min(len(off) - limit, i0 - rng.randrange(limit - span + 1)))
    cs, ce = off[st][0], off[min(len(off), st + limit) - 1][1]
    while 0 < cs < len(t) and not t[cs - 1].isspace(): cs += 1
    while ce < len(t) and ce > cs and not t[ce - 1].isspace() and not t[ce].isspace(): ce -= 1
    txt = t[cs:ce]
    rg = [{'start': max(g['start'], cs) - cs, 'end': min(g['end'], ce) - cs, 'label': g['label']} for g in r['regions'] if min(g['end'], ce) > max(g['start'], cs)]
    return {**r, 'id': f"{r['id']}@{cs}", 'text': txt, 'regions': rg, 'source_start': r.get('source_start', 0) + cs, 'source_end': r.get('source_start', 0) + ce,
            'target_start': max(0, min(len(txt), int(r.get('target_start', 0)) - cs)), 'target_end': max(0, min(len(txt), int(r.get('target_end', len(t))) - cs))}

OUT.mkdir(parents=True, exist_ok=True); man = json.loads((P / 'manifest.json').read_text()); report = {}
for f in sorted(P.glob('*.jsonl.gz')):
    key = f.name[:-len('.jsonl.gz')]; rows = [json.loads(l) for l in gzip.open(f, 'rt')]; n = 0; out = []
    # Batched lengths first (fast tokenizers batch in Rust); only rows over the window get per-row offsets and cropping.
    longest = [0] * len(rows)
    for tk in toks:
        for i in range(0, len(rows), 2000):
            for j, ids in enumerate(tk([r['text'] for r in rows[i:i + 2000]], add_special_tokens=False)['input_ids']):
                longest[i + j] = max(longest[i + j], len(ids))
    for r, k in zip(rows, longest):
        if r.get('supervision') != 'document_only' and k > 510:
            for limit in (500, 470, 440, 400):
                c = crop(r, limit)
                if len(offsets(c['text'])) <= 510: break
            assert len(offsets(c['text'])) <= 510 and c['text'].strip(), r['id']
            n += c is not r; r = c
        out.append(r)
    b = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out).encode(); (OUT / f.name).write_bytes(gzip.compress(b))
    if key in man['files']: man['files'][key] = {**man['files'][key], 'rows': len(out), 'sha256': hashlib.sha256(b).hexdigest()}
    report[key] = {'rows': len(out), 'cropped': n}
(OUT / 'manifest.json').write_text(json.dumps(man, indent=1)); (OUT / 'fit-report.json').write_text(json.dumps(report, indent=1))
for extra in P.iterdir():
    if extra.is_file() and not extra.name.endswith('.jsonl.gz') and extra.name != 'manifest.json' and not (OUT / extra.name).exists(): shutil.copy(extra, OUT / extra.name)
print(json.dumps(report))
