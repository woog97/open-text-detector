"""Space job: unit splits and embedding similarities for the soft n-gram labeler test.

For every source/edited pair it builds the units of T and S for each splitter:
  luna      GPT-6 Luna clause splits (computed beforehand, passed in)
  spacy     dependency-parse clause splitter (split_spacy.cuts_for) with en_core_web_trf
  sentence  spaCy sentence segmentation of the same parse (a neutral sentence splitter)
  paragraph the whole text as one unit (a Pangram-3-like scalar comparison)
and, for each T unit, E = max cosine similarity to any S unit or adjacent pair of S units,
using Qwen3-Embedding-0.6B (last-token pooling, query instruction on the T side).
CPU by default (--device cuda for a GPU). Writes OUT/units.jsonl and OUT/status.json.

Usage (on the Space, inside a venv that has spaCy and en_core_web_trf):
  space_units.py PAIRS.jsonl LUNA_SPLITS.jsonl OUT_DIR --model-cache /data/workspace/model-cache
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_spacy import cuts_for

EMBED = 'Qwen/Qwen3-Embedding-0.6B'
INSTRUCT = 'Instruct: Given a sentence from an edited paragraph, retrieve the passage of the original paragraph that expresses the same idea\nQuery:'


def trim(text, a, b):
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return (a, b)


def spans_from_cuts(text, cuts):
    bounds = [0] + list(cuts) + [len(text)]
    return [s for s in (trim(text, bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)) if s[1] > s[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pairs'); ap.add_argument('luna'); ap.add_argument('out', type=Path)
    ap.add_argument('--model-cache', required=True); ap.add_argument('--threads', type=int, default=32)
    ap.add_argument('--embed-model', default=EMBED); ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    status = lambda **k: (a.out / 'status.json').write_text(json.dumps(dict(k, at=time.strftime('%Y-%m-%dT%H:%M:%S'))))
    status(stage='load')
    pairs = [x for x in map(json.loads, open(a.pairs)) if x['state'] == 'ok']
    luna = {r['item_id']: [tuple(s) for s in r['spans']] for r in map(json.loads, open(a.luna)) if r['state'] == 'ok'}

    import spacy, torch
    torch.set_num_threads(a.threads)
    nlp = spacy.load('en_core_web_trf')
    units = {}  # (pid, side) -> {splitter: [(a, b)]}
    texts = [(p['passage_id'], side, p['source'] if side == 'S' else p['target']) for p in pairs for side in 'ST']
    status(stage='parse', n=len(texts))
    for (pid, side, t), doc in zip(texts, nlp.pipe((t for _, _, t in texts), batch_size=16)):
        u = {'spacy': spans_from_cuts(t, cuts_for(doc, t)),
             'sentence': [trim(t, s.start_char, s.end_char) for s in doc.sents],
             'paragraph': [trim(t, 0, len(t))]}
        if f'{pid}/{side}' in luna:
            u['luna'] = luna[f'{pid}/{side}']
        units[pid, side] = u

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.embed_model, cache_dir=a.model_cache, padding_side='left')
    dtype = torch.float32 if a.device == 'cpu' else torch.bfloat16
    model = AutoModel.from_pretrained(a.embed_model, cache_dir=a.model_cache, dtype=dtype).to(a.device).eval()
    cache = {}

    def embed(strings):
        todo = sorted({s for s in strings if s not in cache}, key=len)
        for i in range(0, len(todo), 32):
            batch = todo[i:i + 32]
            enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(a.device)
            with torch.no_grad():
                h = model(**enc).last_hidden_state[:, -1]
            for s, v in zip(batch, torch.nn.functional.normalize(h, dim=-1)):
                cache[s] = v.float().cpu()
        return torch.stack([cache[s] for s in strings])

    out = open(a.out / 'units.jsonl', 'w')
    for k, p in enumerate(pairs):
        S, T = p['source'], p['target']
        for name in ('luna', 'spacy', 'sentence', 'paragraph'):
            us, ut = units[p['passage_id'], 'S'].get(name), units[p['passage_id'], 'T'].get(name)
            if us is None or ut is None:
                continue
            cands = [S[x:y] for x, y in us] + [S[us[j][0]:us[j + 1][1]] for j in range(len(us) - 1)]
            q = embed([INSTRUCT + T[x:y] for x, y in ut]); d = embed(cands)
            sims = (q @ d.T).max(dim=1).values.tolist()
            out.write(json.dumps({'passage_id': p['passage_id'], 'splitter': name, 'source_units': us,
                                  'target_units': ut, 'E': [round(s, 5) for s in sims]}) + '\n')
        out.flush()
        if k % 10 == 0:
            status(stage='embed', done=k, n=len(pairs), cached=len(cache))
    out.close()
    status(stage='done', n=len(pairs), embed_model=a.embed_model, device=a.device)


if __name__ == '__main__':
    main()
