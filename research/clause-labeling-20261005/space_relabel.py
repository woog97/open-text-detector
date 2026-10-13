"""Space job: spaCy clause units + embedding similarity E for the AI regions of existing pairs.

Input rows (from build_relabel_inputs.py, compacted): {id, t_par_text, t_off, region_rel [a, b], s_par_text}.
For each row: clause units of the T paragraph(s) (split_spacy.cuts_for, en_core_web_trf), clipped
to the AI region; candidates are the clause units of the S paragraph(s) and adjacent pairs; E is
each T unit's max cosine to a candidate (Qwen3-Embedding-0.6B, query instruction on the T side).
The lexical score and the labels are computed afterwards, so thresholds can change without rerunning.
Writes OUT/units.jsonl ({id, t_units (absolute offsets in T), E}) and OUT/status.json; resumable.

Usage: space_relabel.py ROWS.jsonl OUT_DIR --model-cache DIR [--device cuda] [--threads 32]
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_spacy import cuts_for
from space_units import INSTRUCT, spans_from_cuts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rows'); ap.add_argument('out', type=Path)
    ap.add_argument('--model-cache', required=True); ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=32); ap.add_argument('--embed-model', default='Qwen/Qwen3-Embedding-0.6B')
    ap.add_argument('--chunk', type=int, default=200)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    out_path = a.out / 'units.jsonl'
    done = {json.loads(l)['id'] for l in open(out_path)} if out_path.exists() else set()
    rows = [r for r in map(json.loads, open(a.rows)) if r['id'] not in done]
    t0 = time.time()
    status = lambda **k: (a.out / 'status.json').write_text(json.dumps(dict(k, at=time.strftime('%Y-%m-%dT%H:%M:%S'), elapsed_s=round(time.time() - t0))))
    status(stage='load', pending=len(rows), done=len(done))

    import spacy, torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(a.threads)
    if a.device != 'cpu':
        spacy.prefer_gpu()  # GPU parsing needs cupy; without it spaCy stays on CPU and only the embedding model uses the GPU
    nlp = spacy.load('en_core_web_trf')
    tok = AutoTokenizer.from_pretrained(a.embed_model, cache_dir=a.model_cache, padding_side='left')
    dtype = torch.float32 if a.device == 'cpu' else torch.bfloat16
    model = AutoModel.from_pretrained(a.embed_model, cache_dir=a.model_cache, dtype=dtype).to(a.device).eval()

    def embed(strings):
        uniq = sorted(set(strings), key=len); vec = {}
        for i in range(0, len(uniq), 64):
            batch = uniq[i:i + 64]
            enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(a.device)
            with torch.no_grad():
                h = model(**enc).last_hidden_state[:, -1]
            for s, v in zip(batch, torch.nn.functional.normalize(h.float(), dim=-1).cpu()):
                vec[s] = v
        return vec

    n_done = len(done)
    with open(out_path, 'a') as out:
        for c in range(0, len(rows), a.chunk):
            chunk = rows[c:c + a.chunk]
            texts = [t for r in chunk for t in (r['t_par_text'], r['s_par_text'])]
            docs = list(nlp.pipe(texts, batch_size=32))
            plans = []
            for k, r in enumerate(chunk):
                tp, sp = r['t_par_text'], r['s_par_text']
                ra, rb = r['region_rel']
                tu = [(max(x, ra), min(y, rb)) for x, y in spans_from_cuts(tp, cuts_for(docs[2 * k], tp)) if min(y, rb) > max(x, ra)]
                tu = [(x, y) for x, y in tu if tp[x:y].strip()]
                su = spans_from_cuts(sp, cuts_for(docs[2 * k + 1], sp))
                cands = [sp[x:y] for x, y in su] + [sp[su[j][0]:su[j + 1][1]] for j in range(len(su) - 1)]
                plans.append((r, tu, cands))
            vec = embed([INSTRUCT + r['t_par_text'][x:y] for r, tu, _ in plans for x, y in tu] + [s for _, _, cs in plans for s in cs])
            for r, tu, cands in plans:
                if tu and cands:
                    q = torch.stack([vec[INSTRUCT + r['t_par_text'][x:y]] for x, y in tu]); d = torch.stack([vec[s] for s in cands])
                    E = (q @ d.T).max(dim=1).values.tolist()
                else:
                    E = [0.0] * len(tu)
                out.write(json.dumps({'id': r['id'], 't_units': [[x + r['t_off'], y + r['t_off']] for x, y in tu],
                                      'E': [round(e, 5) for e in E]}) + '\n')
            out.flush(); n_done += len(chunk)
            status(stage='run', done=n_done, total=n_done + len(rows) - c - len(chunk), rate_rows_per_s=round((c + len(chunk)) / (time.time() - t0), 2))
    status(stage='done', done=n_done, device=a.device, embed_model=a.embed_model)


if __name__ == '__main__':
    main()
