"""Fast sweep evaluation: score every saved checkpoint of sweeps/<tag> on sweeps/sweep-eval-rows.jsonl.gz.

Usage: sweep_eval.py TAG [--watch]   (--watch keeps polling for new step checkpoints until the run finishes)
Writes sweeps/<tag>/eval/<checkpoint>.json (metrics for dev, test, all) and <checkpoint>-sentences.npz.
BF16 forwards; scoring rules match the suite: sentence score = mean token probability of its
non-whitespace tokens, label kept only when all those tokens share one gold label.
"""
from runtime import ROOT, require_space
import argparse, gzip, json, math, re, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from safetensors.torch import load_file
from data import layout
from modeling import Detector, collate
from adapters_short import attach_lora

S = ROOT / 'sweeps'


def sentences(text):
    return [(m.start(), m.end()) for m in re.finditer(r'\S.*?(?:[.!?](?=\s|$)|$)', text, re.S) if m.group().strip()]


def token_labels(text, offsets, regions):
    out = []
    for a, b in offsets:
        act = {r['label'] for r in regions if min(b, r['end']) > max(a, r['start']) and text[max(a, r['start']):min(b, r['end'])].strip()}
        out.append(next(iter(act)) if len(act) == 1 and next(iter(act)) in (0, 1) else -100)
    return np.asarray(out)


def starts(n, width=510, stride=256):
    return [0] if n <= width else sorted(set(list(range(0, n - width + 1, stride)) + [n - width]))


@torch.inference_mode()
def predict(rows, model, tok, batch_size=8):
    enc = tok([r['text'] for r in rows], add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    sums = [np.zeros(len(x)) for x in enc['input_ids']]; den = [np.zeros(len(x)) for x in enc['input_ids']]; jobs = []
    for i, ids in enumerate(enc['input_ids']):
        for s in starts(len(ids)):
            jobs.append((i, s, min(s + 510, len(ids))))
    jobs.sort(key=lambda j: j[2] - j[1])
    for pos in range(0, len(jobs), batch_size):
        part = jobs[pos:pos + batch_size]; ex = []
        for i, a, b in part:
            e = layout(enc['input_ids'][i][a:b], [-100] * (b - a), model.kind, 2, tok.cls_token_id, tok.sep_token_id)
            e.update(source_labels=[-100] * (b - a), target=[False] * (b - a), segment_label=-100, mixed_label=-100, sentence_groups=[],
                     document_label=-100, document_only=False,
                     document_mask=[bool(rows[i]['text'][c:d].strip()) for c, d in enc['offset_mapping'][i][a:b]])
            ex.append(e)
        batch = collate(ex, tok.pad_token_id)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(batch)['tokens']
        probs = logits.float().softmax(-1)[..., 1].cpu().numpy()
        for (i, a, b), p in zip(part, probs):
            sums[i][a:b] += p[:b - a]; den[i][a:b] += 1
    return [(np.asarray(off), s / d) for off, s, d in zip(enc['offset_mapping'], sums, den)]


def sentence_rows(r, off, prob):
    text = r['text']; valid = np.array([bool(text[a:b].strip()) for a, b in off])
    ys = token_labels(text, off, r['regions'] or [])
    out = []
    for a, b in sentences(text):
        use = valid & (off[:, 1] > a) & (off[:, 0] < b)
        if use.any():
            labs = set(ys[use].tolist())
            out.append((float(prob[use].mean()), next(iter(labs)) if len(labs) == 1 and next(iter(labs)) in (0, 1) else -1))
    return out


def auroc(pos, neg):
    if len(pos) < 5 or len(neg) < 5:
        return None
    s = np.r_[pos, neg]; y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    o = np.argsort(s, kind='mergesort'); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    ss = s[o]; b = np.r_[True, ss[1:] != ss[:-1]]; g = np.cumsum(b) - 1
    r[o] = (np.bincount(g, r[o]) / np.bincount(g))[g]
    return float((r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def recall_at(pos, neg, fpr=.01):
    if len(pos) < 5 or len(neg) < 20:
        return None
    t = np.sort(neg)[::-1][int(math.floor(fpr * len(neg)))]
    return float(np.mean(np.asarray(pos) > t))


def metrics(scored, split):
    rs = [x for x in scored if split == 'all' or x['split'] == split]
    neg_sent = [s for x in rs if x['slot'] in ('edit_context', 'human_untouched', 'human_paper') for s, l in x['sents'] if l == 0]
    out = {'n_rows': len(rs), 'n_human_sentences': len(neg_sent)}
    sizes = {'one': ('sentence',), 'two': ('two_sentence',), 'small': ('sentence', 'two_sentence'), 'para': ('paragraph_concise', 'paragraph_v3'),
             'all': ('sentence', 'two_sentence', 'paragraph_concise', 'paragraph_v3')}
    for k, conds in sizes.items():
        pos = [s for x in rs if x['slot'] == 'edit_context' and x['condition'] in conds for s, l in x['sents'] if l == 1]
        out[f'edits_{k}'] = {'n': len(pos), 'auroc': auroc(pos, neg_sent), 'recall_at_1pct': recall_at(pos, neg_sent),
                             'recall_at_05': float(np.mean(np.asarray(pos) >= .5)) if pos else None}
    out['human_sentence_fpr_at_05'] = float(np.mean(np.asarray(neg_sent) >= .5)) if neg_sent else None
    v3p = [s for x in rs if x['slot'] == 'v3' for s, l in x['sents'] if l == 1]; v3n = [s for x in rs if x['slot'] == 'v3' for s, l in x['sents'] if l == 0]
    out['v3_sentences'] = {'auroc': auroc(v3p, v3n), 'recall_at_1pct': recall_at(v3p, v3n)}
    hp = [x['doc'] for x in rs if x['slot'] in ('human_paper', 'human_untouched')]; sa = [x['doc'] for x in rs if x['slot'] == 'ai_standalone']
    out['standalone_rewrites'] = {'auroc': auroc(sa, hp), 'recall_at_1pct': recall_at(sa, hp)}
    pa = [x['doc'] for x in rs if x['slot'] == 'public' and x['label'] == 'ai']; ph = [x['doc'] for x in rs if x['slot'] == 'public' and x['label'] == 'human']
    out['public_docs'] = {'auroc': auroc(pa, ph), 'recall_at_1pct': recall_at(pa, ph)}
    return out


def load_model(run):
    cfg = run['config']
    model = Detector.load_base({'repo': run['assets'], 'revision': None}, cfg['kind']); attach_lora(model, cfg)
    tok = AutoTokenizer.from_pretrained(run['assets'], local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.physical_microbatch = 8
    return model.to(dtype=torch.bfloat16, device='cuda').eval(), tok


def score_checkpoint(model, tok, path, rows):
    w = load_file(str(path)); assert set(w) == {n for n, p in model.named_parameters() if p.requires_grad}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in w:
                p.copy_(w[n].to(p.dtype))
    preds = predict(rows, model, tok); scored = []
    for r, (off, p) in zip(rows, preds):
        valid = np.array([bool(r['text'][a:b].strip()) for a, b in off])
        scored.append({'slot': r['slot'], 'split': r['split'], 'label': r['label'], 'condition': r.get('condition'),
                       'doc': float(p[valid].mean()) if valid.any() else 0., 'sents': sentence_rows(r, off, p)})
    return scored


def main(tag, watch):
    require_space(); d = S / tag; run = json.loads((d / 'run.json').read_text()); ev = d / 'eval'; ev.mkdir(exist_ok=True)
    rows = [json.loads(l) for l in gzip.open(S / 'sweep-eval-rows.jsonl.gz', 'rt')]
    model, tok = load_model(run)
    status = None
    while True:
        todo = sorted(p for p in d.glob('*-adapters.safetensors') if not (ev / (p.name.split('-adapters')[0] + '.json')).exists() and not p.name.startswith('stage1'))
        for p in todo:
            name = p.name.split('-adapters')[0]; t0 = time.time()
            scored = score_checkpoint(model, tok, p, rows)
            res = {'checkpoint': p.name, 'tag': tag, 'seconds': time.time() - t0, **{s: metrics(scored, s) for s in ('dev', 'test', 'all')}}
            (ev / f'{name}.json').write_text(json.dumps(res, indent=1))
            np.savez_compressed(ev / f'{name}-sentences.npz', **{f'{i}': np.asarray(x['sents'], dtype=np.float32).reshape(-1, 2) for i, x in enumerate(scored)},
                                doc=np.asarray([x['doc'] for x in scored], np.float32))
            print(json.dumps({'checkpoint': name, 'seconds': round(res['seconds']), 'dev_edits_small_r1': res['dev']['edits_small']['recall_at_1pct'],
                              'dev_edits_all_auroc': res['dev']['edits_all']['auroc']}), flush=True)
        status = json.loads((d / 'status.json').read_text()).get('state') if (d / 'status.json').exists() else None
        if not watch or status in ('trained', 'failed'):
            if not watch or not todo:
                break
        time.sleep(60)
    if status == 'trained':
        prune(d, ev)


def prune(d, ev):
    """Finished and scored: keep only the final stage-2 checkpoint; drop the resume state and every other checkpoint."""
    ck = sorted(d.glob('*-adapters.safetensors'))
    final = sorted(p for p in ck if p.name.startswith('stage2-epoch'))[-1:]
    sel = d / 'stage2-selection.json'  # also keep the checkpoint chosen by validation loss, which is often not the last epoch
    if sel.exists():
        final += [p for p in ck if p.name == json.loads(sel.read_text()).get('checkpoint') and p not in final]
    scored = {p.stem for p in ev.glob('*.json')}
    removed = []
    for p in ck:
        name = p.name.split('-adapters')[0]
        if p in final or (not name.startswith('stage1') and name not in scored):
            continue
        removed.append(p.name); p.unlink()
    (d / 'resume-state.pt').unlink(missing_ok=True)
    (d / 'pruned.json').write_text(json.dumps({'kept': [p.name for p in final], 'removed': removed, 'time': time.time()}, indent=1))
    print(json.dumps({'pruned': d.name, 'kept': [p.name for p in final], 'removed': len(removed)}), flush=True)


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('tag'); a.add_argument('--watch', action='store_true'); args = a.parse_args()
    main(args.tag, args.watch)
