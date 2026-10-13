"""Cross-generator evaluation on open-text-detector/heterogeneous-ai-spans (Claude, GPT and Luna writers).

Usage: cross_model_eval.py TAG CHECKPOINT DATA_DIR [NAME]
  TAG         sweep run folder under sweeps/ (its run.json gives backbone and LoRA config)
  CHECKPOINT  adapters .safetensors file
  DATA_DIR    folder with the dataset's data/{mixed,human_controls}/{train,validation,test}.parquet, or a prepared rows .jsonl.gz
  NAME        output prefix (default cross-model), e.g. heldout-edits
  SAMPLE      optional: score a fixed writer-stratified subset of SAMPLE mixed documents plus as many human controls
              (e.g. 500 for routine runs; name the output e.g. cross-model-s500 so it is never mixed with full-set results)

Scores every sentence with the same rules as sweep_eval.py (BF16). Negatives are human sentences from the matched human
controls plus retained human text in mixed documents; positives are AI sentences, broken down by writer and cohort.
Two operating points: 1% FPR on this dataset's human sentences (best case), and the in-domain threshold fixed at 1% FPR
on the sweep-eval dev human sentences (what a deployed paper detector would use). Writes eval/cross-model-<ckpt>.json.
"""
import gzip, json, re, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
try:
    from sweep_eval import S, load_model, score_checkpoint, auroc, recall_at
except ModuleNotFoundError:  # deployed as sweep_eval_run.py on the H200
    from sweep_eval_run import S, load_model, score_checkpoint, auroc, recall_at

LAB = {'ai': 1, 'human': 0}


def load_rows(data_dir, sample=0):
    if str(data_dir).endswith('.jsonl.gz'):  # prepared rows: id, text, regions, slot (mixed/human_controls), split, condition, writer
        return [json.loads(l) for l in gzip.open(data_dir, 'rt')]
    rows = []
    for cfg in ['mixed', 'human_controls']:
        for split in ['train', 'validation', 'test']:
            for r in pq.read_table(Path(data_dir) / 'data' / cfg / f'{split}.parquet').to_pylist():
                regs = [{'start': s['start'], 'end': s['end'], 'label': LAB[s['label']]} for s in (r['spans'] or [])]
                if cfg == 'human_controls' and not regs:
                    regs = [{'start': 0, 'end': len(r['text']), 'label': 0}]
                writer = r.get('reported_model') or r.get('requested_model') or 'none'
                rows.append({'id': r['id'], 'text': r['text'], 'regions': regs, 'slot': cfg, 'split': split, 'label': cfg,
                             'condition': r.get('cohort'), 'writer': writer if cfg == 'mixed' else 'human'})
    if sample:  # fixed, writer-stratified subset for routine runs: same rows every time (hash order), equal human controls
        import hashlib
        hk = lambda r: hashlib.sha256(r['id'].encode()).hexdigest()
        mixed = [r for r in rows if r['slot'] == 'mixed']; ctrl = sorted((r for r in rows if r['slot'] == 'human_controls'), key=hk)
        writers = sorted({r['writer'] for r in mixed}); per = max(1, sample // len(writers)); keep = []
        for w in writers:
            keep += sorted((r for r in mixed if r['writer'] == w), key=hk)[:per]
        rows = keep + ctrl[:len(keep)]
    return rows


def in_domain_threshold(tag, ck_name):
    """Threshold at 1% FPR on sweep-eval dev human sentences, from the checkpoint's saved sweep-eval sentence scores."""
    rows = [json.loads(l) for l in gzip.open(S / 'sweep-eval-rows.jsonl.gz', 'rt')]
    z = np.load(S / tag / 'eval' / f'{ck_name}-sentences.npz'); sents = [z[str(i)] for i in range(len(rows))]
    neg = [s for r, x in zip(rows, sents) if r['split'] == 'dev' and r['slot'] in ('edit_context', 'human_untouched', 'human_paper')
           for s, l in x if l == 0]
    return float(np.sort(neg)[::-1][int(np.floor(.01 * len(neg)))]), len(neg)


def main(tag, ckpt, data_dir, name='cross-model', sample='0'):
    d = S / tag; run = json.loads((d / 'run.json').read_text())
    model, tok = load_model(run); rows = load_rows(data_dir, int(sample)); t0 = time.time()
    scored = score_checkpoint(model, tok, Path(ckpt), rows)
    m = re.search(r'(stage\d-epoch\d+|step\d+)', Path(ckpt).name); ck_name = m.group(1) if m else Path(ckpt).name.split('-adapters')[0]
    thr_in, n_in = in_domain_threshold(tag, ck_name) if (d / 'eval' / f'{ck_name}-sentences.npz').exists() else (None, 0)
    neg = [s for r, x in zip(rows, scored) for s, l in x['sents'] if l == 0]
    neg_ctrl = [s for r, x in zip(rows, scored) if r['slot'] == 'human_controls' for s, l in x['sents'] if l == 0]
    neg_kept = [s for r, x in zip(rows, scored) if r['slot'] == 'mixed' for s, l in x['sents'] if l == 0]  # retained human text beside AI spans
    thr_here = float(np.sort(neg)[::-1][int(np.floor(.01 * len(neg)))])
    groups = defaultdict(list)
    for r, x in zip(rows, scored):
        for s, l in x['sents']:
            if l == 1:
                groups[('writer', r['writer'])].append(s); groups[('cohort', r['condition'])].append(s); groups[('all', 'all')].append(s)
    out = {'tag': tag, 'checkpoint': Path(ckpt).name, 'seconds': round(time.time() - t0), 'n_human_sentences': len(neg),
           'threshold_1pct_here': thr_here, 'threshold_in_domain': thr_in, 'n_in_domain_human_sentences': n_in,
           'human_fpr_at_in_domain_threshold': float(np.mean(np.asarray(neg) > thr_in)) if thr_in is not None else None,
           'control_fpr_at_in_domain_threshold': float(np.mean(np.asarray(neg_ctrl) > thr_in)) if thr_in is not None else None,
           'retained_human_fpr_at_in_domain_threshold': float(np.mean(np.asarray(neg_kept) > thr_in)) if thr_in is not None else None,
           'n_control_sentences': len(neg_ctrl), 'n_retained_human_sentences': len(neg_kept),
           'groups': {}}
    for (kind, gname), pos in sorted(groups.items()):
        out['groups'][f'{kind}:{gname}'] = {'n_ai_sentences': len(pos), 'auroc': auroc(pos, neg), 'recall_at_1pct_here': recall_at(pos, neg),
                                           'auroc_vs_controls': auroc(pos, neg_ctrl), 'recall_at_1pct_controls': recall_at(pos, neg_ctrl),
                                           'recall_at_in_domain_threshold': float(np.mean(np.asarray(pos) > thr_in)) if thr_in is not None else None}
    docs_ai = [x['doc'] for r, x in zip(rows, scored) if r['slot'] == 'mixed']; docs_h = [x['doc'] for r, x in zip(rows, scored) if r['slot'] == 'human_controls']
    out['documents'] = {'auroc_mixed_vs_controls': auroc(docs_ai, docs_h)}
    (d / 'eval').mkdir(exist_ok=True); (d / 'eval' / f'{name}-{ck_name}.json').write_text(json.dumps(out, indent=1))
    np.savez_compressed(d / 'eval' / f'{name}-{ck_name}-sentences.npz', doc=np.asarray([x['doc'] for x in scored], np.float32),
                        **{str(i): np.asarray(x['sents'], np.float32).reshape(-1, 2) for i, x in enumerate(scored)})
    (d / 'eval' / f'{name}-{ck_name}-rows.json').write_text(json.dumps([{k: r[k] for k in ('id', 'slot', 'split', 'condition', 'writer')} for r in rows]))
    print(json.dumps({k: v for k, v in out.items() if k != 'groups'}), flush=True)
    for k, v in out['groups'].items():
        print(k, json.dumps(v), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:6])
