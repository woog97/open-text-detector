"""Score one final run the way the T2/T2.1 runs were scored (H200 or Space; run from the final-20261007 work dir).

Usage: eval_final.py TAG [WORKERS]   (CUDA_VISIBLE_DEVICES picks the GPU; inputs in ./eval-inputs from eval_inputs.py;
       FAST_EVAL=1 for the packed fast path; WORKERS: start that many copies, one per GPU, to share the jobs)

1. Sweep eval (sweep_eval.score_checkpoint/metrics) on every stage-2 epoch checkpoint -> sweeps/TAG/eval/<ck>.json and
   <ck>-sentences.npz. These also give each checkpoint's in-domain threshold (1% FPR on dev human sentences).
2. For the selected checkpoint (stage2-selection.json) and stage2-epoch2 (the checkpoint T2/T2.1 reported), the
   cross_model_eval sets: heldout-writers, t21-heldout-v2, fable-{polish,humanizer,cowrite}, cross-model-s500 and the
   new wide1-heldout (DeepSeek held-out papers, GLM and MiMo).
3. sweeps/TAG/eval/summary.json: headline numbers per checkpoint.
Same scoring rules as sweep_eval.py (BF16, sentence = mean token probability). Differences from the T2.1 runner: the
model is loaded once, the head width / encoder boundary tokens follow modeling_sweep and data.bounds (EmbeddingGemma 2),
and nothing is pruned. Finished outputs are skipped on rerun.
"""
import gzip, json, os, sys, time
from pathlib import Path
import numpy as np
import sweep_eval as se
import modeling_sweep as ms

se.Detector = ms.Detector; se.collate = ms.collate  # the runs were trained with modeling_sweep (768-wide heads on EmbeddingGemma 2)
_load = se.load_model
_cache = {}


FAST = os.environ.get('FAST_EVAL', '')  # '1': packed windows, routed-token expert LoRA; 'merged': LoRA folded into the weights (fast_eval.py; needs ccdshim on PYTHONPATH)


def load_model(run):
    if 'm' not in _cache:
        model, tok = _load(run)
        if tok.cls_token_id is None: tok.cls_token = tok.bos_token  # data.bounds: BOS/EOS when a tokenizer has no CLS/SEP
        if tok.sep_token_id is None: tok.sep_token = tok.eos_token
        if FAST and model.kind == 'causal':
            import fast_eval
            fast_eval.prepare(model, merge=FAST == 'merged')
            se.predict = fast_eval.predict; se.score_checkpoint = cm.score_checkpoint = fast_eval.score_checkpoint
        _cache['m'] = (model, tok)
    return _cache['m']


se.load_model = load_model
import cross_model_eval as cm  # noqa: E402
cm.load_model = load_model

I = Path(os.environ.get('EVAL_INPUTS', 'eval-inputs'))  # e.g. eval-inputs-1020 for the long-context sets
SUFFIX = os.environ.get('EVAL_SUFFIX', '')  # appended to set names, e.g. '-lc', so long-context results never overwrite the standard ones
SETS0 = [('heldout-writers', I / 'heldout-score-rows.jsonl.gz', '0'), ('t21-heldout-v2', I / 't21-heldout-score-rows-v2.jsonl.gz', '0'),
        ('fable-polish', I / 'polish-eval-v1.jsonl.gz', '0'), ('fable-humanizer', I / 'humanizer-eval-v1.jsonl.gz', '0'),
        ('fable-cowrite', I / 'cowrite-eval-v1.jsonl.gz', '0'), ('cross-model-s500', I / 'hetero', '500'),
        ('wide1-heldout', I / 'wide1-heldout-rows.jsonl.gz', '0')]
SETS = [(n + SUFFIX, p, smp) for n, p, smp in SETS0]


def claim(path):
    """Atomically take a job (several workers may share one run). Returns False when another worker holds it."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, f'{os.uname().nodename} {os.getpid()}'.encode()); os.close(fd); return True
    except FileExistsError:
        return False


def main(tag, workers=1):
    """workers > 1: run this same command in that many processes (one GPU each); jobs are shared through claim files in
    eval/ (a crashed worker leaves its .claim behind: delete stale claims before rerunning). The last worker to finish
    writes summary.json."""
    d = se.S / tag; ev = d / 'eval'; ev.mkdir(exist_ok=True); run = json.loads((d / 'run.json').read_text())
    log = open(ev / 'eval-final.log', 'a')
    def say(**k): log.write(json.dumps({'t': time.strftime('%H:%M:%S'), **k}) + '\n'); log.flush(); print(json.dumps(k), flush=True)
    rows = [json.loads(l) for l in gzip.open(se.S / 'sweep-eval-rows.jsonl.gz', 'rt')]
    model, tok = load_model(run); say(event='loaded', tag=tag)
    sel = json.loads((d / 'stage2-selection.json').read_text())['checkpoint'].split('-adapters')[0]
    cks = [c for c in dict.fromkeys([sel, 'stage2-epoch2']) if (d / f'{c}-adapters.safetensors').exists()]  # single-pass runs have only epoch0
    sweeps = [p.name.split('-adapters')[0] for p in sorted(d.glob('stage2-epoch*-adapters.safetensors'))]
    if os.environ.get('EVAL_STEPS'):  # also sweep-eval the --ckpt-fracs step checkpoints (learning curve of a single-pass run)
        sweeps += [p.name.split('-adapters')[0] for p in sorted(d.glob('step*-adapters.safetensors'))]
    sets = [(ck, name, path, sample) for ck in cks for name, path, sample in SETS if path.exists()]
    for name in sweeps:  # sweep jobs first: the set jobs need their checkpoint's in-domain threshold
        if (ev / f'{name}-sentences.npz').exists() or (workers > 1 and not claim(ev / f'{name}.claim')): continue
        t0 = time.time(); scored = se.score_checkpoint(model, tok, d / f'{name}-adapters.safetensors', rows)
        res = {'checkpoint': f'{name}-adapters.safetensors', 'tag': tag, 'seconds': time.time() - t0, **{s: se.metrics(scored, s) for s in ('dev', 'test', 'all')}}
        (ev / f'{name}.json').write_text(json.dumps(res, indent=1))
        np.savez_compressed(ev / f'{name}-sentences.tmp.npz', **{f'{i}': np.asarray(x['sents'], dtype=np.float32).reshape(-1, 2) for i, x in enumerate(scored)},
                            doc=np.asarray([x['doc'] for x in scored], np.float32))
        os.replace(ev / f'{name}-sentences.tmp.npz', ev / f'{name}-sentences.npz')
        say(event='sweep', ck=name, seconds=round(res['seconds']), dev_small_r1=res['dev']['edits_small']['recall_at_1pct'],
            test_small_r1=res['test']['edits_small']['recall_at_1pct'], test_small_auroc=res['test']['edits_small']['auroc'])
    for ck, name, path, sample in sets:
        if (ev / f'{name}-{ck}.json').exists() or (workers > 1 and not claim(ev / f'{name}-{ck}.claim')): continue
        while not (ev / f'{ck}-sentences.npz').exists():
            time.sleep(10)
        t0 = time.time(); cm.main(tag, str(d / f'{ck}-adapters.safetensors'), str(path), name, sample)
        say(event='set', ck=ck, set=name, seconds=round(time.time() - t0))
    needed = [ev / f'{n}.json' for n in sweeps] + [ev / f'{name}-{ck}.json' for ck, name, _, _ in sets]
    if not all(p.exists() for p in needed):
        say(event='worker-done', waiting_for=[p.name for p in needed if not p.exists()]); return
    summ = {'tag': tag, 'selected': sel, 'checkpoints': {}}
    for ck in cks:
        sw = json.loads((ev / f'{ck}.json').read_text()); c = {'sweep': {s: {'small_recall_at_1pct': sw[s]['edits_small']['recall_at_1pct'],
                                                                            'small_auroc': sw[s]['edits_small']['auroc'],
                                                                            'para_recall_at_1pct': sw[s]['edits_para']['recall_at_1pct'],
                                                                            'public_docs_auroc': sw[s]['public_docs']['auroc'],
                                                                            'standalone_auroc': sw[s]['standalone_rewrites']['auroc']} for s in ('dev', 'test')}}
        for name, _, _ in SETS:
            f = ev / f'{name}-{ck}.json'
            if f.exists():
                x = json.loads(f.read_text())
                c[name] = {'control_fpr_in_domain': x['control_fpr_at_in_domain_threshold'], 'doc_auroc': x['documents']['auroc_mixed_vs_controls'],
                           **{g: {'n': v['n_ai_sentences'], 'recall_in_domain': v['recall_at_in_domain_threshold'], 'auroc': v['auroc_vs_controls']}
                              for g, v in x['groups'].items() if g.startswith(('all:', 'writer:'))}}
        summ['checkpoints'][ck] = c
    (ev / f'summary{SUFFIX}.json').write_text(json.dumps(summ, indent=1)); say(event='done', tag=tag)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1)
