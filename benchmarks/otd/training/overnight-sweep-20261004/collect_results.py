"""Pull sweep run/eval summaries from the Space and the H200 into results/ (small JSON only) and summarize arms."""
import json, os, subprocess, sys, statistics, io, contextlib
from pathlib import Path

HERE = Path(__file__).resolve().parent; OUT = HERE / 'results'; OUT.mkdir(exist_ok=True)
GATHER = r'''
import json
from pathlib import Path
S=Path(ROOT)/'sweeps';out={}
for d in sorted(S.iterdir()):
    if not d.is_dir() or d.name.startswith(('_','smoke','evaltest')):continue
    r={'status':json.loads((d/'status.json').read_text()) if (d/'status.json').exists() else None}
    for k in ['run.json','history.json','stage2-selection.json','wandb-tracking.json']:
        if (d/k).exists():
            v=json.loads((d/k).read_text());r[k]=v if k!='run.json' else {'config':v['config'],'tag':v.get('tag'),'model':v['model']['name']}
    r['eval']={f.stem:json.loads(f.read_text()) for f in sorted((d/'eval').glob('*.json'))} if (d/'eval').exists() else {}
    out[d.name]=r
print('<J>'+json.dumps(out)+'</J>')
'''


def from_space():
    sys.path.insert(0, os.path.expanduser('~/.config/open-text-detector'))
    from remote import run
    o = io.StringIO()
    with contextlib.redirect_stdout(o):
        run(GATHER.replace('ROOT', repr('/tmp/pangram-space-fast10')), timeout=300)
    s = o.getvalue(); return json.loads(s[s.index('<J>') + 3:s.index('</J>')])


def from_h200():
    code = GATHER.replace('ROOT', repr('/workspace/woog/pangram/backbones-20261003'))
    s = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'otd-h200', '/workspace/woog/pangram/backbones-20261003/venv/bin/python -'],
                       input=code, capture_output=True, text=True, timeout=300).stdout
    return json.loads(s[s.index('<J>') + 3:s.index('</J>')])


KEYS = [('edits_one', 'recall_at_1pct'), ('edits_two', 'recall_at_1pct'), ('edits_small', 'recall_at_1pct'), ('edits_para', 'recall_at_1pct'),
        ('edits_all', 'recall_at_1pct'), ('edits_small', 'auroc'), ('v3_sentences', 'recall_at_1pct'), ('standalone_rewrites', 'recall_at_1pct'),
        ('public_docs', 'auroc'), ('public_docs', 'recall_at_1pct')]


def pick(run, how):
    """Checkpoint per run: 'loss' = stage-2 best by selection loss; 'dev' = best dev small-edit recall@1%; 'last' = final epoch."""
    ev = {k: v for k, v in run['eval'].items() if k.startswith('stage2')}
    if not ev:
        return None
    if how == 'loss' and run.get('stage2-selection.json'):
        return ev.get(run['stage2-selection.json']['checkpoint'].split('-adapters')[0])
    if how == 'last':
        return ev[sorted(ev)[-1]]
    return max(ev.values(), key=lambda r: r['dev']['edits_small']['recall_at_1pct'] or 0)


def summarize(runs):
    arms = {}
    for tag, r in runs.items():
        if not tag.startswith(('q4b', 'q9b')) or not r['eval']:
            continue
        model, arm = tag.split('-')[0], tag.split('-')[1]
        arms.setdefault((model, arm), []).append(r)
    table = {}
    for (model, arm), rs in sorted(arms.items()):
        row = {'n_seeds': len(rs)}
        for how in ('dev', 'loss', 'last'):
            picks = [pick(r, how) for r in rs]; picks = [p for p in picks if p]
            for sec, met in KEYS:
                vals = [p['test'][sec][met] for p in picks if p['test'][sec][met] is not None]
                if vals:
                    row[f'{how}:{sec}:{met}'] = {'mean': statistics.mean(vals), 'sd': statistics.stdev(vals) if len(vals) > 1 else None, 'values': vals}
        table[f'{model}-{arm}'] = row
    return table


if __name__ == '__main__':
    runs = {}
    for name, f in [('space', from_space), ('h200', from_h200)]:
        try:
            got = f()
            for v in got.values():
                v['host'] = 'H200' if name == 'h200' else 'A100'
            runs.update(got); print(name, len(got), 'runs')
        except Exception as e:
            print(name, 'FAILED', repr(e)[:200])
    (OUT / 'runs.json').write_text(json.dumps(runs, indent=1))
    table = summarize(runs); (OUT / 'arms.json').write_text(json.dumps(table, indent=1))
    for arm, row in table.items():
        k = 'dev:edits_small:recall_at_1pct'
        if k in row:
            print(arm, row['n_seeds'], 'small-edit R@1%', round(row[k]['mean'], 3), row[k]['values'])
